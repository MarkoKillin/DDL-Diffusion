"""
Training utilities for the latent diffusion U-Net.

Owns:
  - EMA model creation + update
  - Optimizer + LR schedule factories
  - AMP dtype selection
  - CFG dropout helper
  - leakage-safe split and the latent/caption dataset
  - train_step (one optimizer step end-to-end)
  - validation_loss (eval-mode, fixed-seed, comparable across epochs)
  - checkpoint save/load

Notes on two defaults:

  min_snr_gamma=None, with v-prediction
      Unweighted v-loss is already balanced across timesteps: at the Bayes optimum it is
      1.0 at every t. Unweighted eps-loss equals alpha_bar_t instead, which puts about a
      third of its signal below t=100 and almost none above t=750 — and above t=500 is
      where global composition is decided. Min-SNR-gamma is available but is the right tool
      for eps, not v; on top of v it re-concentrates the signal at low noise.

  pick_amp_dtype()
      bf16 needs compute capability >= 8.0, so it is unavailable on a T4 (7.5) rather than
      free. This picks bf16 only where the hardware supports it and falls back to fp16.
"""

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# EMA (Exponential Moving Average)
def build_ema(model: nn.Module) -> nn.Module:
    """
    Deep-copied shadow of the model with grads disabled.

    The EMA tracks a smoothed version of the weights over training. Sample from it rather
    than the live model: smoother outputs, less sensitive to the last noisy gradient step.
    """
    ema = copy.deepcopy(model)
    ema.requires_grad_(False)
    ema.eval()
    return ema


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999, step: int | None = None) -> None:
    """
    In-place EMA update:   ema_param = decay * ema_param + (1 - decay) * live_param

    decay=0.9999 absorbs 0.01% of the new weights per step, an averaging window of roughly
    10,000 steps.

    The EMA starts from the random init, so a fixed high decay leaves the shadow weights
    mostly random for thousands of steps (90% random init at step 1000) and early previews
    are noise no matter how training is going. Passing `step` applies the usual warmup ramp

        effective_decay = min(decay, (1 + step) / (10 + step))

    so the EMA tracks the live model closely at first and eases into `decay`. Omit `step`
    for fixed decay.

    The ramp is the binding term until about step 10/(1-decay). Past that, check
    ||ema - live|| / ||live|| before trusting samples.
    """
    if step is not None:
        decay = min(decay, (1.0 + step) / (10.0 + step))

    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.data, alpha=1.0 - decay)

    # Keep buffers in sync too, in case the live model's ever drift. Cheap.
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


# Optimizer + LR schedule
def make_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """AdamW is the default for diffusion. Weight decay helps generalization on small datasets."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_lr_schedule(optimizer, num_warmup_steps: int, num_training_steps: int):
    """
    Linear warmup from 0 -> lr over num_warmup_steps, then cosine decay back to 0.

    Warmup prevents large early gradients from destabilizing GroupNorm stats.
    Cosine decay is the standard "good enough" annealing for fixed-budget runs.
    """
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Mixed precision
def pick_amp_dtype(prefer: str = "auto", device: str = "cuda", verbose: bool = True):
    """
    Choose an autocast dtype that the hardware actually supports.

    bf16 requires compute capability >= 8.0 (Ampere and later); a Colab T4 is 7.5. fp16
    works there but needs a GradScaler, which train_step uses when you pass one.

    Returns (amp_dtype_or_None, label).
    """
    if prefer == "none" or device != "cuda" or not torch.cuda.is_available():
        return None, "none"

    major = torch.cuda.get_device_capability()[0]
    bf16_ok = major >= 8 and torch.cuda.is_bf16_supported()

    if prefer == "bf16" or prefer == "auto":
        if bf16_ok:
            return torch.bfloat16, "bf16"
        if prefer == "bf16" and verbose:
            print(f"bf16 unsupported on {torch.cuda.get_device_name()} (sm_{major}x) — falling back to fp16")
        return torch.float16, "fp16"

    if prefer == "fp16":
        return torch.float16, "fp16"

    raise ValueError(f"unknown AMP preference {prefer!r} (use 'auto', 'bf16', 'fp16' or 'none')")


# Classifier-Free Guidance dropout
def apply_cfg_dropout(
    embeddings: torch.Tensor,
    uncond_embedding: torch.Tensor,
    dropout_prob: float = 0.1,
) -> torch.Tensor:
    """
    With probability dropout_prob, replace each row's text embedding with the uncond embedding.

    Trains the U-Net to denoise both conditionally and unconditionally with the same
    weights, which is what CFG needs at inference.

    At small batch sizes this yields well under one uncond row per step, so it averages
    out only slowly.

    Args:
        embeddings        : (B, 77, 768)
        uncond_embedding  : (1, 77, 768)
    Returns:
        (B, 77, 768) with some rows swapped out.
    """
    B = embeddings.shape[0]
    mask = torch.rand(B, device=embeddings.device) < dropout_prob          # (B,)
    uncond_expanded = uncond_embedding.expand(B, -1, -1)                   # (B, 77, 768)
    # mask[:, None, None] broadcasts over the (77, 768) trailing dims.
    return torch.where(mask[:, None, None], uncond_expanded, embeddings)


# Train / validation split
def make_split(
    group_ids: torch.Tensor,
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Leakage-safe train/val split over the encoded latents.

    Splits on group (the source image), never on row, so every augmented view of one
    Pokemon — all crops, both flips — lands on the same side. Without that, validation
    holds crops and mirrors of training images and its loss measures nothing.

    Args:
        group_ids : (N_lat,) from latent_stats["group_ids"]
    Returns:
        (train_rows, val_rows) as sorted long tensors of row indices into latents.pt.
    """
    groups = torch.unique(group_ids)
    g = torch.Generator().manual_seed(seed)
    order = groups[torch.randperm(len(groups), generator=g)]

    n_val = int(round(len(groups) * val_frac))
    val_groups = set(order[:n_val].tolist())

    is_val = torch.tensor([int(gid) in val_groups for gid in group_ids.tolist()])
    val_rows = torch.nonzero(is_val, as_tuple=True)[0]
    train_rows = torch.nonzero(~is_val, as_tuple=True)[0]
    return train_rows, val_rows


class LatentCaptionDataset(torch.utils.data.Dataset):
    """
    Joins latents to caption embeddings by group id, and samples a caption variant.

    Embeddings are stored once per (image, caption) and joined here rather than duplicated
    per crop, because a token x 768 embedding is large: at 32 tokens, 5 captions each for
    8,189 Flowers images is 2.0 GB in fp16, and duplicating that across crops and flips
    would multiply it again. Notebook writes fp16; __getitem__ casts back to fp32.

    sample_variant=True picks uniformly among that image's captions on every __getitem__,
    so the caption augmentation is fresh each epoch rather than a fixed pairing. Set False
    for deterministic evaluation (always caption 0).

    Args:
        latents        : (N_lat, C, H, W)
        group_ids      : (N_lat,)   source image per latent
        embeddings     : (N_cap, L, 768)
        caption_groups : (N_cap,)   source image per caption
    """

    def __init__(self, latents, group_ids, embeddings, caption_groups,
                 view_params=None, sample_variant: bool = True):
        self.latents = latents
        self.embeddings = embeddings
        self.group_ids = group_ids
        self.sample_variant = sample_variant
        # (N_lat, view_dim) micro-conditioning: which crop box / flip this latent is.
        # Zeros when unused, which the U-Net treats as "no view conditioning".
        self.view_params = view_params

        # group -> LongTensor of caption rows. Variant order is preserved, so column 0
        # is always the original caption.
        n_groups = int(caption_groups.max().item()) + 1
        counts = torch.bincount(caption_groups, minlength=n_groups)
        if counts.min() == 0:
            missing = int((counts == 0).nonzero()[0])
            raise ValueError(f"image {missing} has no captions")
        if counts.min() != counts.max():
            raise ValueError(f"ragged caption variants per image: {counts.min()}..{counts.max()}")

        self.n_variants = int(counts[0].item())
        table = torch.empty(n_groups, self.n_variants, dtype=torch.long)
        fill = torch.zeros(n_groups, dtype=torch.long)
        for row, gid in enumerate(caption_groups.tolist()):
            table[gid, fill[gid]] = row
            fill[gid] += 1
        self.caption_table = table

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, i):
        gid = int(self.group_ids[i])
        row = self.caption_table[gid]
        j = row[torch.randint(self.n_variants, (1,)).item()] if self.sample_variant else row[0]
        view = self.view_params[i] if self.view_params is not None else torch.zeros(0)
        # embeddings.pt is stored fp16 (10 captions per image over 8k images is 4 GB in
        # fp32), so cast here. The model's params are fp32 and autocast does not cover a
        # tensor arriving from the DataLoader.
        return self.latents[i], self.embeddings[j].float(), view


# Loss
def diffusion_loss(
    model: nn.Module,
    noise_scheduler,
    x_0: torch.Tensor,
    context: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    min_snr_gamma: float | None = None,
    view: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Weighted MSE against whatever the scheduler says the target is.

    Returns a scalar. Per-sample MSE is computed first, then weighted, so the weighting is
    per-timestep rather than smeared across the batch.

    `view` is the augmentation micro-conditioning (crop box + flip); see unet.UNet's
    view_dim note for why it matters.
    """
    x_t = noise_scheduler.q_sample(x_0, t, noise)
    target = noise_scheduler.get_target(x_0, noise, t)

    pred = model(x_t, t, context, view)

    per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
    weight = noise_scheduler.loss_weight(t, min_snr_gamma).to(per_sample)
    return (per_sample * weight).mean()


# Training step
def train_step(
    model: nn.Module,
    ema_model: nn.Module,
    noise_scheduler,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    x_0: torch.Tensor,
    context: torch.Tensor,
    uncond_embedding: torch.Tensor,
    cfg_dropout_prob: float = 0.1,
    grad_clip: float = 1.0,
    ema_decay: float = 0.9999,
    min_snr_gamma: float | None = None,
    view: torch.Tensor | None = None,
    amp_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
    step: int | None = None,
) -> float:
    """
    One optimizer step. Returns the scalar loss.

    Inputs:
        x_0              : (B, 4, H, W)     clean latents (per-channel normalized)
        context          : (B, 77, 768)     text embeddings
        uncond_embedding : (1, 77, 768)     empty-string embedding (for CFG dropout)
        min_snr_gamma    : Min-SNR-gamma weighting. None = unweighted.
        amp_dtype        : if set, wraps forward+loss in autocast. Params stay fp32.
        scaler           : required when amp_dtype is torch.float16. fp16 gradients
                           underflow to zero without loss scaling; bf16 has fp32's
                           exponent range and needs no scaler, so pass None there.
        step             : global step, forwarded to ema_update for the warmup ramp.
    """
    model.train()
    B = x_0.shape[0]
    device = x_0.device

    # 1. Sample timesteps uniformly across the schedule. A different t per sample gives
    #    the model exposure to the whole noise range every batch. Any non-uniform emphasis
    #    across t comes from min_snr_gamma, not from biasing this draw.
    t = torch.randint(0, noise_scheduler.T, (B,), device=device, dtype=torch.long)

    # 2. Sample fresh Gaussian noise.
    noise = torch.randn_like(x_0)

    # 3. CFG dropout: randomly drop the text condition on some rows.
    context = apply_cfg_dropout(context, uncond_embedding, cfg_dropout_prob)

    # 4. Forward noising, prediction, weighted MSE.
    if amp_dtype is not None:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            loss = diffusion_loss(model, noise_scheduler, x_0, context, t, noise,
                                  min_snr_gamma, view)
    else:
        loss = diffusion_loss(model, noise_scheduler, x_0, context, t, noise,
                              min_snr_gamma, view)

    # 5. Backward + optimizer step. Grad clip prevents the occasional spike
    #    from blowing up training (mostly an issue early on with random init).
    optimizer.zero_grad(set_to_none=True)

    if scaler is not None:
        scaler.scale(loss).backward()
        # Unscale first so grad_clip measures true gradient norms, not scaled ones.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    lr_scheduler.step()

    # 6. Update the EMA shadow.
    ema_update(ema_model, model, ema_decay, step=step)

    return loss.item()


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    noise_scheduler,
    x_0: torch.Tensor,
    context: torch.Tensor,
    min_snr_gamma: float | None = None,
    view: torch.Tensor | None = None,
    batch_size: int = 32,
    seed: int = 1234,
) -> dict[str, float]:
    """
    Eval-mode loss on a fixed set of latents, with a fixed t and noise draw.

    Both matter: eval mode so dropout does not inflate the number, and a fixed (t, noise)
    draw so epoch-to-epoch changes are the model changing rather than which timesteps
    happened to come up.

    Returns the weighted loss (comparable to the training number) and the unweighted one
    (comparable to published eps-MSE figures).
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    g = torch.Generator().manual_seed(seed)
    n = x_0.shape[0]
    t_all = torch.randint(0, noise_scheduler.T, (n,), generator=g)
    noise_all = torch.randn(x_0.shape, generator=g)

    tot_w, tot_u, seen = 0.0, 0.0, 0
    for i in range(0, n, batch_size):
        xb = x_0[i : i + batch_size].to(device)
        cb = context[i : i + batch_size].to(device)
        tb = t_all[i : i + batch_size].to(device)
        nb = noise_all[i : i + batch_size].to(device)

        vb = view[i : i + batch_size].to(device) if view is not None else None
        x_t = noise_scheduler.q_sample(xb, tb, nb)
        target = noise_scheduler.get_target(xb, nb, tb)
        pred = model(x_t, tb, cb, vb)

        per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
        w = noise_scheduler.loss_weight(tb, min_snr_gamma).to(per_sample)

        tot_w += (per_sample * w).sum().item()
        tot_u += per_sample.sum().item()
        seen += xb.shape[0]

    if was_training:
        model.train()
    return {"weighted": tot_w / seen, "unweighted": tot_u / seen}


# Checkpointing
def save_checkpoint(
    path: str,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    step: int,
    epoch: int,
    loss_history: list | None = None,
    scaler: torch.amp.GradScaler | None = None,
    config: dict | None = None,
    slim: bool = False,
) -> None:
    """
    Save everything needed to resume training.

    `config` should carry the model and scheduler kwargs so a checkpoint can be rebuilt
    without guessing what it was trained with.

    slim=True drops the optimizer, LR-schedule and scaler state — everything only needed to
    resume. A full checkpoint holds four copies of the weights (model, EMA, and AdamW's two
    moments), which at 35M params is over 500 MB each and fills a Drive quickly. Keep one
    full checkpoint as the resume point and make the rest slim; slim ones still load in
    every eval cell.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "step": step,
        "epoch": epoch,
        "loss_history": loss_history or [],
        "config": config or {},
        "slim": slim,
    }
    if not slim:
        payload.update({
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
        })
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    map_location: str = "cpu",
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, int, list]:
    """
    Restore states in place. Returns (step, epoch, loss_history).

    Raises on a slim checkpoint, which has no optimizer state and cannot resume — load
    its weights directly instead (`ckpt["ema_model"]`), as the eval cells do.
    """
    ckpt = torch.load(path, map_location=map_location)
    if ckpt.get("slim") or "optimizer" not in ckpt:
        raise ValueError(
            f"{path} is a slim checkpoint (no optimizer state) and cannot resume training. "
            f"Use it for evaluation via ckpt['ema_model'], or resume from a full one."
        )
    model.load_state_dict(ckpt["model"])
    ema_model.load_state_dict(ckpt["ema_model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"], ckpt["epoch"], ckpt.get("loss_history", [])