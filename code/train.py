"""
Training utilities for the latent diffusion U-Net: EMA, optimizer and LR factories, AMP
selection, CFG dropout, the leakage-safe split, the latent/caption dataset, train_step,
validation_loss and checkpoint save/load.

Loss weighting lives in scheduler.loss_weight, including why min_snr_gamma stays None for
v-prediction.
"""

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# EMA (Exponential Moving Average)
def build_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model)
    ema.requires_grad_(False)
    ema.eval()
    return ema


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999, step: int | None = None) -> None:
    """
    ema_param = decay * ema_param + (1 - decay) * live_param
    decay=0.9999 averages over roughly 10,000 steps.
    """
    if step is not None:
        decay = min(decay, (1.0 + step) / (10.0 + step))

    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.data, alpha=1.0 - decay)

    # Keep buffers in sync too, in case the live model's ever drift
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


# Optimizer + LR schedule
def make_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_lr_schedule(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Linear warmup from 0 -> lr over num_warmup_steps, then cosine decay back to 0."""
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Mixed precision
def pick_amp_dtype(prefer: str = "auto", device: str = "cuda", verbose: bool = True):
    """
    Returns (amp_dtype_or_None, label). bf16 needs compute capability >= 8.0, so a T4 (7.5)
    falls back to fp16, which needs the GradScaler train_step takes.
    """
    if prefer == "none" or device != "cuda" or not torch.cuda.is_available():
        return None, "none"

    major = torch.cuda.get_device_capability()[0]
    bf16_ok = major >= 8 and torch.cuda.is_bf16_supported()

    if prefer == "bf16" or prefer == "auto":
        if bf16_ok:
            return torch.bfloat16, "bf16"
        if prefer == "bf16" and verbose:
            print(f"bf16 unsupported on {torch.cuda.get_device_name()} (sm_{major}x), falling back to fp16")
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
    Replaces each row's text embedding with the uncond embedding at probability
    dropout_prob, so the U-Net learns to denoise with and without text.

    embeddings (B, 77, 768), uncond_embedding (1, 77, 768). Returns (B, 77, 768).
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
    Joins latents to caption embeddings by group id, and picks a caption per __getitem__.
    Embeddings are stored once per (image, caption) rather than duplicated per crop: at 32 tokens.

    sample_variant=True picks uniformly among the image's captions each time, so the caption
    augmentation is fresh per epoch. False always gives caption 0, for deterministic eval.
    """
    def __init__(self, latents, group_ids, embeddings, caption_groups,
                 view_params=None, sample_variant: bool = True):
        self.latents = latents
        self.embeddings = embeddings
        self.group_ids = group_ids
        self.sample_variant = sample_variant
        # (N_lat, view_dim) crop box and flip per latent. Zeros when unused.
        self.view_params = view_params

        # group -> LongTensor of caption rows. Column 0 is always the original caption.
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
    Weighted MSE against the scheduler's target. Returns a scalar.
    Per-sample MSE first, then the weight, so the weighting is per-timestep rather than smeared across the batch.
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

        x_0              : (B, 4, H, W)     clean latents, per-channel normalized
        context          : (B, 77, 768)     text embeddings
        uncond_embedding : (1, 77, 768)     empty-string embedding, for CFG dropout
        amp_dtype        : if set, wraps forward and loss in autocast. Params stay fp32.
        scaler           : required for fp16, whose gradients underflow without loss
                           scaling. bf16 has fp32's exponent range, so pass None there.
        step             : forwarded to ema_update for the warmup ramp.
    """
    model.train()
    B = x_0.shape[0]
    device = x_0.device

    t = torch.randint(0, noise_scheduler.T, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(x_0)
    context = apply_cfg_dropout(context, uncond_embedding, cfg_dropout_prob)
    if amp_dtype is not None:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            loss = diffusion_loss(model, noise_scheduler, x_0, context, t, noise,
                                  min_snr_gamma, view)
    else:
        loss = diffusion_loss(model, noise_scheduler, x_0, context, t, noise,
                              min_snr_gamma, view)

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