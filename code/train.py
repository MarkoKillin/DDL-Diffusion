"""
Training utilities for the latent diffusion U-Net.

Owns:
  - EMA model creation + update
  - Optimizer + LR schedule factories
  - CFG dropout helper
  - train_step (one optimizer step end-to-end)
  - checkpoint save/load
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
    Create a deep-copied shadow of the model with grads disabled.

    EMA tracks a smoothed version of model weights over training. At
    inference we use the EMA model, not the live one — smoother outputs,
    less sensitive to the most recent noisy gradient step.
    """
    ema = copy.deepcopy(model)
    ema.requires_grad_(False)
    ema.eval()
    return ema


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
    """
    In-place EMA update:   ema_param = decay * ema_param + (1 - decay) * live_param

    decay=0.9999 means each step the EMA absorbs 0.01% of the new weights.
    Effective averaging window is ~10000 steps. Use 0.999 for shorter runs
    so EMA actually tracks something.
    """
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.data, alpha=1.0 - decay)

    # Buffers (e.g., NoiseScheduler constants) should also be kept in sync
    # in case the live model's buffers somehow drift. Cheap.
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


# Classifier-Free Guidance dropout
def apply_cfg_dropout(
    embeddings: torch.Tensor,
    uncond_embedding: torch.Tensor,
    dropout_prob: float = 0.1,
) -> torch.Tensor:
    """
    With probability dropout_prob, replace each row's text embedding with the uncond embedding.

    This trains the U-Net to denoise both conditionally and unconditionally
    using the SAME weights — required for CFG at inference time.

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
    amp_dtype: torch.dtype | None = None,
) -> float:
    """
    One optimizer step. Returns the scalar loss.

    Inputs:
        x_0              : (B, 4, 64, 64)   clean latents
        context          : (B, 77, 768)     text embeddings
        uncond_embedding : (1, 77, 768)     empty-string embedding (for CFG dropout)
        amp_dtype        : if torch.bfloat16, wraps forward+loss in autocast.
                           Params stay fp32; no GradScaler needed for bf16.
    """
    model.train()
    B = x_0.shape[0]
    device = x_0.device

    # 1. Sample timesteps uniformly across the schedule. Different t per sample
    #    gives the model exposure to the entire noise range every batch.
    t = torch.randint(0, noise_scheduler.T, (B,), device=device, dtype=torch.long)

    # 2. Sample fresh Gaussian noise.
    noise = torch.randn_like(x_0)

    # 3. Forward noising (one shot via q_sample).
    x_t = noise_scheduler.q_sample(x_0, t, noise)

    # 4. CFG dropout: randomly drop the text condition on some rows.
    context = apply_cfg_dropout(context, uncond_embedding, cfg_dropout_prob)

    # 5. Predict noise. MSE against the actual noise.
    if amp_dtype is not None:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            eps_pred = model(x_t, t, context)
            loss = F.mse_loss(eps_pred, noise)
    else:
        eps_pred = model(x_t, t, context)
        loss = F.mse_loss(eps_pred, noise)

    # 6. Backward + optimizer step. Grad clip prevents the occasional spike
    #    from blowing up training (mostly an issue early on with random init).
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    lr_scheduler.step()

    # 7. Update the EMA shadow.
    ema_update(ema_model, model, ema_decay)

    return loss.item()


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
) -> None:
    """Save everything needed to resume training."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "loss_history": loss_history or [],
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    map_location: str = "cpu",
) -> tuple[int, int, list]:
    """Restore states in place. Returns (step, epoch, loss_history)."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    ema_model.load_state_dict(ckpt["ema_model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    return ckpt["step"], ckpt["epoch"], ckpt.get("loss_history", [])
