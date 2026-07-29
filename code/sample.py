"""
Inference samplers for the latent diffusion U-Net.

DDPM = the slow, faithful sampler from the original paper. 1000 steps.
DDIM = the fast deterministic sampler. ~50 steps, same trained model.

Both use Classifier-Free Guidance: each step batches the conditional and
unconditional predictions through one forward pass, then combines them as
    eps = eps_uncond + w * (eps_cond - eps_uncond)
The model itself is unchanged — CFG lives entirely in the sampler.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# CFG helper
def _predict_eps_with_cfg(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    One forward pass that batches uncond + cond together, then combines.

    Shapes:
        x_t        : (B, 4, 64, 64)
        t          : (B,)
        cond_emb   : (B, 77, 768)
        uncond_emb : (B, 77, 768)   already broadcast to batch size B

    Returns predicted noise (B, 4, 64, 64) after CFG combine.
    """
    x_in = torch.cat([x_t, x_t], dim=0)
    t_in = torch.cat([t, t], dim=0)
    ctx_in = torch.cat([uncond_emb, cond_emb], dim=0)

    eps_both = model(x_in, t_in, ctx_in)
    eps_uncond, eps_cond = eps_both.chunk(2, dim=0)

    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


# Timestep grid
def _make_timesteps(T: int, num_steps: int, device: torch.device | str) -> torch.Tensor:
    """
    Decreasing timestep grid of length num_steps, always spanning the full [0, T-1] range.

    Sampling MUST start at t = T-1, where the schedule expects pure noise — which is
    what we initialize x to. Asking for fewer steps means a COARSER STRIDE over the
    whole schedule, not a shorter walk through its low-noise tail.
    """
    if num_steps >= T:
        return torch.arange(T - 1, -1, -1, device=device)
    return torch.linspace(T - 1, 0, num_steps, device=device).long()


# DDPM sampler (slow, 1000 steps)
@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    scheduler,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    guidance_scale: float = 7.5,
    num_steps: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate latents by reversing the DDPM noising process.

    Math per transition t -> prev (prev is the next entry in the timestep grid):
        alpha_eff = alpha_bar_t / alpha_bar_prev      # signal kept over this jump
        beta_eff  = 1 - alpha_eff
        mean      = (1 / sqrt(alpha_eff)) * (x_t - (beta_eff / sqrt(1 - alpha_bar_t)) * eps)
        x_prev    = mean + sqrt(beta_eff) * z         except on the final step
        x_0       = mean                              on the final step

    Written in terms of alpha_bar ratios so a strided grid (num_steps < T) is handled
    correctly. On the full 1000-step grid alpha_eff and beta_eff reduce exactly to
    alphas[t] and betas[t], so this is identical to the textbook per-step update.

    Args:
        model         : the U-Net (typically the EMA model in eval mode)
        scheduler     : NoiseScheduler, holds beta / alpha / alpha_bar buffers
        cond_emb      : (B, 77, 768) text embeddings per sample
        uncond_emb    : (1, 77, 768) or (B, 77, 768) empty-string embedding
        guidance_scale: w in CFG. 7.5 is the SD default.
        num_steps     : how many denoising steps; defaults to scheduler.T (the full 1000).
                        Fewer steps stride the whole schedule rather than truncating it.
        generator     : torch.Generator for reproducible sampling

    Returns:
        (B, 4, 64, 64) latents, still in scaled space (divide by 0.18215
        before passing to vae.decode).
    """
    model.eval()
    B = cond_emb.shape[0]
    device = cond_emb.device

    if uncond_emb.shape[0] == 1:
        uncond_emb = uncond_emb.expand(B, -1, -1)

    timesteps = _make_timesteps(scheduler.T, scheduler.T if num_steps is None else num_steps, device)

    x = torch.randn(B, 4, 64, 64, device=device, generator=generator)

    alphas_cumprod = scheduler.alphas_cumprod
    sqrt_one_minus_ab = scheduler.sqrt_one_minus_alphas_cumprod
    one = torch.tensor(1.0, device=device)

    for i, step in enumerate(timesteps):
        is_last = i + 1 == len(timesteps)
        t = torch.full((B,), step.item(), device=device, dtype=torch.long)

        eps = _predict_eps_with_cfg(model, x, t, cond_emb, uncond_emb, guidance_scale)

        ab_t = alphas_cumprod[step]
        ab_prev = one if is_last else alphas_cumprod[timesteps[i + 1]]

        # Effective per-jump alpha/beta. Equals alphas[t] / betas[t] on the full grid.
        alpha_eff = ab_t / ab_prev
        beta_eff = 1.0 - alpha_eff

        mean = (1.0 / torch.sqrt(alpha_eff)) * (x - (beta_eff / sqrt_one_minus_ab[step]) * eps)

        if is_last:
            x = mean
        else:
            z = torch.randn(x.shape, device=device, generator=generator)
            x = mean + torch.sqrt(beta_eff) * z

    return x


# DDIM sampler (fast, deterministic)
@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    scheduler,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Deterministic (eta=0) sampler using a subset of timesteps.

    For each transition tau_i -> tau_{i-1}:
        x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        x_{t-1}  = sqrt(alpha_bar_{t-1}) * x_0_pred + sqrt(1 - alpha_bar_{t-1}) * eps

    With eta=0 we skip the stochastic term entirely — fully deterministic
    given a fixed starting noise.

    Args identical to sample_ddpm except:
        num_steps : how many denoising steps (typically 25-100)
        eta       : 0.0 = deterministic DDIM, 1.0 = stochastic (DDPM-like)
    """
    model.eval()
    B = cond_emb.shape[0]
    device = cond_emb.device

    if uncond_emb.shape[0] == 1:
        uncond_emb = uncond_emb.expand(B, -1, -1)

    # Evenly-spaced subset, decreasing from T-1 down to 0.
    timesteps = _make_timesteps(scheduler.T, num_steps, device)

    alphas_cumprod = scheduler.alphas_cumprod

    x = torch.randn(B, 4, 64, 64, device=device, generator=generator)

    for i, step in enumerate(timesteps):
        is_last = i + 1 == len(timesteps)
        t = torch.full((B,), step.item(), device=device, dtype=torch.long)

        eps = _predict_eps_with_cfg(model, x, t, cond_emb, uncond_emb, guidance_scale)

        ab_t = alphas_cumprod[step]
        ab_prev = torch.tensor(1.0, device=device) if is_last else alphas_cumprod[timesteps[i + 1]]

        # Estimate x_0 from current x_t and predicted noise.
        x0_pred = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)

        # Stochastic term — sigma=0 by default (eta=0).
        sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)) if eta > 0 else torch.tensor(0.0, device=device)

        # Direction pointing to x_t.
        dir_xt = torch.sqrt(1 - ab_prev - sigma ** 2) * eps

        if eta > 0 and not is_last:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = torch.sqrt(ab_prev) * x0_pred + dir_xt + sigma * noise
        else:
            x = torch.sqrt(ab_prev) * x0_pred + dir_xt

    return x


# Decode helpers (require a VAE)
@torch.no_grad()
def latents_to_images(latents: torch.Tensor, vae, scale: float = 0.18215) -> torch.Tensor:
    """
    Decode latents back to RGB images in [0, 1].

    Args:
        latents : (B, 4, 64, 64) in scaled space
        vae     : a diffusers AutoencoderKL in eval mode
        scale   : the 0.18215 SD scaling factor — we divide before decoding.

    Returns:
        (B, 3, 512, 512) float tensor in [0, 1] on the same device as the vae.
    """
    decoded = vae.decode(latents / scale).sample
    return (decoded.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def sample_to_image(
    model: nn.Module,
    scheduler,
    vae,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    method: str = "ddim",
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    End-to-end: text embedding -> noise -> denoise -> VAE decode -> RGB.

    Returns (B, 3, 512, 512) in [0, 1]. Useful as a training-time preview hook.
    """
    if method == "ddim":
        latents = sample_ddim(
            model, scheduler, cond_emb, uncond_emb,
            guidance_scale=guidance_scale, num_steps=num_steps, generator=generator,
        )
    elif method == "ddpm":
        latents = sample_ddpm(
            model, scheduler, cond_emb, uncond_emb,
            guidance_scale=guidance_scale, num_steps=num_steps, generator=generator,
        )
    else:
        raise ValueError(f"unknown sampling method: {method!r} (use 'ddim' or 'ddpm')")

    return latents_to_images(latents, vae)