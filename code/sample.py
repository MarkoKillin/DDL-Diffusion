"""
Inference samplers for the latent diffusion U-Net.

DDPM = the slow, faithful sampler from the original paper. 1000 steps.
DDIM = the fast deterministic sampler. ~50 steps, same trained model.

Both use Classifier-Free Guidance: each step batches the conditional and
unconditional predictions through one forward pass, then combines them as
    pred = pred_uncond + w * (pred_cond - pred_uncond)
The model itself is unchanged — CFG lives entirely in the sampler.

Changes since run 1, and why:

  Both samplers now go through scheduler.to_x0_and_eps()
      so they work with prediction_type="eps" or "v" without knowing which.

  DDPM is written in x_0-posterior form rather than the eps shortcut
      The old form computed mean = (1/sqrt(alpha_eff)) * (...), which divides by
      zero on the first step when zero_terminal_snr makes alphas_cumprod[t] = 0.
      The posterior form is finite everywhere and is algebraically identical on
      the full 1000-step grid.

  guidance_rescale (Lin et al. section 3.4)
      Run 1's measured latent statistics showed CFG inflating the standard
      deviation well past the training data's (std 1.30 at w=7.5 vs 1.12 in the
      data) while INVERTING the per-channel mean pattern (Pearson r vs the data's
      channel means: +0.83 at w=1.0, -0.63 at w=7.5). Rescaling the guided
      prediction back to the conditional prediction's std counteracts that.

  latent_shape is a parameter
      Run 1 hardcoded randn(B, 4, 64, 64) in both samplers, which silently
      blocked training at any other resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scheduler import make_timesteps


# CFG helpers
def _apply_guidance_rescale(pred_cfg: torch.Tensor, pred_cond: torch.Tensor, phi: float) -> torch.Tensor:
    """
    Lin et al. section 3.4. Rescale the guided prediction so its per-sample standard
    deviation matches the plain conditional prediction, then blend by phi.

    phi=0.0 is plain CFG; phi=0.7 is the paper's recommendation. Attacks the
    over-exposure / over-saturation that high guidance causes.
    """
    dims = tuple(range(1, pred_cfg.ndim))
    std_cond = pred_cond.std(dim=dims, keepdim=True)
    std_cfg = pred_cfg.std(dim=dims, keepdim=True).clamp(min=1e-12)
    return phi * (pred_cfg * (std_cond / std_cfg)) + (1.0 - phi) * pred_cfg


def _predict_with_cfg(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    guidance_scale: float,
    guidance_rescale: float = 0.0,
    view: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    One forward pass that batches uncond + cond together, then combines.

    Shapes:
        x_t        : (B, 4, H, W)
        t          : (B,)
        cond_emb   : (B, 77, 768)
        uncond_emb : (B, 77, 768)   already broadcast to batch size B

    Returns the combined model prediction (B, 4, H, W) — eps or v, whichever the
    model was trained on.
    """
    x_in = torch.cat([x_t, x_t], dim=0)
    t_in = torch.cat([t, t], dim=0)
    ctx_in = torch.cat([uncond_emb, cond_emb], dim=0)
    view_in = torch.cat([view, view], dim=0) if view is not None else None

    both = model(x_in, t_in, ctx_in, view_in)
    pred_uncond, pred_cond = both.chunk(2, dim=0)

    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    if guidance_rescale > 0.0:
        pred = _apply_guidance_rescale(pred, pred_cond, guidance_rescale)
    return pred


def _init_latents(B, latent_shape, device, generator, x_T=None):
    """
    Starting noise. Pass x_T to control it explicitly.

    Any test that compares PROMPTS must hold the starting noise fixed across rows —
    otherwise the difference between two samples is mostly the noise draw. Run 1's
    "prompt-independent collapse" check drew fresh noise per row, so its pairwise
    latent distances measured nothing about the prompt.
    """
    if x_T is not None:
        expected = (B, *latent_shape)
        if tuple(x_T.shape) != expected:
            raise ValueError(f"x_T must be {expected}, got {tuple(x_T.shape)}")
        return x_T.to(device)
    return torch.randn(B, *latent_shape, device=device, generator=generator)


# DDPM sampler (slow, 1000 steps)
@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    scheduler,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    guidance_scale: float = 2.0,
    guidance_rescale: float = 0.7,
    num_steps: int | None = None,
    latent_shape: tuple[int, int, int] = (4, 32, 32),
    spacing: str = "trailing",
    clip_x0: float | None = None,
    x_T: torch.Tensor | None = None,
    view: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate latents by reversing the DDPM noising process.

    Per transition t -> prev (prev is the next entry in the timestep grid):
        alpha_eff = alpha_bar_t / alpha_bar_prev        # signal kept over this jump
        beta_eff  = 1 - alpha_eff
        mean      = sqrt(alpha_bar_prev) * beta_eff / (1 - alpha_bar_t)  * x0_pred
                  + sqrt(alpha_eff) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x_t
        var       = beta_eff * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        x_prev    = mean + sqrt(var) * z

    This is the true posterior q(x_prev | x_t, x_0) with x_0 replaced by the model's
    estimate. Written with alpha_bar ratios so a strided grid is handled correctly, and
    in x_0 form so alpha_bar_t = 0 (zero terminal SNR) is finite. On the final step
    alpha_bar_prev = 1, which reduces the mean to x0_pred and the variance to 0.

    Args:
        model            : the U-Net (typically the EMA model in eval mode)
        scheduler        : NoiseScheduler, holds the schedule buffers and prediction_type
        cond_emb         : (B, 77, 768) text embeddings per sample
        uncond_emb       : (1, 77, 768) or (B, 77, 768) empty-string embedding
        guidance_scale   : w in CFG. See guidance_rescale; 2.0 is a sane default for a
                           small model on a small dataset, not the SD default of 7.5.
        guidance_rescale : phi in Lin et al. section 3.4. 0.0 disables.
        num_steps        : denoising steps; defaults to scheduler.T (the full 1000).
        latent_shape     : (C, H, W) of the latents — must match what was trained on.
        clip_x0          : if set, clamp each x_0 estimate to [-clip_x0, clip_x0].
                           Cheap guard against high guidance running away. Pick it
                           from the training latents' actual range, not from 1.0.
        generator        : torch.Generator for reproducible sampling

    Returns:
        (B, C, H, W) latents in the SAME normalized space the model trained on.
        Pass through latents_to_images (with the saved lat_mean / lat_std) to decode.
    """
    model.eval()
    B = cond_emb.shape[0]
    device = cond_emb.device

    if uncond_emb.shape[0] == 1:
        uncond_emb = uncond_emb.expand(B, -1, -1)

    steps = scheduler.T if num_steps is None else num_steps
    timesteps = make_timesteps(scheduler.T, steps, device, spacing=spacing)

    x = _init_latents(B, latent_shape, device, generator, x_T=x_T)
    ab = scheduler.alphas_cumprod
    one = torch.tensor(1.0, device=device)

    for i, step in enumerate(timesteps):
        is_last = i + 1 == len(timesteps)
        t = torch.full((B,), step.item(), device=device, dtype=torch.long)

        pred = _predict_with_cfg(model, x, t, cond_emb, uncond_emb, guidance_scale,
                                 guidance_rescale, view)
        x0_pred, _ = scheduler.to_x0_and_eps(pred, x, t)
        if clip_x0 is not None:
            x0_pred = x0_pred.clamp(-clip_x0, clip_x0)

        ab_t = ab[step]
        ab_prev = one if is_last else ab[timesteps[i + 1]]

        alpha_eff = ab_t / ab_prev
        beta_eff = 1.0 - alpha_eff
        one_minus_ab_t = 1.0 - ab_t

        mean = (
            (ab_prev.sqrt() * beta_eff / one_minus_ab_t) * x0_pred
            + (alpha_eff.sqrt() * (1.0 - ab_prev) / one_minus_ab_t) * x
        )

        if is_last:
            x = mean
        else:
            var = beta_eff * (1.0 - ab_prev) / one_minus_ab_t
            z = torch.randn(x.shape, device=device, generator=generator)
            x = mean + var.clamp(min=0.0).sqrt() * z

    return x


# DDIM sampler (fast, deterministic)
@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    scheduler,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    guidance_scale: float = 2.0,
    guidance_rescale: float = 0.7,
    num_steps: int = 50,
    eta: float = 0.0,
    latent_shape: tuple[int, int, int] = (4, 32, 32),
    spacing: str = "trailing",
    clip_x0: float | None = None,
    x_T: torch.Tensor | None = None,
    view: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Deterministic (eta=0) sampler using a subset of timesteps.

    For each transition tau_i -> tau_{i-1}:
        x_prev = sqrt(alpha_bar_prev) * x0_pred
               + sqrt(1 - alpha_bar_prev - sigma^2) * eps_pred
               + sigma * z

    x0_pred and eps_pred come from the scheduler, so this works for eps- and
    v-prediction alike. With eta=0 the stochastic term drops entirely.

    Worth knowing: run 1's diagnostics showed DDIM at eta=0 drifting toward the
    conditional mean (per-channel means 13-27% of the data's spread) while DDPM-1000
    tracked the data far better (38%, r=+0.88). If DDIM samples look flat, try
    eta=1.0 before assuming the model is at fault.

    Args as sample_ddpm, plus:
        num_steps : how many denoising steps (typically 25-100)
        eta       : 0.0 = deterministic DDIM, 1.0 = stochastic (DDPM-like)
    """
    model.eval()
    B = cond_emb.shape[0]
    device = cond_emb.device

    if uncond_emb.shape[0] == 1:
        uncond_emb = uncond_emb.expand(B, -1, -1)

    timesteps = make_timesteps(scheduler.T, num_steps, device, spacing=spacing)

    x = _init_latents(B, latent_shape, device, generator, x_T=x_T)
    ab = scheduler.alphas_cumprod
    one = torch.tensor(1.0, device=device)

    for i, step in enumerate(timesteps):
        is_last = i + 1 == len(timesteps)
        t = torch.full((B,), step.item(), device=device, dtype=torch.long)

        pred = _predict_with_cfg(model, x, t, cond_emb, uncond_emb, guidance_scale,
                                 guidance_rescale, view)
        x0_pred, eps_pred = scheduler.to_x0_and_eps(pred, x, t)
        if clip_x0 is not None:
            x0_pred = x0_pred.clamp(-clip_x0, clip_x0)

        ab_t = ab[step]
        ab_prev = one if is_last else ab[timesteps[i + 1]]

        if eta > 0.0 and not is_last:
            sigma = eta * torch.sqrt(
                (1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev)
            )
        else:
            sigma = torch.zeros((), device=device)

        dir_xt = (1.0 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps_pred
        x = ab_prev.sqrt() * x0_pred + dir_xt

        if sigma > 0.0:
            x = x + sigma * torch.randn(x.shape, device=device, generator=generator)

    return x


# Decode helpers (require a VAE)
@torch.no_grad()
def latents_to_images(
    latents: torch.Tensor,
    vae,
    lat_mean: torch.Tensor | None = None,
    lat_std: torch.Tensor | None = None,
    scale: float = 0.18215,
) -> torch.Tensor:
    """
    Decode latents back to RGB images in [0, 1].

    Args:
        latents  : (B, C, H, W) in whatever space the model trained on
        vae      : a diffusers AutoencoderKL in eval mode
        lat_mean : (1, C, 1, 1) per-channel mean saved by precompute.py, or None
        lat_std  : (1, C, 1, 1) per-channel std saved by precompute.py, or None
        scale    : the 0.18215 SD scaling factor — we divide before decoding.

    If lat_mean / lat_std are given, the per-channel normalization is undone first.
    Skipping them on latents that WERE normalized decodes grey mush, so pass them.
    """
    lat = latents
    if lat_mean is not None and lat_std is not None:
        lat = lat * lat_std.to(lat) + lat_mean.to(lat)
    decoded = vae.decode(lat / scale).sample
    return (decoded.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def sample_to_image(
    model: nn.Module,
    scheduler,
    vae,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    method: str = "ddim",
    guidance_scale: float = 2.0,
    guidance_rescale: float = 0.7,
    num_steps: int = 50,
    latent_shape: tuple[int, int, int] = (4, 32, 32),
    lat_mean: torch.Tensor | None = None,
    lat_std: torch.Tensor | None = None,
    eta: float = 0.0,
    clip_x0: float | None = None,
    x_T: torch.Tensor | None = None,
    view: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    End-to-end: text embedding -> noise -> denoise -> VAE decode -> RGB.

    Returns (B, 3, H*8, W*8) in [0, 1]. Useful as a training-time preview hook.
    """
    common = dict(
        guidance_scale=guidance_scale, guidance_rescale=guidance_rescale,
        num_steps=num_steps, latent_shape=latent_shape, clip_x0=clip_x0,
        x_T=x_T, view=view, generator=generator,
    )
    if method == "ddim":
        latents = sample_ddim(model, scheduler, cond_emb, uncond_emb, eta=eta, **common)
    elif method == "ddpm":
        latents = sample_ddpm(model, scheduler, cond_emb, uncond_emb, **common)
    else:
        raise ValueError(f"unknown sampling method: {method!r} (use 'ddim' or 'ddpm')")

    return latents_to_images(latents, vae, lat_mean=lat_mean, lat_std=lat_std)