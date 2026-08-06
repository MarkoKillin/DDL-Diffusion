"""
Noise schedule + forward (q_sample) for the diffusion process.

Owns the precomputed schedule constants, the one-shot forward noising function,
the prediction-target conversions, and the loss weighting. Not a neural net —
just deterministic math.

Two knobs matter a lot and are new since run 1:

  zero_terminal_snr
      Run 1 had alphas_cumprod[-1] = 0.00158, so sqrt(alphas_cumprod[-1]) = 0.0397
      and training at t=999 still leaked 0.0397 * x_0 into x_t. The model learned to
      READ the output's DC level off that leak instead of GENERATING it. At inference
      x_T = randn has exactly zero per-channel mean, the cue is gone, and every
      channel collapses toward a common value. Rescaling so alphas_cumprod[-1] = 0
      removes the leak (Lin et al., "Common Diffusion Noise Schedules and Sample
      Steps are Flawed", Algorithm 1).

  prediction_type
      "eps" is the classic DDPM parameterization and what plan.md derives.
      "v" predicts the velocity v = sqrt(ab) * eps - sqrt(1-ab) * x_0.
      v is REQUIRED when zero_terminal_snr=True: at ab=0 the eps route needs
      x_0 = (x_t - sqrt(1-ab) eps) / sqrt(ab), which divides by zero. The v route
      is x_0 = sqrt(ab) x_t - sqrt(1-ab) v, which is finite everywhere.

      v also fixes the loss weighting for free. Since ||v - v_hat||^2 equals
      ||eps - eps_hat||^2 / alphas_cumprod, and the Bayes-optimal eps loss is
      ~alphas_cumprod for unit-variance data, v-loss is roughly flat across t.
      Run 1's unweighted eps loss put 91% of its magnitude in t < 500.
"""

import math

import torch
import torch.nn as nn


def make_betas(T: int, beta_start: float, beta_end: float, schedule: str) -> torch.Tensor:
    """Beta schedule in float64 — the cumprod over 1000 terms is worth the precision."""
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

    if schedule == "scaled_linear":
        # Stable Diffusion's default: linear in sqrt(beta).
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T, dtype=torch.float64) ** 2

    if schedule == "cosine":
        # Nichol & Dhariwal. Spends more of the schedule at moderate noise levels,
        # which is where image structure is decided.
        s = 0.008
        u = torch.arange(T + 1, dtype=torch.float64) / T
        ab = torch.cos((u + s) / (1.0 + s) * math.pi * 0.5) ** 2
        ab = ab / ab[0]
        return (1.0 - ab[1:] / ab[:-1]).clamp(max=0.999)

    raise ValueError(f"unknown schedule {schedule!r} (use 'linear', 'scaled_linear' or 'cosine')")


def rescale_betas_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """
    Lin et al. 2023, Algorithm 1.

    Shifts and scales sqrt(alphas_cumprod) so that its last entry is exactly 0 while
    its first entry is unchanged. Result: x_T is pure noise with no residual signal,
    which is what inference actually samples from.

    Side effect worth knowing: betas[-1] becomes 1.0. That is expected, not a bug —
    the final forward step destroys all remaining signal by construction.
    """
    ab = torch.cumprod(1.0 - betas, dim=0)
    sqrt_ab = ab.sqrt()

    first, last = sqrt_ab[0].clone(), sqrt_ab[-1].clone()
    sqrt_ab = (sqrt_ab - last) * (first / (first - last))

    ab = sqrt_ab ** 2
    alphas = torch.cat([ab[0:1], ab[1:] / ab[:-1]])
    return 1.0 - alphas


def make_timesteps(T: int, num_steps: int, device, spacing: str = "trailing") -> torch.Tensor:
    """
    Decreasing timestep grid of length num_steps.

    Sampling MUST include the highest timestep, where the schedule expects pure noise —
    which is what we initialize x to. Fewer steps means a COARSER STRIDE over the whole
    schedule, not a shorter walk through its low-noise tail.

      "trailing" : Lin et al.'s recommendation. round(arange(T, 0, -T/n)) - 1, so the
                   grid starts at exactly T-1. Ends at T/n - 1 rather than 0; the
                   samplers set alpha_bar_prev = 1 on the final step, so x still lands
                   on x_0.
      "linspace" : evenly spaced from T-1 down to 0. What run 1 used. Also starts at
                   T-1, so it never had the "leading" bug Lin et al. describe.
    """
    if num_steps >= T:
        return torch.arange(T - 1, -1, -1, device=device)

    if spacing == "trailing":
        ts = torch.round(torch.arange(T, 0, -T / num_steps)) - 1
        return ts.clamp(min=0).long().to(device)

    if spacing == "linspace":
        return torch.linspace(T - 1, 0, num_steps, device=device).long()

    raise ValueError(f"unknown spacing {spacing!r} (use 'trailing' or 'linspace')")


class NoiseScheduler(nn.Module):
    """
    DDPM noise schedule. nn.Module so the constants can ride along via
    register_buffer — moves with .to(device), included in state_dict,
    invisible to the optimizer.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        schedule: str = "linear",
        zero_terminal_snr: bool = True,
        prediction_type: str = "v",
    ):
        super().__init__()
        self.T = T
        self.schedule = schedule
        self.zero_terminal_snr = zero_terminal_snr
        self.prediction_type = prediction_type

        if prediction_type not in ("eps", "v"):
            raise ValueError(f"prediction_type must be 'eps' or 'v', got {prediction_type!r}")
        if zero_terminal_snr and prediction_type == "eps":
            raise ValueError(
                "zero_terminal_snr=True is incompatible with prediction_type='eps': at "
                "alphas_cumprod=0 the x_0 estimate (x_t - sqrt(1-ab) eps) / sqrt(ab) divides "
                "by zero. Use prediction_type='v', or set zero_terminal_snr=False to "
                "reproduce run 1."
            )

        betas = make_betas(T, beta_start, beta_end, schedule)
        if zero_terminal_snr:
            betas = rescale_betas_zero_terminal_snr(betas)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Pre-compute the square roots used in q_sample so the hot path is
        # just two index-and-multiply ops at train time.
        for name, buf in [
            ("betas", betas),
            ("alphas", alphas),
            ("alphas_cumprod", alphas_cumprod),
            ("sqrt_alphas_cumprod", alphas_cumprod.sqrt()),
            ("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt()),
        ]:
            self.register_buffer(name, buf.float())

    # Forward process
    def _ab_terms(self, t: torch.Tensor):
        """sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t), shaped (B, 1, 1, 1)."""
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return a, b

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion in one shot:
            x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * noise

        Shapes:
            x_0   : (B, C, H, W)
            t     : (B,)   long tensor of per-sample timesteps
            noise : same shape as x_0, sampled from N(0, I)
        """
        a, b = self._ab_terms(t)
        return a * x_0 + b * noise

    def get_velocity(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """v = sqrt(alpha_bar_t) * eps - sqrt(1 - alpha_bar_t) * x_0."""
        a, b = self._ab_terms(t)
        return a * noise - b * x_0

    def get_target(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """What the U-Net is trained to output, per prediction_type."""
        if self.prediction_type == "eps":
            return noise
        return self.get_velocity(x_0, noise, t)

    # Prediction conversions
    def to_x0_and_eps(self, model_out: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """
        Turn whatever the model predicted into (x_0_hat, eps_hat).

        (x_t, v) -> (x_0, eps) is an exact rotation, so no division is involved:
            x_0 = sqrt(ab) * x_t - sqrt(1-ab) * v
            eps = sqrt(1-ab) * x_t + sqrt(ab) * v
        The eps route needs a divide by sqrt(ab), which is why it cannot be paired
        with zero_terminal_snr.
        """
        a, b = self._ab_terms(t)
        if self.prediction_type == "eps":
            return (x_t - b * model_out) / a, model_out
        return a * x_t - b * model_out, b * x_t + a * model_out

    # Loss weighting
    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """alpha_bar / (1 - alpha_bar). Goes to 0 at t=T-1 under zero_terminal_snr."""
        ab = self.alphas_cumprod[t]
        return ab / (1.0 - ab).clamp(min=1e-12)

    def loss_weight(self, t: torch.Tensor, min_snr_gamma: float | None = None) -> torch.Tensor:
        """
        Per-sample loss weight, shaped (B,). None (the default) means unweighted.

        For prediction_type="v", UNWEIGHTED IS ALREADY THE BALANCED OBJECTIVE. At the
        Bayes optimum the v-loss is 1.0 at every t, so the training signal is spread
        uniformly per timestep. Run 1's unweighted eps-loss instead equalled alpha_bar_t,
        which put 31.5% of the signal below t=100 and 0.3% above t=750.

        min_snr_gamma applies Min-SNR-gamma (Hang et al. 2023):
            eps : min(SNR, gamma) / SNR
            v   : min(SNR, gamma) / (SNR + 1)
        It is the right tool for eps-prediction. On top of v-prediction it peaks where
        SNR = gamma (alpha_bar = gamma/(1+gamma), so t~116 for gamma=5 on this schedule)
        and decays in both directions — which puts the emphasis BACK on low noise, the
        opposite of what run 1 needed. Left in for experiments; leave it None for v.
        """
        if min_snr_gamma is None:
            return torch.ones_like(self.alphas_cumprod[t])

        snr = self.snr(t)
        clamped = snr.clamp(max=min_snr_gamma)
        if self.prediction_type == "eps":
            return clamped / snr.clamp(min=1e-12)
        return clamped / (snr + 1.0)