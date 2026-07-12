"""
Noise schedule + forward (q_sample) for the diffusion process.

Owns the precomputed schedule constants and the one-shot forward noising
function. Not a neural net — just deterministic math.
"""

import torch
import torch.nn as nn


class NoiseScheduler(nn.Module):
    """
    Linear-beta DDPM schedule. nn.Module so the constants can ride along
    via register_buffer — moves with .to(device), included in state_dict,
    invisible to the optimizer.
    """

    def __init__(self, T: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        super().__init__()
        self.T = T

        # Linear beta schedule. Small at t=0, larger at t=T-1.
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

        # alpha_t = 1 - beta_t ("how much signal survives one step")
        alphas = 1.0 - betas

        # alpha_bar_t = prod_{s=1..t} alpha_s — lets us jump from x_0 to x_t in one shot.
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Pre-compute the square roots used in q_sample so the hot path is
        # just two index-and-multiply ops at train time.
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion in one shot:
            x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * noise

        Shapes:
            x_0   : (B, C, H, W)
            t     : (B,)   long tensor of per-sample timesteps
            noise : same shape as x_0, sampled from N(0, I)
        """
        # Index by t -> (B,), then reshape to (B, 1, 1, 1) so it broadcasts against (B, C, H, W).
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_ab * x_0 + sqrt_one_minus_ab * noise
