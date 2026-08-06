"""
U-Net for latent diffusion.

Predicts the diffusion target (eps or v, see scheduler.prediction_type) given a
noisy latent, its timestep, and a text embedding. Same shape in, same shape out:
(B, 4, H, W) -> (B, 4, H, W).

Components (bottom-up):
  - sinusoidal_embedding / TimeEmbedding : encode integer t as a vector
  - ResNetBlock                          : conv -> conv with time injection
  - SelfAttention                        : every spatial position attends to every other
  - CrossAttention                       : image queries text — how prompts steer generation
  - Downsample / Upsample                : strided conv / NN-interp + conv
  - DownStage / UpStage                  : ResNet(s) + SelfAttn + CrossAttn group
  - UNet                                 : full assembly with skip connections

Changes since run 1, and why:

  top_self_attn=False (was: always on)
      Self-attention over the highest-resolution feature map was 44.2% of the
      forward pass (two layers out of ~28) because it attends over H*W tokens:
      4096 at 64x64 vs 1024 at 32x32 vs 256 at 16x16. Stable Diffusion has no
      attention at its highest resolution for exactly this reason. Cross-attention
      is kept there — it is only 0.8% and it is how text reaches full resolution.

  num_res_blocks=2 (was: 1)
      DDPM and SD use 2+. One block per stage left the skip connections dominating
      and gave the network very little depth to compute with.

  dropout=0.0 (was: 0.1)
      Run 1 was underfitting by a wide margin — it fit its own training set worse
      than a closed-form Gaussian projection. Dropout on an underfitting model is
      pure damage, and it also inflated every logged training loss.

  time_scale_shift=True (was: additive bias)
      FiLM-style conditioning (ADM / SD): scale and shift the normalized activations
      instead of adding a bias before GroupNorm, which partly normalizes the bias away.
      Set False to restore run 1's additive path — needed to load a run-1 checkpoint,
      since the shape of ResNetBlock.time_proj differs.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    """GroupNorm that degrades gracefully if channels isn't divisible by `groups`."""
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


# Time embedding
def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Transformer-style positional encoding adapted for timesteps.
    Input  t : shape (B,), integer
    Output   : shape (B, dim), float
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None, :]            # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbedding(nn.Module):
    """Sinusoidal encoding -> 2-layer MLP. Output dim is 4 * input dim by convention."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.out_dim = dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, self.out_dim),
            nn.SiLU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(_sinusoidal_embedding(t, self.dim))


# ResNet block (with time conditioning)
class ResNetBlock(nn.Module):
    """
    GroupNorm -> SiLU -> Conv -> GroupNorm -> (time scale/shift) -> SiLU -> Dropout -> Conv -> + skip

    GroupNorm (not BatchNorm) because diffusion uses tiny batches and GroupNorm
    is per-sample. Skip uses 1x1 conv when channels change, identity otherwise.

    time_scale_shift=True applies FiLM AFTER norm2: h = norm2(h) * (1 + scale) + shift.
    time_scale_shift=False adds the projected time embedding as a bias BEFORE norm2,
    which is what DDPM and run 1 did — simpler, but GroupNorm removes part of it.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        time_scale_shift: bool = True,
    ):
        super().__init__()
        self.time_scale_shift = time_scale_shift

        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels * 2 if time_scale_shift else out_channels)

        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        if time_scale_shift:
            # Start as the identity transform so training begins from a clean residual block.
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t_out = self.time_proj(F.silu(t_emb))
        if self.time_scale_shift:
            scale, shift = t_out.chunk(2, dim=1)
            h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        else:
            h = self.norm2(h + t_out[:, :, None, None])

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


# Attention
class SelfAttention(nn.Module):
    """
    Each spatial position attends to every other spatial position.
    Multi-head, with head_dim=32 -> num_heads = channels // 32.

    Cost is O(H*W * H*W * C), so this is affordable only at low resolution.
    """

    def __init__(self, channels: int, head_dim: int = 32):
        super().__init__()
        assert channels % head_dim == 0, f"channels ({channels}) must be divisible by head_dim ({head_dim})"
        self.channels = channels
        self.num_heads = channels // head_dim
        self.head_dim = head_dim

        self.norm = _group_norm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        # (B, C, H, W) -> (B, H*W, C)
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, N, C) -> (B, num_heads, N, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj_out(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out + residual


class CrossAttention(nn.Module):
    """
    Q from image features, K and V from text embeddings (B, 77, 768).
    This is THE mechanism by which text controls the generated image.

    Cost is O(H*W * 77 * C) — cheap even at full resolution, because the text
    sequence is short. Keep this everywhere self-attention gets dropped.
    """

    def __init__(self, channels: int, context_dim: int = 768, head_dim: int = 32):
        super().__init__()
        assert channels % head_dim == 0, f"channels ({channels}) must be divisible by head_dim ({head_dim})"
        self.channels = channels
        self.num_heads = channels // head_dim
        self.head_dim = head_dim

        self.norm = _group_norm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(context_dim, channels)
        self.to_v = nn.Linear(context_dim, channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, C, H, W)
        context : (B, 77, 768)
        """
        B, C, H, W = x.shape
        residual = x

        h = self.norm(x).view(B, C, H * W).transpose(1, 2)

        q = self.to_q(h)
        k = self.to_k(context)
        v = self.to_v(context)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj_out(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out + residual


# Resampling
class Downsample(nn.Module):
    """Strided 3x3 conv. Learned downsample is better than MaxPool."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample + 3x3 conv. Avoids ConvTranspose checkerboard artifacts."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# Stages
class Stage(nn.Module):
    """
    num_res_blocks x ResNet -> [SelfAttn] -> CrossAttn.

    The channel change happens in the first ResNet; the rest are out_ch -> out_ch.
    Used for both the encoder and the decoder — an UpStage just receives an in_ch
    that already includes the concatenated skip channels.

    Note this keeps ONE skip per stage regardless of num_res_blocks, rather than
    DDPM's one-skip-per-block. Less standard, but it keeps the assembly below
    readable and still buys the depth, which is the point.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        context_dim: int,
        num_res_blocks: int = 2,
        use_self_attn: bool = True,
        dropout: float = 0.0,
        time_scale_shift: bool = True,
    ):
        super().__init__()
        self.res = nn.ModuleList([
            ResNetBlock(
                in_ch if i == 0 else out_ch, out_ch, time_emb_dim,
                dropout=dropout, time_scale_shift=time_scale_shift,
            )
            for i in range(num_res_blocks)
        ])
        self.self_attn = SelfAttention(out_ch) if use_self_attn else None
        self.cross_attn = CrossAttention(out_ch, context_dim=context_dim)

    def forward(self, x, t_emb, context):
        for block in self.res:
            x = block(x, t_emb)
        if self.self_attn is not None:
            x = self.self_attn(x)
        x = self.cross_attn(x, context)
        return x


# Full U-Net
class UNet(nn.Module):
    """
    Latent diffusion U-Net.

    Encoder: 3 stages at channel widths (c1, c2, c3) = (base, 2*base, 4*base),
             each followed by a 2x downsample. Skips saved after init_conv and
             after each stage.
    Bottleneck: ResNet -> SelfAttn -> CrossAttn -> ResNet at the deepest level.
    Decoder: mirror of the encoder; each stage concatenates the matching skip
             along the channel dim before its ResNets.

    Fully resolution-agnostic — the same weights work on 32x32 or 64x64 latents.
    Only the samplers need to know the spatial size.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 128,
        time_dim: int = 128,
        context_dim: int = 768,
        num_res_blocks: int = 2,
        top_self_attn: bool = False,
        dropout: float = 0.0,
        time_scale_shift: bool = True,
    ):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.time_embedding = TimeEmbedding(time_dim)
        t_emb_dim = self.time_embedding.out_dim

        self.init_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        stage_kwargs = dict(
            time_emb_dim=t_emb_dim, context_dim=context_dim, num_res_blocks=num_res_blocks,
            dropout=dropout, time_scale_shift=time_scale_shift,
        )

        # Encoder. Stage 1 runs at the full latent resolution — self-attention there
        # is the single most expensive thing in the network, hence the flag.
        self.down1 = Stage(c1, c1, use_self_attn=top_self_attn, **stage_kwargs)
        self.down1_sample = Downsample(c1)

        self.down2 = Stage(c1, c2, use_self_attn=True, **stage_kwargs)
        self.down2_sample = Downsample(c2)

        self.down3 = Stage(c2, c3, use_self_attn=True, **stage_kwargs)
        self.down3_sample = Downsample(c3)

        # Bottleneck
        self.mid_res1 = ResNetBlock(c3, c3, t_emb_dim, dropout=dropout, time_scale_shift=time_scale_shift)
        self.mid_self_attn = SelfAttention(c3)
        self.mid_cross_attn = CrossAttention(c3, context_dim=context_dim)
        self.mid_res2 = ResNetBlock(c3, c3, t_emb_dim, dropout=dropout, time_scale_shift=time_scale_shift)

        # Decoder. Up stages receive in_ch = upsample_ch + skip_ch.
        self.up3_sample = Upsample(c3)
        self.up3 = Stage(c3 + c3, c3, use_self_attn=True, **stage_kwargs)

        self.up2_sample = Upsample(c3)
        self.up2 = Stage(c3 + c2, c2, use_self_attn=True, **stage_kwargs)

        self.up1_sample = Upsample(c2)
        self.up1 = Stage(c2 + c1, c1, use_self_attn=top_self_attn, **stage_kwargs)

        self.out_norm = _group_norm(c1 + c1)
        self.out_conv = nn.Conv2d(c1 + c1, out_channels, kernel_size=3, padding=1)

        # Zero-init the output layer: the network starts by predicting exactly 0,
        # so the first steps only have to learn a correction.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, 4, H, W)     noisy latent
        t       : (B,)             timestep per sample
        context : (B, 77, 768)     text embedding (or uncond_embedding)
        Returns : (B, 4, H, W)     predicted eps or v
        """
        t_emb = self.time_embedding(t)

        # Encoder.
        h = self.init_conv(x)
        skip_0 = h

        h = self.down1(h, t_emb, context); skip_1 = h
        h = self.down1_sample(h)

        h = self.down2(h, t_emb, context); skip_2 = h
        h = self.down2_sample(h)

        h = self.down3(h, t_emb, context); skip_3 = h
        h = self.down3_sample(h)

        # Bottleneck.
        h = self.mid_res1(h, t_emb)
        h = self.mid_self_attn(h)
        h = self.mid_cross_attn(h, context)
        h = self.mid_res2(h, t_emb)

        # Decoder.
        h = self.up3_sample(h)
        h = torch.cat([h, skip_3], dim=1)
        h = self.up3(h, t_emb, context)

        h = self.up2_sample(h)
        h = torch.cat([h, skip_2], dim=1)
        h = self.up2(h, t_emb, context)

        h = self.up1_sample(h)
        h = torch.cat([h, skip_1], dim=1)
        h = self.up1(h, t_emb, context)

        h = torch.cat([h, skip_0], dim=1)
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return h