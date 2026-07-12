"""
U-Net for latent diffusion.

Predicts noise added to a latent given (x_t, t, text_context).
Same shape in, same shape out: (B, 4, 64, 64) -> (B, 4, 64, 64).

Components (bottom-up):
  - sinusoidal_embedding / TimeEmbedding : encode integer t as a vector
  - ResNetBlock                          : conv -> conv with time injection
  - SelfAttention                        : every spatial position attends to every other
  - CrossAttention                       : image queries text — how prompts steer generation
  - Downsample / Upsample                : strided conv / NN-interp + conv
  - DownStage / UpStage                  : ResNet + SelfAttn + CrossAttn group
  - UNet                                 : full assembly with skip connections
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    GroupNorm -> SiLU -> Conv -> (add time) -> GroupNorm -> SiLU -> Dropout -> Conv -> + skip

    GroupNorm (not BatchNorm) because diffusion uses tiny batches and GroupNorm
    is per-sample. Time embedding is projected and broadcast across spatial dims.
    Skip uses 1x1 conv when channels change, identity otherwise.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Broadcast time bias across spatial dims.
        t_bias = self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = h + t_bias

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


# Attention
class SelfAttention(nn.Module):
    """
    Each spatial position attends to every other spatial position.
    Multi-head, with head_dim=32 -> num_heads = channels // 32.
    """

    def __init__(self, channels: int, head_dim: int = 32):
        super().__init__()
        assert channels % head_dim == 0, f"channels ({channels}) must be divisible by head_dim ({head_dim})"
        self.channels = channels
        self.num_heads = channels // head_dim
        self.head_dim = head_dim

        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
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
    """

    def __init__(self, channels: int, context_dim: int = 768, head_dim: int = 32):
        super().__init__()
        assert channels % head_dim == 0, f"channels ({channels}) must be divisible by head_dim ({head_dim})"
        self.channels = channels
        self.num_heads = channels // head_dim
        self.head_dim = head_dim

        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
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
class DownStage(nn.Module):
    """ResNet -> SelfAttn -> CrossAttn. Channel change happens in the ResNet."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, context_dim: int):
        super().__init__()
        self.res = ResNetBlock(in_ch, out_ch, time_emb_dim)
        self.self_attn = SelfAttention(out_ch)
        self.cross_attn = CrossAttention(out_ch, context_dim=context_dim)

    def forward(self, x, t_emb, context):
        x = self.res(x, t_emb)
        x = self.self_attn(x)
        x = self.cross_attn(x, context)
        return x


class UpStage(nn.Module):
    """Same as DownStage; expects in_ch to already include the skip channels."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, context_dim: int):
        super().__init__()
        self.res = ResNetBlock(in_ch, out_ch, time_emb_dim)
        self.self_attn = SelfAttention(out_ch)
        self.cross_attn = CrossAttention(out_ch, context_dim=context_dim)

    def forward(self, x, t_emb, context):
        x = self.res(x, t_emb)
        x = self.self_attn(x)
        x = self.cross_attn(x, context)
        return x


# Full U-Net
class UNet(nn.Module):
    """
    Latent diffusion U-Net.

    Encoder: 3 down stages at channel widths (c1, c2, c3) = (64, 128, 256).
    Skips saved after init_conv and after each down stage.
    Bottleneck: ResNet -> SelfAttn -> CrossAttn -> ResNet at the deepest level.
    Decoder: mirror of encoder, each stage concatenates the matching skip
             along the channel dim before the ResNet.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 64,
        time_dim: int = 128,
        context_dim: int = 768,
    ):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.time_embedding = TimeEmbedding(time_dim)
        t_emb_dim = self.time_embedding.out_dim

        self.init_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        # Encoder.
        self.down1 = DownStage(c1, c1, t_emb_dim, context_dim)
        self.down1_sample = Downsample(c1)

        self.down2 = DownStage(c1, c2, t_emb_dim, context_dim)
        self.down2_sample = Downsample(c2)

        self.down3 = DownStage(c2, c3, t_emb_dim, context_dim)
        self.down3_sample = Downsample(c3)

        # Bottleneck
        self.mid_res1 = ResNetBlock(c3, c3, t_emb_dim)
        self.mid_self_attn = SelfAttention(c3)
        self.mid_cross_attn = CrossAttention(c3, context_dim=context_dim)
        self.mid_res2 = ResNetBlock(c3, c3, t_emb_dim)

        # Decoder. Up stages receive in_ch = upsample_ch + skip_ch.
        self.up3_sample = Upsample(c3)
        self.up3 = UpStage(c3 + c3, c3, t_emb_dim, context_dim)

        self.up2_sample = Upsample(c3)
        self.up2 = UpStage(c3 + c2, c2, t_emb_dim, context_dim)

        self.up1_sample = Upsample(c2)
        self.up1 = UpStage(c2 + c1, c1, t_emb_dim, context_dim)

        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=c1 + c1)
        self.out_conv = nn.Conv2d(c1 + c1, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, 4, 64, 64)   noisy latent
        t       : (B,)             timestep per sample
        context : (B, 77, 768)     text embedding (or uncond_embedding)
        Returns : (B, 4, 64, 64)   predicted noise
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
