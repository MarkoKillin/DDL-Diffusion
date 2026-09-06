from __future__ import annotations

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF


# Image preprocessing
_NORMALIZE = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# The view vector fed to UNet as micro-conditioning: [top, left, height, width, flip],
# with the box normalized to [0, 1] against the source image. The canonical full-frame,
# unflipped view is therefore [0, 0, 1, 1, 0].
VIEW_DIM = 5
CANONICAL_VIEW = (0.0, 0.0, 1.0, 1.0, 0.0)


def build_preprocess(resolution: int = 256, crop: bool = False, scale=(0.8, 1.0)):
    if crop:
        geom = T.RandomResizedCrop(resolution, scale=scale, ratio=(0.9, 1.1), antialias=True)
    else:
        geom = T.Compose([T.Resize(resolution, antialias=True), T.CenterCrop(resolution)])
    return T.Compose([geom, _NORMALIZE])


def crop_with_params(img, resolution: int, crop: bool, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                     flip: bool = False):
    """
    Returns (tensor in [-1,1], view vector) so the crop geometry can be recorded and used
    as conditioning.
    """
    W, H = img.size
    if crop:
        top, left, h, w = T.RandomResizedCrop.get_params(img, list(scale), list(ratio))
    else:
        # Resize(shorter side) + CenterCrop is a centred square of side min(H, W).
        s = min(H, W)
        top, left, h, w = (H - s) // 2, (W - s) // 2, s, s
    out = TF.resized_crop(img, top, left, h, w, [resolution, resolution], antialias=True)
    if flip:
        out = TF.hflip(out)
    view = (top / H, left / W, h / H, w / W, float(flip))
    return _NORMALIZE(out), view


def build_multi_captions(
    caption_lists: list[list[str]],
    n_per_image: int,
) -> tuple[list[str], torch.Tensor, int]:
    """Flatten per-image caption lists down to exactly n_per_image each."""
    flat, groups, n_cycled = [], [], 0
    for i, caps in enumerate(caption_lists):
        if not caps:
            raise ValueError(f"image {i} has no captions")
        if len(caps) < n_per_image:
            n_cycled += 1
        for k in range(n_per_image):
            flat.append(caps[k % len(caps)])
            groups.append(i)
    return flat, torch.tensor(groups, dtype=torch.long), n_cycled


def normalize_per_channel(latents: torch.Tensor):
    """Returns (normalized, mean, std) with mean/std shaped (1, C, 1, 1)."""
    mean = latents.mean(dim=(0, 2, 3), keepdim=True)
    std = latents.std(dim=(0, 2, 3), keepdim=True)
    return (latents - mean) / std, mean, std