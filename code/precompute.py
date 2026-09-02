"""
Helpers for turning a captioned image dataset into latents and caption embeddings.

The notebook (`diffuser.ipynb`) owns the encoding pipeline itself, since it needs the
intermediate tensors in memory to report view noise and preview the crops. This module
holds the pure functions it imports:

    VIEW_DIM, CANONICAL_VIEW    the micro-conditioning vector's shape and its identity value
    build_preprocess            resize/crop transform, for display
    crop_with_params            crop and return the box, so the geometry can be conditioned on
    strip_pokemon_name          caption variant 1: the proper name replaced by "creature"
    build_caption_variants      flat caption list plus the group id each caption belongs to
    build_multi_captions        the same, for datasets with several real captions per image
    normalize_per_channel       per-channel zero-mean unit-std, SD3/Flux style

Why the augmentation looks like this:

  Crops multiply a small dataset K-fold for one extra VAE pass each, which is the cheapest
  defence available when the model has more parameters than the training set has numbers.
  Crop 0 is always the deterministic centre crop, so the canonical view is always present.
  Run 3 measured the price: 8 views per caption left 59.9% of the target variance
  unpredictable from the text, and an L2 objective answers that with a blurry average. Set
  CROPS=1 once the dataset is large enough to stand on its own.

  The view vector records where each crop came from, turning that ambiguity into signal.

  Caption variants stop the model using a proper name as a lookup key and push it onto the
  descriptive words. Only needed when the dataset has one caption per image, as Pokemon
  does. Flowers-102, CUB-200 and COCO ship several real captions, which is strictly better,
  and build_multi_captions handles those.

  Latents and captions are stored separately and joined by group id. A 77x768 embedding is
  237 KB, so duplicating one per crop would dominate the output. Latents carry `group_ids`,
  captions carry `caption_groups`, and train.LatentCaptionDataset joins them, picking a
  random variant per __getitem__ so the caption augmentation is free and per-epoch.
"""

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
    """
    crop=False : Resize -> CenterCrop. The canonical view (crop 0).
    crop=True  : RandomResizedCrop applied to the original image, so the crop is taken
                 at full source resolution and downsampled once rather than twice.

    Both end at [-1, 1], which is what the SD VAE expects.

    Used for display and decoding comparisons. The encoding path uses crop_with_params
    below, which returns the box alongside the tensor.
    """
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


# Caption variants
_KEEP = {"Pokemon", "Pokémon", "A", "An", "The", "Its", "This", "It", "Forme",
         "Legendary", "Mythical", "Therian", "Alolan", "Galarian", "Mega"}


def strip_pokemon_name(caption: str, replacement: str = "creature") -> tuple[str, bool]:
    """
    Replace the Pokemon's proper name with a generic noun.

    Heuristic: a capitalised word that is not sentence-initial, not the franchise word,
    not a franchise qualifier, and not hyphenated (so "Water-type" and "egg-themed"
    survive — those carry real visual information). On the 833 captions it fires on about
    60%, and most of the words it finds appear once, i.e. they really are names.

    Returns (variant, changed).
    """
    toks = caption.split()
    out, changed = [], False
    for i, tok in enumerate(toks):
        word = tok.strip(".,!?;:'\"()")
        if (
            i > 0
            and word
            and word[:1].isupper()
            and word not in _KEEP
            and "-" not in word
        ):
            out.append(tok.replace(word, replacement))
            changed = True
        else:
            out.append(tok)
    return " ".join(out), changed


def build_caption_variants(captions: list[str], n_variants: int) -> tuple[list[str], torch.Tensor]:
    """
    Returns (flat_caption_list, caption_groups) where caption_groups[j] is the index of
    the source image for caption j. Variant 0 is always the original.

    For the Pokemon set, which has one caption per image. Use build_multi_captions below
    when the dataset ships several real captions instead.
    """
    flat, groups = [], []
    for i, c in enumerate(captions):
        flat.append(c)
        groups.append(i)
        if n_variants > 1:
            stripped, _ = strip_pokemon_name(c)
            flat.append(stripped)
            groups.append(i)
        if n_variants > 2:
            raise ValueError("only 2 caption variants are implemented (original + name-stripped)")
    return flat, torch.tensor(groups, dtype=torch.long)


def build_multi_captions(
    caption_lists: list[list[str]],
    n_per_image: int,
) -> tuple[list[str], torch.Tensor, int]:
    """
    Flatten per-image caption lists down to exactly n_per_image each.

    For datasets that ship several human captions per image: Flowers-102 and CUB-200 have
    10, COCO has 5. Every one is a different description of the same picture, so the model
    learns that many phrasings map to one image. That is the thing the Pokemon set could
    not teach, since 62% of its captions carried a memorizable proper name and the only
    second variant was that name removed.

    Returns (flat, caption_groups, n_cycled). caption_groups[j] is the source image index
    of caption j, which is what train.LatentCaptionDataset joins on.

    Images with fewer than n_per_image captions have theirs cycled rather than raising,
    because LatentCaptionDataset needs a rectangular (image, caption) table. `n_cycled`
    counts them so the notebook can report it. Extra captions past n_per_image are dropped.
    """
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
    """
    Returns (normalized, mean, std) with mean/std shaped (1, C, 1, 1).

    After this every channel is zero-mean unit-std across the dataset, so the x_T ~ N(0, I)
    that inference starts from really is where the forward process ends. The single global
    0.18215 scale does not do that: it leaves per-channel means as high as +1.5, which a
    model can read off its input instead of generating.
    """
    mean = latents.mean(dim=(0, 2, 3), keepdim=True)
    std = latents.std(dim=(0, 2, 3), keepdim=True)
    return (latents - mean) / std, mean, std