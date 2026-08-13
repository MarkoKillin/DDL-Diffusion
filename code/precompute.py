"""
Helpers for turning the Pokemon dataset into latents and caption embeddings.

The notebook (`diffuser.ipynb`) owns the actual encoding pipeline — it needs the
intermediate tensors in memory to report view noise and preview the crops, so running the
pipeline twice (here and there) only creates two things that can drift apart. This module
holds the pure functions both the notebook and the training code import:

    VIEW_DIM, CANONICAL_VIEW    the micro-conditioning vector's shape and its identity value
    build_preprocess            resize/crop transform, for display
    crop_with_params            crop AND return the box, so the geometry can be conditioned on
    strip_pokemon_name          caption variant 1: the proper name replaced by "creature"
    build_caption_variants      flat caption list + the group id each caption belongs to
    normalize_per_channel       per-channel zero-mean unit-std, SD3/Flux style

Design notes, kept because the reasons are not obvious from the code:

  Crops (`crop_with_params` with crop=True)
      Run 2 memorized hard: held-out loss ended 50% WORSE than predicting zero, and 3 of
      4 samples were near-pixel copies of training images. The cause was arithmetic —
      35.5M parameters against 1,500 x 4,096 = 6.1M training scalars. K random-resized
      crops per image multiply the data K-fold for one extra VAE pass each. Crop 0 is
      always the deterministic centre crop, so the canonical view is guaranteed present.

  The view vector
      Run 3 threw the crop geometry away, which made 59.9% of its target variance
      unpredictable from the caption, and the model answered by producing blurry averages.
      Recording `[top, left, height, width, flip]` and feeding it to the U-Net as
      micro-conditioning (SDXL-style) is what fixed that in run 4.

  Caption variants (`strip_pokemon_name`)
      Run 2's model keyed on the Pokemon NAME as a lookup token: shuffling captions cost
      it 68x the matched loss, while the empty caption cost only 11x. Given no caption it
      produced something generic and sane; given a WRONG one it confidently produced the
      wrong image. Variant 1 replaces the name with "creature", forcing the model onto the
      descriptive words. The captions support it — 66% contain a colour word, 35% a visual
      noun, mean length 19.5 words.

  Latents and captions are stored SEPARATELY, joined by group id
      A 77x768 embedding is 237 KB; duplicating it per crop would dominate the output
      (K=4 plus flips would push embeddings.pt past 1.5 GB). Instead latents carry a
      `group_ids` array naming their source image, captions carry `caption_groups`, and
      train.LatentCaptionDataset joins them — picking a random variant per __getitem__,
      which makes the caption augmentation free and per-epoch.
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
    crop=True  : RandomResizedCrop applied to the ORIGINAL image, so the crop is taken
                 at full source resolution and downsampled once rather than twice.

    Both end at [-1, 1], which is what the SD VAE expects.

    Used for display and for decoding comparisons. The encoding path uses
    crop_with_params below, which returns the box alongside the tensor.
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
    survive — those carry real visual information). Measured on the 833 captions this
    fires on ~60%, and 442 of the words it finds are singletons, i.e. actual names.

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


def normalize_per_channel(latents: torch.Tensor):
    """
    Returns (normalized, mean, std) with mean/std shaped (1, C, 1, 1).

    After this, every channel is zero-mean unit-std across the dataset, so x_T ~ N(0, I)
    at inference is genuinely the distribution the forward process ends at. Run 1's
    single global 0.18215 left per-channel means up to +1.50, which the model learned to
    read off its input rather than generate.
    """
    mean = latents.mean(dim=(0, 2, 3), keepdim=True)
    std = latents.std(dim=(0, 2, 3), keepdim=True)
    return (latents - mean) / std, mean, std