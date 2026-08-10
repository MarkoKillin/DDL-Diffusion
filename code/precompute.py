"""
One-off pre-computation: encode the Pokemon dataset to latents + text embeddings.

Run this ONCE. Saves four files to disk that the training loop loads at startup:
    latents.pt           (N_lat, 4, S, S)   — VAE-encoded crops, PER-CHANNEL NORMALIZED
    latent_stats.pt      dict               — mean/std, index arrays, run metadata
    embeddings.pt        (N_cap, 77, 768)   — CLIP-encoded caption variants
    uncond_embedding.pt  (1, 77, 768)       — CLIP-encoded "" (for CFG)

Frees the VAE and text encoder from VRAM during training.

Usage:
    python code/precompute.py --out-dir code/ --resolution 256 --crops 4 --hflip

Changes since run 2, and why (see RUNS.md run-2 findings):

  --crops K  (new)
      Run 2 memorized hard: held-out loss ended 50% WORSE than predicting zero, and 3 of
      4 samples were near-pixel copies of training images. Root cause was arithmetic —
      35.5M params against 1,500 x 4,096 = 6.1M training scalars, i.e. 5.8x more
      parameters than numbers in the dataset. K random-resized crops per image multiply
      the data K-fold for one extra VAE pass each. Crop 0 is always the deterministic
      centre crop, so the canonical view is guaranteed present.

  --caption-variants  (new)
      Run 2's model keyed on the Pokemon NAME as a lookup token: shuffling captions cost
      it 68x the matched loss, while the empty caption cost only 11x. Given no caption it
      produced something generic and sane; given a WRONG one it confidently produced the
      wrong image. Variant 1 replaces the name with "creature", forcing the model onto the
      descriptive words. The captions support it — 66% contain a colour word, 35% a visual
      noun, mean length 19.5 words.

  Latents and captions are stored SEPARATELY, joined by group id
      A 77x768 embedding is 237 KB; duplicating it per crop would dominate the output
      (K=4 + flip would push embeddings.pt past 1.5 GB). Instead latents carry a
      `group_ids` array naming their source image, captions carry `caption_groups`, and
      train.LatentCaptionDataset joins them — picking a random variant per __getitem__,
      which makes the caption augmentation free and per-epoch.

  Per-channel normalization and zero-terminal-SNR were run 2's wins and are unchanged.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from torchvision import transforms as T
from tqdm import tqdm


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Image preprocessing
_NORMALIZE = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def build_preprocess(resolution: int = 256, crop: bool = False, scale=(0.8, 1.0)):
    """
    crop=False : Resize -> CenterCrop. The canonical view (crop 0).
    crop=True  : RandomResizedCrop applied to the ORIGINAL image, so the crop is taken
                 at full source resolution and downsampled once rather than twice.

    Both end at [-1, 1], which is what the SD VAE expects.
    """
    if crop:
        geom = T.RandomResizedCrop(resolution, scale=scale, ratio=(0.9, 1.1), antialias=True)
    else:
        geom = T.Compose([T.Resize(resolution, antialias=True), T.CenterCrop(resolution)])
    return T.Compose([geom, _NORMALIZE])


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


# Encoding
def encode_latents(
    vae,
    dataset,
    preprocess,
    device: str,
    batch_size: int = 16,
    scale: float = 0.18215,
    flip: bool = False,
    stochastic: bool = False,
    desc: str = "VAE encode",
) -> torch.Tensor:
    """One pass over the dataset through the VAE encoder. Returns (N, 4, S, S) on CPU."""
    out = []
    for i in tqdm(range(0, len(dataset), batch_size), desc=desc):
        batch = dataset[i : i + batch_size]
        imgs = torch.stack([preprocess(im) for im in batch["image"]])
        if flip:
            imgs = torch.flip(imgs, dims=[-1])
        imgs = imgs.to(device)
        with torch.no_grad():
            dist = vae.encode(imgs).latent_dist
            lat = (dist.sample() if stochastic else dist.mode()) * scale
        out.append(lat.cpu())  # .cpu() is critical — don't hoard on device
    return torch.cat(out, dim=0)


def encode_text(text_encoder, tokenizer, captions: list[str], device: str, batch_size: int = 32) -> torch.Tensor:
    """Run all captions through CLIP. Returns (N, 77, 768) on CPU."""
    all_ids = tokenizer(
        captions,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    chunks = []
    for i in tqdm(range(0, len(captions), batch_size), desc="CLIP encode"):
        ids = all_ids[i : i + batch_size].to(device)
        with torch.no_grad():
            emb = text_encoder(ids).last_hidden_state
        chunks.append(emb.cpu())
    return torch.cat(chunks, dim=0)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="diffusers/pokemon-gpt4-captions")
    ap.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out-dir", default="code")
    ap.add_argument("--resolution", type=int, default=256, help="image side in pixels; latents are /8")
    ap.add_argument("--crops", type=int, default=4, help="crops per image; crop 0 is the centre crop")
    ap.add_argument("--crop-scale", type=float, nargs=2, default=(0.8, 1.0))
    ap.add_argument("--hflip", action="store_true", help="also encode horizontally flipped views")
    ap.add_argument("--caption-variants", type=int, default=2, help="1 = original only, 2 = + name-stripped")
    ap.add_argument("--latent-sample", action="store_true", help="sample the VAE posterior instead of its mode")
    ap.add_argument("--no-normalize", action="store_true", help="skip per-channel normalization")
    ap.add_argument("--seed", type=int, default=0, help="seeds the random crops, for reproducibility")
    ap.add_argument("--vae-batch-size", type=int, default=16)
    ap.add_argument("--clip-batch-size", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"device: {device}")

    # Deferred imports — these are heavy and not needed elsewhere.
    import datasets
    from diffusers import AutoencoderKL
    from transformers import CLIPTokenizer, CLIPTextModel

    print(f"loading dataset {args.dataset}")
    poke = datasets.load_dataset(args.dataset)["train"]
    captions = [row["text"] for row in poke]
    n_images = len(poke)
    flips = (False, True) if args.hflip else (False,)
    print(f"dataset size: {n_images}   crops: {args.crops}   flips: {len(flips)}   "
          f"-> {n_images * args.crops * len(flips)} latents")

    print(f"loading VAE {args.vae}")
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    vae.requires_grad_(False)

    lat_chunks, group_ids, crop_ids, flip_ids = [], [], [], []
    for crop_i in range(args.crops):
        # Seed per crop pass so the geometry is reproducible across reruns.
        torch.manual_seed(args.seed + crop_i)
        pre = build_preprocess(args.resolution, crop=crop_i > 0, scale=tuple(args.crop_scale))
        for flip in flips:
            tag = f"crop {crop_i}{' flipped' if flip else ''}"
            lat_chunks.append(encode_latents(
                vae, poke, pre, device, batch_size=args.vae_batch_size,
                flip=flip, stochastic=args.latent_sample, desc=tag,
            ))
            group_ids.append(torch.arange(n_images))
            crop_ids.append(torch.full((n_images,), crop_i))
            flip_ids.append(torch.full((n_images,), int(flip)))

    latents = torch.cat(lat_chunks, dim=0)
    group_ids = torch.cat(group_ids).long()
    crop_ids = torch.cat(crop_ids).long()
    flip_ids = torch.cat(flip_ids).long()

    print(f"\nlatents: {tuple(latents.shape)}  (raw, before normalization)")
    raw_stats = {
        "overall_mean": latents.mean().item(),
        "overall_std": latents.std().item(),
        "per_channel_mean": latents.mean(dim=(0, 2, 3)).tolist(),
        "per_channel_std": latents.std(dim=(0, 2, 3)).tolist(),
    }
    print(f"  per-channel mean {[round(v, 3) for v in raw_stats['per_channel_mean']]}")
    print(f"  per-channel std  {[round(v, 3) for v in raw_stats['per_channel_std']]}")

    if args.no_normalize:
        lat_mean = torch.zeros(1, latents.shape[1], 1, 1)
        lat_std = torch.ones(1, latents.shape[1], 1, 1)
    else:
        latents, lat_mean, lat_std = normalize_per_channel(latents)
        print(f"  normalized -> mean {latents.mean().item():+.5f}  std {latents.std().item():.4f}  "
              f"abs max {latents.abs().max().item():.2f}")

    torch.save(latents, out_dir / "latents.pt")

    # Free VAE VRAM before loading CLIP.
    del vae
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"loading CLIP {args.clip}")
    tokenizer = CLIPTokenizer.from_pretrained(args.clip)
    text_encoder = CLIPTextModel.from_pretrained(args.clip).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    flat_captions, caption_groups = build_caption_variants(captions, args.caption_variants)
    n_changed = sum(1 for c in captions if strip_pokemon_name(c)[1])
    if args.caption_variants > 1:
        print(f"caption variants: {args.caption_variants} per image "
              f"({n_changed}/{n_images} had a name to strip)")
    embeddings = encode_text(text_encoder, tokenizer, flat_captions, device, batch_size=args.clip_batch_size)
    print(f"embeddings: {tuple(embeddings.shape)}")
    torch.save(embeddings, out_dir / "embeddings.pt")

    torch.save(
        {
            "mean": lat_mean,
            "std": lat_std,
            "scale": 0.18215,
            "resolution": args.resolution,
            "latent_size": latents.shape[-1],
            "latent_channels": latents.shape[1],
            "n_images": n_images,
            "crops": args.crops,
            "crop_scale": tuple(args.crop_scale),
            "hflip": bool(args.hflip),
            "caption_variants": args.caption_variants,
            "normalized": not args.no_normalize,
            "group_ids": group_ids,
            "crop_ids": crop_ids,
            "flip_ids": flip_ids,
            "caption_groups": caption_groups,
            "vae": args.vae,
            "seed": args.seed,
            "raw_stats": raw_stats,
        },
        out_dir / "latent_stats.pt",
    )

    # Uncond embedding: CLIP encoding of "".
    uncond_ids = tokenizer(
        "",
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        uncond = text_encoder(uncond_ids).last_hidden_state.cpu()
    print(f"uncond_embedding: {tuple(uncond.shape)}")
    torch.save(uncond, out_dir / "uncond_embedding.pt")

    print("done.")


if __name__ == "__main__":
    main()