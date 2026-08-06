"""
One-off pre-computation: encode the Pokemon dataset to latents + text embeddings.

Run this ONCE. Saves four files to disk that the training loop loads at startup:
    latents.pt           (N, 4, S, S)   — VAE-encoded images, PER-CHANNEL NORMALIZED
    latent_stats.pt      dict           — mean/std to undo that, plus run metadata
    embeddings.pt        (N, 77, 768)   — CLIP-encoded captions
    uncond_embedding.pt  (1, 77, 768)   — CLIP-encoded "" (for CFG)

Frees the VAE and text encoder from VRAM during training.

Usage:
    python code/precompute.py --out-dir code/ --resolution 256 --hflip

Changes since run 1, and why:

  Per-channel normalization  (the big one)
      Run 1 scaled latents by the single global 0.18215, which only normalizes the
      OVERALL std. The SD latent space has large per-channel offsets, so the run-1
      latents had per-channel means [+1.50, +0.86, +0.03, -0.64]. Decomposing the
      latent energy: 65.8% of it was a fixed mean pattern, and 89% of THAT was just
      those four constants — 58.6% of the total signal in four numbers.

      Those four numbers are also what the trained model got most wrong: generated
      channel means came out [0.66, 0.37, 0.52, 0.43], an 83% error, with the spread
      across channels collapsed to 13% of the data's. Normalizing per channel drops
      the fixed-pattern share of latent energy from 65.8% to 15.7% and removes the
      burden entirely. This is what SD3 / Flux do instead of a single scalar.

  --resolution 256 (was: 512 -> 64x64 latents)
      64x64 latents are 16384 dimensions to model from 833 images, and the highest
      U-Net stage then attends over 4096 tokens. 32x32 latents are 4x cheaper per
      conv, 16x cheaper in top-stage attention, and plenty for a centred creature
      on a plain background.

  --hflip
      Doubles the dataset for the cost of one more VAE pass. Flipping happens in
      PIXEL space before encoding — flipping a latent is not the same thing.

  latent_dist.mode() by default (was: .sample())
      Run 1 froze a single stochastic VAE posterior sample per image, baking a fixed
      noise draw into every training target forever. The mode is deterministic.
      Pass --latent-sample to restore the old behaviour.
"""

from __future__ import annotations

import argparse
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


def build_preprocess(resolution: int = 256):
    """Resize -> CenterCrop -> ToTensor -> Normalize to [-1, 1]. Matches SD's VAE."""
    return T.Compose([
        T.Resize(resolution),
        T.CenterCrop(resolution),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def encode_latents(
    vae,
    dataset,
    preprocess,
    device: str,
    batch_size: int = 16,
    scale: float = 0.18215,
    flip: bool = False,
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Run the whole dataset through the VAE encoder. Returns (N, 4, S, S) on CPU.

    flip=True horizontally flips every image first — call twice and concatenate
    to get the augmented set.
    """
    desc = "VAE encode (flipped)" if flip else "VAE encode"
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
    at inference is genuinely the distribution the forward process ends at.
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
    ap.add_argument("--hflip", action="store_true", help="also encode horizontally flipped images")
    ap.add_argument("--latent-sample", action="store_true", help="sample the VAE posterior instead of taking its mode")
    ap.add_argument("--no-normalize", action="store_true", help="skip per-channel normalization (run-1 behaviour)")
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
    n_original = len(poke)
    print(f"dataset size: {n_original}")

    print(f"loading VAE {args.vae}")
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    vae.requires_grad_(False)

    preprocess = build_preprocess(args.resolution)
    latents = encode_latents(
        vae, poke, preprocess, device,
        batch_size=args.vae_batch_size, flip=False, stochastic=args.latent_sample,
    )
    if args.hflip:
        flipped = encode_latents(
            vae, poke, preprocess, device,
            batch_size=args.vae_batch_size, flip=True, stochastic=args.latent_sample,
        )
        # Layout is [originals; flipped], so row i and row i + n_original are the SAME
        # Pokemon. train.make_split relies on that to keep a pair on one side of the
        # train/val boundary — otherwise the validation loss leaks.
        latents = torch.cat([latents, flipped], dim=0)

    print(f"latents: {tuple(latents.shape)}  (raw, before normalization)")
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
        print(f"  normalized -> mean {latents.mean().item():+.4f}  std {latents.std().item():.4f}  "
              f"abs max {latents.abs().max().item():.2f}")

    torch.save(latents, out_dir / "latents.pt")
    torch.save(
        {
            "mean": lat_mean,
            "std": lat_std,
            "scale": 0.18215,
            "resolution": args.resolution,
            "latent_size": latents.shape[-1],
            "latent_channels": latents.shape[1],
            "n_original": n_original,
            "hflip": bool(args.hflip),
            "normalized": not args.no_normalize,
            "vae": args.vae,
            "raw_stats": raw_stats,
        },
        out_dir / "latent_stats.pt",
    )

    # Free VAE VRAM before loading CLIP.
    del vae
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"loading CLIP {args.clip}")
    tokenizer = CLIPTokenizer.from_pretrained(args.clip)
    text_encoder = CLIPTextModel.from_pretrained(args.clip).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    embeddings = encode_text(text_encoder, tokenizer, captions, device, batch_size=args.clip_batch_size)
    if args.hflip:
        # A flipped Pokemon has the same caption. Tile to match the latent layout.
        embeddings = torch.cat([embeddings, embeddings], dim=0)
    assert embeddings.shape[0] == latents.shape[0], "embedding count must match latent count"
    print(f"embeddings: {tuple(embeddings.shape)}")
    torch.save(embeddings, out_dir / "embeddings.pt")

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