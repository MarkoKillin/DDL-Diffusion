"""
One-off pre-computation: encode the Pokemon dataset to latents + text embeddings.

Run this ONCE. Saves three files to disk that the training loop loads at startup:
    latents.pt           (N, 4, 64, 64)   — VAE-encoded images, scaled by 0.18215
    embeddings.pt        (N, 77, 768)     — CLIP-encoded captions
    uncond_embedding.pt  (1, 77, 768)     — CLIP-encoded "" (for CFG)

Frees the VAE and text encoder from VRAM during training.

Usage:
    python code/precompute.py --out-dir code/
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


def build_preprocess():
    """Resize -> CenterCrop -> ToTensor -> Normalize to [-1, 1]. Matches SD's VAE."""
    return T.Compose([
        T.Resize(512),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def encode_latents(vae, dataset, preprocess, device: str, batch_size: int = 16, scale: float = 0.18215) -> torch.Tensor:
    """Run the whole dataset through the VAE encoder. Returns (N, 4, 64, 64) on CPU."""
    all_latents = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="VAE encode"):
        batch = dataset[i : i + batch_size]
        imgs = torch.stack([preprocess(im) for im in batch["image"]]).to(device)
        with torch.no_grad():
            lat = vae.encode(imgs).latent_dist.sample() * scale
        all_latents.append(lat.cpu())
    return torch.cat(all_latents, dim=0)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="diffusers/pokemon-gpt4-captions")
    ap.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out-dir", default="code")
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
    print(f"dataset size: {len(poke)}")

    print(f"loading VAE {args.vae}")
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    vae.requires_grad_(False)

    preprocess = build_preprocess()
    latents = encode_latents(vae, poke, preprocess, device, batch_size=args.vae_batch_size)
    print(f"latents: {tuple(latents.shape)}")
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

    assert len(captions) == latents.shape[0], "caption count must match latent count"
    embeddings = encode_text(text_encoder, tokenizer, captions, device, batch_size=args.clip_batch_size)
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