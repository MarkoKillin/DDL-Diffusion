"""
Training driver: stand-alone CLI version of the notebook's training cell.

Loads precomputed tensors, builds the model/optimizer/scheduler/EMA, runs
the training loop with checkpointing + resume. Optional --preview generates
a DDIM sample every N epochs so you can SEE training progress, not just loss.

Usage (smoke):
    python code/main.py --epochs 1 --batch-size 4

Usage (full run, on Colab T4):
    python code/main.py --epochs 200 --batch-size 8 --amp bf16

Usage (with preview every 10 epochs):
    python code/main.py --epochs 200 --preview --preview-every 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

from scheduler import NoiseScheduler
from unet import UNet
from train import build_ema, make_optimizer, make_lr_schedule, train_step, save_checkpoint, load_checkpoint


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_amp(flag: str) -> torch.dtype | None:
    if flag == "none":
        return None
    if flag == "bf16":
        return torch.bfloat16
    if flag == "fp16":
        return torch.float16
    raise ValueError(f"unknown --amp {flag!r}")


def maybe_preview(
    out_path: Path,
    model: torch.nn.Module,
    scheduler: NoiseScheduler,
    vae,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    device: str,
    num_steps: int,
    guidance_scale: float,
    seed: int,
) -> None:
    """Generate one DDIM sample with the EMA model and save it as a PNG."""
    from sample import sample_to_image

    g = torch.Generator(device=device).manual_seed(seed)
    img = sample_to_image(
        model=model,
        scheduler=scheduler,
        vae=vae,
        cond_emb=cond_emb,
        uncond_emb=uncond_emb,
        method="ddim",
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        generator=g,
    )[0].cpu()  # (3, 512, 512) in [0, 1]

    from torchvision.utils import save_image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(img, str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="code", help="dir with latents.pt / embeddings.pt / uncond_embedding.pt")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--resume", default=None, help="path to a checkpoint .pt to resume from")

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=1000)

    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--log-every", type=int, default=50)

    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--preview", action="store_true", help="generate a DDIM sample every --preview-every epochs")
    ap.add_argument("--preview-every", type=int, default=10)
    ap.add_argument("--preview-steps", type=int, default=50)
    ap.add_argument("--preview-cfg", type=float, default=7.5)
    ap.add_argument("--preview-row", type=int, default=0, help="which embedding row to use as the preview prompt")
    ap.add_argument("--preview-dir", default="previews")
    ap.add_argument("--preview-vae", default="stabilityai/sd-vae-ft-mse")

    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = pick_device()
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    latents_tensor = torch.load(data_dir / "latents.pt")
    embeddings_tensor = torch.load(data_dir / "embeddings.pt")
    uncond_embedding = torch.load(data_dir / "uncond_embedding.pt").to(device)
    print(f"latents: {tuple(latents_tensor.shape)}  embeddings: {tuple(embeddings_tensor.shape)}")

    train_dataset = TensorDataset(latents_tensor, embeddings_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    total_steps = args.epochs * len(train_loader)
    print(f"batches/epoch: {len(train_loader)}  total steps: {total_steps:,}")

    model = UNet().to(device)
    noise_scheduler = NoiseScheduler().to(device)
    ema_model = build_ema(model)
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = make_lr_schedule(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    start_step, start_epoch = 0, 0
    loss_history: list[float] = []
    if args.resume:
        start_step, start_epoch, loss_history = load_checkpoint(
            args.resume, model, ema_model, optimizer, lr_scheduler, map_location=device,
        )
        model.to(device); ema_model.to(device)
        print(f"resumed from {args.resume}: step={start_step}, epoch={start_epoch}")

    amp_dtype = parse_amp(args.amp)
    if amp_dtype is not None:
        print(f"AMP enabled: {amp_dtype}")

    vae = None
    preview_cond = None
    if args.preview:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.preview_vae).to(device).eval()
        vae.requires_grad_(False)
        preview_cond = embeddings_tensor[args.preview_row : args.preview_row + 1].to(device)
        print(f"preview enabled: every {args.preview_every} epochs, embedding row {args.preview_row}")

    global_step = start_step
    running_loss, running_count = 0.0, 0

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        for x_0, context in pbar:
            x_0 = x_0.to(device)
            context = context.to(device)

            loss = train_step(
                model=model,
                ema_model=ema_model,
                noise_scheduler=noise_scheduler,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                x_0=x_0,
                context=context,
                uncond_embedding=uncond_embedding,
                amp_dtype=amp_dtype,
            )

            loss_history.append(loss)
            running_loss += loss
            running_count += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg = running_loss / running_count
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                running_loss, running_count = 0.0, 0

        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
            ckpt_path = Path(args.ckpt_dir) / f"epoch_{epoch + 1}.pt"
            save_checkpoint(
                str(ckpt_path), model, ema_model, optimizer, lr_scheduler,
                step=global_step, epoch=epoch + 1, loss_history=loss_history,
            )
            print(f"saved checkpoint -> {ckpt_path}")

        if args.preview and (epoch + 1) % args.preview_every == 0:
            out_path = Path(args.preview_dir) / f"epoch_{epoch + 1}.png"
            maybe_preview(
                out_path, ema_model, noise_scheduler, vae,
                cond_emb=preview_cond, uncond_emb=uncond_embedding,
                device=device, num_steps=args.preview_steps,
                guidance_scale=args.preview_cfg, seed=args.seed,
            )
            print(f"saved preview -> {out_path}")

    print("training complete.")


if __name__ == "__main__":
    main()