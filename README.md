# DDL-Diffusion

A latent text-to-image diffusion model built from scratch in PyTorch, trained on the Pokemon dataset. Educational project — every component (noise schedule, U-Net, training loop, samplers) is hand-written.

See [`plan.md`](plan.md) for the full design doc and math derivation. The Jupyter notebook at [`code/diffuser.ipynb`](code/diffuser.ipynb) is the recommended way to step through the project; the `.py` files under `code/` are the same components extracted for stand-alone use.

[`RUNS.md`](RUNS.md) is the training run log — per-run config, what the numbers said, and what to change next. Read it before starting a run.

## Layout

| File | What it is |
|---|---|
| `code/scheduler.py` | `NoiseScheduler` — linear-β DDPM schedule + `q_sample` |
| `code/unet.py` | `UNet` — the only thing being trained. ResNet + self/cross-attention, 3 down/up stages, ~12.7M params |
| `code/train.py` | EMA, optimizer/LR factories, CFG dropout, `train_step`, checkpoint save/load |
| `code/sample.py` | `sample_ddpm` (1000 steps, faithful) and `sample_ddim` (50 steps, fast). Both do CFG. |
| `code/precompute.py` | CLI: encode dataset → `latents.pt` / `embeddings.pt` / `uncond_embedding.pt` |
| `code/main.py` | CLI: training driver with checkpoint/resume and an optional sample-preview hook |
| `code/diffuser.ipynb` | The walkthrough notebook |
| `RUNS.md` | Training run log: config, results, findings, next actions |

The three `.pt` tensor files are not committed (see `.gitignore`); regenerate them with `precompute.py`.

## How to reproduce

```bash
uv sync                                            # install deps
python code/precompute.py --out-dir code/          # one-off: encode dataset → .pt files
python code/main.py --epochs 200 --batch-size 8 --amp bf16 --preview
```

`--preview` decodes a DDIM sample of `embeddings[0]` every 10 epochs into `previews/epoch_N.png` — useful because the loss curve alone is a poor signal for diffusion quality.

## Notes on the design

A couple of intentional deviations from `plan.md`:

- **One block per Down/Up stage** rather than the two the plan calls for. 833 Pokemon images don't need more capacity, and 2× doubles wall-clock on a T4.
- **Extra skip from `init_conv` into the final layer** (concat before `out_norm`). Cheap; the plan diagram doesn't include it but it doesn't hurt.

Everything else matches the plan: linear β schedule, CFG with w=7.5, EMA at 0.9999, AdamW with cosine-decay LR after warmup.