# DDL-Diffusion

A latent text-to-image diffusion model built from scratch in PyTorch, trained on the Pokemon dataset. Educational project — every component (noise schedule, U-Net, training loop, samplers) is hand-written.

See [`plan.md`](plan.md) for the full design doc and math derivation. The Jupyter notebook at [`code/diffuser.ipynb`](code/diffuser.ipynb) is the recommended way to step through the project; the `.py` files under `code/` are the same components extracted for stand-alone use.

[`RUNS.md`](RUNS.md) is the training run log — per-run config, what the numbers said, and what to change next. Read it before starting a run.

## Layout

| File | What it is |
|---|---|
| `code/scheduler.py` | `NoiseScheduler` — beta schedules, `q_sample`, zero-terminal-SNR rescale, eps/v target + conversions, loss weighting |
| `code/unet.py` | `UNet` — the only thing being trained. ResNet + self/cross-attention, 3 down/up stages. Resolution-agnostic; width and depth are constructor args |
| `code/train.py` | EMA, optimizer/LR factories, AMP selection, CFG dropout, leakage-safe split, `train_step`, `validation_loss`, checkpoint save/load |
| `code/sample.py` | `sample_ddpm` (faithful) and `sample_ddim` (fast). Both do CFG with optional guidance rescale, and work with either prediction type |
| `code/precompute.py` | CLI: encode dataset → `latents.pt` / `latent_stats.pt` / `embeddings.pt` / `uncond_embedding.pt` |
| `code/diffuser.ipynb` | The walkthrough notebook |
| `RUNS.md` | Training run log: config, results, findings, next actions |

The four `.pt` tensor files are not committed (see `.gitignore`); regenerate them with `precompute.py`.

## How to reproduce

```bash
uv sync                                            # install deps
python code/precompute.py --out-dir code/ \
    --resolution 256 --hflip                       # one-off: encode dataset → .pt files
```

`latents.pt` is **per-channel normalized** (zero mean, unit std per channel); `latent_stats.pt`
holds the mean/std needed to undo that before `vae.decode`, plus the resolution and
augmentation metadata. Pass them to `latents_to_images` — skipping them decodes grey mush.

`--preview` in the notebook decodes a DDIM sample every N epochs. Use it: the loss curve
alone is a poor signal for diffusion quality, and run 1 spent 200 epochs proving it.

## Notes on the design

Deviations from `plan.md`, all deliberate:

- **v-prediction is the default**, not the ε-prediction `plan.md` derives. ε-prediction is
  still fully supported (`prediction_type="eps"`) and is the better path for reading the
  math, but it is *degenerate* at zero terminal SNR — recovering `x_0` needs a divide by
  `sqrt(alpha_bar)`. v also spreads the training signal uniformly across timesteps, which
  unweighted ε does not.
- **Zero terminal SNR by default.** Run 1's schedule left `sqrt(alpha_bar[-1]) = 0.0397`,
  so training always leaked a little `x_0` into `x_T` and the model learned to read the
  output's DC level off its input instead of generating it. See `RUNS.md` finding 6.
- **One skip per stage**, not per ResNet block as DDPM does. Keeps the assembly in
  `UNet.forward` readable while still buying the depth.
- **No self-attention at the top resolution** (`top_self_attn=False`), matching Stable
  Diffusion. At 64×64 latents it was 50% of the forward pass; dropping it is a 1.79×
  speedup. Cross-attention stays — it is ~1% and it is how text reaches full resolution.
- **Extra skip from `init_conv` into the final layer** (concat before `out_norm`). Cheap;
  the plan diagram doesn't include it but it doesn't hurt.
- **Zero-init `out_conv`**, so the network predicts exactly 0 at step 0 and the first loss
  is a clean 1.0 baseline.

CFG defaults are 2.0 with `guidance_rescale=0.7`, not the SD default of 7.5 — see the
guidance sweep in `RUNS.md`, where everything above ~4 inverted the latent statistics.