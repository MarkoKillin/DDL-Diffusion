# Training run log

Newest run first. One entry per training run: the config, what the numbers actually
said, and what to change next. See [`plan.md`](plan.md) for the design and
[`README.md`](README.md) for layout.

---

## Run 1 — 2026-08-04 · 200 epochs · completed, plateaued early

Full run on Colab from `code/diffuser.ipynb`. Checkpoints on Drive, not committed.

### Config

| Knob | Value | Note |
|---|---|---|
| Dataset | `diffusers/pokemon-gpt4-captions`, 833 images | all in train, **no held-out split** |
| Epochs | 200 | |
| Batch size | 4 | 208 batches/epoch → 41,600 steps |
| Params | 12,658,628 | `base_channels=64`, 1 ResNetBlock per stage |
| LR | 1e-4, 1000-step linear warmup → cosine to 0 | |
| Optimizer | AdamW, weight decay 0.01 | |
| EMA decay | 0.9999 with warmup ramp | |
| CFG dropout | 0.1 | |
| Grad clip | 1.0 | |
| AMP | `none` (full fp32) | `bf16` was available and free |
| Schedule | linear β, 0.00085 → 0.012, T=1000 | `alphas_cumprod[-1] = 0.0015790` |
| Preview | **disabled** | 200 epochs with zero images seen |
| Seed | 42 | |

### Result

All 200 epochs finished, all 8 checkpoints saved (`epoch_25` … `epoch_200`), no NaN,
cosine LR annealed cleanly to 0.

| Step | Moving-avg loss |
|---|---|
| 0 | ~1.05 |
| ~1,500 | ~0.10 |
| 41,600 | ~0.07 |

The 1.05 start is the correct baseline: a model predicting ≈0 scores
$\mathbb{E}[\epsilon^2] = 1$. So the loss is real, and the run is mechanically
healthy — but **96% of the compute bought a 0.10 → 0.07 improvement.**

### Verdict

Nothing broke. Nothing was verified either. The loss curve cannot answer the three
questions that decide whether this run worked, and none of them were checked:

1. Did cross-attention learn anything, or is this now an unconditional denoiser?
2. Is the EMA — the thing we sample from — a real average, or still near its init?
3. Generation or memorization? 833 images × 200 epochs × 12.6M params.

Diagnostics for all three are now in the notebook (see below).

### Findings

**1. Uniform-`t` MSE is a poor progress metric, and partly explains the flatness.**
Most per-step variance is just *which `t` was drawn* — at `t≈0` loss is ~0.002, at
`t≈999` even a perfect model scores ~1. That is the entire 0.002–0.5 band in the
plot. A flat average can hide real progress in the mid-`t` range that decides image
structure. Bin the loss by `t` instead.

**2. The reported 0.07 is inflated by dropout.** `ResNetBlock` carries `p=0.1`
(`code/unet.py:74`) and the loop logs train-mode loss. Eval-mode loss reads lower, so
0.07 is not comparable to published figures.

**3. Self-attention at 64×64 is the compute sink.** `down1` and `up1`
(`code/unet.py:272`, `code/unet.py:295`) run self-attention over 4096 tokens.

| Stage | Tokens | Attention entries | vs 64×64 |
|---|---|---|---|
| 64×64 | 4096 | 16,777,216 | 1× |
| 32×32 | 1024 | 1,048,576 | 1/16 |
| 16×16 | 256 | 65,536 | 1/256 |

Stable Diffusion deliberately has **no** attention at its highest resolution. This is
almost certainly most of the wall-clock, and it is what forced `BATCH_SIZE = 4`.

**4. Depth is the likely plateau cause.** One `ResNetBlock` per Down/Up stage
(`code/unet.py:216`, `code/unet.py:232`); DDPM/SD use 2+. This is a documented
deviation in `README.md`, but at 12.6M params it is probably the binding constraint,
not the 833 images.

**5. The EMA is fine — for a non-obvious reason.** `EMA_DECAY = 0.9999` implies a
10,000-step window, which looks far too slow for a 41,600-step run. But the warmup
ramp `min(decay, (1 + step) / (10 + step))` (`code/train.py:55`) is *still the binding
term at the end*: at step 41,600 it yields 0.999784 → a ~4,623-step window. The EMA
does track the model. Worth verifying numerically anyway, since it is what we sample.

**6. Non-zero terminal SNR.** `sqrt(alphas_cumprod[-1]) = 0.0397`, so at `t=999`
training still leaks `0.0397 · x_0` into `x_t`. Combined with the per-channel latent
means below, that is a DC offset of roughly `[+0.060, +0.034, +0.001, −0.025]` —
detectable at ~4σ once averaged over a channel's 4096 positions. Inference starts from
`torch.randn`, whose per-channel mean is exactly 0, so the first denoising step sees a
slightly off-distribution input. This is the flaw in *Common Diffusion Noise Schedules
and Sample Steps are Flawed* (Lin et al.). Milder here than stock SD
(`scaled_linear` gives 0.0047), but a real cause of shifted brightness in samples.

**7. Samplers reviewed, no bugs found.** The alpha-bar-ratio formulation in
`code/sample.py` handles strided grids correctly, `_make_timesteps` spans the full
range starting at `T−1`, and the final step reduces to `x_0 = x0_pred` in both
samplers.

### Measured reference numbers

Computed from `code/latents.pt` — use these when judging whether a sample is sane.

**Training latents** (833 × 4 × 64 × 64, VAE output × 0.18215):

```
overall    mean +0.4403   std 1.1226   min -3.82   max +4.20
per-chan   mean [+1.503, +0.862, +0.032, -0.636]
per-chan   std  [ 0.731,  0.984,  0.604,  0.735]
```

These are **not** zero-mean / unit-std per channel. The per-channel offset is a known
property of the SD VAE latent space, and 0.18215 is a single global scalar that only
normalizes the overall std. **A generated latent at mean 0.00 / std 1.00 is wrong, not
right** — it decodes to off-colour output.

**Pairwise latent L2 between two different training images**, all 833 × 832 pairs:

```
mean 126.98   median 126.17   p1 98.69   min 18.79   max 204.97
```

The min of 18.79 means the dataset contains its own near-duplicates (alternate forms,
recolours), so "closer than any two training images" is not a usable memorization
threshold. Two tiers instead: `< 98.69` (p1) is suspicious and worth inspecting;
`< 18.79` is an unambiguous copy.

### Action items for run 2

Ordered by expected value per unit of effort.

- [ ] **Run the notebook's inference checks against `epoch_200.pt` first.** Do not
      retrain until the conditioning check has passed — if cross-attention is dead,
      every other change is wasted.
- [ ] **Set `PREVIEW = True`.** A 200-epoch run with no images is not worth repeating.
- [ ] **Drop self-attention from the 64×64 stage** (`down1`/`up1`). Biggest win
      available: spend the freed compute on batch size and depth.
- [ ] **Go to 2 ResNetBlocks per stage** with the reclaimed budget.
- [ ] **Set `AMP = "bf16"`.** Free on a Colab GPU.
- [ ] **Raise `BATCH_SIZE`** once attention is cheaper. At 4, CFG dropout at `p=0.1`
      yields 0.4 uncond rows per step — it averages out over 41.6k steps, but noisily.
- [ ] **Hold out ~80 latents as a validation split.** Currently everything is in
      `train_dataset` (`code/main.py:127`), so memorization is undetectable from loss.
- [ ] **Lower CFG.** `PREVIEW_CFG = 7.5` is the SD default for an 860M model trained
      on billions of images; 12.6M on 833 usually oversaturates well before that.
      The notebook's sweep covers 1.0–15.0; expect 2–4 to win.
- [ ] **Consider rescaling β for zero terminal SNR** (finding 6). Low priority — fix
      it only after the model demonstrably generates something.
- [ ] **Log eval-mode loss periodically**, so the headline number isn't dropout-inflated.

### Added to the notebook

10 cells appended to `code/diffuser.ipynb` (cells 27–36). All load weights from a
checkpoint, so they run in a fresh runtime without retraining. Ordered so each is only
worth reading if the previous one passed. Two `# TODO:` markers to set: `CKPT_PATH`
(cell 28) and `CFG` (cell 34, after seeing the sweep).

| Cell | Check | Failure it catches |
|---|---|---|
| 28 | Checkpoint load, `‖ema − live‖/‖live‖` | EMA still near random init → samples are noise regardless of loss |
| 29 | Loss binned by timestep, EMA vs live vs predict-zero | *where* on the schedule the model learned; the profile the aggregate curve hides |
| 30 | Matched vs shuffled vs empty caption at fixed noise/`t` | **dead cross-attention** — the most common silent failure, and the loss curve looks identical |
| 31 | `encode_prompt`, `generate`, `show_row`, `latent_stats` | — |
| 32 | Samples from training captions + ground truth | the easiest possible ask |
| 33 | CFG sweep 1.0 / 2.0 / 4.0 / 7.5 / 15.0 | oversaturation; identical columns ⇒ guidance is a no-op |
| 34 | DDIM 25/50/250 vs DDPM 1000 | separates sampler bugs from training bugs |
| 35 | Novel prompts + pairwise latent distance | prompt-independent collapse |
| 36 | Nearest training latent per sample | memorization |