# Training run log

Newest run first. One entry per training run: the config, what the numbers actually
said, and what to change next. See [`plan.md`](plan.md) for the design and
[`README.md`](README.md) for layout.

---

## Run 2 — planned · code landed, not yet trained

Diagnosis from run 1's checkpoint: **the model underfits by a wide margin; 833 images is
not the binding constraint.** Three independent lines of evidence:

1. Sampling from training caption #0 lands 124.68 from training latent #0 — **0.99× the
   median distance between two random different Pokemon.** After 200 exposures, its output
   for a memorized caption is indistinguishable from an unrelated one.
2. A closed-form Gaussian fitted to the 833 latents scores uniform-t **0.0257** where the
   trained network scores **0.0629** — 2.4× better on the network's own training set, 6.5×
   at t=25. (That Gaussian is degenerate and does not generalize; the point is only that
   the network has not *fit* what it was shown.) Empirical-posterior floor is ~0.
3. Four numbers — the per-channel latent means — carry **58.6%** of total latent energy.
   Generated channel means were 83% off, with the spread across channels collapsed to 13%
   of the data's.

### Code changes landed

| Area | Change | Why |
|---|---|---|
| `precompute.py` | per-channel latent normalization → `latent_stats.pt` | fixed-pattern share of latent energy 65.8% → 15.7% |
| `precompute.py` | `--resolution` (default 256), `--hflip`, `latent_dist.mode()` | 4× cheaper modeling task; 2× data for one VAE pass |
| `scheduler.py` | `zero_terminal_snr=True` (Lin et al. Alg. 1) | kills the `0.0397 · x_0` leak at t=999 |
| `scheduler.py` | `prediction_type="v"` + `to_x0_and_eps` | required at ᾱ=0; also flattens the loss across t |
| `scheduler.py` | `cosine` schedule option, `trailing` timestep spacing | |
| `unet.py` | `top_self_attn=False`, `num_res_blocks=2`, `dropout=0.0` | 1.79× speedup; depth; dropout hurts an underfitting model |
| `unet.py` | FiLM time conditioning, zero-init `out_conv`, resolution-agnostic | |
| `sample.py` | `guidance_rescale` (Lin et al. §3.4), `latent_shape` param, DDPM in x₀-posterior form | fixes CFG over-exposure; unblocks 32×32; finite at ᾱ=0 |
| `train.py` | `validation_loss`, `make_split` (mirror-leakage safe), `pick_amp_dtype` | run 1 had no val split and logged dropout-inflated train loss |

### Recommended config

Measured params and FLOPs per forward (attention matmuls included):

| config | latent | params | GFLOP | attn share | samples/hr vs run 1 |
|---|---|---|---|---|---|
| run 1: base64, 1 blk, top-attn | 64×64 | 12.7M | 20.06 | **50%** | 1.00× |
| base64, 2 blk, no top-attn | 32×32 | 17.4M | 3.69 | 4% | 5.44× |
| **base96, 2 blk, no top-attn** | **32×32** | **35.5M** | **7.97** | **3%** | **2.52×** |
| base128, 2 blk, no top-attn | 32×32 | 60.1M | 13.87 | 2% | 1.45× |

`base96 @ 32×32` is the pick: 2.8× run 1's capacity at 0.40× the cost per sample. Then
batch 32 (activations are 4× smaller, so this is finally affordable), `clip_x0=4.6`
(99.99th percentile of |x₀| on the normalized latents), CFG 2.0 with rescale 0.7,
`PREVIEW = True`, and a 10% held-out split via `make_split`.

Run 1 spent 166,400 sample-presentations. Budget at least 5–10× that.

### Two corrections to the run-1 findings below

- **Finding 1 has the t-direction backwards.** It claims "at t≈0 loss is ~0.002, at t≈999
  even a perfect model scores ~1." It is the reverse: the Bayes-optimal ε-loss for
  unit-variance data is exactly ᾱ_t, so it is *highest* at low t. The measured profile
  agrees — 0.3254 at t=25 falling to 0.0014 at t=975. This matters because it makes "the
  model is bad at low t" read like the expected shape. The real implication: unweighted ε
  puts 31.5% of its signal below t=100 and 0.3% above t=750.
- **Min-SNR-γ=5 is the wrong tool once you switch to v-prediction.** On top of v it peaks
  at SNR=γ (ᾱ=0.833, t≈116) and decays both ways, pushing 45% of the signal into
  t=100–250 and 0.4% above t=750 — reintroducing the low-noise bias. Unweighted v-loss is
  already exactly uniform per timestep at the optimum. `min_snr_gamma` defaults to `None`.
- **`AMP = "bf16"` is not free on a T4.** bf16 needs compute capability ≥ 8.0; Turing is
  7.5. `train.pick_amp_dtype("auto")` picks bf16 only where supported, else fp16.

### Note on run-1 checkpoints

`epoch_200.pt` will not load into the new `UNet` — `Stage.res` is now a `ModuleList`, so
keys gained a `.0`. The parameter count is unchanged (12,658,628) with
`UNet(base_channels=64, num_res_blocks=1, top_self_attn=True, dropout=0.1,
time_scale_shift=False)` plus `NoiseScheduler(zero_terminal_snr=False,
prediction_type="eps")`, so a `.res.` → `.res.0.` key remap is enough if you want to
re-evaluate run 1 against the fixed samplers. Otherwise `git show b4f6080:code/unet.py`.

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
Most per-step variance is just *which `t` was drawn* — and the direction is the opposite of
what you might guess: at `t≈25` the measured loss is 0.325, falling to 0.0014 at `t≈975`.
The Bayes-optimal ε-loss for unit-variance data is exactly `alphas_cumprod[t]`, so ε is
*hardest* to recover when the latent is nearly clean and trivial when `x_t` is almost pure
noise. That is the entire 0.002–0.5 band in the plot. Consequence: the aggregate number is
dominated by the low-noise tail (31.5% of the signal below `t=100`, 0.3% above `t=750`), so
it says almost nothing about the mid/high-`t` range that decides image structure. Bin the
loss by `t` instead — and see the run-2 note above on why v-prediction fixes the weighting.

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