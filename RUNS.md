# Training run log

Newest run first. One entry per training run: the config, what the numbers actually
said, and what to change next. See [`plan.md`](plan.md) for the design and
[`README.md`](README.md) for layout.

---

## Run 3 — configured, not yet trained

Run 2's checkpoints were deleted, so this is a clean start. Run 2's diagnosis: **it fit,
then memorized hard** — held-out loss finished 50% worse than predicting zero while 3 of 4
samples were near-pixel copies of training images. The cause was arithmetic: 35.5M params
against 1,500 × 4,096 = 6.1M training scalars, **5.8× more parameters than numbers in the
dataset**. Run 3 moves that ratio ~12× by attacking both sides.

### Config

| Knob | Run 2 | Run 3 | Why |
|---|---|---|---|
| Latents | 1,666 | **6,664** | 4 crops (scale 0.8–1.0) × 2 flips |
| Caption variants | 1 | **2** | original + Pokemon name → "creature" |
| Params | 35.5M | **17.4M** | `base_channels` 96 → 64 |
| Params / training scalar | 5.78× | **0.71×** | the headline number |
| Dropout | 0.0 | **0.1** | run 1's underfit justified 0; run 2 does not |
| Weight decay | 0.01 | **0.05** | |
| CFG dropout | 0.10 | **0.15** | |
| Epochs | 600 | **150** | run 2's optimum was epoch ~55–100 of 600 |
| Batch | 32 | 32 | → 187 batches/epoch, 28,050 steps |
| Checkpoints | 12 × 569 MB | **`best.pt` 139 MB + `last.pt` 278 MB** | constant, not per-epoch |
| GFLOP/forward | 7.97 | **3.69** | 2.2× faster per sample |

Unchanged because they worked: 256px → 32×32 latents, per-channel normalization, zero
terminal SNR, v-prediction, `min_snr_gamma=None`, LR 2e-4, EMA 0.9999, CFG 2.0 with
`guidance_rescale=0.7`.

### What changed in the code

| Area | Change |
|---|---|
| `precompute.py` | `--crops` (crop 0 = centre, rest random-resized), `--caption-variants` with `strip_pokemon_name`, `group_ids`/`crop_ids`/`flip_ids`/`caption_groups` index arrays |
| `train.py` | `make_split` now splits on **group id**, so all crops+flips of one image stay together; `LatentCaptionDataset` joins latents to captions by group and samples a variant per `__getitem__`; `save_checkpoint(slim=True)` drops optimizer state; `load_checkpoint` refuses slim files |
| notebook | `best.pt` written on every held-out improvement; `last.pt` for resume; checkpoint-sweep cell; `stripped` column in the conditioning probe; 3-way `latent_stats`; `CFG_IMAGE = 1` |

Caption augmentation detail: the name-stripping heuristic (non-initial capitalised word,
excluding the franchise word, qualifiers like "Forme"/"Legendary", and anything hyphenated so
"Water-type" survives) fires on **62% of the 833 captions**, and 442 of the words it finds are
singletons — i.e. actual names. Storage stays flat because captions are held once per
(image, variant) and joined by group id rather than duplicated across the 8 views.

### What to watch, in order

1. **Smoke-test first loss ≈ 1.0.** Anything else means the latents are not normalized.
2. **`params-per-training-scalar` printed by the loader** — should read ~0.71×, not 5.78×.
3. **The held-out curve.** If it rises again, the run is done — stop it; `best.pt` already
   holds the good weights.
4. **The `stripped` column in the conditioning probe.** Close to `matched` means the model
   is reading the *description*; collapsing toward `uncond` means it is still keying on the
   name. This is the single most informative new number in run 3.
5. **`per-sample std` and `mean-spread`** from `latent_stats` (finding 11) — targets 0.987
   and 0.122.
6. **Novel prompts.** Run 2 got colour right and structure wrong on all four. Any structure
   here is the real win.
7. **Memorization cell:** want ~1.0× median and a train/held-out ratio near 1.0, against
   run 2's 0.14–0.20× and 0.345.

Not yet done, in reserve if run 3 underfits: 2 downsamples instead of 3 (the 32×32
bottleneck is 4×4 vs SD's 8×8).

---

## Run 2 — 2026-08-08 · 600 epochs · underfit fixed, now hard memorization

Full run on Colab from `code/diffuser.ipynb`. Checkpoints on Drive, not committed.

Pre-run diagnosis from run 1's `epoch_200.pt`: **the model underfit by a wide margin; 833
images was not the binding constraint.** Sampling from training caption #0 landed 124.68
from training latent #0 — 0.99× the median distance between two *random different* Pokemon.
A closed-form Gaussian fitted to the 833 latents scored uniform-t 0.0257 where the network
scored 0.0629, i.e. 2.4× better on the network's own training set. And four numbers (the
per-channel latent means) carried 58.6% of total latent energy, which the model got 83%
wrong. Every one of those is now resolved.

### Config

| Knob | Value | Note |
|---|---|---|
| Dataset | `diffusers/pokemon-gpt4-captions` | 833 images → **1,666 latents** (hflip) |
| Resolution | 256px → **32×32** latents | was 512px → 64×64 |
| Latents | **per-channel normalized** | raw per-chan mean `[1.343, 0.767, 0.024, -0.572]` |
| Split | 1,500 train / 166 val | leakage-safe via `make_split` |
| Epochs | 600 | 46 batches/epoch → 27,600 steps |
| Batch size | 32 | was 4 |
| Presentations | 883,200 | run 1: 166,400 |
| Params | 35,540,836 | `base_channels=96`, 2 ResNetBlocks/stage, no top-res self-attn |
| Schedule | linear β **rescaled to zero terminal SNR** | `alphas_cumprod[-1] = 0.0000000` exactly |
| Prediction | **v** | required at ᾱ=0 |
| Loss weighting | none (`min_snr_gamma=None`) | unweighted v is already uniform per t |
| LR | 2e-4, 500-step warmup → cosine to 0 | |
| Dropout | 0.0 | |
| AMP | **fp16 + GradScaler** | `pick_amp_dtype("auto")` correctly rejected bf16 on this GPU |
| CFG (eval) | 2.0, `guidance_rescale=0.7`, `clip_x0=4.4` | |

### Result

Mechanically clean: first loss 0.9816 (the predict-zero baseline, as designed by the
zero-init `out_conv`), EMA healthy at `‖ema−live‖/‖live‖ = 0.0059`, no NaN.

| | Run 1 | Run 2 |
|---|---|---|
| Train ε-equivalent loss, mean over t | 0.0629 | **0.0292** |
| Generated per-channel means | 83% off, spread 13% of data's | **`[-0.04, -0.06, 0.02, 0.17]`** vs target 0 |
| Batch latent mean / std | — | **+0.022 / 1.010** vs 0 / 1.0 |
| Samples from training captions | unrecognizable | **near-exact Caterpie, Voltorb, Electrode** |

(The loss comparison is indicative, not exact — run 1 was 64×64 raw-space, run 2 is 32×32
normalized. The images need no caveat.)

**But the held-out curve tells the real story.** Val loss bottomed at **~0.62 around step
2,500–4,600 (epoch 55–100)**, rose monotonically after, crossed the predict-zero baseline
at ~step 13,500 (epoch ~293), and finished at **1.5157** against a train loss of 0.1079 —
a 14× gap.

**So `epoch_600.pt` is the worst checkpoint of the run, and ~85% of the 600 epochs made
the model actively worse on unseen captions.**

### Verdict

The run-1 diagnosis was correct and its fixes all landed. The model now fits — and
overshot straight into memorizing. It is a caption→image lookup table with a weak generic
colour prior, not a generator.

### Findings

**1. Val loss exceeds the predict-zero baseline at 14 of 20 timesteps.** At t=25 the
held-out v-loss is 3.0835 (ε-equivalent 3.00) against a trivial baseline of 1.0. The model
is not merely failing to generalize on unseen captions, it is *confidently wrong* — it
retrieves a wrong memorized latent.

**2. Three of four samples are near-pixel copies.** Nearest-training-latent distances
11.66 / 16.09 / 16.90 against p1 = 68.13 and median = 83.06. Visually confirmed against
ground truth. They match rows 834/835/836 — the **mirrored** copies — and the generated
images are correspondingly flipped relative to the originals.

**3. hflip did not buy invariance, it bought a second mode.** Each caption now has exactly
two memorizable targets, and the deterministic sampler picks one. It also broke the
memorization threshold: train-vs-train `min` fell from 18.77 (833 originals) to **8.99**,
because a near-symmetric Pokemon's mirror is nearly identical to itself. Use p1, not min.

**4. The starting noise has no effect.** `prompt effect / total variation = 100.2%` —
sampling the same prompt from different `x_T` gives the same image. Zero diversity; one
prompt = one output.

**5. Memorization, quantified in one number.** Train v-loss at t=975 is **0.3446**. At ᾱ=0
there is *no image information in the input*, so from pure noise plus a caption the model
reconstructs 66% of the training image's variance. Held-out: 0.9531, i.e. 5%.

**6. The 6702% conditioning gap is an index, not comprehension.** At t=300: matched
0.0311, shuffled 2.1158, **uncond 0.3380**. Given *no* caption it produces something
generic and sane; given a *wrong* caption it confidently produces the wrong image — uncond
beats shuffled by 6×. Compare run 1's best relative gap of 13.6%.

**7. Capacity vs data is now the binding constraint.** 35,540,836 params against
1,500 × 4,096 = 6,144,000 training scalars: **5.8× more parameters than there are numbers
in the training set**, or 47,388 params per unique image. Memorization is the expected
solution, not a surprise.

**8. `guidance_rescale` works as intended.** At CFG 7.5 without it, std blows to 1.6228 and
the per-channel means invert to `[1.03, 0.63, -0.73, -0.21]` — run 1's exact failure. With
`rescale=0.7`, std holds at 0.8752 and means stay `[0.32, 0.24, 0.33, 0.13]`.

**9. No sampler bug.** DDIM 25/50/250, DDIM eta=1.0, and DDPM 1000 agree within ~3.6% on
every latent statistic (std 0.6996 → 0.7251). The x₀-posterior DDPM form is finite at ᾱ=0
as intended, and run 1's DDIM-drifts-to-the-mean behaviour is gone.

**10. Novel prompts get colour right, structure wrong.** "red fire dragon" → red, "blue
water turtle" → blue dome, "yellow electric mouse" → yellow, "green grass" → green. Real
text signal, amorphous shape. That is the ceiling of a lookup table asked to extrapolate.

**11. Samples are over-smoothed AND drift in brightness — two errors that cancelled.**
This was initially written off as a non-issue and that was wrong. Total latent variance
decomposes as within-image + between-image-means; measured on 300 real Pokemon latents that
split is **0.985 + 0.015**, i.e. 98.5% of the variance is texture inside an image and
overall brightness barely varies between Pokemon (`per-image mean std 0.122`). Run 2's
samples reported batch std 1.0096 against a batch reference of 1.000, which looks perfect.
Split apart:

| | run 2 | reference | |
|---|---|---|---|
| per-sample std (texture) | 0.706 | 0.987 | **−28%, over-smoothed** |
| between-sample mean spread | 0.722 | 0.122 | **5.9× too much brightness drift** |

The two defects happened to cancel in the lumped number. `latent_stats` now reports them
separately. Both are consistent with a model that is still weak at high `t` — the DDIM
deterministic trajectory collapsing toward conditional means, plus a residual per-sample
DC that per-channel normalization does not fix (it normalizes the *dataset's* channel means;
each individual sample still has to generate its own level).

### Action items for run 3

Ordered by expected value per unit of effort.

- [ ] **Sweep `CKPT_PATH` over `epoch_50 … epoch_600` and take the val minimum.** Free —
      the checkpoints already exist, and `epoch_50`/`epoch_100` sit at the bottom of the
      curve. Do this before anything else; you may already have a usable model.
- [ ] **~100 epochs, not 600.** You overshot by roughly 10×. Spend the saved compute on
      3–5 below instead of on more steps.
- [ ] **Shrink to `base_channels=64`** (17.4M params) — 3× fewer params per training scalar,
      and 2.2× cheaper per sample on top.
- [ ] **Turn regularization back on:** `dropout=0.1`, `weight_decay=0.05`,
      `cfg_dropout_prob=0.15`. Run 1's underfit justified stripping these; run 2 does not.
- [ ] **Real augmentation.** Precompute 4–8 random-resized-crops per image (scale 0.8–1.0)
      → 6–13k latents instead of 1,666. Largest remaining data lever.
- [ ] **Caption augmentation — high value for this dataset specifically.** The model keys on
      the Pokemon *name* as a lookup token. Add a second caption variant per image with the
      name stripped, forcing it onto the descriptive words. The captions support it: 66%
      contain a colour word, 35% a visual noun, mean length 19.5 words, vocab 2,367. Finding
      10 is direct evidence the descriptive channel already carries signal.
- [ ] **Save a `best.pt` on val improvement**, and checkpoint every 10 epochs — the entire
      interesting region is epochs 30–100.
- [ ] **Consider 2 downsamples instead of 3.** At 32×32 the bottleneck is 4×4 (16 tokens);
      SD's is 8×8. Only worth doing if run 3 underfits again after the shrink.

### Notebook fixes needed

- **`CFG_ROW = 0` is the one training caption that fails to retrieve.** Bulbasaur comes out
  as green mush while rows 1–3 are near-exact copies, so the whole CFG sweep ran on the
  model's worst case and reads far worse than the model is. Set `CFG_ROW = 1`.
- **`latent_stats` lumped two different statistics into one number, and it hid finding 11.**
  A single generated sample's std should be compared to the *per-sample* (within-image)
  reference of 0.987, not to the dataset-wide 1.000. Doing it correctly shows run 2's
  samples were 28% under-dispersed while their brightness drifted 5.9× too much — the
  errors cancelled in the lumped figure. `latent_stats` now prints per-sample std and
  between-sample mean spread against separate data-derived references.
- **Added: a checkpoint-sweep cell** (`1b`, right after the checkpoint load). Scores every
  saved checkpoint on train and held-out loss and reports the argmin. Given finding 1,
  no checkpoint should ever be chosen by recency again.

### Reference: measured architecture costs

Params and FLOPs per forward, attention matmuls included (`FlopCounterMode` silently skips
SDPA, so those were added analytically):

| config | latent | params | GFLOP | attn share | samples/hr vs run 1 |
|---|---|---|---|---|---|
| run 1: base64, 1 blk, top-attn | 64×64 | 12.7M | 20.06 | **50%** | 1.00× |
| base64, 2 blk, no top-attn | 32×32 | 17.4M | 3.69 | 4% | 5.44× |
| run 2: base96, 2 blk, no top-attn | 32×32 | 35.5M | 7.97 | 3% | 2.52× |
| base128, 2 blk, no top-attn | 32×32 | 60.1M | 13.87 | 2% | 1.45× |

### Three corrections to the run-1 findings below

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
  7.5. `train.pick_amp_dtype("auto")` picks bf16 only where supported, else fp16 — and on
  run 2's Colab GPU it did in fact select fp16 + GradScaler.

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