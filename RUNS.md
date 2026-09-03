# Training run log

Newest run first. One entry per training run: the config, what the numbers actually
said, and what to change next. See [`plan.md`](plan.md) for the design and
[`README.md`](README.md) for layout.

---

## Run 5 · 2026-09-03 · 40 epochs · generalization gap closed, attribute binding weak

CelebA-HQ (`Ryan-sjtu/celebahq-caption`), 30,000 images, `base_channels=128` (60.1M params),
60,000 latents (1 crop x 2 flips), 1 caption each, batch 32, 40 epochs, ~67k steps, on a
rented RTX 5090.

**The eval section was never run.** The memorization cell OOM-killed the instance and the
checkpoints went with it, so everything below is from the training log and the inline
previews. Numbers for the eval cells will have to come from run 6.

### Result

| | Run 4 (Pokemon) | Run 5 (CelebA-HQ) |
|---|---|---|
| Train/val gap, epoch 2 | — | **+0.0003** |
| Train/val gap, epoch 4 | — | **+0.0009** |
| Gap at end of run | +0.6447 | **~0, never diverged** |
| Val minimum at | epoch 16 of 150 | did not bottom by epoch 40 |

Epoch 2 was train 0.5708 / val 0.5711, epoch 4 train 0.5574 / val 0.5582.

**Do not compare 0.57 here against run 3's 0.5658.** The resemblance is coincidence; held-out
v-loss is not comparable across datasets. The gap is the measurement, not the level.

### Findings

**1. Distinct images was the constraint, and 30,000 removes it.** The decision criterion set
out below was "if the train/val gap stays near zero for the full run". It did. Steps make the
comparison fair: run 4 had 187 steps/epoch and diverged at ~2,992 steps, while run 5 passed
that at epoch 2 and stayed gapless through ~67k steps with a 3.4x larger model.

**2. The throughput estimates were wrong by 10x, in the good direction.** Measured 30-40 s per
epoch, so ~25 min for the full run against the 4.0 h predicted. The table below scales run 4's
T4 throughput by GFLOP ratio, which bakes in run 4's 2% utilization. That figure came from a
17.6M model at batch 16 on Turing, where kernel launches dominated; at base128 on Blackwell the
kernels are large enough to feed the GPU, and the measured ~1,545 img/s is ~64 TFLOPS, or
20-30% utilization. **Compute is no longer a constraint on this project.**

**3. Flattening curves at epoch 40 say nothing on their own.** The cosine schedule anneals LR
to exactly 0 at `NUM_EPOCHS`, so every run flattens at the end by construction. Convergence and
schedule-end are not separable without changing the epoch count.

**4. Unseen prompts mix facial features.** The model learned the face manifold but blends modes
instead of binding attributes to them. This is the weakness predicted when the dataset was
chosen: one caption per image, a narrow visual domain, and formulaic BLIP captions nearly all
opening "a photography of". Untested at higher CFG, since the checkpoints were lost, and
guidance is exactly the lever that sharpens attribute binding.

**5. The memorization cell cannot run at this scale.** With `CROPS=1` every row is crop 0, so
`mem_rows` was all 54,000 training rows and the cell asked for a 11.7 GB distance matrix, a
2.9 GB mask and an 11.7 GB masked copy, ~27 GB live, in CPU RAM. The box swapped, the OOM
killer fired, jupyter-server stopped answering (hence "failed to save"), and the instance died.
`off_diag.quantile(0.01)` would have thrown separately at 2.9e9 elements against quantile's
2^24 cap, which is run 3's bug surviving in a second place. Now fixed: the reference
distribution is estimated from a 4,000-row subsample with `kthvalue`, and the
nearest-neighbour search still runs against every training row.

### Action items for run 6

- [ ] **`base_channels=192`, 60 epochs.** Applied to the notebook. Flattening without
      overfitting is the capacity-limited signature, and the model-size table says go to 192.
      ~1.5-2 h at measured throughput.
- [ ] **Run the CFG sweep before drawing any conclusion about attribute binding.** Previews
      ran at 2.0. The "above ~4 inverts the latent statistics" warning comes from a 17.6M model
      on 750 images and probably does not bind here.
- [ ] **Pull `checkpoints/` off the box at the halfway mark**, not only at the end.
- [ ] If mixing survives higher CFG and 192 channels, the next suspect is cross-attention
      density: `Stage` injects text once per stage, where SD interleaves attention with every
      ResNet block. That is an architecture change and breaks checkpoint compatibility.

---

## Dataset choice for run 5

The generalization floor has not moved in two runs: best held-out v-loss 0.5658 (run 3,
epoch 34) then 0.5754 (run 4, epoch 16). Every fix in between landed, and the ceiling stayed
put. Run 4's held-out loss bottomed at **epoch 16 of 150** while train kept falling to
0.2617, and its samples sit 0.587× as far from training latents as from held-out ones. With
750 distinct caption→image pairs against 17.6M parameters, more epochs, more crops and more
architecture will not help. **Distinct images is the constraint.**

### What makes a dataset harder

MSE is minimized by predicting the *average* of every image that fits the caption. Run 3
measured this directly: 8 views per caption left 59.9% of the target variance unpredictable
from the caption, and the average of 8 views decodes as blur. Two properties follow.

1. **Images per caption.** One on Pokemon, so the average *is* that image and can be sharp.
   Millions on COCO, and no micro-conditioning fixes it, because the caption genuinely does
   not specify which one. Expect blur to return on COCO, correctly this time.
2. **Scene structure the model must learn.** Pokemon hands it a white background and a
   centred subject for free. COCO requires perspective, occlusion, a ground plane, multiple
   instances, varied lighting. That costs parameters.

Pokemon is an easy target with too little data. COCO is a hard target with enough data.
Single-subject datasets in between move only the first of the two.

One refinement to point 1, because the plain "MSE predicts the average" story is incomplete for
a diffusion model. The prediction is conditioned on the noisy latent as well as the text, so at
low noise the latent already carries the identity and the model can be sharp even when the
caption is vague. An ambiguous caption mainly shows up as diversity across seeds and as weak
prompt fidelity. It turns into visible mush only when the model is *also* data-starved or
under-trained, which is exactly the run 3 and run 4 situation. Read the 59.9% figure as "the
conditioning is carrying little of the load here", not as "blur is arithmetically forced".

### Options

| Dataset | Images | Captions/image | Structure | Captions ship with the images? |
|---|---|---|---|---|
| current: `pokemon-gpt4-captions` | 833 | 2 (1 synthetic) | single subject, plain bg | yes |
| Oxford Flowers-102 | 8,189 | 10 | single subject, plain bg | **no** |
| CUB-200-2011 birds | 11,788 | 10 | single subject, natural bg | **no** |
| **Multi-Modal CelebA-HQ** | **30,000** | **10 on paper, 1 on the Hub** | aligned faces, single subject | yes |
| Flickr30k | 31,783 | 5 | scenes | yes |
| COCO 2017 | 118,287 | 5 | multi-object scenes | yes |

**The last column is the one that decided this, and it was not obvious.** Flowers-102 and
CUB-200 are the standard small-scale text-to-image benchmarks (StackGAN, AttnGAN, DF-GAN), so
they looked like the clean choice. But their captions are not part of the image dataset. The
Oxford and Caltech distributions are classification data: `torchvision.datasets.Flowers102`
returns `(image, int_label)` and carries no caption text and no species names at all. The 10
captions per image come from a separate Reed et al. 2016 archive with a history of dead
mirrors, joined by filename. Same for CUB.

So the "easy target with more images" middle step is harder to obtain than it looks. Of the
datasets that ship captions, only CelebA-HQ keeps the single-centred-subject structure that
makes the target easy; the rest are scenes.

### Recommendation: CelebA-HQ as run 5

It changes close to one variable, the number of distinct images, which is what makes the
result readable. COCO changes four at once (image count, scene complexity, caption ambiguity,
required capacity), so bad COCO samples would not say which of the four caused it.

- **36× the distinct images**: 30,000 against 750.
- **Aligned, centred faces**, so the structural prior stays as easy as Pokemon's. Arguably
  easier: CelebA-HQ is landmark-aligned, so the eyes sit in nearly the same place every time.
- **No proper nouns to key on**, so `strip_pokemon_name` and the caption-variant path go away
  and run 2's finding 6 cannot recur.

What it costs, stated plainly. One caption per image and a narrow visual domain, so this is a
weaker test of language understanding than Flowers or COCO would be. See **Run 5 config** below
for what that does and does not rule out.

**`CAPTIONS_PER_IMAGE` is a RAM knob, not a quality knob.** At 30,000 images and
`MAX_LENGTH=32`, each caption per image costs ~1.5 GB of fp16 embeddings. The dataset only has
one, so `N_CAPS` clamps to 1 and the file is ~1.5 GB. The knob matters again on Flickr30k.

**Read the result off two numbers.** If the train/val gap stays near zero for the full run,
distinct images were the constraint and COCO is worth the day. Note that held-out v-loss is not
comparable across datasets, so run 3's 0.5658 is not a bar CelebA-HQ can be measured against.
The gap, not the level.

### What is actually on the Hub

Checked on 2026-08-31 against `datasets-server.huggingface.co/info?dataset=<id>`, so the
columns and row counts below are the ones the viewer reports, not the ones the paper claims.

The caveat above turned out to be half right. `Ryan-sjtu/celebahq-caption` does exist and does
carry sentences: 30,000 rows, columns `image` and `text`, 2.76 GB parquet. But it stores **one**
caption per image, not ten, and `IIGROUP/MM-CelebA-HQ-Dataset` returns 401, so the ten-caption
MM-CelebA text cannot be fetched and joined. Take CelebA-HQ as a 30,000-image, one-caption
dataset or not at all. `korexyz/celeba-hq-256x256` is 30,000 images already at 256 px but has
only a male/female label.

Of the datasets that ship several real captions with the images, three are usable:

| Dataset | Images | Captions/image | Caption column | Size | Catch |
|---|---|---|---|---|---|
| `jxie/flickr8k` | 8,000 | 5 | `caption_0`..`caption_4` | 1.1 GB | only 8k distinct images |
| `nlphuji/flickr30k` | 31,014 | 5 | `caption`, a list | 4.3 GB | zip + loading script, so `datasets` 3.x needs the `refs/convert/parquet` branch |
| `HuggingFaceM4/NoCaps` | 15,100 | ~11 | `annotations_captions` | 4.8 GB | splits are validation 4,500 and test 10,600, no train split |

`build_multi_captions` already handles all three shapes once the caption cell is normalized to a
list, which cell 3 of the notebook does.

All three are scenes rather than single subjects, so they move both variables at once and the
blur argument above applies to them the same way it applies to COCO. Flickr30k is the closest
in scale to CelebA-HQ, so running it after CelebA-HQ isolates scene structure with image count
roughly held fixed.

Rejected, with the reason:

- `jxie/coco_captions`: stores one row per caption with the image duplicated, 566,747 rows and
  87 GB for 113k distinct images.
- `phiyodr/coco2017`: has the 5-caption list but images are `coco_url` strings only, so the
  images are a separate download.
- `efekankavalci/flowers102-captions` (8,189) and `Multimodal-Fatima/CUB_train` (5,994): one
  caption per image, both synthetic. This confirms the point made above, the Reed 10-caption
  archives for Flowers and CUB are still not on the Hub.

### Run 5 config, decided

Two things changed after the section above was written. The captions are not what it assumed,
and training moved off Colab onto a local RTX 4090-class card.

**The captions are BLIP-style free text, not attribute templates.** First rows of
`Ryan-sjtu/celebahq-caption`, verbatim:

- "a photography of a woman with a very long blond hair"
- "a photography of a young man with a necklace and a black shirt"
- "a photography of a woman with a smile on her face"

That is better than the attribute-template vocabulary the recommendation feared. Gender, hair,
expression, glasses, hats and clothing are all named in ordinary English. Nearly every caption
opens with "a photography of", so inference prompts should carry that prefix.

**The goal for this run is sample quality**, specifically: an unseen attribute combination such
as "a photography of a man with glasses wearing a hat" should produce a recognizable
approximation. That is the easy case of compositional generalization. Glasses and hats are each
around 5% of CelebA, so the model sees each attribute thousands of times alone and the
conjunction a few hundred times. It composes at inference; it is not being asked to invent a
concept.

| Knob | Value | Why |
|---|---|---|
| `DATASET_ID` | `Ryan-sjtu/celebahq-caption` | Verified: 30,000 rows, `image` + `text`, 2.76 GB parquet |
| `CAPTIONS_PER_IMAGE` | 1 | Only one exists; `N_CAPS = min(...)` clamps it already |
| `RESOLUTION` | 256 | 32×32 latents, unchanged |
| `CROPS` | 1 | Run 3 finding 5. Several views per caption is what made Pokemon muddy |
| hflip | keep, with the view vector | The flip flag in `CANONICAL_VIEW` makes the flip predictable from conditioning, so it doubles the data without re-creating the ambiguity |
| `base_channels` | 128 | 60.1M params, 1,940 per image, still 12× below where run 4 memorized |
| Batch | 32 | VRAM is free at 32×32 latents on 24 GB |
| `EMA_DECAY` | 0.999 | Run 4's fix |
| `NUM_EPOCHS` | 40 | `HFLIP` doubles the latents to 60,000, so 1,688 steps/epoch at batch 32 after the 10% val split. ~67k steps, ~6h. Drop to 25 for a ~3.7h run |
| CFG at sampling | 2 to 4 | Run 4: above ~4 inverts the latent statistics |

`embeddings.pt` is 1.5 GB and `latents.pt` 491 MB (60,000 rows, because `HFLIP` adds a mirrored
pass), so both stay in RAM comfortably.

**What to expect.** Sharp aligned faces, correct eye placement (CelebA-HQ is landmark-aligned,
which is most of why this works), plausible hair and skin. Usual failure modes are ears, teeth,
earrings, background, and asymmetric eye colour. Nothing outside the face crop: no scenes, no
full bodies, no backgrounds on request, because the dataset is head-and-shoulders only.

### Hardware, and what it does to the other options

Roughly 6× a T4 for this workload at bf16, which is native on Ada and Blackwell, so plan.md
step 8 applies as written with no `GradScaler`. VRAM stops binding; wall-clock and dataset
download become the constraints. Estimates below scale run 4's measured ~14,200 steps/h at
base64 batch 16 by the GFLOP column, so they are optimistic.

| Dataset @ size | 4090 | 5090 | 5080 |
|---|---|---|---|
| CelebA-HQ @ base128, 40 ep | 6.0h | 4.0h | 9.1h |
| Flickr30k @ base128, 20 ep | 1.7h | 1.1h | 2.6h |
| COCO @ base128, 10 ep | 3.3h | 2.2h | 5.0h |
| COCO @ base192, 10 ep | 7.2h | 4.8h | 11h |

Cheap compute re-ranks the alternatives. The case against COCO was mostly a cost argument, and
at 3.3h it largely dissolves; what survives is that COCO still changes four variables at once,
so a blurry COCO result would not say which one caused it. Flickr30k at 1.7h is the control
that makes a COCO run readable, and it is the run to do next if the question becomes language
rather than image quality. Its samples will look worse than CelebA-HQ's while its numbers look
better, and both are true at once.

Two things that bite on a local box:

- **Embeddings, not VRAM, are the ceiling on COCO.** 118k images × 2 captions × `MAX_LENGTH=32`
  in fp16 is 11.6 GB, and the notebook holds `embeddings.pt` in RAM. Flickr30k is 3.0 GB and
  fine. COCO needs either a memory-map or storing token ids and running CLIP in the dataloader.
  Decide before the precompute pass, not after.
- **Blackwell needs torch 2.7+ on cu128** for sm_120. A 5090 on an older wheel fails at import.

### Running it on a rented GPU

**Nothing needs uploading.** `datasets.load_dataset("Ryan-sjtu/celebahq-caption")` pulls the
2.76 GB parquet from the Hub at runtime, and the VAE and CLIP come from the Hub the same way.
The dataset is public and ungated, so no token. It caches to `~/.cache/huggingface` on the
instance disk, so a fresh instance re-downloads it.

Run the notebook from the repo's `code/` directory, since the setup cell checks for
`scheduler.py` beside it and puts the working directory on `sys.path`. Everything it writes,
the four `.pt` files and `checkpoints/`, lands in that directory. **Pull `best.pt` and `last.pt`
off the box before you stop it**, the disk is ephemeral.

**What the run costs**, 40 epochs at base128:

| Card | Time | Note |
|---|---|---|
| RTX 4090 | ~6h | ~$0.40/h, so about $2.50 |
| RTX 5090 | ~4h | needs a torch 2.7+/cu128 image for sm_120 |
| RTX 5080 | ~9h | 16 GB is enough at base128, but caps batch size |
| T4 | ~36h | what runs 1-4 used. Not viable at base128 |

`FULL_EVERY = 5` writes a resumable `last.pt` every 5 epochs. `best.pt` and the `MILESTONES`
snapshots are slim, meaning no optimizer state, so they can evaluate but not resume. If the
instance dies, set `RESUME_PATH = "checkpoints/last.pt"` and re-run the training cell. A full
checkpoint at 60.1M params is ~960 MB and `last.pt` is overwritten in place, so it stays one
file.

### Model size, if the target is COCO

Fitting the measured architecture table below gives
`params ≈ 0.0032·c² + 0.058·c + 0.7` million for `base_channels = c`. First three rows
measured, the rest are the fit:

| `base_channels` | Params | GFLOP @ 32×32 | vs run 4 compute |
|---|---|---|---|
| 64 (run 4) | 17.6M | 3.69 | 1× |
| 96 (run 2) | 35.5M | 7.97 | 2.2× |
| 128 | 60.1M | 13.87 | 3.8× |
| 160 | 91M | 21.2 | 5.7× |
| 192 | 129M | 30.7 | 8.3× |
| 256 | 224M | 53.5 | 14.5× |

Start at 128, go to 192 if still capacity-limited. Anchors: SD 1.5 is 860M at 64×64 latents,
and from-scratch COCO work generally lands between 100M and 400M.

Params per distinct image says memorization stops being the risk:

| | Params/image | Outcome |
|---|---|---|
| Run 2 | 42,600 | memorized hard |
| Run 4 | 23,500 | memorized |
| Flowers at base64 | 2,390 | |
| COCO at base128 | 565 | |
| COCO at base192 | 1,210 | |
| SD 1.5 | 0.43 | |

COCO sits 20–40× below the ratio where this model memorized and still 3,000× above SD, so the
ratio stops discriminating. Watch the train/val gap instead.

### Config changes for a COCO run

| Knob | Change | Why |
|---|---|---|
| `DROPOUT` | 0.1 → **0.0** | a run-3 patch for overfitting 750 images |
| `WEIGHT_DECAY` | 0.05 → **0.01** | same |
| `CFG_DROPOUT` | keep 0.15 | CFG needs it regardless of dataset size |
| `CROPS` | 4 → **1** | synthetic data for a starved set; cost run 3 its sharpness |
| `VIEW_DIM` | 5 → **7–8** | add original size and aspect ratio, as SDXL does. COCO is ~640×480 at varied aspect, so square centre-cropping cuts objects. Here the micro-conditioning does real work rather than patching self-inflicted damage |
| `max_length` | 77 → **24** | COCO captions average ~11 tokens. At 77, `embeddings.pt` is 14 GB for one caption per image and 70 GB for all five. At 24 it is 4.4 GB, and memmapping it removes the problem entirely |
| Resolution | keep 256 → 32×32 for the first run | validate the pipeline before paying 4× for 512 → 64×64, which also reopens the top-resolution attention question |
| `NUM_EPOCHS` | far more than 16 | early stopping is calibrated to a regime that no longer exists |
| Eval | add **FID on 30k val-caption samples + CLIP score** | the memorization cell and nearest-latent checks stop meaning anything at 118k images, and latent L2 does not separate photographs the way it separated centred sprites. Keep the timestep-binned loss and the matched/shuffled/uncond check |

### Throughput, if moving off Colab

Run 4 hit 125 img/s, which is ~1.4 TFLOPS against a T4's 65 TFLOPS fp16 peak, so **~2%
utilization**. A 17.6M U-Net at 32×32 with batch 32 is bound by kernel launches and memory
traffic, not matmuls, so the raw FLOPS ratio of a bigger card is not what you get. Only the
first row is measured:

| Setup | img/s | COCO epoch (212,900 rows) | 30 epochs |
|---|---|---|---|
| T4, batch 32 | 125 | 28 min | 14.2 h |
| 4090, batch 32, nothing else changed | ~450 | 8 min | 4 h |
| 4090, batch 256 + bf16 + channels_last + compile | ~1,500 | 2.4 min | 1.2 h |
| 5090, same tuning | ~2,400 | 1.5 min | 0.75 h |

The tuning matters more than the card. bf16 instead of fp16 (Ada and Blackwell are both
cc ≥ 8.0, so `pick_amp_dtype("auto")` selects it and `GradScaler` disappears; run 4 got fp16
only because a T4 is Turing), batch 256–512, `channels_last`, and `torch.compile` for ~1.3–1.8×
from fusing the norm/activation chains. Scale LR with batch, roughly 3e-4 at 256 with 1000–2000
warmup steps. On a 5090, PyTorch needs sm_120 (torch 2.7+, cu128) or it will not launch.

At vast.ai's ~$0.35/hr for a 4090, base128 for 30 epochs is ~4.4h and ~$1.50; base192 is ~10h
and ~$3.50, plus ~1–2h of one-off precompute. **Model size is not the budget constraint.**

### What to skip

Expanded Pokemon sets (all forms, shinies, cross-generation sprites) reach 10–20k images that
are mostly near-duplicates. That is run 3's crop mistake at larger scale: more rows, no new
concepts. Run 3 finding 4 already measured why this does not work.

### Action items

- [ ] **Re-run cells 26–34 against `best.pt`, not `epoch_100.pt`.** Run 4's eval describes a
      checkpoint 84 epochs past the val minimum, so its −21.9% at t=975 and 0.587 memorization
      ratio are both worse than the run's actual best.
- [ ] **Run 5 on Multi-Modal CelebA-HQ**, `base_channels=64`, no crops, 20 epochs. The
      notebook is already configured for it.
- [ ] Confirm the chosen Hub mirror actually has caption sentences, not just attribute columns.
- [ ] Only after run 5 answers the question: COCO at `base_channels=128`, with the config table
      above.

---

## Run 4 · 150 epochs · sharpness fixed, memorization back

### Result (summary; full writeup pending a re-run of the eval cells against `best.pt`)

Everything below this block is the pre-run plan, and it is unchanged. 17,634,372 params,
6,664 latents, 28,050 steps, first loss 1.0153, `‖ema−live‖/‖live‖ = 0.0275`.

| | Run 3 | Run 4 |
|---|---|---|
| Best held-out v-loss | 0.5658 (ep 34) | **0.5754 (ep 16)** |
| Final held-out | 0.6193 | 0.9064 |
| Final train | 0.468 | **0.2617** |
| Final train/val gap | +0.1148 | **+0.6447** |
| Per-sample std (sharpness) | blurry | **0.989 vs 0.987 ref** ✓ |
| Nearest-train / nearest-held-out | ~1.0 | **0.587** |
| Held-out vs no-text baseline, t ≥ 325 | +25% to +43% | **−22%** (at `epoch_100.pt`) |

Three readings:

1. **View conditioning worked, and the "honest tension" below was right.** Sharpness is fixed
   (0.989 against a 0.987 reference, versus run 2's 0.706). It paid for that by removing the
   regularizer, and memorization came straight back: all four samples flagged suspicious,
   nearest-train ratio 0.587.
2. **Held-out bottomed at epoch 16 of 150.** Train fell to 0.2617 while held-out climbed to
   0.9064. That is a data-limited model, not an under-trained one. 91% of the run made it
   worse. Run 2 overshot ~10×, run 4 overshot ~9×.
3. **The floor did not move** (0.5658 → 0.5754). Two rounds of objective and architecture work
   have not shifted generalization. See the dataset section above.

Caveat on the numbers: cell 26 has `CKPT_PATH = "checkpoints/epoch_100.pt"`, so cells 27–34
describe a checkpoint 84 epochs past the val minimum. The −22% at high t and the 0.587
ratio are epoch-100 figures and will read less bad against `best.pt`. The trend will not change.

### Plan (written before the run)

Run 3's failure mode was fixed but the samples were blurry. Measured cause: augmenting each
image into 8 views made **59.9% of the target variance unpredictable from the caption**. The
model cannot know which framing it is being asked for, so an L2 objective is minimized by
predicting the *average* of the 8 views, and decoding an average is blur.

| config | views/img | view noise |
|---|---|---|
| run 1 | 1 | 0.0% |
| run 2 (flip) | 2 | 38.6% |
| 4 crops, no flip | 4 | 38.8% |
| **run 3** (4 crops + flip) | 8 | **59.9%** |

Flip alone accounts for 38.6%, so trimming crops does not fix it. Run 4 keeps all 6,664
latents and instead **conditions on the view**, the way SDXL's micro-conditioning does.

### Changes

| Area | Change | Why |
|---|---|---|
| `unet.py` | `view_dim=5`, a small MLP added to the time embedding, **zero-init** | turns the framing ambiguity into signal; +266k params; a fresh model behaves exactly as if absent |
| `precompute.py` | `crop_with_params` records `[top, left, h, w, flip]` normalized; saved as `view_params` | run 3 discarded this information |
| `train.py`/`sample.py` | `view` threaded through `diffusion_loss`, `train_step`, `validation_loss`, both samplers, `_predict_with_cfg` | at inference you request the canonical view `[0,0,1,1,0]` |
| `EMA_DECAY` | 0.9999 → **0.999** | run 3's best checkpoint was epoch 34, where the 0.9999 ramp still lagged the live weights by 8%, itself a source of blur |
| notebook | `MILESTONES = [30, 60, 100, 150]` slim snapshots | run 3 kept only best.pt + last.pt, and best.pt was the blurriest |
| notebook | previews mix 3 training captions **+ 1 unseen prompt** | run 3's previews were training-only, so the generalization gap was invisible until eval |
| notebook | no-text baseline (~0.84) computed and plotted inline | see finding 3 below |
| notebook | `torch.quantile` → `kthvalue` | **the bug that killed run 3's entire eval section** |

Unchanged because they worked: 256px → 32×32, per-channel normalization, zero terminal SNR,
v-prediction, `min_snr_gamma=None`, `base_channels=64`, dropout 0.1, weight decay 0.05,
CFG dropout 0.15, caption variants, 4 crops + flip.

### The honest tension

Micro-conditioning makes the canonical view single-valued again, which is what buys
sharpness. It also removes some of the regularization that stopped run 3 memorizing.
Watch the memorization cell. If nearest-train distances collapse toward run 2's 0.14–0.20×
median, the sharpness came at the cost of copying.

### What to watch

1. Smoke-test first loss ≈ 1.0, and `view batch shape (32, 5)` printed beside it.
2. **View noise printed by cell 10.** Same 59.9%, but now conditioned on rather than averaged over.
3. **The `stripped` column.** Run 3 had it within 0.1% of `matched`. Keep it there.
4. **Previews: the 4th image is an unseen prompt.** Run 3 got colour right and structure wrong.
5. **Compare `best.pt` against the milestones BY EYE.** Lowest val MSE is the blurriest checkpoint.
6. **`per-sample std` and `mean-spread`.** Targets 0.987 and 0.122.
7. **`t=975` held-out vs the no-text baseline.** Run 3 scored +2.0%. This is the number that
   says whether caption→layout generalizes at all.

---

## Run 3 · 2026-08-11 · 150 epochs · memorization fixed, samples blurry

Full run on Colab. `base_channels=64` (17.4M params), 6,664 latents (4 crops × 2 flips),
2 caption variants, dropout 0.1, weight decay 0.05, CFG dropout 0.15, batch 32,
28,050 steps. `params-per-training-scalar` 0.71× (run 2: 5.78×).

### Result

**The failure mode is fixed.**

| | Run 2 | Run 3 |
|---|---|---|
| Best held-out v-loss | ~0.62 | **0.5658** (epoch 34) |
| Final held-out | 1.5157 | **0.6193** |
| Train/val gap | +1.4077 | **+0.1148** |
| Held-out above predict-zero | 14/20 timesteps | **0/20** |
| Conditioning shape | `uncond` ≪ `shuffled` (lookup) | **`matched < uncond < shuffled`** ✓ |
| Largest relative conditioning gap | 6702% | **81%** |

The eval section crashed at cell 31 (`torch.quantile` element limit), so there are **no sample
images, no CFG sweep and no memorization check** for this run.

### Findings

**1. Caption augmentation worked, cleanly.** `stripped` came within **0.1%** of `matched` at
every timestep (0.4305 vs 0.4306, 0.5763 vs 0.5769). Removing the Pokemon name costs the
model nothing, so run 2's name-as-lookup-key habit is gone. And the ordering
`matched < uncond < shuffled` now holds at every t, where run 2 had `uncond` 6× *better*
than `shuffled`.

**2. The "learned nothing" line is 0.84, not 1.0.** The optimal predictor knowing only the
per-dim mean and variance of the training latents, with no text at all, scores **0.8426** train
and **0.8404** held-out. Per-channel normalization zeroes the *channel* means but leaves the
spatial mean pattern, so predicting the average latent already beats predicting zero. Every
earlier "% of variance explained" figure judged against 1.0 was too generous.

**3. Generalization is real at low t and absent at high t.** Against that 0.84 baseline:

| band | held-out improvement |
|---|---|
| t = 25–225 (texture) | **+45% to +60%** |
| t = 425–575 | +25% to +43% |
| t = 725–975 (caption → layout) | **+2% to +9%** |

Mean: train +45.8%, held-out +32.0%. But at t=975, where the input carries zero information
and only the caption remains, held-out is 0.8029 against a 0.8191 floor, i.e. **+2.0%**
(train +28.8%). So the caption→layout mapping barely transfers. This predicts exactly what
run 2's samples showed: correct colour, generic global shape.

**4. 4× more data bought little, because crops add geometry not concepts.** Best held-out
moved only 0.62 → 0.566. For the high-t task the effective dataset is still **750 distinct
caption→image pairs**, since the 8 views share one caption. Measured: two views of the same
Pokemon sit at latent distance 68.7 vs 80.6 for two *different* Pokemon (ratio 0.85), so the
views are not redundant, they just carry no new concepts. **Number of distinct images is now
the binding constraint.**

**5. The augmentation is what made the samples blurry.** See the run-4 section above: 59.9%
of the target variance became unpredictable from the caption.

**6. Early stopping on val MSE selects for blur.** MSE is minimized by predicting the
conditional *mean*, so the lowest-val-MSE checkpoint is systematically the smoothest.
`best.pt` was epoch 34 of 150; `last.pt` fit far better (train 0.386 vs 0.468) at only 9%
worse val. Use val loss to detect divergence, then choose between survivors by eye.

**7. EMA lag added blur.** `‖ema−live‖/‖live‖ = 0.0807` at epoch 34, vs run 2's 0.0059 at
epoch 600. Early in training the weights move fast and a 0.9999 decay cannot keep up.

**8. The memorization threshold is now blunt.** Same-image different-view distance (68.7)
exceeds the p1 of different images (66.9), so latent L2 can no longer sharply separate "copy
at a different crop" from "distinct image".

---

## Run 2 · 2026-08-08 · 600 epochs · underfit fixed, now hard memorization

Full run on Colab from `code/diffuser.ipynb`. Checkpoints on Drive, not committed.

Pre-run diagnosis from run 1's `epoch_200.pt`: **the model underfit by a wide margin; 833
images was not the binding constraint.** Sampling from training caption #0 landed 124.68
from training latent #0, which is 0.99× the median distance between two *random different*
Pokemon. A closed-form Gaussian fitted to the 833 latents scored uniform-t 0.0257 where the
network scored 0.0629, i.e. 2.4× better on the network's own training set. And four numbers
(the per-channel latent means) carried 58.6% of total latent energy, which the model got 83%
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

(The loss comparison is indicative, not exact. Run 1 was 64×64 raw-space, run 2 is 32×32
normalized. The images need no caveat.)

**But the held-out curve tells the real story.** Val loss bottomed at **~0.62 around step
2,500–4,600 (epoch 55–100)**, rose monotonically after, crossed the predict-zero baseline
at ~step 13,500 (epoch ~293), and finished at **1.5157** against a train loss of 0.1079,
a 14× gap.

**So `epoch_600.pt` is the worst checkpoint of the run, and ~85% of the 600 epochs made
the model actively worse on unseen captions.**

### Verdict

The run-1 diagnosis was correct and its fixes all landed. The model now fits, and it
overshot straight into memorizing. It is a caption→image lookup table with a weak generic
colour prior, not a generator.

### Findings

**1. Val loss exceeds the predict-zero baseline at 14 of 20 timesteps.** At t=25 the
held-out v-loss is 3.0835 (ε-equivalent 3.00) against a trivial baseline of 1.0. The model
is not merely failing to generalize on unseen captions, it is *confidently wrong*. It
retrieves a wrong memorized latent.

**2. Three of four samples are near-pixel copies.** Nearest-training-latent distances
11.66 / 16.09 / 16.90 against p1 = 68.13 and median = 83.06. Visually confirmed against
ground truth. They match rows 834/835/836, the **mirrored** copies, and the generated
images are correspondingly flipped relative to the originals.

**3. hflip did not buy invariance, it bought a second mode.** Each caption now has exactly
two memorizable targets, and the deterministic sampler picks one. It also broke the
memorization threshold: train-vs-train `min` fell from 18.77 (833 originals) to **8.99**,
because a near-symmetric Pokemon's mirror is nearly identical to itself. Use p1, not min.

**4. The starting noise has no effect.** `prompt effect / total variation = 100.2%`.
Sampling the same prompt from different `x_T` gives the same image. Zero diversity; one
prompt = one output.

**5. Memorization, quantified in one number.** Train v-loss at t=975 is **0.3446**. At ᾱ=0
there is *no image information in the input*, so from pure noise plus a caption the model
reconstructs 66% of the training image's variance. Held-out: 0.9531, i.e. 5%.

**6. The 6702% conditioning gap is an index, not comprehension.** At t=300: matched
0.0311, shuffled 2.1158, **uncond 0.3380**. Given *no* caption it produces something
generic and sane; given a *wrong* caption it confidently produces the wrong image, and
uncond beats shuffled by 6×. Compare run 1's best relative gap of 13.6%.

**7. Capacity vs data is now the binding constraint.** 35,540,836 params against
1,500 × 4,096 = 6,144,000 training scalars: **5.8× more parameters than there are numbers
in the training set**, or 47,388 params per unique image. Memorization is the expected
solution, not a surprise.

**8. `guidance_rescale` works as intended.** At CFG 7.5 without it, std blows to 1.6228 and
the per-channel means invert to `[1.03, 0.63, -0.73, -0.21]`, which is run 1's exact failure.
With `rescale=0.7`, std holds at 0.8752 and means stay `[0.32, 0.24, 0.33, 0.13]`.

**9. No sampler bug.** DDIM 25/50/250, DDIM eta=1.0, and DDPM 1000 agree within ~3.6% on
every latent statistic (std 0.6996 → 0.7251). The x₀-posterior DDPM form is finite at ᾱ=0
as intended, and run 1's DDIM-drifts-to-the-mean behaviour is gone.

**10. Novel prompts get colour right, structure wrong.** "red fire dragon" → red, "blue
water turtle" → blue dome, "yellow electric mouse" → yellow, "green grass" → green. Real
text signal, amorphous shape. That is the ceiling of a lookup table asked to extrapolate.

**11. Samples are over-smoothed AND drift in brightness, two errors that cancelled.**
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
separately. Both are consistent with a model that is still weak at high `t`: the DDIM
deterministic trajectory collapsing toward conditional means, plus a residual per-sample
DC that per-channel normalization does not fix (it normalizes the *dataset's* channel means;
each individual sample still has to generate its own level).

### Action items for run 3

Ordered by expected value per unit of effort.

- [ ] **Sweep `CKPT_PATH` over `epoch_50 … epoch_600` and take the val minimum.** Free,
      since the checkpoints already exist and `epoch_50`/`epoch_100` sit at the bottom of the
      curve. Do this before anything else; you may already have a usable model.
- [ ] **~100 epochs, not 600.** You overshot by roughly 10×. Spend the saved compute on
      3–5 below instead of on more steps.
- [ ] **Shrink to `base_channels=64`** (17.4M params). 3× fewer params per training scalar,
      and 2.2× cheaper per sample on top.
- [ ] **Turn regularization back on:** `dropout=0.1`, `weight_decay=0.05`,
      `cfg_dropout_prob=0.15`. Run 1's underfit justified stripping these; run 2 does not.
- [ ] **Real augmentation.** Precompute 4–8 random-resized-crops per image (scale 0.8–1.0)
      for 6–13k latents instead of 1,666. Largest remaining data lever.
- [ ] **Caption augmentation, high value for this dataset specifically.** The model keys on
      the Pokemon *name* as a lookup token. Add a second caption variant per image with the
      name stripped, forcing it onto the descriptive words. The captions support it: 66%
      contain a colour word, 35% a visual noun, mean length 19.5 words, vocab 2,367. Finding
      10 is direct evidence the descriptive channel already carries signal.
- [ ] **Save a `best.pt` on val improvement**, and checkpoint every 10 epochs. The entire
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
  samples were 28% under-dispersed while their brightness drifted 5.9× too much, and the
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
  agrees, 0.3254 at t=25 falling to 0.0014 at t=975. This matters because it makes "the
  model is bad at low t" read like the expected shape. The real implication: unweighted ε
  puts 31.5% of its signal below t=100 and 0.3% above t=750.
- **Min-SNR-γ=5 is the wrong tool once you switch to v-prediction.** On top of v it peaks
  at SNR=γ (ᾱ=0.833, t≈116) and decays both ways, pushing 45% of the signal into
  t=100–250 and 0.4% above t=750, which reintroduces the low-noise bias. Unweighted v-loss
  is already exactly uniform per timestep at the optimum. `min_snr_gamma` defaults to `None`.
- **`AMP = "bf16"` is not free on a T4.** bf16 needs compute capability ≥ 8.0; Turing is
  7.5. `train.pick_amp_dtype("auto")` picks bf16 only where supported, else fp16, and on
  run 2's Colab GPU it did in fact select fp16 + GradScaler.

### Note on run-1 checkpoints

`epoch_200.pt` will not load into the new `UNet`, because `Stage.res` is now a `ModuleList`
and the keys gained a `.0`. The parameter count is unchanged (12,658,628) with
`UNet(base_channels=64, num_res_blocks=1, top_self_attn=True, dropout=0.1,
time_scale_shift=False)` plus `NoiseScheduler(zero_terminal_snr=False,
prediction_type="eps")`, so a `.res.` → `.res.0.` key remap is enough if you want to
re-evaluate run 1 against the fixed samplers. Otherwise `git show b4f6080:code/unet.py`.

---

## Run 1 · 2026-08-04 · 200 epochs · completed, plateaued early

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
healthy. But **96% of the compute bought a 0.10 → 0.07 improvement.**

### Verdict

Nothing broke. Nothing was verified either. The loss curve cannot answer the three
questions that decide whether this run worked, and none of them were checked:

1. Did cross-attention learn anything, or is this now an unconditional denoiser?
2. Is the EMA, the thing we sample from, a real average, or still near its init?
3. Generation or memorization? 833 images × 200 epochs × 12.6M params.

Diagnostics for all three are now in the notebook (see below).

### Findings

**1. Uniform-`t` MSE is a poor progress metric, and partly explains the flatness.**
Most per-step variance is just *which `t` was drawn*, and the direction is the opposite of
what you might guess: at `t≈25` the measured loss is 0.325, falling to 0.0014 at `t≈975`.
The Bayes-optimal ε-loss for unit-variance data is exactly `alphas_cumprod[t]`, so ε is
*hardest* to recover when the latent is nearly clean and trivial when `x_t` is almost pure
noise. That is the entire 0.002–0.5 band in the plot. Consequence: the aggregate number is
dominated by the low-noise tail (31.5% of the signal below `t=100`, 0.3% above `t=750`), so
it says almost nothing about the mid/high-`t` range that decides image structure. Bin the
loss by `t` instead, and see the run-2 note above on why v-prediction fixes the weighting.

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

**5. The EMA is fine, for a non-obvious reason.** `EMA_DECAY = 0.9999` implies a
10,000-step window, which looks far too slow for a 41,600-step run. But the warmup
ramp `min(decay, (1 + step) / (10 + step))` (`code/train.py:55`) is *still the binding
term at the end*: at step 41,600 it yields 0.999784, so a ~4,623-step window. The EMA
does track the model. Worth verifying numerically anyway, since it is what we sample.

**6. Non-zero terminal SNR.** `sqrt(alphas_cumprod[-1]) = 0.0397`, so at `t=999`
training still leaks `0.0397 · x_0` into `x_t`. Combined with the per-channel latent
means below, that is a DC offset of roughly `[+0.060, +0.034, +0.001, −0.025]`,
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

Computed from `code/latents.pt`. Use these when judging whether a sample is sane.

**Training latents** (833 × 4 × 64 × 64, VAE output × 0.18215):

```
overall    mean +0.4403   std 1.1226   min -3.82   max +4.20
per-chan   mean [+1.503, +0.862, +0.032, -0.636]
per-chan   std  [ 0.731,  0.984,  0.604,  0.735]
```

These are **not** zero-mean / unit-std per channel. The per-channel offset is a known
property of the SD VAE latent space, and 0.18215 is a single global scalar that only
normalizes the overall std. **A generated latent at mean 0.00 / std 1.00 is wrong, not
right.** It decodes to off-colour output.

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
      retrain until the conditioning check has passed. If cross-attention is dead,
      every other change is wasted.
- [ ] **Set `PREVIEW = True`.** A 200-epoch run with no images is not worth repeating.
- [ ] **Drop self-attention from the 64×64 stage** (`down1`/`up1`). Biggest win
      available: spend the freed compute on batch size and depth.
- [ ] **Go to 2 ResNetBlocks per stage** with the reclaimed budget.
- [ ] **Set `AMP = "bf16"`.** Free on a Colab GPU.
- [ ] **Raise `BATCH_SIZE`** once attention is cheaper. At 4, CFG dropout at `p=0.1`
      yields 0.4 uncond rows per step. It averages out over 41.6k steps, but noisily.
- [ ] **Hold out ~80 latents as a validation split.** Currently everything is in
      `train_dataset` (`code/main.py:127`), so memorization is undetectable from loss.
- [ ] **Lower CFG.** `PREVIEW_CFG = 7.5` is the SD default for an 860M model trained
      on billions of images; 12.6M on 833 usually oversaturates well before that.
      The notebook's sweep covers 1.0–15.0; expect 2–4 to win.
- [ ] **Consider rescaling β for zero terminal SNR** (finding 6). Low priority, fix
      it only after the model demonstrably generates something.
- [ ] **Log eval-mode loss periodically**, so the headline number isn't dropout-inflated.

### Added to the notebook

10 cells appended to `code/diffuser.ipynb` (cells 27–36). All load weights from a
checkpoint, so they run in a fresh runtime without retraining. Ordered so each is only
worth reading if the previous one passed. Two `# TODO:` markers to set: `CKPT_PATH`
(cell 28) and `CFG` (cell 34, after seeing the sweep).

| Cell | Check | Failure it catches |
|---|---|---|
| 28 | Checkpoint load, `‖ema − live‖/‖live‖` | EMA still near random init, so samples are noise regardless of loss |
| 29 | Loss binned by timestep, EMA vs live vs predict-zero | *where* on the schedule the model learned; the profile the aggregate curve hides |
| 30 | Matched vs shuffled vs empty caption at fixed noise/`t` | **dead cross-attention**, the most common silent failure, and the loss curve looks identical |
| 31 | `encode_prompt`, `generate`, `show_row`, `latent_stats` | — |
| 32 | Samples from training captions + ground truth | the easiest possible ask |
| 33 | CFG sweep 1.0 / 2.0 / 4.0 / 7.5 / 15.0 | oversaturation; identical columns ⇒ guidance is a no-op |
| 34 | DDIM 25/50/250 vs DDPM 1000 | separates sampler bugs from training bugs |
| 35 | Novel prompts + pairwise latent distance | prompt-independent collapse |
| 36 | Nearest training latent per sample | memorization |