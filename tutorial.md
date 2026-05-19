x# Tutorial: Building a Latent Diffusion Model from Scratch (PyTorch)

This tutorial walks you through `plan.md` as a hands-on learning exercise. The rule: **you write every line yourself**. I only step in to explain concepts, review code, or unblock you when you're truly stuck.

For each task below:
1. Read the relevant section in `plan.md`.
2. Try to implement it yourself first.
3. When stuck, ask me a specific question (not "write this for me").
4. Run the verification step before moving on.

---

## Phase 0 — Understand the Math (no code yet)

**Goal:** be able to explain, in your own words, what a diffusion model does.

### Tasks
- [ ] Read `plan.md` Step 0 carefully.
- [ ] On paper, derive `x_t` from `x_0` using the reparameterization trick. Why does it work in one step instead of `t` steps?
- [ ] Write down what `epsilon_theta(x_t, t, c)` outputs and why we predict noise rather than the clean image.
- [ ] Explain CFG to yourself: what does `w=1` mean? `w=0`? `w=7.5`?
- [ ] Explain why we work in latent space (not pixel space).

**Checkpoint:** Open a notebook cell and write a markdown summary (4-6 bullet points) of the diffusion process in your own words. If you can't, re-read.

---

## Phase 1 — Environment Setup

**Goal:** working PyTorch + CUDA environment on Colab.

### Tasks
- [x] Open a fresh Colab notebook, set runtime to GPU.
- [x] Install dependencies (Step 1 of `plan.md`). Most are already on Colab; you'll mostly need `diffusers` and `accelerate`.
- [x] Run `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)` — confirm a GPU is visible.
- [ ] Mount Google Drive and create a folder like `/content/drive/MyDrive/ddl-diffusion/` for checkpoints + precomputed data.
- [x] Set seeds: `torch.manual_seed(42)`, `np.random.seed(42)`.

**Checkpoint:** `torch.zeros(4, 4, device="cuda")` runs without error.

---

## Phase 2 — Data Pre-computation (run once)

**Goal:** turn the Pokemon dataset into three tensor files: `latents.pt`, `embeddings.pt`, `uncond_embedding.pt`.

You only run this once. After it's done, restart the runtime to free VRAM and never touch the VAE/CLIP again until inference.

### Sub-steps
1. **Load the dataset** with `datasets.load_dataset("lambdalabs/pokemon-blip-captions")`. Inspect one sample — what fields are there? What size are the images?
2. **Load the VAE** from `stabilityai/sd-vae-ft-mse` with `AutoencoderKL.from_pretrained(...)`. Move to GPU, `.eval()`, set `requires_grad_(False)`. No HF login needed.
3. **Write an image preprocessing function** using `torchvision.transforms`: PIL → resize 512 → CenterCrop → ToTensor → Normalize to [-1, 1].
4. **Encode one image** through the VAE first. Use `vae.encode(x).latent_dist.sample()`. Check the output shape — should be `(1, 4, 64, 64)`. Multiply by `0.18215`.
5. **Now loop the whole dataset** in batches of 4-8 (use `torch.no_grad()` and `torch.autocast` to save memory). Collect all latents on CPU as you go.
6. **Stack into `(N, 4, 64, 64)`** and `torch.save` as `latents.pt`.
7. **Load CLIP tokenizer + text encoder** from `openai/clip-vit-large-patch14`.
8. **Encode all captions:** tokenize with `padding="max_length", max_length=77, truncation=True`, then take `text_encoder(input_ids).last_hidden_state`. Shape `(N, 77, 768)`. Save as `embeddings.pt`.
9. **Encode the empty string** `""` the same way. Shape `(1, 77, 768)`. Save as `uncond_embedding.pt`.

### Tasks
- [ ] Implement and run each sub-step.
- [ ] Verify shapes after each save: `torch.load(path).shape`.
- [ ] Sanity-check: take one latent, divide by `0.18215`, decode through `vae.decode(latent).sample`, denormalize, display. Should look like the original Pokemon.
- [ ] After saving, restart runtime. From here on, you don't need diffusers/transformers loaded during training.

**Checkpoint:** All three `.pt` files exist on Drive with correct shapes. You can free VRAM by restarting the runtime.

---

## Phase 3 — Diffusion Math

**Goal:** a `NoiseScheduler` (or set of functions) for forward noising. No neural net yet.

### Tasks
- [ ] Write `NoiseScheduler` (a small `nn.Module` is convenient — register tensors as buffers so they move with `.to(device)`).
- [ ] In `__init__`, compute `betas` (linear from 0.00085 to 0.012, length 1000), `alphas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`.
- [ ] Write `q_sample(x_0, t, noise)` that returns `x_t`. Index per-sample with `self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)` so it broadcasts over `(B, C, H, W)`.
- [ ] **Test it:** load one latent, sample noise, run `q_sample` at `t=0`, `t=500`, `t=999`. Decode each through the VAE (in a separate cell, reload VAE briefly). You should see: clean Pokemon, very noisy Pokemon, pure noise.

**Checkpoint:** the visual sanity test passes. If `t=999` doesn't look like pure Gaussian noise, your schedule is wrong.

---

## Phase 4 — U-Net (the big one)

**Goal:** an `nn.Module` that takes `(x_t, t, c)` and returns predicted noise of the same shape as `x_t`.

Build bottom-up. Test each piece with random inputs before composing.

### 4a. TimeEmbedding
- [ ] Write a function `sinusoidal_embedding(t, dim)` that takes integer timesteps `(B,)` and returns `(B, dim)`.
- [ ] Wrap in `nn.Module`: `Linear(dim, dim*4) → SiLU → Linear(dim*4, dim*4)`.
- [ ] **Test:** pass `t = torch.tensor([0, 500, 999])`. Output shape should be `(3, dim*4)`.

### 4b. ResNetBlock
- [ ] Inputs: `(x, t_emb)`. Output: same spatial size, possibly different channels.
- [ ] Structure: `GroupNorm(8, c_in) → SiLU → Conv3x3 → (add Linear(t_emb)[:, :, None, None]) → GroupNorm(8, c_out) → SiLU → Dropout → Conv3x3 → add skip`.
- [ ] If `c_in != c_out`, the skip path needs `nn.Conv2d(c_in, c_out, 1)`.
- [ ] **Test:** input `(2, 64, 32, 32)`, output channels 128 → expect `(2, 128, 32, 32)`.

### 4c. Self-Attention
- [ ] Reshape `(B, C, H, W) → (B, H*W, C)` for attention, reshape back at the end.
- [ ] Project Q, K, V with `nn.Linear`. Split into heads.
- [ ] Use `F.scaled_dot_product_attention(q, k, v)` — it's fast and fused.
- [ ] Output projection + skip connection.
- [ ] **Test:** input `(2, 128, 16, 16)`, output should match.

### 4d. Cross-Attention
- [ ] Q from image features `(B, H*W, C)`, K and V from text `(B, 77, 768)`.
- [ ] Project text via `nn.Linear(768, C)` so K/V live in image-feature space.
- [ ] Same multi-head pattern, but K/V have sequence length 77, Q has length H*W.
- [ ] **Test:** image `(2, 128, 16, 16)` + text `(2, 77, 768)` → output `(2, 128, 16, 16)`.

### 4e. Down/Up sampling
- [ ] `Downsample`: `nn.Conv2d(c, c, 3, stride=2, padding=1)`.
- [ ] `Upsample`: `F.interpolate(scale_factor=2, mode="nearest")` then `nn.Conv2d(c, c, 3, padding=1)`.
- [ ] **Test:** roundtrip — downsample then upsample should preserve shape.

### 4f. Assemble the U-Net
- [ ] Follow the architecture diagram in `plan.md` Step 4. Channels: 64 → 128 → 256.
- [ ] Track skip connections carefully — every encoder block output must be saved and concatenated to the matching decoder input via `torch.cat([x, skip], dim=1)`.
- [ ] **Test:** pass a fake batch `x = torch.randn(2, 4, 64, 64)`, `t = torch.randint(0, 1000, (2,))`, `c = torch.randn(2, 77, 768)`. Output must be `(2, 4, 64, 64)`.
- [ ] Print `sum(p.numel() for p in model.parameters())`. Should be in the low millions (5-30M).

**Checkpoint:** forward pass works on real precomputed latents and embeddings without crashing. The output is garbage — that's expected. Move on.

---

## Phase 5 — Training Loop

**Goal:** a training step that runs and decreases the loss.

### Setup tasks
- [ ] Build a `TensorDataset(latents, embeddings)` and wrap in `DataLoader(batch_size=4, shuffle=True, num_workers=0)`.
- [ ] Move `latents`, `embeddings`, `uncond_embedding` to GPU once if they fit (833 × 4 × 64 × 64 × 4 bytes ≈ 50MB — fine).
- [ ] Instantiate `model`, `scheduler` (NoiseScheduler), `optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)`.
- [ ] Build LR schedule: warmup 1000 steps → cosine to 0. `transformers.get_cosine_schedule_with_warmup` is one liner.
- [ ] Build EMA model: a deep-copied `nn.Module` with `requires_grad_(False)` on all params.

### Training step
Write a function `train_step(batch)`:
1. Unpack `(x_0, c)`. Move to GPU if not already there.
2. `t = torch.randint(0, T, (B,), device=device)`.
3. `noise = torch.randn_like(x_0)`.
4. `x_t = scheduler.q_sample(x_0, t, noise)`.
5. CFG dropout: build a mask `(B,)` of `Bernoulli(0.1)`, replace those rows of `c` with `uncond_embedding.expand_as(...)`.
6. `eps_pred = model(x_t, t, c)`.
7. `loss = F.mse_loss(eps_pred, noise)`.
8. `loss.backward()`, `clip_grad_norm_(params, 1.0)`, `optimizer.step()`, `scheduler.step()`, `optimizer.zero_grad()`.
9. EMA update: `for p, ep in zip(model.parameters(), ema_model.parameters()): ep.mul_(0.9999).add_(p.data, alpha=0.0001)`.

### Tasks
- [ ] Implement `train_step` and run it for 10 batches. Loss should drop from ~1.0 to noticeably less.
- [ ] Wrap with `torch.compile(model)` and confirm it still trains (first step is slow as it compiles).
- [ ] Add running-loss logging every 50 steps.
- [ ] Add checkpoint save every 25 epochs: `{model_state, ema_state, optimizer_state, lr_scheduler_state, step, epoch}` to Drive.
- [ ] Add resume-from-checkpoint logic.
- [ ] Train for 50 epochs as a first real run.

**Checkpoint:** loss curve goes down and plateaus around some non-zero value. Save checkpoint to Drive.

---

## Phase 6 — DDPM Sampler (slow, faithful)

**Goal:** generate one Pokemon image from a text prompt.

### Tasks
- [ ] Write `ddpm_sample(ema_model, prompt_embedding, uncond_embedding, scheduler, cfg_scale=7.5)`.
- [ ] Use `ema_model.eval()` and wrap the whole thing in `torch.no_grad()`.
- [ ] Steps:
  1. `x_t = torch.randn(1, 4, 64, 64, device=device)`.
  2. For `t` from `T-1` down to `0`:
     - `t_batch = torch.full((1,), t, device=device, dtype=torch.long)`.
     - Forward with `c_text` and with `uncond_embedding` (two forward passes, OR concat into a batch of 2 and split).
     - CFG combine: `eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)`.
     - DDPM update:
       - `alpha_t = scheduler.alphas[t]`, `alpha_bar_t = scheduler.alphas_cumprod[t]`, `beta_t = scheduler.betas[t]`.
       - `mean = (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) / sqrt(alpha_t)`.
       - If `t > 0`: add `sqrt(beta_t) * z` where `z = torch.randn_like(x_t)`. Else: `x_{t-1} = mean`.
- [ ] Take final `x_0`, divide by `0.18215`, reload VAE briefly, decode, denormalize to [0,1], display with matplotlib.

**Checkpoint:** image is no longer pure noise. Probably blurry, possibly cursed-looking. That's fine — push training longer if needed.

---

## Phase 7 — DDIM Sampler (fast)

**Goal:** same model, ~50 steps instead of 1000.

### Tasks
- [ ] Pick 50 timesteps evenly spaced in `[0, T-1]` (e.g., `torch.linspace(0, T-1, 50).long().flip(0)`).
- [ ] Implement DDIM update with `eta=0` (deterministic):
  - `x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)`.
  - `x_{t_prev} = sqrt(alpha_bar_t_prev) * x_0_pred + sqrt(1 - alpha_bar_t_prev) * eps`.
- [ ] Same CFG combine as DDPM.
- [ ] Verify image quality is reasonably close to DDPM at 1000 steps.

**Checkpoint:** sampling takes ~3-5s instead of ~60s.

---

## Phase 8 — Mixed Precision (only if needed)

Skip unless you're hitting OOM or want bigger batches.

### Option A — autocast (safer)
- [ ] Wrap forward+loss in `with torch.autocast(device_type="cuda", dtype=torch.bfloat16):`.
- [ ] No `GradScaler` needed for bfloat16 (only float16 needs it).
- [ ] Params and optimizer state stay float32.

### Option B — full bfloat16 (more savings, slightly less stable)
- [ ] Cast model: `model.to(dtype=torch.bfloat16)`. Cast inputs to bf16 before forward.
- [ ] Confirm training still converges. If loss spikes, fall back to Option A.

---

## Debugging Cheatsheet

| Symptom | Likely cause |
|---|---|
| OOM during training | Reduce batch size; reduce U-Net channels; bf16; `torch.utils.checkpoint` on ResNet blocks |
| Loss stuck near 1.0 | Wrong noise schedule; broadcasting bug in `q_sample` (forgot `.view(-1, 1, 1, 1)`); LR too low |
| Loss = NaN | LR too high; missing GroupNorm; need grad clip; division-by-zero in attention |
| `RuntimeError: shape mismatch` in U-Net | Skip connection cat is wrong; first ResNet of UpBlock must accept `2*C` input |
| Generated images = noise | Train longer; check CFG actually swaps to uncond; verify EMA model is the one used at inference |
| Generated images = solid color | Sign error in DDPM update; or model is in `train()` mode at inference (BN/dropout active) |
| Forward pass works, training breaks | A `.detach()` somewhere is killing grads; or a tensor was moved to wrong device |
| `torch.compile` errors | Disable it temporarily — debug eager first, recompile after |

---

## What to ask me

Good questions:
- "Why does GroupNorm work better than BatchNorm in diffusion U-Nets?"
- "My loss is 0.8 and not moving — here's my `train_step`, what looks wrong?"
- "I don't understand the DDPM update equation, can you derive it?"
- "Should I use `register_buffer` or just keep tensors as attributes?"

Less good:
- "Write the U-Net for me."
- "Just give me the code."

Push through the discomfort of writing it yourself — that's where the learning lives.