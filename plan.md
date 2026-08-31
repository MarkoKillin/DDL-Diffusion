# Latent Diffusion Model from Scratch in PyTorch

**Goal:** build a text-to-image Latent Diffusion Model trained on `lambdalabs/pokemon-blip-captions`. The diffusion math, the U-Net and the training loop are all written by hand in PyTorch. This is a learning project, so we implement together, step by step.

**Environment:** Google Colab (single GPU, limited VRAM)

**Constraint:** we do NOT train the VAE or Text Encoder. We pre-compute latents and text embeddings using HuggingFace, save them as tensors, then train only the U-Net.

---

## Step 0: theory and math foundation

Before writing any code, understand what we're building.

### What is diffusion?
A diffusion model learns to reverse a gradual noising process. Given a clean image $x_0$, we add Gaussian noise over $T$ timesteps until it becomes pure noise $x_T$. The model learns to predict the noise at each step, so we can start from random noise and iteratively denoise into a coherent image.

### Forward process (adding noise)
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

In practice, this means:
$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Where:
- $\beta_t$ is the noise schedule (small values, increasing over time)
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ (cumulative product)

### Reverse process (denoising)
The model $\epsilon_\theta(x_t, t, c)$ predicts the noise $\epsilon$ given:
- $x_t$: the noisy input
- $t$: the current timestep
- $c$: conditioning (text embeddings)

### Training objective
Simple MSE between predicted and actual noise:
$$L = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]$$

### Classifier-free guidance (CFG)
During training, randomly drop the conditioning (replace the text embedding with the empty-string embedding) 10% of the time. At inference, combine conditional and unconditional predictions:
$$\epsilon = \epsilon_{uncond} + w \cdot (\epsilon_{cond} - \epsilon_{uncond})$$

Where $w$ is the guidance scale (typically 7.5).

### Latent space
We don't diffuse in pixel space, which is too expensive. Instead a pre-trained VAE encodes images into a compressed latent space (512×512×3 → 64×64×4), all the diffusion happens there, and we decode back at the end.

---

## Step 1: dependencies and setup

**Libraries:**
- `torch`, `torchvision` (CUDA-enabled)
- `diffusers`, `transformers` (for VAE and CLIP)
- `datasets` (HuggingFace datasets)
- `accelerate` (optional, for cleaner device handling)
- `numpy`, `tqdm`, `matplotlib`

**Colab setup:**
- Verify GPU with `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`
- Mount Google Drive for saving checkpoints and pre-computed data
- Set seeds (`torch.manual_seed`, `numpy.random.seed`) for reproducibility

---

## Step 2: data pre-computation pipeline (run once)

A one-off script. Run it once, save results to Drive, never run again. It frees the VAE and text encoder from VRAM during training.

1. **Load pre-trained models:**
   - VAE from `stabilityai/sd-vae-ft-mse` (no gated access, no HF login required)
   - CLIP text encoder + tokenizer from `openai/clip-vit-large-patch14`
   - Both in `eval()` mode, gradients disabled, moved to GPU

2. **Process images → latents:**
   - Resize to 512×512, normalize to [-1, 1]
   - Encode through `vae.encode(x).latent_dist.sample()`
   - Scale latents by `0.18215` (Stable Diffusion convention)
   - Keep PyTorch's NCHW: `(N, 4, 64, 64)`

3. **Process captions → embeddings:**
   - Tokenize with the CLIP tokenizer (`max_length=77`, `padding="max_length"`, `truncation=True`)
   - Encode through the CLIP text encoder, take `last_hidden_state`
   - Output shape: `(N, 77, 768)`

4. **Compute the unconditional embedding:**
   - Encode the empty string `""` through CLIP
   - Save separately. CFG needs it during training and inference

5. **Save to disk:**
   - `latents.pt`, shape `(N, 4, 64, 64)` where N ≈ 833
   - `embeddings.pt`, shape `(N, 77, 768)`
   - `uncond_embedding.pt`, shape `(1, 77, 768)`
   - Use `torch.save(tensor.cpu(), path)`

---

## Step 3: diffusion core

Implement as a small `NoiseScheduler` class or a set of pure functions:

- **Timesteps:** $T = 1000$
- **Beta schedule:** linear from $\beta_1 = 0.00085$ to $\beta_T = 0.012$
- **Pre-compute and register as buffers:** `betas`, `alphas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`
- **Forward noising function:** `q_sample(x_0, t, noise)` returns $x_t$
- **All tensors live on the same device as the model** (use `.to(device)` or register as buffers)
- Index per-sample with `tensor[t].view(-1, 1, 1, 1)` so it broadcasts over `(B, C, H, W)`

---

## Step 4: PyTorch U-Net architecture

The U-Net predicts noise $\epsilon_\theta(x_t, t, c)$. Built entirely with `torch.nn` modules.

### Sub-modules to implement

**TimeEmbedding:**
- Sinusoidal positional encoding of timestep $t$ into a vector
- 2-layer MLP: `nn.Linear → nn.SiLU → nn.Linear`
- Output projected and added inside each ResNet block

**ResNetBlock:**
- `GroupNorm → SiLU → Conv3×3 → (add projected t_emb) → GroupNorm → SiLU → Dropout → Conv3×3`
- Skip connection (with 1×1 conv if channels change)

**AttentionBlock (self + cross):**
- Self-attention: Q=K=V from image features (spatial self-attention)
- Cross-attention: Q from image features, K/V from text embeddings `(B, 77, 768)`
- Multi-head attention: 4 heads at 128-dim, 8 heads at 256-dim
- Pre-norm with `GroupNorm` for spatial, `LayerNorm` for the sequence dim
- Use `torch.nn.functional.scaled_dot_product_attention` (fast and memory-efficient)

**Downsample:** `nn.Conv2d(c, c, 3, stride=2, padding=1)` (not pooling)
**Upsample:** `F.interpolate(scale_factor=2, mode="nearest")` + Conv3×3

### Architecture (NCHW throughout)

```
Input: (B, 4, 64, 64)
│
├─ Conv3×3 → (B, 64, 64, 64)
│
├─ DownBlock1: 2×(ResNet + SelfAttn + CrossAttn) → (B, 64, 64, 64) → Downsample → (B, 64, 32, 32)
├─ DownBlock2: 2×(ResNet + SelfAttn + CrossAttn) → (B, 128, 32, 32) → Downsample → (B, 128, 16, 16)
├─ DownBlock3: 2×(ResNet + SelfAttn + CrossAttn) → (B, 256, 16, 16) → Downsample → (B, 256, 8, 8)
│
├─ MidBlock: ResNet + SelfAttn + CrossAttn + ResNet → (B, 256, 8, 8)
│
├─ UpBlock3: 2×(ResNet + SelfAttn + CrossAttn) → (B, 256, 8, 8) → Upsample → (B, 256, 16, 16)
├─ UpBlock2: 2×(ResNet + SelfAttn + CrossAttn) → (B, 128, 16, 16) → Upsample → (B, 128, 32, 32)
├─ UpBlock1: 2×(ResNet + SelfAttn + CrossAttn) → (B, 64, 32, 32) → Upsample → (B, 64, 64, 64)
│
├─ GroupNorm → SiLU → Conv3×3 → (B, 4, 64, 64)
│
Output: predicted noise (B, 4, 64, 64)
```

**Skip connections:** concatenate encoder features with decoder features along the channel dim (`torch.cat(dim=1)`). The first ResNet of each UpBlock takes `2*C` channels in.

**Channel dims kept small** (64/128/256) to fit in Colab VRAM.

---

## Step 5: training loop

### Hyperparameters
- **Batch size:** 4 (maybe 8 if memory allows)
- **Learning rate:** 1e-4
- **LR schedule:** linear warmup (1000 steps) then cosine decay (use `torch.optim.lr_scheduler.LambdaLR` or `transformers.get_cosine_schedule_with_warmup`)
- **Optimizer:** `torch.optim.AdamW`
- **Epochs:** 200-300. 833 samples is small and needs many passes
- **CFG dropout:** p=0.1 (replace the text embedding with the uncond embedding)
- **Grad clipping:** `torch.nn.utils.clip_grad_norm_` at 1.0

### EMA (exponential moving average)
- Maintain a shadow model: same architecture, no gradients
- Update every step: `ema_param.mul_(0.9999).add_(param.data, alpha=0.0001)`
- Or use `torch.optim.swa_utils.AveragedModel` with a custom `avg_fn`
- Use the EMA model for inference. Samples come out smoother and better

### Training step
1. Sample random timesteps $t \sim \text{Uniform}(0, T)$ per batch item
2. Sample random noise $\epsilon \sim \mathcal{N}(0, I)$
3. Compute $x_t$ via `q_sample`
4. CFG dropout: with probability 0.1, swap each row's text embedding for the uncond embedding
5. Forward pass: $\hat{\epsilon} = \text{UNet}(x_t, t, c)$
6. Loss = `F.mse_loss(eps_pred, eps)`
7. `loss.backward()`, clip grads, `optimizer.step()`, `optimizer.zero_grad()`
8. `scheduler.step()`, update EMA

### Performance
- Wrap the model with `torch.compile(model)` after init for a 20-30% speedup
- Use `torch.set_float32_matmul_precision("high")` to enable TF32 on Ampere+

### Monitoring
- Log loss every N steps. A running average is more useful than the instantaneous value
- Generate a sample image every 10-20 epochs, using the EMA model
- Plot the loss curve

### Checkpointing
- Save `{model_state, ema_state, optimizer_state, scheduler_state, step, epoch}` every 25 epochs
- Save to Google Drive, since Colab sessions die
- Implement resume-from-checkpoint

---

## Step 6: inference with the DDPM sampler

The slow but faithful sampler (1000 steps):

1. Start with pure noise $x_T \sim \mathcal{N}(0, I)$, shape `(1, 4, 64, 64)`
2. For $t = T-1, T-2, ..., 0$:
   - Predict noise with conditioning: $\epsilon_{cond} = \text{UNet}(x_t, t, c_{text})$
   - Predict noise without conditioning: $\epsilon_{uncond} = \text{UNet}(x_t, t, c_{empty})$
   - CFG combine: $\epsilon = \epsilon_{uncond} + 7.5 \cdot (\epsilon_{cond} - \epsilon_{uncond})$
   - Compute $x_{t-1}$ using the DDPM update equations
3. Wrap the loop in `torch.no_grad()` and run the model in `eval()` mode
4. Scale the final latent: divide by `0.18215`
5. Decode through the VAE decoder into an RGB image
6. Display with matplotlib

---

## Step 7: inference with the DDIM sampler (fast)

Same model, 50 steps instead of 1000:

- Select a subset of timesteps (e.g. 50 evenly spaced from 1000)
- The DDIM update is deterministic (no added noise between steps when `eta=0`)
- Same CFG logic as DDPM
- Much faster to iterate with during development

---

## Step 8: mixed precision

After everything works in float32:
- Use `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around the forward and loss
- Keep params and optimizer state in float32 (autocast handles the rest)
- For more savings, cast the model directly to `bfloat16`. No GradScaler needed, since bf16 has fp32's exponent range
- Either way VRAM drops a lot, which might allow a larger batch size

---

## Implementation notes

**Collaboration style:** we implement this together. I explain, you write. I review, suggest fixes, and provide code when you're stuck. The goal is to understand every line.

**Order of implementation:**
1. Steps 1-2 first (setup + data pipeline). Run once, verify shapes
2. Step 3 (diffusion math). Test with dummy data
3. Step 4 (U-Net). Build incrementally, verify each sub-module
4. Step 5 (training). Start training, monitor loss
5. Steps 6-7 (inference). Generate samples once the loss looks good
6. Step 8 (optimization). Only if memory forces it

**When things go wrong:**
- OOM → reduce batch size, reduce channels, try bfloat16, enable `torch.utils.checkpoint` on ResNet blocks
- Loss not decreasing → check learning rate, check the noise schedule, verify tensor shapes, check that a stray `.detach()` isn't zeroing grads
- Loss is NaN → grad clipping, lower LR, check for division by zero in attention scaling
- Generated images are noise → train longer, check the CFG implementation, verify the EMA weights are the ones loaded for inference
- Generated images are blurry → check how many images share one caption first. MSE is
  minimized by predicting their average, and averages decode as blur (run 3: 8 augmented views
  per caption made 59.9% of the target variance unpredictable). Then check EMA lag: run 4
  moved `EMA_DECAY` to 0.999 because 0.9999 trailed the live weights by 8% at the best
  checkpoint. Raising the guidance scale hides blur rather than fixing it, and above ~4 it
  inverts the latent statistics

**Dataset info:**
- `lambdalabs/pokemon-blip-captions`: ~833 Pokemon images with text captions
- A small dataset means fast iteration and a real risk of overfitting, which is fine for learning
- Runs 1-4 confirmed the overfitting risk and then hit its ceiling: 750 distinct
  caption→image pairs cap held-out v-loss at ~0.57 regardless of architecture. See
  **Dataset choice for run 5** in [`RUNS.md`](RUNS.md) for the measured argument and the
  candidate replacements (Flowers-102, CUB-200, COCO), with model sizes for each