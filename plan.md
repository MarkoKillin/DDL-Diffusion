# Latent Diffusion Model from Scratch in PyTorch

**Goal:** Build a text-to-image Latent Diffusion Model trained on `lambdalabs/pokemon-blip-captions`. Everything from diffusion math to U-Net to training loop is implemented by hand in PyTorch. This is a learning project — we implement together, step by step.

**Environment:** Google Colab (single GPU, limited VRAM)

**Key Constraint:** We do NOT train the VAE or Text Encoder. We pre-compute latents and text embeddings using HuggingFace, save them as tensors, then train only the U-Net.

---

## Step 0: Theory & Math Foundation

Before writing any code, understand what we're building:

### What is Diffusion?
A diffusion model learns to reverse a gradual noising process. Given a clean image $x_0$, we add Gaussian noise over $T$ timesteps until it becomes pure noise $x_T$. The model learns to predict the noise at each step, allowing us to start from random noise and iteratively denoise into a coherent image.

### Forward Process (Adding Noise)
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

In practice, this means:
$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Where:
- $\beta_t$ is the noise schedule (small values, increasing over time)
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ (cumulative product)

### Reverse Process (Denoising)
The model $\epsilon_\theta(x_t, t, c)$ predicts the noise $\epsilon$ given:
- $x_t$: the noisy input
- $t$: the current timestep
- $c$: conditioning (text embeddings)

### Training Objective
Simple MSE between predicted and actual noise:
$$L = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]$$

### Classifier-Free Guidance (CFG)
During training, randomly drop the conditioning (replace text embedding with empty string embedding) 10% of the time. At inference, combine conditional and unconditional predictions:
$$\epsilon = \epsilon_{uncond} + w \cdot (\epsilon_{cond} - \epsilon_{uncond})$$

Where $w$ is the guidance scale (typically 7.5).

### Latent Space
We don't diffuse in pixel space (too expensive). Instead we use a pre-trained VAE to encode images into a compressed latent space (512×512×3 → 64×64×4), do all the diffusion there, then decode back.

---

## Step 1: Dependencies & Setup

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

## Step 2: Data Pre-computation Pipeline (run once)

This is a one-off script. Run it once, save results to Drive, never run again. Frees the VAE and text encoder from VRAM during training.

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
   - Tokenize with CLIP tokenizer (`max_length=77`, `padding="max_length"`, `truncation=True`)
   - Encode through CLIP text encoder, take `last_hidden_state`
   - Output shape: `(N, 77, 768)`

4. **Compute unconditional embedding:**
   - Encode empty string `""` through CLIP
   - Save separately — needed for CFG during training and inference

5. **Save to disk:**
   - `latents.pt` — shape `(N, 4, 64, 64)` where N ≈ 833
   - `embeddings.pt` — shape `(N, 77, 768)`
   - `uncond_embedding.pt` — shape `(1, 77, 768)`
   - Use `torch.save(tensor.cpu(), path)`

---

## Step 3: Diffusion Core

Implement as a small `NoiseScheduler` class or a set of pure functions:

- **Timesteps:** $T = 1000$
- **Beta schedule:** Linear from $\beta_1 = 0.00085$ to $\beta_T = 0.012$
- **Pre-compute and register as buffers:** `betas`, `alphas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`
- **Forward noising function:** `q_sample(x_0, t, noise)` → returns $x_t`
- **All tensors live on the same device as the model** (use `.to(device)` or register as buffers)
- Index per-sample with `tensor[t].view(-1, 1, 1, 1)` so it broadcasts over `(B, C, H, W)`

---

## Step 4: PyTorch U-Net Architecture

The U-Net predicts noise $\epsilon_\theta(x_t, t, c)$. Built entirely with `torch.nn` modules.

### Sub-modules to implement:

**TimeEmbedding:**
- Sinusoidal positional encoding of timestep $t$ → vector
- 2-layer MLP: `nn.Linear → nn.SiLU → nn.Linear`
- Output projected and added inside each ResNet block

**ResNetBlock:**
- `GroupNorm → SiLU → Conv3×3 → (add projected t_emb) → GroupNorm → SiLU → Dropout → Conv3×3`
- Skip connection (with 1×1 conv if channels change)

**AttentionBlock (Self + Cross):**
- Self-Attention: Q=K=V from image features (spatial self-attention)
- Cross-Attention: Q from image features, K/V from text embeddings `(B, 77, 768)`
- Multi-head attention: 4 heads at 128-dim, 8 heads at 256-dim
- Pre-norm with `GroupNorm` for spatial, `LayerNorm` for sequence dim
- Use `torch.nn.functional.scaled_dot_product_attention` (fast + memory-efficient)

**Downsample:** `nn.Conv2d(c, c, 3, stride=2, padding=1)` (not pooling)
**Upsample:** `F.interpolate(scale_factor=2, mode="nearest")` + Conv3×3

### Architecture (NCHW throughout):

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

**Skip connections:** Concatenate encoder features with decoder features along the channel dim (`torch.cat(dim=1)`). The first ResNet of each UpBlock takes `2*C` channels in.

**Channel dims kept small** (64/128/256) to fit in Colab VRAM.

---

## Step 5: Training Loop

### Hyperparameters:
- **Batch size:** 4 (maybe 8 if memory allows)
- **Learning rate:** 1e-4
- **LR schedule:** Linear warmup (1000 steps) → cosine decay (use `torch.optim.lr_scheduler.LambdaLR` or `transformers.get_cosine_schedule_with_warmup`)
- **Optimizer:** `torch.optim.AdamW`
- **Epochs:** 200-300 (833 samples is small, needs many passes)
- **CFG dropout:** p=0.1 (replace text embedding with uncond embedding)
- **Grad clipping:** `torch.nn.utils.clip_grad_norm_` at 1.0

### EMA (Exponential Moving Average):
- Maintain a shadow model: same architecture, no gradients
- Update every step: `ema_param.mul_(0.9999).add_(param.data, alpha=0.0001)`
- Or use `torch.optim.swa_utils.AveragedModel` with custom `avg_fn`
- Use EMA model for inference (smoother, higher-quality samples)

### Training step:
1. Sample random timesteps $t \sim \text{Uniform}(0, T)$ per batch item
2. Sample random noise $\epsilon \sim \mathcal{N}(0, I)$
3. Compute $x_t$ via `q_sample`
4. CFG dropout: with probability 0.1, swap each row's text embedding for uncond embedding
5. Forward pass: $\hat{\epsilon} = \text{UNet}(x_t, t, c)$
6. Loss = `F.mse_loss(eps_pred, eps)`
7. `loss.backward()`, clip grads, `optimizer.step()`, `optimizer.zero_grad()`
8. `scheduler.step()`, update EMA

### Performance:
- Wrap the model with `torch.compile(model)` after init for ~20-30% speedup
- Use `torch.set_float32_matmul_precision("high")` to enable TF32 on Ampere+

### Monitoring:
- Log loss every N steps (running average is more useful than instantaneous)
- Generate a sample image every 10-20 epochs (using EMA model)
- Plot loss curve

### Checkpointing:
- Save `{model_state, ema_state, optimizer_state, scheduler_state, step, epoch}` every 25 epochs
- Save to Google Drive (Colab sessions die)
- Implement resume-from-checkpoint

---

## Step 6: Inference — DDPM Sampler

The slow but faithful sampler (1000 steps):

1. Start with pure noise $x_T \sim \mathcal{N}(0, I)$, shape `(1, 4, 64, 64)`
2. For $t = T-1, T-2, ..., 0$:
   - Predict noise with conditioning: $\epsilon_{cond} = \text{UNet}(x_t, t, c_{text})$
   - Predict noise without conditioning: $\epsilon_{uncond} = \text{UNet}(x_t, t, c_{empty})$
   - CFG combine: $\epsilon = \epsilon_{uncond} + 7.5 \cdot (\epsilon_{cond} - \epsilon_{uncond})$
   - Compute $x_{t-1}$ using DDPM update equations
3. Wrap the loop in `torch.no_grad()` and run model in `eval()` mode
4. Scale final latent: divide by `0.18215`
5. Decode through VAE decoder → RGB image
6. Display with matplotlib

---

## Step 7: Inference — DDIM Sampler (Fast)

Same model, faster sampling (50 steps instead of 1000):

- Select a subset of timesteps (e.g., 50 evenly spaced from 1000)
- DDIM update is deterministic (no added noise between steps when `eta=0`)
- Same CFG logic as DDPM
- Much faster iteration during development

---

## Step 8: Mixed Precision

After everything works in float32:
- Use `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around the forward+loss
- Keep params and optimizer state in float32 (autocast handles the rest)
- Or for more aggressive savings: cast model directly to `bfloat16` (no GradScaler needed since bf16 has fp32's exponent range)
- Significantly reduces VRAM — might allow larger batch size

---

## Implementation Notes

**Collaboration style:** We implement this together. I explain, you write. I review, suggest fixes, provide code when you're stuck. The goal is to understand every line.

**Order of implementation:**
1. Step 1-2 first (setup + data pipeline) — run once, verify shapes
2. Step 3 (diffusion math) — test with dummy data
3. Step 4 (U-Net) — build incrementally, verify each sub-module
4. Step 5 (training) — start training, monitor loss
5. Step 6-7 (inference) — generate samples once loss looks good
6. Step 8 (optimization) — only if needed for memory

**When things go wrong:**
- OOM → reduce batch size, reduce channels, try bfloat16, enable `torch.utils.checkpoint` on ResNet blocks
- Loss not decreasing → check learning rate, check noise schedule, verify tensor shapes, check that grads aren't getting zeroed by `.detach()` somewhere
- Loss is NaN → grad clipping, lower LR, check for division by zero in attention scaling
- Generated images are noise → train longer, check CFG implementation, verify EMA is loaded for inference
- Generated images are blurry → increase guidance scale, check EMA decay (0.9999 is right; 0.99 is too aggressive)

**Dataset info:**
- `lambdalabs/pokemon-blip-captions`: ~833 Pokemon images with text captions
- Small dataset = fast iteration but risk of overfitting (which is fine for learning)