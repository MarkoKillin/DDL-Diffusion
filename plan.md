# Latent Diffusion Model from Scratch in JAX/Flax

**Goal:** Build a text-to-image Latent Diffusion Model trained on `lambdalabs/pokemon-blip-captions`. Everything from diffusion math to U-Net to training loop is implemented by hand in JAX/Flax. This is a learning project — we implement together, step by step.

**Environment:** Google Colab (single GPU, limited VRAM)

**Key Constraint:** We do NOT train the VAE or Text Encoder. We pre-compute latents and text embeddings using HuggingFace (PyTorch), save them as NumPy arrays, then work purely in JAX/Flax for everything else.

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
- `jax`, `jaxlib` (GPU-enabled)
- `flax` (neural network library)
- `optax` (optimizers)
- `diffusers`, `transformers` (for pre-computing latents/embeddings)
- `datasets` (HuggingFace datasets)
- `numpy`, `tqdm`, `matplotlib`

**Colab setup:**
- Verify GPU with `jax.devices()`
- Mount Google Drive for saving checkpoints and pre-computed data

---

## Step 2: Data Pre-computation Pipeline (PyTorch → NumPy)

This is a one-off script. Run it once, save results to Drive, never run again.

1. **Load pre-trained models:**
   - VAE from `CompVis/stable-diffusion-v1-4`
   - CLIP text encoder from `openai/clip-vit-large-patch14`
   - Both in `eval()` mode, gradients disabled

2. **Process images → latents:**
   - Resize to 512×512, normalize to [-1, 1]
   - Encode through VAE encoder
   - Scale latents by `0.18215` (Stable Diffusion convention)
   - Permute from PyTorch NCHW `(B, 4, 64, 64)` → JAX NHWC `(B, 64, 64, 4)`

3. **Process captions → embeddings:**
   - Tokenize with CLIP tokenizer (max_length=77, padding, truncation)
   - Encode through CLIP text encoder
   - Output shape: `(B, 77, 768)`

4. **Compute unconditional embedding:**
   - Encode empty string `""` through CLIP
   - Save separately — needed for CFG during training and inference

5. **Save to disk:**
   - `latents.npy` — shape `(N, 64, 64, 4)` where N ≈ 833
   - `embeddings.npy` — shape `(N, 77, 768)`
   - `uncond_embedding.npy` — shape `(1, 77, 768)`

---

## Step 3: Diffusion Core (Pure JAX)

Implement as pure functions (no classes needed):

- **Timesteps:** $T = 1000$
- **Beta schedule:** Linear from $\beta_1 = 0.00085$ to $\beta_T = 0.012$
- **Pre-compute:** `betas`, `alphas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`
- **Forward noising function:** `q_sample(x_0, t, noise, noise_schedule)` → returns $x_t$
- **All arrays as JAX arrays**, indexed by timestep `t`

---

## Step 4: Flax U-Net Architecture

The U-Net predicts noise $\epsilon_\theta(x_t, t, c)$. Built entirely in `flax.linen`.

### Sub-modules to implement:

**TimeEmbedding:**
- Sinusoidal positional encoding of timestep $t$ → vector
- 2-layer MLP: Linear → SiLU → Linear
- Output injected into each ResNet block

**ResNetBlock:**
- GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Conv3×3
- Time embedding added between the two conv layers (project + add)
- Skip connection (with 1×1 conv if channels change)

**AttentionBlock (Cross-Attention):**
- Self-Attention: Q=K=V from image features (spatial self-attention)
- Cross-Attention: Q from image features, K/V from text embeddings `(B, 77, 768)`
- Multi-head attention: 4 heads at 128-dim, 8 heads at 256-dim
- Pre-norm with GroupNorm/LayerNorm

**Downsample:** Conv with stride 2 (not pooling)
**Upsample:** Nearest-neighbor interpolation + Conv

### Architecture:

```
Input: (B, 64, 64, 4)
│
├─ Conv3×3 → (B, 64, 64, 64)
│
├─ DownBlock1: 2×(ResNet + SelfAttn + CrossAttn) → (B, 64, 64, 64) → Downsample → (B, 32, 32, 64)
├─ DownBlock2: 2×(ResNet + SelfAttn + CrossAttn) → (B, 32, 32, 128) → Downsample → (B, 16, 16, 128)
├─ DownBlock3: 2×(ResNet + SelfAttn + CrossAttn) → (B, 16, 16, 256) → Downsample → (B, 8, 8, 256)
│
├─ MidBlock: ResNet + SelfAttn + CrossAttn + ResNet → (B, 8, 8, 256)
│
├─ UpBlock3: 2×(ResNet + SelfAttn + CrossAttn) → (B, 8, 8, 256) → Upsample → (B, 16, 16, 256)
├─ UpBlock2: 2×(ResNet + SelfAttn + CrossAttn) → (B, 16, 16, 128) → Upsample → (B, 32, 32, 128)
├─ UpBlock1: 2×(ResNet + SelfAttn + CrossAttn) → (B, 32, 32, 64) → Upsample → (B, 64, 64, 64)
│
├─ GroupNorm → SiLU → Conv3×3 → (B, 64, 64, 4)
│
Output: predicted noise (B, 64, 64, 4)
```

**Skip connections:** Concatenate encoder features with decoder features (double channels at each UpBlock input, handled by first ResNet block).

**Channel dims kept small** (64/128/256) to fit in Colab VRAM.

---

## Step 5: Training Loop

### Hyperparameters:
- **Batch size:** 4 (maybe 8 if memory allows)
- **Learning rate:** 1e-4
- **LR schedule:** Linear warmup (1000 steps) → cosine decay
- **Optimizer:** AdamW (optax)
- **Epochs:** 200-300 (833 samples is small, needs many passes)
- **CFG dropout:** p=0.1 (replace text embedding with uncond embedding)

### EMA (Exponential Moving Average):
- Keep a shadow copy of model weights: `ema_params = 0.9999 * ema_params + 0.0001 * params`
- Update every training step
- Use EMA params for inference (produces smoother, higher-quality samples)

### Training step (JIT-compiled):
1. Sample random timesteps $t$ for each item in batch
2. Sample random noise $\epsilon$
3. Compute $x_t$ using forward process
4. CFG dropout: with probability 0.1, swap text embeddings for uncond embedding
5. Predict noise: $\hat{\epsilon} = \text{UNet}(x_t, t, c)$
6. Loss = MSE($\epsilon$, $\hat{\epsilon}$)
7. Backprop, update params, update EMA

### PRNG key management:
- Split keys properly at every step (noise key, dropout key, CFG key)
- Pass new keys into `train_step` — never reuse

### Monitoring:
- Log loss every N steps
- Generate a sample image every 10-20 epochs (using EMA params)
- Plot loss curve

### Checkpointing:
- Save model params + EMA params + optimizer state every 25 epochs
- Save to Google Drive (Colab sessions die)
- Implement resume-from-checkpoint

---

## Step 6: Inference — DDPM Sampler

The slow but faithful sampler (1000 steps):

1. Start with pure noise $x_T \sim \mathcal{N}(0, I)$, shape `(1, 64, 64, 4)`
2. For $t = T, T-1, ..., 1$:
   - Predict noise with conditioning: $\epsilon_{cond} = \text{UNet}(x_t, t, c_{text})$
   - Predict noise without conditioning: $\epsilon_{uncond} = \text{UNet}(x_t, t, c_{empty})$
   - CFG combine: $\epsilon = \epsilon_{uncond} + 7.5 \cdot (\epsilon_{cond} - \epsilon_{uncond})$
   - Compute $x_{t-1}$ using DDPM update equations
3. Scale final latent: divide by `0.18215`
4. Permute back to NCHW: `(1, 64, 64, 4)` → `(1, 4, 64, 64)`
5. Decode through VAE decoder → RGB image
6. Display with matplotlib

---

## Step 7: Inference — DDIM Sampler (Fast)

Same model, faster sampling (50 steps instead of 1000):

- Select a subset of timesteps (e.g., 50 evenly spaced from 1000)
- DDIM update is deterministic (no added noise between steps)
- Same CFG logic as DDPM
- Much faster iteration during development

---

## Step 8: bfloat16 Mixed Precision

After everything works in float32:
- Convert model params to bfloat16
- Keep optimizer state in float32 (prevents underflow)
- Significantly reduces VRAM usage — might allow larger batch size

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
- OOM → reduce batch size, reduce channels, try bfloat16
- Loss not decreasing → check learning rate, check noise schedule, verify tensor shapes
- Generated images are noise → train longer, check CFG implementation
- Generated images are blurry → increase guidance scale, check EMA decay

**Dataset info:**
- `lambdalabs/pokemon-blip-captions`: ~833 Pokemon images with text captions
- Small dataset = fast iteration but risk of overfitting (which is fine for learning)