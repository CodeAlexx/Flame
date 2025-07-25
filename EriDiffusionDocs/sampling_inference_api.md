# EriDiffusion Sampling and Inference API Documentation

## Overview

EriDiffusion provides unified sampling and inference capabilities for generating images from trained models. The system supports multiple diffusion model architectures (SDXL, SD 3.5, Flux) with various sampling algorithms and schedulers.

## Core Sampling Components

### 1. Unified Sampling (`src/trainers/unified_sampling.rs`)

Provides a unified interface for sampling across different model types.

```rust
/// Configuration for sampling
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub num_inference_steps: usize,    // Number of denoising steps (default: 30)
    pub guidance_scale: f64,           // CFG scale (default: 7.5)
    pub height: usize,                 // Output height (default: 1024)
    pub width: usize,                  // Output width (default: 1024)
    pub seed: Option<u64>,             // Random seed for reproducibility
    pub output_dir: PathBuf,           // Output directory
}

/// Generate validation samples during training
pub fn generate_validation_samples(
    model_type: &str,
    vae: &Tensor,                      // VAE weights
    text_embeddings: &Tensor,          // Encoded text embeddings
    pooled_embeddings: Option<&Tensor>, // Pooled embeddings (for some models)
    prompts: &[String],                // Text prompts
    config: &SamplingConfig,           // Sampling configuration
    step: usize,                       // Current training step
    device: &Device,                   // Compute device
) -> flame_core::Result<Vec<PathBuf>>;
```

**Supported Model Types:**
- `"sdxl"` - Stable Diffusion XL
- `"sd35"` - Stable Diffusion 3.5
- `"flux"` - Flux

### 2. Image Saving Functions

```rust
/// Save tensor as image
pub fn save_image(
    image_tensor: &Tensor,   // Image tensor [B, C, H, W] in [-1, 1]
    output_dir: &Path,       // Output directory
    model_name: &str,        // Model identifier
    step: usize,             // Training step
    idx: usize,              // Sample index
    prompt: &str,            // Text prompt used
) -> flame_core::Result<PathBuf>;

/// Helper function to save a tensor as an image file
pub fn save_tensor_as_image(
    tensor: &Tensor,         // Image tensor [C, H, W] in [0, 255]
    path: &PathBuf,          // Output path
) -> flame_core::Result<()>;
```

**Image Format:**
- Supports RGB (3 channel) and grayscale (1 channel) images
- Automatically converts from tensor format to image format
- Handles tensor normalization from [-1, 1] to [0, 255]

### Usage Example:

```rust
// Create sampling config
let config = SamplingConfig {
    num_inference_steps: 50,
    guidance_scale: 7.5,
    height: 1024,
    width: 1024,
    seed: Some(42),
    output_dir: PathBuf::from("outputs"),
};

// Generate samples
let prompts = vec![
    "a beautiful sunset".to_string(),
    "a cat in space".to_string(),
];

let saved_paths = generate_validation_samples(
    "sdxl",
    &vae_weights,
    &text_embeddings,
    Some(&pooled_embeddings),
    &prompts,
    &config,
    1000, // step
    &Device::cuda(0)?
)?;
```

## SDXL Sampling

### 1. SDXL Forward Sampling (`src/trainers/sdxl_forward_sampling.rs`)

```rust
pub fn forward_sdxl_sampling(
    weights: &HashMap<String, Tensor>,
    config: &SDXLConfig,
    latent: &Tensor,           // Input latent [B, 4, H/8, W/8]
    text_embeddings: &Tensor,  // Text embeddings [B, 77, 2048]
    pooled_embeddings: &Tensor,// Pooled embeddings [B, 1280]
    timestep: f64,             // Timestep value
    added_cond_kwargs: Option<HashMap<String, Tensor>>,
) -> flame_core::Result<Tensor>;
```

### 2. SDXL Complete Sampling (`src/trainers/sdxl_sampling_complete.rs`)

Provides complete sampling pipeline with scheduler integration.

**Key Components:**
- Noise scheduling
- Classifier-free guidance
- Time embedding
- Conditioning preparation

## Scheduler Integration

### 1. DDPM Scheduler (`src/trainers/ddpm_scheduler.rs`)

```rust
pub struct DDPMScheduler {
    num_train_timesteps: usize,
    betas: Vec<f32>,
    alphas: Vec<f32>,
    alphas_cumprod: Vec<f32>,
}

impl DDPMScheduler {
    pub fn new(num_train_timesteps: usize) -> Self;
    
    /// Add noise to samples
    pub fn add_noise(
        &self,
        original_samples: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> flame_core::Result<Tensor>;
    
    /// Single denoising step
    pub fn step(
        &self,
        model_output: &Tensor,
        timestep: usize,
        sample: &Tensor,
        eta: f64,
        use_clipped_model_output: bool,
        generator: Option<&mut StdRng>,
    ) -> flame_core::Result<Tensor>;
}
```

### 2. Flow Matching Schedulers

For SD 3.5 and Flux models that use flow matching:

```rust
/// Shifted sigmoid schedule for Flux
pub fn get_flux_schedule(
    num_steps: usize,
    shift: f64,
) -> Vec<f64>;

/// Linear schedule for SD 3.5
pub fn get_linear_schedule(
    num_steps: usize,
) -> Vec<f64>;
```

## Model-Specific Sampling

### 1. SDXL Sampling Pipeline

```rust
// 1. Initialize random latent
let latent = Tensor::randn(0.0, 1.0, &[batch_size, 4, h/8, w/8], &device)?;

// 2. Prepare conditioning
let text_cond = encode_prompt(&prompt)?;
let uncond = encode_prompt("")?;  // Uses zeros for SDXL

// 3. Sampling loop
for t in scheduler.timesteps() {
    // Classifier-free guidance
    let noise_pred_cond = unet_forward(&latent, &text_cond, t)?;
    let noise_pred_uncond = unet_forward(&latent, &uncond, t)?;
    let noise_pred = noise_pred_uncond + guidance_scale * 
                     (noise_pred_cond - noise_pred_uncond);
    
    // Scheduler step
    latent = scheduler.step(&noise_pred, t, &latent)?;
}

// 4. Decode with VAE
let image = vae.decode(&latent)?;
```

### 2. SD 3.5 Sampling (Flow Matching)

```rust
// SD 3.5 uses flow matching with 16-channel latents
let latent = Tensor::randn(0.0, 1.0, &[batch_size, 16, h/8, w/8], &device)?;

// Linear timestep schedule
let sigmas = get_linear_schedule(num_steps);

for (i, &sigma) in sigmas.iter().enumerate() {
    let timestep = sigma * 1000.0;
    
    // MMDiT forward pass
    let v_pred = mmdit_forward(&latent, &text_embeddings, timestep)?;
    
    // Flow matching update
    latent = latent + v_pred * (sigmas[i+1] - sigma);
}
```

### 3. Flux Sampling

```rust
// Flux uses patchified latents
let latent = Tensor::randn(0.0, 1.0, &[batch_size, 64, h/16, w/16], &device)?;

// Shifted sigmoid schedule
let sigmas = get_flux_schedule(num_steps, shift);

for &sigma in &sigmas {
    // Double and single stream processing
    let noise_pred = flux_forward(&latent, &text_embeddings, sigma)?;
    
    // Update latent
    latent = scheduler.step(&noise_pred, sigma, &latent)?;
}

// Unpatchify before VAE
let unpatchified = unpatchify(&latent)?; // [B, 16, H/8, W/8]
```

## Advanced Sampling Features

### 1. Guidance Embedding

For models that support guidance embedding (like Flux):

```rust
pub struct GuidanceConfig {
    pub guidance_scale: f64,
    pub guidance_embedding: Option<Tensor>,
    pub bypass_guidance: bool,
}
```

### 2. Dynamic Thresholding

For better sample quality:

```rust
pub fn dynamic_thresholding(
    sample: &Tensor,
    threshold: f64,
    percentile: f64,
) -> flame_core::Result<Tensor>;
```

### 3. Ancestral Sampling

For stochastic samplers:

```rust
pub struct AncestralSamplerConfig {
    pub eta: f64,              // Noise level (0 = deterministic)
    pub s_churn: f64,          // Churn amount
    pub s_tmin: f64,           // Min timestep for churn
    pub s_tmax: f64,           // Max timestep for churn
    pub s_noise: f64,          // Noise multiplier
}
```

## Integration with Training

### 1. Validation During Training

```rust
// In training loop
if step % validation_interval == 0 {
    let samples = generate_validation_samples(
        model_type,
        &vae,
        &cached_embeddings,
        Some(&pooled),
        &validation_prompts,
        &sampling_config,
        step,
        &device
    )?;
    
    // Log samples to tensorboard/wandb
    for path in samples {
        logger.log_image(&path, step)?;
    }
}
```

### 2. LoRA Inference

```rust
/// Apply LoRA weights during inference
pub fn apply_lora_for_inference(
    base_weights: &HashMap<String, Tensor>,
    lora_weights: &HashMap<String, Tensor>,
    scale: f64,
) -> HashMap<String, Tensor>;
```

## Performance Optimization

### 1. Memory-Efficient Sampling

```rust
/// Sample with reduced memory usage
pub struct MemoryEfficientSamplingConfig {
    pub enable_cpu_offload: bool,
    pub enable_sequential_cpu_offload: bool,
    pub enable_attention_slicing: bool,
    pub slice_size: Option<usize>,
}
```

### 2. Batch Sampling

```rust
/// Generate multiple samples in parallel
pub fn batch_sample(
    prompts: &[String],
    config: &SamplingConfig,
    batch_size: usize,
) -> flame_core::Result<Vec<Tensor>>;
```

## Best Practices

1. **Seed Management**: Always set seeds for reproducible results
2. **Memory Management**: Use CPU offloading for large batch sizes
3. **Guidance Scale**: 
   - SDXL: 7-9
   - SD 3.5: 4-7
   - Flux: 3.5 (or 1.0 for Schnell)
4. **Step Count**:
   - Quality: 50-100 steps
   - Speed: 20-30 steps
   - Turbo models: 4-8 steps
5. **Resolution**: Match training resolution for best results

## Error Handling

Common errors and solutions:

```rust
// Handle OOM during sampling
match generate_samples(&config) {
    Err(e) if e.to_string().contains("out of memory") => {
        // Reduce batch size or enable CPU offload
        config.batch_size /= 2;
        config.enable_cpu_offload = true;
        generate_samples(&config)?
    }
    Err(e) => return Err(e),
    Ok(samples) => samples,
}
```