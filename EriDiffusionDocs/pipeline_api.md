# EriDiffusion Pipeline API Documentation

## Overview

EriDiffusion provides comprehensive training and inference pipelines for different diffusion models. The pipeline system follows a modular architecture with model-specific implementations for SDXL, SD 3.5, and Flux.

## Pipeline Architecture

### Naming Convention

All pipelines follow the standard naming pattern:
- `pipeline_<model>_<type>.rs`
- Examples: `pipeline_sdxl_lora.rs`, `pipeline_sd35_lora.rs`, `pipeline_flux_lora.rs`

## SDXL LoRA Pipeline (`src/trainers/pipeline_sdxl_lora.rs`)

### 1. Main Trainer Class

```rust
pub struct SDXLLoRATrainerFixed {
    // Core components
    device: Device,
    dtype: DType,
    
    // Model weights loaded directly
    unet_weights: Option<HashMap<String, Tensor>>,
    
    // Models
    vae_encoder: Option<SDXLVAENative>,
    text_encoders: Option<TextEncoders>,
    
    // Scheduler
    noise_scheduler: DDPMScheduler,
    
    // LoRA adapters
    lora_collection: Option<LoRACollection>,
    
    // Optimizer
    optimizer: Option<Adam8bit>,
    cpu_offload: Option<CPUOffloadManager>,
    
    // Training state
    global_step: usize,
    gradient_accumulator: GradientAccumulator,
    gradient_checkpoint: Option<SDXLGradientCheckpoint>,
    
    // Configuration
    config: ProcessConfig,
    output_dir: PathBuf,
}
```

### 2. Core Methods

```rust
impl SDXLLoRATrainerFixed {
    /// Create new trainer instance
    pub fn new(
        config: ProcessConfig,
        device: Device,
        dtype: DType,
        output_dir: PathBuf,
    ) -> flame_core::Result<Self>;
    
    /// Setup compute device
    fn setup_device(device_str: &str) -> flame_core::Result<Device>;
    
    /// Load all models
    pub fn load_models(&mut self) -> flame_core::Result<()>;
    
    /// Initialize LoRA adapters
    pub fn init_lora_adapters(&mut self) -> flame_core::Result<()>;
    
    /// Main training loop
    pub fn train(
        &mut self,
        dataloader: DataLoader,
        num_epochs: usize,
    ) -> flame_core::Result<()>;
    
    /// Single training step
    pub fn training_step(
        &mut self,
        batch: &DataBatch,
    ) -> flame_core::Result<f32>;
    
    /// Save checkpoint
    pub fn save_checkpoint(&self, step: usize) -> flame_core::Result<()>;
    
    /// Generate validation samples
    pub fn generate_samples(
        &self,
        prompts: &[String],
        step: usize,
    ) -> flame_core::Result<Vec<PathBuf>>;
}
```

### 3. Configuration Structure

```rust
pub struct ProcessConfig {
    pub model: ModelConfig,
    pub network: NetworkConfig,
    pub train: TrainConfig,
    pub save: SaveConfig,
    pub sample: SampleConfig,
}

pub struct ModelConfig {
    pub name_or_path: String,
    pub vae_path: Option<String>,
    pub clip_l_path: Option<String>,
    pub clip_g_path: Option<String>,
}

pub struct NetworkConfig {
    pub type_: String,              // "lora", "lokr", "dora"
    pub linear: Option<usize>,      // LoRA rank
    pub linear_alpha: Option<f32>,  // LoRA alpha
}

pub struct TrainConfig {
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub learning_rate: f64,
    pub max_train_steps: usize,
    pub gradient_checkpointing: bool,
    pub mixed_precision: Option<String>,
    pub optimizer: String,
}
```

### 4. Training Step Implementation

```rust
fn training_step(&mut self, batch: &DataBatch) -> flame_core::Result<f32> {
    // 1. Encode images to latents
    let latents = self.vae_encoder.encode(&batch.images)?;
    
    // 2. Sample noise
    let noise = Tensor::randn_like(&latents)?;
    let timesteps = self.sample_timesteps(batch.batch_size())?;
    
    // 3. Add noise to latents
    let noisy_latents = self.noise_scheduler.add_noise(
        &latents, &noise, &timesteps
    )?;
    
    // 4. Encode text
    let (text_embeddings, pooled) = self.text_encoders.encode_batch(
        &batch.captions, 77
    )?;
    
    // 5. Forward pass with LoRA
    let noise_pred = self.unet_forward_with_lora(
        &noisy_latents,
        &timesteps,
        &text_embeddings,
        &pooled,
    )?;
    
    // 6. Compute loss
    let loss = mse_loss(&noise_pred, &noise)?;
    
    // 7. Backward pass
    let grads = loss.backward()?;
    
    // 8. Update LoRA weights
    self.optimizer.step(&grads)?;
    
    Ok(loss.to_scalar::<f32>()?)
}
```

### Usage Example:

```rust
// Load configuration
let config = ProcessConfig::from_yaml("config/sdxl_lora.yaml")?;

// Create trainer
let mut trainer = SDXLLoRATrainerFixed::new(
    config,
    Device::cuda(0)?,
    DType::F16,
    PathBuf::from("outputs"),
)?;

// Load models
trainer.load_models()?;
trainer.init_lora_adapters()?;

// Create dataloader
let dataset = ImageDataset::new(&config.dataset)?;
let dataloader = DataLoader::new(dataset, config.train.batch_size);

// Train
trainer.train(dataloader, num_epochs)?;
```

## SD 3.5 LoRA Pipeline (`src/trainers/pipeline_sd35_lora.rs`)

### Key Differences from SDXL:

1. **MMDiT Architecture**: Uses Multimodal Diffusion Transformer
2. **16-channel VAE**: Different latent space
3. **Triple Text Encoding**: CLIP-L + CLIP-G + T5-XXL
4. **Flow Matching**: Different training objective

```rust
pub struct SD35LoRATrainer {
    // Similar structure to SDXL but with:
    mmdit_weights: HashMap<String, Tensor>,
    t5_encoder: Option<T5EncoderModel>,
    flow_matching: bool,
    snr_gamma: Option<f32>,
}

impl SD35LoRATrainer {
    /// SD 3.5 specific forward pass
    fn mmdit_forward(
        &self,
        latents: &Tensor,        // [B, 16, H/8, W/8]
        text_embeddings: &Tensor, // [B, seq_len, 4096]
        timesteps: &Tensor,
    ) -> flame_core::Result<Tensor>;
    
    /// Flow matching loss
    fn compute_flow_matching_loss(
        &self,
        model_output: &Tensor,
        target: &Tensor,
        timesteps: &Tensor,
    ) -> flame_core::Result<Tensor>;
}
```

## Flux LoRA Pipeline (`src/trainers/pipeline_flux_lora_complete.rs`)

### Key Features:

1. **Patchified Processing**: 2x2 patches for 16-channel VAE
2. **Double/Single Stream Blocks**: Hybrid architecture
3. **Guidance Embedding**: Optional guidance bypass
4. **Custom RoPE**: Multiple axis rotary embeddings

```rust
pub struct FluxLoRATrainer {
    // Flux-specific components
    double_blocks: Vec<DoubleStreamBlock>,
    single_blocks: Vec<SingleStreamBlock>,
    guidance_embedder: Option<GuidanceEmbedder>,
    patch_size: usize, // Usually 2
}

impl FluxLoRATrainer {
    /// Patchify latents for Flux processing
    fn patchify(&self, latents: &Tensor) -> flame_core::Result<Tensor>;
    
    /// Flux forward with LoRA
    fn flux_forward_with_lora(
        &self,
        latents: &Tensor,        // [B, 64, H/16, W/16] after patchify
        text_embeddings: &Tensor,
        timesteps: &Tensor,
        guidance: Option<&Tensor>,
    ) -> flame_core::Result<Tensor>;
}
```

## Common Pipeline Components

### 1. Training Loop (`src/trainers/training/mod.rs`)

```rust
pub struct TrainingLoop {
    pub state: TrainingState,
    pub config: TrainingConfig,
    pub callbacks: Vec<Box<dyn TrainingCallback>>,
}

pub trait TrainingCallback {
    fn on_epoch_start(&mut self, epoch: usize);
    fn on_step_end(&mut self, step: usize, loss: f32);
    fn on_validation(&mut self, step: usize, metrics: &HashMap<String, f32>);
}
```

### 2. Checkpoint Management

```rust
pub struct CheckpointManager {
    output_dir: PathBuf,
    max_checkpoints: usize,
    save_optimizer: bool,
}

impl CheckpointManager {
    pub fn save_checkpoint(
        &self,
        step: usize,
        model_state: &HashMap<String, Tensor>,
        optimizer_state: Option<&OptimizerState>,
        metrics: &HashMap<String, f32>,
    ) -> Result<PathBuf>;
    
    pub fn load_checkpoint(
        &self,
        checkpoint_path: &Path,
    ) -> Result<(HashMap<String, Tensor>, Option<OptimizerState>)>;
}
```

### 3. Loss Computation

```rust
pub enum LossType {
    MSE,
    L1,
    Huber(f32),
    SNRWeighted(f32),
}

pub fn compute_loss(
    predictions: &Tensor,
    targets: &Tensor,
    loss_type: LossType,
    timesteps: Option<&Tensor>,
) -> Result<Tensor>;
```

### 4. Gradient Accumulation

```rust
pub struct GradientAccumulator {
    accumulated_grads: HashMap<String, Tensor>,
    accumulation_steps: usize,
    current_step: usize,
}

impl GradientAccumulator {
    pub fn accumulate(&mut self, grads: &HashMap<String, Tensor>) -> Result<()>;
    pub fn step(&mut self) -> Result<Option<HashMap<String, Tensor>>>;
    pub fn zero_grad(&mut self);
}
```

## Advanced Pipeline Features

### 1. Mixed Precision Training

```rust
pub struct MixedPrecisionConfig {
    pub enabled: bool,
    pub dtype: DType,              // BF16 or F16
    pub loss_scale: f32,
    pub dynamic_loss_scaling: bool,
}
```

### 2. Distributed Training

```rust
pub struct DistributedConfig {
    pub backend: String,           // "nccl", "gloo"
    pub world_size: usize,
    pub rank: usize,
    pub local_rank: usize,
}
```

### 3. EMA (Exponential Moving Average)

```rust
pub struct EMAModel {
    decay: f32,
    model_weights: HashMap<String, Tensor>,
    shadow_weights: HashMap<String, Tensor>,
}

impl EMAModel {
    pub fn update(&mut self, model: &HashMap<String, Tensor>) -> Result<()>;
    pub fn apply_shadow(&self) -> HashMap<String, Tensor>;
}
```

## Pipeline Integration

### 1. With Data Loading

```rust
// Create dataset with augmentations
let dataset = ImageDataset::new(config.dataset_path)?
    .with_augmentations(vec![
        RandomCrop(1024),
        RandomHorizontalFlip(0.5),
    ])?;

// Create dataloader
let dataloader = DataLoader::new(
    dataset,
    DataLoaderConfig {
        batch_size: config.batch_size,
        shuffle: true,
        num_workers: 4,
        drop_last: true,
    },
);

// Pass to pipeline
pipeline.train(dataloader, config.num_epochs)?;
```

### 2. With Validation

```rust
// Validation callback
struct ValidationCallback {
    validation_prompts: Vec<String>,
    validation_interval: usize,
}

impl TrainingCallback for ValidationCallback {
    fn on_step_end(&mut self, step: usize, _loss: f32) {
        if step % self.validation_interval == 0 {
            // Generate validation samples
            let samples = pipeline.generate_samples(
                &self.validation_prompts,
                step
            )?;
        }
    }
}
```

### 3. With Monitoring

```rust
// Tensorboard logging
let monitor = TensorboardMonitor::new("./logs")?;
pipeline.add_callback(Box::new(monitor));

// Weights & Biases
let wandb = WandbMonitor::new(project_name)?;
pipeline.add_callback(Box::new(wandb));
```

## Best Practices

1. **Memory Management**: Enable gradient checkpointing for large models
2. **Learning Rate**: Use warmup for stable training
3. **Batch Size**: Use gradient accumulation for effective larger batches
4. **Validation**: Regular validation with diverse prompts
5. **Checkpointing**: Save regularly, keep best checkpoints
6. **Mixed Precision**: Use BF16 for better stability than F16

## Error Handling

```rust
// Graceful OOM handling
match pipeline.training_step(&batch) {
    Err(e) if e.to_string().contains("out of memory") => {
        // Clear cache and retry with smaller batch
        clear_cuda_cache()?;
        let smaller_batch = batch.split(2)?;
        pipeline.training_step(&smaller_batch[0])?
    }
    Err(e) => return Err(e),
    Ok(loss) => loss,
}
```