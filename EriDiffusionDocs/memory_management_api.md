# EriDiffusion Memory Management API Documentation

## Overview

EriDiffusion provides comprehensive memory management capabilities optimized for training large diffusion models on GPUs with limited VRAM (particularly 24GB cards). The system includes memory pooling, CPU offloading, gradient checkpointing, and various optimization techniques.

## Core Memory Management

### 1. Memory Utilities (`src/trainers/memory_utils.rs`)

Basic memory management functions for CUDA operations.

```rust
/// Set environment variables for better CUDA memory management
pub fn setup_cuda_memory_management() -> flame_core::Result<()>;

/// Clear CUDA cache if available
pub fn clear_cuda_cache(device: &Device) -> flame_core::Result<()>;

/// Get current GPU memory usage
pub fn get_gpu_memory_info() -> flame_core::Result<String>;

/// Print memory usage at key points
pub fn log_memory_usage(stage: &str) -> flame_core::Result<()>;
```

**Environment Variables Set:**
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` - Prevents excessive memory reservation
- `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` - Enables TF32 for better performance
- `CUDA_LAUNCH_BLOCKING=0` - Reduces memory fragmentation

**Usage Example:**
```rust
// Setup at program start
setup_cuda_memory_management()?;

// Monitor memory during training
log_memory_usage("Before forward pass")?;
let output = model.forward(&input)?;
log_memory_usage("After forward pass")?;

// Clear cache when needed
clear_cuda_cache(&device)?;
```

### 2. Memory Pool System (`src/memory/pool.rs`)

Advanced memory pooling for efficient tensor allocation.

```rust
pub struct MemoryPool {
    device: Device,
    block_size: usize,
    free_blocks: Vec<MemoryBlock>,
    allocated_blocks: HashMap<usize, MemoryBlock>,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(device: Device, block_size: usize) -> Self;
    
    /// Allocate memory block
    pub fn allocate(&mut self, size: usize) -> Result<MemoryBlock>;
    
    /// Free memory block
    pub fn free(&mut self, block_id: usize) -> Result<()>;
    
    /// Get pool statistics
    pub fn stats(&self) -> PoolStats;
}
```

### 3. CUDA Memory Manager (`src/memory/cuda_allocator.rs`)

Low-level CUDA memory management.

```rust
pub struct CudaAllocator {
    device_id: usize,
    allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
}

impl CudaAllocator {
    /// Allocate CUDA memory
    pub fn allocate(&self, size: usize) -> Result<*mut u8>;
    
    /// Deallocate CUDA memory
    pub fn deallocate(&self, ptr: *mut u8, size: usize);
    
    /// Get current allocation
    pub fn current_allocated(&self) -> usize;
    
    /// Get peak allocation
    pub fn peak_allocated(&self) -> usize;
}
```

## CPU Offloading

### 1. CPU Offload Manager (`src/trainers/cpu_offload_manager.rs`)

Manages offloading of model weights to CPU RAM.

```rust
pub struct CPUOffloadManager {
    device: Device,
    cpu_device: Device,
    offloaded_weights: HashMap<String, Tensor>,
    active_weights: HashSet<String>,
}

impl CPUOffloadManager {
    /// Create new offload manager
    pub fn new(gpu_device: Device) -> Result<Self>;
    
    /// Offload weights to CPU
    pub fn offload_weights(
        &mut self,
        weights: HashMap<String, Tensor>,
        keep_on_gpu: &[String],
    ) -> Result<()>;
    
    /// Load weights to GPU on demand
    pub fn load_to_gpu(&mut self, weight_names: &[String]) -> Result<()>;
    
    /// Clear GPU weights not in use
    pub fn clear_inactive(&mut self) -> Result<()>;
}
```

**Usage Example:**
```rust
let mut offload = CPUOffloadManager::new(Device::cuda(0)?)?;

// Offload UNet weights except currently needed layers
offload.offload_weights(
    unet_weights,
    &["down_blocks.0", "mid_block"]  // Keep these on GPU
)?;

// Load specific layers when needed
offload.load_to_gpu(&["up_blocks.3"])?;

// Clear after use
offload.clear_inactive()?;
```

## Gradient Checkpointing

### 1. SDXL Gradient Checkpoint (`src/trainers/gpu_gradient_checkpoint.rs`)

Memory-efficient gradient computation for SDXL.

```rust
pub struct SDXLGradientCheckpoint {
    enabled: bool,
    checkpoint_blocks: Vec<String>,
}

impl SDXLGradientCheckpoint {
    /// Create with default checkpointing strategy
    pub fn new() -> Self;
    
    /// Enable/disable checkpointing
    pub fn set_enabled(&mut self, enabled: bool);
    
    /// Checkpoint a forward pass
    pub fn checkpoint_forward<F>(
        &self,
        name: &str,
        forward_fn: F,
        inputs: &[&Tensor],
    ) -> Result<Tensor>
    where
        F: Fn(&[&Tensor]) -> Result<Tensor>;
}
```

### 2. Generic Gradient Checkpointing

```rust
/// Checkpoint any forward function
pub fn checkpoint<F, I, O>(
    forward_fn: F,
    inputs: I,
    enabled: bool,
) -> Result<O>
where
    F: Fn(I) -> Result<O>,
    I: Clone;
```

## Memory-Efficient Model Loading

### 1. Memory Efficient Loader (`src/loaders/memory_efficient_loader.rs`)

Loads models without duplicating memory.

```rust
pub struct MemoryEfficientLoader {
    device: Device,
    dtype: DType,
    use_mmap: bool,
}

impl MemoryEfficientLoader {
    /// Load weights with memory mapping
    pub fn load_safetensors(
        &self,
        path: &Path,
        prefix: Option<&str>,
    ) -> Result<HashMap<String, Tensor>>;
    
    /// Load and convert dtype on-the-fly
    pub fn load_with_dtype_conversion(
        &self,
        path: &Path,
        target_dtype: DType,
    ) -> Result<HashMap<String, Tensor>>;
    
    /// Stream weights from disk
    pub fn stream_weights<F>(
        &self,
        path: &Path,
        callback: F,
    ) -> Result<()>
    where
        F: FnMut(&str, &Tensor) -> Result<()>;
}
```

### 2. Lazy SafeTensors Loading (`src/loaders/lazy_safetensors.rs`)

Load tensors on-demand.

```rust
pub struct LazySafetensors {
    file_path: PathBuf,
    metadata: SafeTensorsMetadata,
}

impl LazySafetensors {
    /// Open file without loading tensors
    pub fn open(path: &Path) -> Result<Self>;
    
    /// Get tensor names without loading
    pub fn tensor_names(&self) -> Vec<&str>;
    
    /// Load specific tensor
    pub fn load_tensor(&self, name: &str) -> Result<Tensor>;
    
    /// Load multiple tensors
    pub fn load_tensors(&self, names: &[&str]) -> Result<HashMap<String, Tensor>>;
}
```

## Model-Specific Memory Optimizations

### 1. SDXL Memory Efficient (`src/trainers/sdxl_memory_efficient.rs`)

SDXL-specific memory optimizations.

```rust
pub struct SDXLMemoryConfig {
    pub enable_cpu_offload: bool,
    pub enable_gradient_checkpointing: bool,
    pub enable_attention_slicing: bool,
    pub attention_slice_size: Option<usize>,
    pub vae_tiling: bool,
    pub vae_tile_size: usize,
}

/// Apply memory optimizations to SDXL
pub fn optimize_sdxl_memory(
    config: SDXLMemoryConfig,
) -> Result<()>;
```

### 2. Block Swapping (`src/memory/block_swapping.rs`)

Swap model blocks between GPU and CPU during forward pass.

```rust
pub struct BlockSwapper {
    gpu_device: Device,
    cpu_device: Device,
    block_schedule: Vec<BlockScheduleEntry>,
}

impl BlockSwapper {
    /// Create swapping schedule
    pub fn create_schedule(
        model_blocks: &[String],
        max_gpu_blocks: usize,
    ) -> Vec<BlockScheduleEntry>;
    
    /// Execute forward pass with swapping
    pub fn forward_with_swapping<F>(
        &mut self,
        input: &Tensor,
        forward_fn: F,
    ) -> Result<Tensor>
    where
        F: Fn(&Tensor, &HashMap<String, Tensor>) -> Result<Tensor>;
}
```

## Quantization for Memory Reduction

### 1. Quanto Integration (`src/memory/quanto.rs`)

8-bit and 4-bit quantization support.

```rust
pub struct QuantoConfig {
    pub weights_dtype: QuantoDType,  // Int8, Int4
    pub activations_dtype: Option<QuantoDType>,
    pub modules_to_not_quantize: Vec<String>,
}

/// Quantize model weights
pub fn quantize_model(
    weights: HashMap<String, Tensor>,
    config: QuantoConfig,
) -> Result<HashMap<String, Tensor>>;

/// Dequantize for computation
pub fn dequantize_tensor(
    tensor: &Tensor,
    scale: &Tensor,
    zero_point: Option<&Tensor>,
) -> Result<Tensor>;
```

### 2. Int8 Training Support

```rust
pub struct Int8TrainingConfig {
    pub use_8bit_adam: bool,
    pub gradient_accumulation_dtype: DType,
}

/// Setup INT8 training
pub fn setup_int8_training(
    config: Int8TrainingConfig,
) -> Result<()>;
```

## Memory Profiling

### 1. Memory Profiler (`src/memory/profiler.rs`)

Track memory usage over time.

```rust
pub struct MemoryProfiler {
    device: Device,
    sample_interval: Duration,
    history: Vec<MemorySnapshot>,
}

impl MemoryProfiler {
    /// Start profiling
    pub fn start(&mut self);
    
    /// Stop profiling
    pub fn stop(&mut self);
    
    /// Get memory report
    pub fn report(&self) -> MemoryReport;
    
    /// Save profile to file
    pub fn save_profile(&self, path: &Path) -> Result<()>;
}

pub struct MemorySnapshot {
    pub timestamp: Instant,
    pub allocated: usize,
    pub reserved: usize,
    pub active: usize,
}
```

### 2. Memory Monitoring

```rust
/// Monitor memory during operation
pub fn with_memory_tracking<F, R>(
    operation_name: &str,
    f: F,
) -> Result<(R, MemoryStats)>
where
    F: FnOnce() -> Result<R>;

pub struct MemoryStats {
    pub peak_allocated: usize,
    pub total_allocated: usize,
    pub num_allocations: usize,
}
```

## VAE Tiling for Large Images

### 1. VAE Tiling (`src/trainers/vae_tiling.rs`)

Process large images in tiles to save memory.

```rust
pub struct VAETiling {
    tile_size: usize,
    overlap: usize,
}

impl VAETiling {
    /// Encode image with tiling
    pub fn encode_tiled(
        &self,
        vae: &VAEEncoder,
        image: &Tensor,
    ) -> Result<Tensor>;
    
    /// Decode latents with tiling
    pub fn decode_tiled(
        &self,
        vae: &VAEDecoder,
        latents: &Tensor,
    ) -> Result<Tensor>;
}
```

## Best Practices

1. **Setup Early**: Call `setup_cuda_memory_management()` at program start
2. **Monitor Usage**: Use profiler to identify memory bottlenecks
3. **Gradient Checkpointing**: Essential for large models on 24GB
4. **CPU Offloading**: Offload inactive layers during training
5. **Quantization**: Use INT8 for inference, mixed precision for training
6. **Batch Size**: Start small, increase with optimizations
7. **Clear Cache**: Clear CUDA cache between epochs

## Configuration Examples

### 24GB VRAM Configuration:

```rust
// SDXL on 24GB
let memory_config = SDXLMemoryConfig {
    enable_cpu_offload: true,
    enable_gradient_checkpointing: true,
    enable_attention_slicing: true,
    attention_slice_size: Some(4),
    vae_tiling: true,
    vae_tile_size: 512,
};

// Flux on 24GB
let config = FluxTrainingConfig {
    batch_size: 1,
    gradient_checkpointing: true,
    mixed_precision: "bf16",
    cpu_offload_optimizer: true,
    quantize_base_model: true,
};
```

## Error Recovery

```rust
// Automatic memory recovery
fn train_with_memory_recovery(batch: &Batch) -> Result<f32> {
    let mut attempts = 3;
    let mut batch_size = batch.size();
    
    while attempts > 0 {
        match train_step(batch) {
            Ok(loss) => return Ok(loss),
            Err(e) if is_oom_error(&e) => {
                clear_cuda_cache(&device)?;
                batch_size /= 2;
                
                if batch_size == 0 {
                    return Err(anyhow!("Cannot reduce batch size further"));
                }
                
                attempts -= 1;
            }
            Err(e) => return Err(e),
        }
    }
    
    Err(anyhow!("Failed after memory recovery attempts"))
}
```