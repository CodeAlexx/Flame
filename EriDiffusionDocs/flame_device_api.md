# FLAME Device API Documentation

This document describes FLAME's device management and CUDA memory operations for EriDiffusion.

## Device Management

### Creating and Managing Devices

```rust
use flame_core::device::Device;
use flame_core::memory_pool::MemoryPool;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

// Create a CUDA device
let device = Device::cuda(0)?; // GPU 0
let device = Device::cuda(1)?; // GPU 1

// Get underlying CUDA device
let cuda_device: &Arc<CudaDevice> = device.cuda_device();

// Get device ordinal
let gpu_id = device.ordinal();

// From existing CudaDevice
let cuda_dev = Arc::new(CudaDevice::new(0)?);
let device = Device::from(cuda_dev);
```

### Memory Pool Management

FLAME uses memory pooling for efficient GPU memory allocation:

```rust
use flame_core::memory_pool::{MemoryPool, MemoryConfig};

// Create memory pool with default config
let pool = MemoryPool::new(device.clone());

// Create with custom config
let config = MemoryConfig {
    initial_size: 1 << 30,      // 1GB initial allocation
    growth_factor: 2.0,         // Double size when growing
    max_size: Some(8 << 30),    // 8GB maximum
    defrag_threshold: 0.7,      // Defragment at 70% fragmentation
};
let pool = MemoryPool::with_config(device.clone(), config);

// Get pool statistics
let stats = pool.stats();
println!("Allocated: {} bytes", stats.allocated);
println!("Reserved: {} bytes", stats.reserved);
println!("Fragmentation: {:.2}%", stats.fragmentation * 100.0);

// Force defragmentation
pool.defragment()?;

// Clear pool (deallocate all)
pool.clear()?;
```

### Memory Allocation Patterns

```rust
// Allocate through pool (automatic)
let tensor = Tensor::zeros(shape, device.clone())?;
// Memory is automatically allocated from pool

// Manual allocation (advanced)
let bytes = shape.numel() * std::mem::size_of::<f32>();
let allocation = pool.allocate(bytes)?;

// Get memory usage
let used = pool.used_memory();
let reserved = pool.reserved_memory();
```

## CUDA Operations

### Stream Management

```rust
use cudarc::driver::{CudaStream, LaunchAsync};

// Default stream (synchronous)
let result = tensor.add(&other)?;

// Custom stream (asynchronous)
let stream = device.cuda_device().fork_default_stream()?;
let result = tensor.add_async(&other, &stream)?;

// Synchronize stream
stream.synchronize()?;
```

### Device Synchronization

```rust
// Synchronize device (wait for all operations)
device.cuda_device().synchronize()?;

// Check if operations are complete
if device.cuda_device().is_idle()? {
    println!("All operations complete");
}
```

### Multi-GPU Operations

```rust
// Create devices for multiple GPUs
let device0 = Device::cuda(0)?;
let device1 = Device::cuda(1)?;

// Move tensor between devices
let tensor_gpu0 = Tensor::randn(shape, 0.0, 1.0, device0.cuda_device().clone())?;
let tensor_gpu1 = tensor_gpu0.to_device(device1.cuda_device())?;

// Peer-to-peer access (if supported)
if device0.cuda_device().can_access_peer(1)? {
    device0.cuda_device().enable_peer_access(1)?;
    // Can now copy directly between GPUs
}
```

## Memory Efficiency Patterns

### Gradient Checkpointing with Memory Pool

```rust
use flame_core::gradient_checkpointing::checkpoint_with_pool;

// Checkpoint a layer to save memory
let output = checkpoint_with_pool(
    || expensive_layer.forward(&input),
    &input,
    &pool
)?;
```

### Memory Profiling

```rust
use flame_core::memory_pool::MemoryProfiler;

let profiler = MemoryProfiler::new(&pool);
profiler.start();

// Run operations
let output = model.forward(&input)?;

let profile = profiler.stop();
println!("Peak memory: {} MB", profile.peak_usage / (1024 * 1024));
println!("Allocations: {}", profile.num_allocations);
```

### Efficient Batch Processing

```rust
// Pre-allocate buffers for batch processing
let batch_size = 32;
let buffer_shape = Shape::from_dims(&[batch_size, 3, 224, 224]);
let buffer = Tensor::zeros(buffer_shape, device.cuda_device().clone())?;

// Reuse buffer for each batch
for batch_data in dataloader {
    // Copy data into pre-allocated buffer
    buffer.copy_from_slice(&batch_data)?;
    
    // Process without new allocations
    let output = model.forward(&buffer)?;
}
```

## CUDA Kernel Integration

FLAME provides direct CUDA kernel access:

```rust
use flame_core::cuda_kernels::KernelManager;

// Get kernel manager
let kernels = KernelManager::get(&device)?;

// Launch custom kernel
let block_size = 256;
let grid_size = (tensor.numel() + block_size - 1) / block_size;
let config = LaunchConfig {
    grid: (grid_size as u32, 1, 1),
    block: (block_size as u32, 1, 1),
    shared_mem: 0,
};

// Example: custom ReLU kernel
kernels.launch_custom(
    "relu_kernel",
    config,
    &[&tensor.storage.as_arg(), tensor.numel() as i32]
)?;
```

## Best Practices

### 1. Device Consistency

Always ensure tensors are on the same device:

```rust
// Check device compatibility
fn ensure_same_device(a: &Tensor, b: &Tensor) -> Result<()> {
    if a.device().ordinal() != b.device().ordinal() {
        return Err(FlameError::DeviceMismatch {
            expected: a.device().ordinal(),
            got: b.device().ordinal(),
        });
    }
    Ok(())
}

// Move to same device if needed
let b_moved = if a.device().ordinal() != b.device().ordinal() {
    b.to_device(a.device())?
} else {
    b.clone()?
};
```

### 2. Memory Pool Usage

```rust
// Good: Let FLAME manage memory
let tensors: Vec<Tensor> = (0..100)
    .map(|_| Tensor::randn(shape.clone(), 0.0, 1.0, device.clone()))
    .collect::<Result<Vec<_>>>()?;

// Memory is efficiently reused through pool

// Bad: Manual allocation without pool
// Don't bypass the memory pool unless absolutely necessary
```

### 3. Synchronization

```rust
// Explicit synchronization when needed
let result = heavy_computation(&input)?;
device.cuda_device().synchronize()?; // Wait for completion
let cpu_value = result.to_cpu()?;

// Avoid unnecessary synchronization
// Don't synchronize between every operation
```

### 4. Multi-GPU Training

```rust
pub struct MultiGPUTrainer {
    devices: Vec<Device>,
    models: Vec<Model>,
}

impl MultiGPUTrainer {
    pub fn new(gpu_ids: &[usize]) -> Result<Self> {
        let devices: Vec<Device> = gpu_ids
            .iter()
            .map(|&id| Device::cuda(id))
            .collect::<Result<Vec<_>>>()?;
        
        // Create model replicas
        let models = devices
            .iter()
            .map(|dev| Model::new(dev.cuda_device().clone()))
            .collect::<Result<Vec<_>>>()?;
        
        Ok(Self { devices, models })
    }
    
    pub fn train_step(&mut self, batch: &Batch) -> Result<f32> {
        // Split batch across GPUs
        let batch_per_gpu = batch.size / self.devices.len();
        
        let mut losses = vec![];
        for (i, (device, model)) in self.devices.iter().zip(&mut self.models).enumerate() {
            let start = i * batch_per_gpu;
            let end = (i + 1) * batch_per_gpu;
            
            let sub_batch = batch.slice(start, end)?;
            let loss = model.forward(&sub_batch)?;
            losses.push(loss);
        }
        
        // Aggregate losses
        Ok(losses.iter().map(|l| l.to_scalar().unwrap()).sum::<f32>() / losses.len() as f32)
    }
}
```

### 5. Memory Debugging

```rust
// Enable memory debugging
std::env::set_var("FLAME_MEMORY_DEBUG", "1");

// Track allocations
let tracker = pool.track_allocations();
let output = model.forward(&input)?;
let allocations = tracker.get_allocations();

for (size, backtrace) in allocations {
    println!("Allocated {} bytes at:", size);
    println!("{:?}", backtrace);
}
```

## Common Pitfalls

1. **Device mismatch**: Always check tensor devices before operations
2. **Memory leaks**: Ensure tensors go out of scope when not needed
3. **Synchronization overhead**: Don't synchronize unnecessarily
4. **Pool fragmentation**: Monitor and defragment when needed
5. **Cross-device operations**: Move tensors explicitly before operations

## Integration Example

```rust
// EriDiffusion device management
use flame_core::device::Device;
use flame_core::memory_pool::MemoryPool;

pub struct EriDiffusionTrainer {
    device: Device,
    pool: MemoryPool,
    model: DiffusionModel,
}

impl EriDiffusionTrainer {
    pub fn new(gpu_id: usize, memory_gb: usize) -> Result<Self> {
        let device = Device::cuda(gpu_id)?;
        
        let config = MemoryConfig {
            initial_size: memory_gb << 30,
            growth_factor: 1.5,
            max_size: None,
            defrag_threshold: 0.8,
        };
        let pool = MemoryPool::with_config(device.cuda_device().clone(), config);
        
        let model = DiffusionModel::new(device.cuda_device().clone())?;
        
        Ok(Self { device, pool, model })
    }
    
    pub fn train(&mut self) -> Result<()> {
        // Monitor memory usage
        println!("Memory before training: {} MB", 
                 self.pool.used_memory() / (1024 * 1024));
        
        // Training loop with memory management
        for epoch in 0..epochs {
            // Defragment if needed
            if self.pool.stats().fragmentation > 0.7 {
                self.pool.defragment()?;
            }
            
            // Train...
            self.train_epoch()?;
            
            // Clear intermediate tensors
            self.pool.trim()?;
        }
        
        Ok(())
    }
}
```