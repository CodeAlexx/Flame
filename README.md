# FLAME (Fast Learning Accelerated Matrix Engine) Version 2

A pure Rust tensor framework built for GPU-accelerated deep learning with full gradient support. FLAME targets NVIDIA GPUs only and is evolving to replace EriDiffusion’s legacy backend so the training pipeline can mutate gradients freely.

## Highlights

- **GPU-first design** – assumes CUDA; there is no CPU fallback.
- **Training-ready autograd** – tensors carry mutable gradients; full backward pass works.
- **Runtime kernel compilation** – NVRTC JITs CUDA kernels on demand.
- **Zero-copy tensors** – `Arc`-based device buffers keep clones cheap.
- **Pure Rust** – no Python, no bindings; just `cargo build`.

## Why FLAME?

Legacy Rust tensor stacks often assumed immutable tensors, making backprop hacks painful. Diffusion training needs to tweak gradients, clip them, add noise, etc. FLAME starts from a training use case:

- Gradients are first-class (`requires_grad`, mutable hooks).
- CUDA buffer layout matches training workflows (no implicit CPU syncs).
- Kernel seams are explicit, so EriDiffusion can bolt on new ops.
- Safety via Rust types without giving up raw CUDA performance.

## Current State

### ✅ Works today
- Core math: add/mul/matmul, reductions, broadcast helpers.
- Activations: ReLU/Sigmoid/GELU/SiLU/Tanh/LeakyReLU.
- Autograd engine, tensor `requires_grad`, manual grad edits.
- CUDA memory manager, NVRTC JIT for custom kernels.
- Conv2D (NHWC + NCHW variants) forward/backward on GPU.
- Gradient tooling: clipping, normalization, noise, stats.
- Tensor utilities: `min_all`, `max_all`, `sum_all`, `floor`, `ceil`, `round`, `triu`, `flip`, `sub_scalar`.
- Device management via shared `CudaDevice` handles.
- `anyhow::Error` integration, Debug formatting for inspection.

### 🚧 In progress
- Autograd ergonomics (fewer manual hooks).
- LayerNorm / BatchNorm kernels.
- Example migrations for real models.
- Full EriDiffusion Flux integration.

### ❌ Not yet
- Distributed training.
- Mixed precision (FP16/BF16) – planned.
- FlashAttention kernels – planned.
- CPU execution – out of scope.

## Quick start

```rust
use flame_core::{Tensor, Shape};
use cudarc::driver::CudaDevice;

fn main() -> anyhow::Result<()> {
    let device = CudaDevice::new(0)?;
    let a = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?;
    let b = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?;
    let c = a.add(&b)?.relu()?;
    println!("result shape = {:?}", c.shape().dims());
    Ok(())
}
```

## Build & test

Requirements: Rust 1.70+, CUDA 11+, NVIDIA GPU (SM 7.0+).

```bash
cargo build --release
cargo test  --release
# examples are opt-in via feature flag
cargo run -p flame-core --example simple_training_test \
  --features legacy_examples
```

## Architecture at a glance

- `flame-core` – tensor API, autograd, kernels, CUDA plumbing.
- NVRTC kernels – JIT compiled per device.
- Gradient store – separate buffers for clean APIs.
- Autograd tape – records ops, drives backward.

## Example snippets

```rust
let x = Tensor::randn(Shape::from_dims(&[32, 64]), 0.0, 1.0, device.clone())?;
let w = Tensor::randn(Shape::from_dims(&[64, 128]), 0.0, 0.02, device.clone())?;
let y = x.matmul(&w)?.relu()?;
let loss = y.sum()?;
```

```rust
let mut weight = Tensor::randn(Shape::from_dims(&[10, 5]), 0.0, 0.02, device.clone())?;
for _ in 0..epochs {
    let output = input.matmul(&weight)?;
    let loss = compute_mse_loss(&output, &target)?;
    let grad = compute_gradients(&loss, &weight)?; // via autograd API
    weight = weight.sub(&grad.mul_scalar(lr)?)?;
}
```

## EriDiffusion integration

FLAME will power EriDiffusion’s training backends:
- Enables gradient edits (LoRA, DoRA, adapters).
- Supports gradient checkpointing for big UNets/DiTs.
- Custom CUDA kernels for diffusion-only ops.

## Roadmap

### Near term
- Friendlier autograd surface.
- Finish Conv2D + LayerNorm coverage for UNet/DiT.
- Port Flux / SDXL training loops.

### Future
- Kernel perf passes.
- Mixed precision (FP16/BF16).
- FlashAttention kernels.
- Multi-GPU / distributed support.

## Contributing

Issue reports and PRs are welcome. Please run the clippy/cuda smoke gates before submitting:

```bash
cargo clippy -p flame-core --lib --bins --tests \
  -- -D warnings -A clippy::too_many_arguments -A clippy::type_complexity
bash ci/smoke_cuda.sh
```

## License

MIT License. See `LICENSE`.

## Credits

- [cudarc](https://github.com/coreylowman/cudarc)
- NVIDIA CUDA Toolkit
