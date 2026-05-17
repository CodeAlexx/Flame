//! BF16-native scalar reductions (sum / mean) with F32 accumulation.
//!
//! Eliminates the BF16→F32 cast → F32-reduce → F32→BF16 cast triple-pass
//! that `GpuOps::sum` used for BF16 inputs. PyTorch's reduce framework
//! (`ATen/native/cuda/Reduce.cuh` → `ReduceSumProdKernel.cu`) reads BF16
//! directly, accumulates in F32 inside the kernel, and writes BF16 out
//! in a single kernel — this module mirrors that.
//!
//! ## Kernel structure (`sum_bf16_to_bf16_kernel`)
//!
//! Two-stage reduce:
//!   1. Each block does a grid-stride load of BF16 inputs, accumulates
//!      partial sums in F32 registers, then warp-shuffle + shared-mem
//!      block reduce.
//!   2. Thread 0 of each block does an `atomicAdd` into a single F32
//!      scratch scalar.
//!   3. A trivial second kernel converts the F32 scalar to BF16. (We
//!      could do this in-place with one extra atomic but the conversion
//!      is one float on the host side, free.)
//!
//! ## Tolerance
//!
//! F32 reductions are **not bit-identical** under different orderings
//! (atomicAdd is non-deterministic across runs). PyTorch does not
//! guarantee this either. Relative tolerance vs the old cast-then-reduce
//! path is ~1e-3 (BF16 has ~3 decimal digits); see
//! `tests/bf16_reduction_parity.rs`.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::{
    driver::{CudaDevice, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};

use crate::dtype::DType;
use crate::tensor::alloc_zeros_from_pool;
use crate::tensor_storage::{ensure_unique_slice, slice_ref, TensorStorage};
use crate::{Error, Result, Shape, Tensor, TensorId};

// Single-pass reduce kernel.
//
// Block size is 256, grid is bounded to (numel + block_size*ITEMS_PER_THREAD - 1) /
// (block_size*ITEMS_PER_THREAD), then capped to keep launch sane on huge
// tensors. Each thread does a grid-stride loop so any element count is
// covered correctly, unlike the legacy F32 `sum_kernel` whose grid was
// hard-capped at 1024 and silently dropped elements past 262144.
const CUDA_REDUCE_SUM_BF16: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void sum_bf16_to_f32_scalar_kernel(
    const __nv_bfloat16* __restrict__ X,
    float* __restrict__ out_scalar,
    long n
) {
    extern __shared__ float sdata[];

    long tid = threadIdx.x;
    long bsz = blockDim.x;

    // Grid-stride accumulate into per-thread F32 register.
    float thread_sum = 0.0f;
    long stride = (long)gridDim.x * bsz;
    for (long i = (long)blockIdx.x * bsz + tid; i < n; i += stride) {
        thread_sum += __bfloat162float(X[i]);
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    // Shared-memory tree reduce (block size assumed power-of-two; we
    // launch with block_size == 256 so this holds).
    for (long s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_scalar, sdata[0]);
    }
}
"#;

// Convert a single F32 scalar to BF16, optionally multiplying by a
// constant scale first (used for mean = sum * (1/n) in-kernel so we
// avoid a host-side D2H sync that would serialize the training stream).
const CUDA_F32_SCALAR_TO_BF16: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void f32_scalar_to_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    float scale
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[0] = __float2bfloat16_rn(src[0] * scale);
    }
}
"#;

fn ensure(dev: &Arc<CudaDevice>, name: &'static str, code: &'static str) -> Result<()> {
    if dev.get_func(name, name).is_some() {
        return Ok(());
    }
    let include_dir = std::env::var("CUDA_INCLUDE_DIR")
        .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
        .unwrap_or_else(|_| "/usr/local/cuda/include".into());
    let mut opts = CompileOptions::default();
    opts.include_paths.push(include_dir.into());
    let ptx = compile_ptx_with_opts(code, opts)
        .map_err(|e| Error::Cuda(format!("nvrtc {}: {:?}", name, e)))?;
    dev.load_ptx(ptx, name, &[name])
        .map_err(|e| Error::Cuda(format!("load_ptx {}: {:?}", name, e)))?;
    Ok(())
}

const BLOCK_SIZE: u32 = 256;
const ITEMS_PER_THREAD: u32 = 8;
// Cap the grid to keep atomicAdd contention bounded. With grid=4096 each
// thread covers ~10 elements for a 10M-element reduce — plenty of
// parallelism, low atomic contention.
const MAX_GRID: u32 = 4096;

/// Sum reduction over all elements of a BF16 tensor, producing a
/// 0-dim BF16 scalar (matching the legacy `GpuOps::sum` shape and dtype
/// for BF16 inputs).
pub fn sum_bf16(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();

    // Scratch F32 scalar, zero-initialized.
    let scratch = alloc_zeros_from_pool(&x.device, 1)?;

    ensure(
        &x.device,
        "sum_bf16_to_f32_scalar_kernel",
        CUDA_REDUCE_SUM_BF16,
    )?;
    let f = x
        .device
        .get_func(
            "sum_bf16_to_f32_scalar_kernel",
            "sum_bf16_to_f32_scalar_kernel",
        )
        .ok_or_else(|| Error::Cuda("sum_bf16_to_f32_scalar_kernel missing".into()))?;

    let xs = match &x.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => {
            return Err(Error::InvalidOperation(
                "sum_bf16 expects BF16 storage".into(),
            ))
        }
    };

    // Grid sizing: aim for one block per ~ITEMS_PER_THREAD*BLOCK_SIZE
    // elements, capped at MAX_GRID. Even if n is huge, the grid-stride
    // loop in the kernel covers everything.
    let work_per_block = (BLOCK_SIZE * ITEMS_PER_THREAD) as usize;
    let raw_grid = n.div_ceil(work_per_block).max(1);
    let grid = raw_grid.min(MAX_GRID as usize) as u32;

    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: BLOCK_SIZE * std::mem::size_of::<f32>() as u32,
    };

    // Build a temporary tensor wrapping the F32 scratch so we can
    // borrow the slice the same way other launchers do.
    unsafe {
        f.launch(cfg, (slice_ref(xs), &scratch, n as i64))?;
    }

    // Convert F32 scalar → BF16 scalar via dedicated kernel.
    let bf16_data = crate::cuda_alloc_pool::pool_alloc_u16(&x.device, 1)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: bf16_data.into(),
            numel: 1,
        },
        shape: Shape::from_dims(&[]),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,
        #[cfg(feature = "autograd_v2")]
        autograd_meta: None,
    };

    ensure(
        &x.device,
        "f32_scalar_to_bf16_kernel",
        CUDA_F32_SCALAR_TO_BF16,
    )?;
    let f_cast = x
        .device
        .get_func("f32_scalar_to_bf16_kernel", "f32_scalar_to_bf16_kernel")
        .ok_or_else(|| Error::Cuda("f32_scalar_to_bf16_kernel missing".into()))?;
    let ys = match &mut out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    let ys = ensure_unique_slice(ys)?;
    let cfg1 = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let scale: f32 = 1.0;
    unsafe {
        f_cast.launch(cfg1, (&scratch, ys, scale))?;
    }

    Ok(out)
}

/// Mean reduction over all elements of a BF16 tensor, producing a 0-dim
/// BF16 scalar. Implemented as `sum_bf16 / numel` with the division
/// fused into the F32→BF16 conversion (avoids one round-trip).
pub fn mean_bf16(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();
    if n == 0 {
        return Err(Error::InvalidInput("mean of empty tensor".into()));
    }

    // Run the reduce kernel.
    let scratch = alloc_zeros_from_pool(&x.device, 1)?;
    ensure(
        &x.device,
        "sum_bf16_to_f32_scalar_kernel",
        CUDA_REDUCE_SUM_BF16,
    )?;
    let f = x
        .device
        .get_func(
            "sum_bf16_to_f32_scalar_kernel",
            "sum_bf16_to_f32_scalar_kernel",
        )
        .ok_or_else(|| Error::Cuda("sum_bf16_to_f32_scalar_kernel missing".into()))?;
    let xs = match &x.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => {
            return Err(Error::InvalidOperation(
                "mean_bf16 expects BF16 storage".into(),
            ))
        }
    };
    let work_per_block = (BLOCK_SIZE * ITEMS_PER_THREAD) as usize;
    let raw_grid = n.div_ceil(work_per_block).max(1);
    let grid = raw_grid.min(MAX_GRID as usize) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: BLOCK_SIZE * std::mem::size_of::<f32>() as u32,
    };
    unsafe {
        f.launch(cfg, (slice_ref(xs), &scratch, n as i64))?;
    }

    // Divide on-device by fusing `* (1/n)` into the F32→BF16 cast.
    // This keeps the whole reduction on a single CUDA stream without
    // forcing a host-side D2H sync that would serialize the training
    // pipeline.
    let bf16_data = crate::cuda_alloc_pool::pool_alloc_u16(&x.device, 1)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: bf16_data.into(),
            numel: 1,
        },
        shape: Shape::from_dims(&[]),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,
        #[cfg(feature = "autograd_v2")]
        autograd_meta: None,
    };
    ensure(
        &x.device,
        "f32_scalar_to_bf16_kernel",
        CUDA_F32_SCALAR_TO_BF16,
    )?;
    let f_cast = x
        .device
        .get_func("f32_scalar_to_bf16_kernel", "f32_scalar_to_bf16_kernel")
        .ok_or_else(|| Error::Cuda("f32_scalar_to_bf16_kernel missing".into()))?;
    let ys = match &mut out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    let ys = ensure_unique_slice(ys)?;
    let cfg1 = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let inv_n: f32 = 1.0 / (n as f32);
    unsafe {
        f_cast.launch(cfg1, (&scratch, ys, inv_n))?;
    }

    Ok(out)
}
