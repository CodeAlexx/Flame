//! PyTorch-compatible `torch.randn` on CUDA.
//!
//! Goal: produce bit-identical F32 normal samples to
//! `torch.randn(shape, generator=torch.Generator(device='cuda').manual_seed(seed))`
//! for any given `seed`.
//!
//! PyTorch's CUDA normal kernel lives in
//! `aten/src/ATen/native/cuda/DistributionTemplates.h`. The key facts we have
//! to mirror exactly:
//!
//!   1. **Per-thread Philox state.** Each thread calls
//!      `curand_init(seed, idx, /*offset=*/0, &state)`, where `idx` is the
//!      global thread index `blockIdx.x * blockDim.x + threadIdx.x`.
//!   2. **Quad-at-a-time.** The unroll factor is 4 (`sizeof(float4)/sizeof(float)`),
//!      so each thread calls `curand_normal4(&state)` once per grid-stride
//!      iteration to produce four normals.
//!   3. **Grid-stride loop.** Strides are `blockDim.x * gridDim.x` (per
//!      *element*), and unroll iterations of one Philox quad map to elements
//!      `[idx, idx+stride, idx+2*stride, idx+3*stride]`.
//!   4. **Box-Muller in `_curand_box_muller`** (curand_normal.h):
//!         u = x * 2^-32 + 2^-33   (i.e. `(x + 0.5) * 2^-32`)
//!         v = y * 2^-32 * 2*PI + 2^-32 * PI
//!         s = sqrtf(-2 * logf(u));
//!         result.x = s * sinf(v);
//!         result.y = s * cosf(v);
//!      Two normals per `(x,y)` u32 pair; one `curand4` quad → two pairs →
//!      four normals.
//!   5. **Execution policy.** `block_size = 256`,
//!      `grid.x = min(SMs * blocks_per_sm, ceil(numel / block_size))` where
//!      `blocks_per_sm = maxThreadsPerSM / 256`. For a fresh generator with
//!      `manual_seed(s)` the `philox_offset` is 0.
//!
//! Caveat: the output of `torch.randn` is therefore a function of the *GPU
//! SM count* (via grid size). On the test machine (RTX 3090 Ti, 84 SMs,
//! 1536 threads/SM) `blocks_per_sm = 6`, so `grid.x = min(504, ceil(N/256))`.
//! Numel <= 504*256 = 129024 maps to grid=ceil(N/256), and each thread does
//! exactly *one* quad iteration (no grid-stride re-roll). The test
//! fixtures all live below this size to keep validation tractable.

use crate::{Error, Shape, Tensor};
use cudarc::{
    driver::{CudaDevice, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::sync::{Arc, OnceLock};

/// CUDA source mirroring `curand_init` + `curand_normal4` for the
/// `curandStatePhilox4_32_10` engine. We avoid linking against curand at
/// NVRTC time — the constants and the algorithm are well known. The
/// `__umulhi` intrinsic is available in NVRTC.
const CUDA_SRC: &str = r#"
extern "C" {

// Philox4x32_10 constants (curand_philox4x32_x.h).
#define PHILOX_W32_0   (0x9E3779B9u)
#define PHILOX_W32_1   (0xBB67AE85u)
#define PHILOX_M4x32_0 (0xD2511F53u)
#define PHILOX_M4x32_1 (0xCD9E8D57u)

// 2^-32 and 2 * pi * 2^-32 — must match curand's CURAND_2POW32_INV /
// CURAND_2POW32_INV_2PI exactly. PyTorch's `at::native` uses the curand
// header constants, defined in `curand_globals.h`:
//
//   #define CURAND_2POW32_INV (2.3283064e-10f)
//   #define CURAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)
//
// We reproduce those literals verbatim (single-precision constants).
#define CURAND_2POW32_INV (2.3283064e-10f)
#define CURAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)

__device__ __forceinline__ unsigned int flame_mulhilo32(unsigned int a, unsigned int b, unsigned int* hip) {
    *hip = __umulhi(a, b);
    return a * b;
}

__device__ __forceinline__ void flame_philox_round(unsigned int* c0, unsigned int* c1,
                                                    unsigned int* c2, unsigned int* c3,
                                                    unsigned int k0, unsigned int k1) {
    unsigned int hi0, hi1;
    unsigned int lo0 = flame_mulhilo32(PHILOX_M4x32_0, *c0, &hi0);
    unsigned int lo1 = flame_mulhilo32(PHILOX_M4x32_1, *c2, &hi1);
    unsigned int n0 = hi1 ^ *c1 ^ k0;
    unsigned int n1 = lo1;
    unsigned int n2 = hi0 ^ *c3 ^ k1;
    unsigned int n3 = lo0;
    *c0 = n0; *c1 = n1; *c2 = n2; *c3 = n3;
}

// Ten-round Philox4x32_10. Returns the quad in (*c0, *c1, *c2, *c3).
__device__ __forceinline__ void flame_philox_10(unsigned int* c0, unsigned int* c1,
                                                 unsigned int* c2, unsigned int* c3,
                                                 unsigned int k0, unsigned int k1) {
    unsigned int kx = k0, ky = k1;
    // 9 bumps + 10 rounds = the curand layout.
    flame_philox_round(c0, c1, c2, c3, kx, ky); kx += PHILOX_W32_0; ky += PHILOX_W32_1; // 1
    flame_philox_round(c0, c1, c2, c3, kx, ky); kx += PHILOX_W32_0; ky += PHILOX_W32_1; // 2
    flame_philox_round(c0, c1, c2, c3, kx, ky); kx += PHILOX_W32_0; ky += PHILOX_W32_1; // 3
    flame_philox_round(c0, c1, c2, c3, kx, ky); kx += PHILOX_W32_0; ky += PHILOX_W32_1; // 4
    flame_philox_round(c0, c1, c2, c3, kx, ky); kx += PHILOX_W32_0; ky += PHILOX_W32_1; // 5
    flame_philox_round(c0, c1, c2, c3, kx, ky); kx += PHILOX_W32_0; ky += PHILOX_W32_1; // 6
    flame_philox_round(c0, c1, c2, c3, kx, ky); kx += PHILOX_W32_0; ky += PHILOX_W32_1; // 7
    flame_philox_round(c0, c1, c2, c3, kx, ky); kx += PHILOX_W32_0; ky += PHILOX_W32_1; // 8
    flame_philox_round(c0, c1, c2, c3, kx, ky); kx += PHILOX_W32_0; ky += PHILOX_W32_1; // 9
    flame_philox_round(c0, c1, c2, c3, kx, ky);                                          // 10
}

// One Box-Muller pair from two u32s — bit-identical to curand's
// `_curand_box_muller` (curand_normal.h:70-87). NV_IS_DEVICE branch uses
// __sincosf.
__device__ __forceinline__ void flame_box_muller(unsigned int x, unsigned int y,
                                                  float* out_x, float* out_y) {
    float u = ((float)x) * CURAND_2POW32_INV + (CURAND_2POW32_INV * 0.5f);
    float v = ((float)y) * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI * 0.5f);
    float s = sqrtf(-2.0f * logf(u));
    float sv, cv;
    __sincosf(v, &sv, &cv);
    *out_x = s * sv;
    *out_y = s * cv;
}

// PyTorch's distribution_elementwise_grid_stride_kernel specialised for
// curand_normal4 + unroll_factor=4.
//
// Thread layout: idx = blockIdx.x * blockDim.x + threadIdx.x. We init the
// philox state with curand_init(seed, idx, /*offset=*/0).
//
// For numel small enough that the grid covers the tensor in a single
// stride (numel <= grid.x * block_size * 4), each thread does exactly one
// Philox quad. That's the regime our fixtures cover.
__global__ void flame_randn_torch_f32(float* __restrict__ out,
                                       int numel,
                                       unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int unroll = 4;
    // rounded_size = ((numel-1) / (stride*unroll) + 1) * stride * unroll.
    // For our test regime stride*unroll >= numel, so rounded_size = stride*unroll
    // and the for-loop body executes exactly once per thread.
    int rounded_size = ((numel - 1) / (stride * unroll) + 1) * stride * unroll;

    // curand_init replicated.
    //   state.ctr = (0, 0, 0, 0)
    //   state.key.x = (uint32_t)seed
    //   state.key.y = (uint32_t)(seed >> 32)
    //   skipahead_sequence(idx, &state):
    //       ctr.z += (uint32_t)idx;  ctr.w += (uint32_t)(idx >> 32);
    //       output = Philox(ctr, key)   ← we don't store, recompute per call
    //   skipahead(0, &state):   no-op besides STATE += 0; recomputes output.
    // Effective initial state: ctr=(0,0,idx_lo,idx_hi), key=(seed_lo,seed_hi),
    // STATE=0, output=Philox(ctr,key).
    unsigned int k0 = (unsigned int)(seed & 0xFFFFFFFFu);
    unsigned int k1 = (unsigned int)((seed >> 32) & 0xFFFFFFFFu);
    unsigned int idx_u = (unsigned int)idx;            // idx fits in 32 bits
    // Counter quads for successive curand4() calls. After curand_init the
    // *first* curand4 returns Philox((0,0,idx,0), key) — see analysis in
    // module doc.

    // Iteration counter for grid-stride loop. Each iteration consumes one
    // Philox quad and writes up to `unroll` elements.
    unsigned long long curand4_calls = 0;

    for (int linear_index = idx; linear_index < rounded_size; linear_index += stride * unroll) {
        // ctr for THIS curand4 call. The first call (curand4_calls==0) uses
        // the freshly-initialised state's output, which equals
        //   Philox(ctr=(0,0,idx,0), key).
        // Each subsequent curand4 call applies Philox_State_Incr (ctr.x++,
        // carrying through .y, .z, .w) BEFORE returning the prior output,
        // but since we don't store outputs between iterations we just
        // compute the Nth output, where N = curand4_calls.
        //
        // The Nth quad (0-indexed) is Philox(ctr=(N, 0, idx, 0), key)
        // assuming no overflow at the 32-bit boundary (true for tiny test
        // sizes).
        unsigned int c0 = (unsigned int)curand4_calls;
        unsigned int c1 = 0;
        unsigned int c2 = idx_u;
        unsigned int c3 = 0;
        flame_philox_10(&c0, &c1, &c2, &c3, k0, k1);

        // curand_box_muller4: two box-muller pairs per quad.
        //   (x, y) = (c0, c1) → (rx, ry)
        //   (z, w) = (c2, c3) → (rz, rw)
        float rx, ry, rz, rw;
        flame_box_muller(c0, c1, &rx, &ry);
        flame_box_muller(c2, c3, &rz, &rw);

        // Scatter to elements [linear_index, linear_index+stride,
        // linear_index+2*stride, linear_index+3*stride] in unroll order
        // (matches PyTorch's `transform_func(li, (&rand.x)[ii])`).
        int li0 = linear_index;
        int li1 = linear_index + stride;
        int li2 = linear_index + 2 * stride;
        int li3 = linear_index + 3 * stride;
        if (li0 < numel) out[li0] = rx;
        if (li1 < numel) out[li1] = ry;
        if (li2 < numel) out[li2] = rz;
        if (li3 < numel) out[li3] = rw;

        curand4_calls += 1;
    }
}

} // extern "C"
"#;

static MOD_ONCE: OnceLock<()> = OnceLock::new();

fn ensure_module(dev: &Arc<CudaDevice>) -> Result<(), Error> {
    if dev
        .get_func("flame_rng_torch", "flame_randn_torch_f32")
        .is_some()
    {
        return Ok(());
    }
    if MOD_ONCE.get().is_none() {
        let include_path = std::env::var("CUDA_HOME")
            .map(|p| format!("{}/include", p))
            .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
        let mut opts = CompileOptions::default();
        opts.include_paths.push(include_path);
        let ptx = compile_ptx_with_opts(CUDA_SRC, opts)
            .map_err(|e| Error::KernelError(format!("torch_randn NVRTC: {e:?}")))?;
        dev.load_ptx(ptx, "flame_rng_torch", &["flame_randn_torch_f32"])
            .map_err(|e| Error::KernelError(format!("torch_randn load_ptx: {e:?}")))?;
        let _ = MOD_ONCE.set(());
    }
    Ok(())
}

/// Compute PyTorch's grid size for a CUDA distribution kernel.
///
/// Mirrors `calc_execution_policy` in DistributionTemplates.h:
///   block_size = 256
///   grid.x = min(SMs * (maxThreadsPerSM / block_size), ceil(numel / block_size))
fn calc_grid(dev: &Arc<CudaDevice>, numel: usize) -> Result<(u32, u32), Error> {
    let block_size: u32 = 256;
    let mut grid = ((numel as u64 + block_size as u64 - 1) / block_size as u64) as u32;
    use cudarc::driver::sys::CUdevice_attribute as A;
    let sm_count = dev
        .attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(|e| Error::KernelError(format!("query SM count: {e:?}")))?
        as u32;
    let max_threads_per_sm = dev
        .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
        .map_err(|e| Error::KernelError(format!("query maxThreadsPerSM: {e:?}")))?
        as u32;
    let blocks_per_sm = max_threads_per_sm / block_size;
    let cap = sm_count.saturating_mul(blocks_per_sm).max(1);
    if grid > cap {
        grid = cap;
    }
    if grid == 0 {
        grid = 1;
    }
    Ok((grid, block_size))
}

/// Allocate an F32 tensor of the given shape and fill it with samples
/// bit-identical to `torch.randn(shape, generator=torch.Generator(device='cuda').manual_seed(seed))`.
///
/// **Caveat (very important).** The exact bytes depend on the launching
/// GPU's SM count via the grid-size calculation; running this on a GPU
/// with a different SM count than the one that produced the reference
/// fixture will *not* match. This mirrors PyTorch's own behaviour — torch.randn
/// is not portable across GPUs.
pub fn randn_torch(
    seed: u64,
    shape: Shape,
    device: Arc<CudaDevice>,
) -> Result<Tensor, Error> {
    let n = shape.elem_count();
    let mut tensor = Tensor::zeros_dtype(shape, crate::DType::F32, Arc::clone(&device))?;
    if n == 0 {
        return Ok(tensor);
    }
    ensure_module(&device)?;
    let (grid, block) = calc_grid(&device, n)?;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let slice = tensor.storage_mut().try_as_mut_slice_f32()?;
        let func = device
            .get_func("flame_rng_torch", "flame_randn_torch_f32")
            .ok_or_else(|| Error::KernelError("flame_randn_torch_f32 missing".into()))?;
        // Pass `seed` as u64 — DeviceRepr is implemented for u64 in cudarc.
        let seed_arg: u64 = seed;
        let numel_arg: i32 = n as i32;
        func.launch(cfg, (slice, numel_arg, seed_arg))
            .map_err(|e| Error::KernelError(format!("flame_randn_torch_f32 launch: {e:?}")))?;
    }
    tensor.storage_mut().bump_version();
    Ok(tensor)
}
