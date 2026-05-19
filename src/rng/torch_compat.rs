//! PyTorch-compatible CUDA RNG primitives.
//!
//! Goal: produce bit-identical CUDA samples to PyTorch's
//! `torch.randn` / `torch.rand` / `torch.bernoulli` / `torch.randint` and the
//! `torch.nn.init.{kaiming_uniform_, xavier_uniform_}` initialisers, for any
//! given `seed`.
//!
//! All functions in this module share the same Philox4x32-10 state setup
//! that mirrors curand's `curand_init(seed, idx, 0)` and PyTorch's
//! `distribution_nullary_kernel` (DistributionTemplates.h). The "u32 →
//! float", "u32 → integer", and "u32 → normal via Box-Muller" transforms
//! are all derived from the same Philox quad stream — only the per-element
//! transform differs.
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

// ---------------------------------------------------------------------------
// `torch.rand` — uniform [0, 1).
//
// PyTorch's uniform_kernel (DistributionTemplates.h:458) uses curand_uniform4
// (which returns floats in (0, 1]), then computes:
//     value = rand * (to - from) + from
//     reverse_bound_value = (value == to) ? from : value
// For (from=0, to=1) this reduces to `value = rand; if (value == 1.0) value = 0.0;`
// i.e. flip the (0,1] -> [0,1) bound by mapping the 1.0 entry to 0.0.
// _curand_uniform formula (curand_uniform.h:69-72):
//     y = x * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f)
__device__ __forceinline__ float flame_curand_uniform_u32(unsigned int x) {
    return ((float)x) * CURAND_2POW32_INV + (CURAND_2POW32_INV * 0.5f);
}

__global__ void flame_rand_torch_f32(float* __restrict__ out,
                                      int numel,
                                      unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int unroll = 4;
    int rounded_size = ((numel - 1) / (stride * unroll) + 1) * stride * unroll;
    unsigned int k0 = (unsigned int)(seed & 0xFFFFFFFFu);
    unsigned int k1 = (unsigned int)((seed >> 32) & 0xFFFFFFFFu);
    unsigned int idx_u = (unsigned int)idx;
    unsigned long long curand4_calls = 0;

    for (int linear_index = idx; linear_index < rounded_size; linear_index += stride * unroll) {
        unsigned int c0 = (unsigned int)curand4_calls;
        unsigned int c1 = 0;
        unsigned int c2 = idx_u;
        unsigned int c3 = 0;
        flame_philox_10(&c0, &c1, &c2, &c3, k0, k1);

        float u0 = flame_curand_uniform_u32(c0);
        float u1 = flame_curand_uniform_u32(c1);
        float u2 = flame_curand_uniform_u32(c2);
        float u3 = flame_curand_uniform_u32(c3);
        // (0,1] -> [0,1) reverse-bound: if u==1.0 emit 0.0. range=1, from=0.
        if (u0 == 1.0f) u0 = 0.0f;
        if (u1 == 1.0f) u1 = 0.0f;
        if (u2 == 1.0f) u2 = 0.0f;
        if (u3 == 1.0f) u3 = 0.0f;

        int li0 = linear_index;
        int li1 = linear_index + stride;
        int li2 = linear_index + 2 * stride;
        int li3 = linear_index + 3 * stride;
        if (li0 < numel) out[li0] = u0;
        if (li1 < numel) out[li1] = u1;
        if (li2 < numel) out[li2] = u2;
        if (li3 < numel) out[li3] = u3;

        curand4_calls += 1;
    }
}

// ---------------------------------------------------------------------------
// `torch.empty(shape).bernoulli_(p, generator=g)` — scalar Bernoulli.
//
// PyTorch's bernoulli_kernel (DistributionTemplates.h:649) for scalar p uses
// uniform_and_transform with `bernoulli(rand, p) = (rand < p) ? 1.0 : 0.0`,
// where rand is in (0, 1] from curand_uniform4. Output dtype here is F32
// (matches `torch.empty(shape, dtype=torch.float32).bernoulli_(p)`).
__global__ void flame_bernoulli_torch_f32(float* __restrict__ out,
                                           int numel,
                                           unsigned long long seed,
                                           float p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int unroll = 4;
    int rounded_size = ((numel - 1) / (stride * unroll) + 1) * stride * unroll;
    unsigned int k0 = (unsigned int)(seed & 0xFFFFFFFFu);
    unsigned int k1 = (unsigned int)((seed >> 32) & 0xFFFFFFFFu);
    unsigned int idx_u = (unsigned int)idx;
    unsigned long long curand4_calls = 0;

    for (int linear_index = idx; linear_index < rounded_size; linear_index += stride * unroll) {
        unsigned int c0 = (unsigned int)curand4_calls;
        unsigned int c1 = 0;
        unsigned int c2 = idx_u;
        unsigned int c3 = 0;
        flame_philox_10(&c0, &c1, &c2, &c3, k0, k1);

        float u0 = flame_curand_uniform_u32(c0);
        float u1 = flame_curand_uniform_u32(c1);
        float u2 = flame_curand_uniform_u32(c2);
        float u3 = flame_curand_uniform_u32(c3);
        float b0 = (u0 < p) ? 1.0f : 0.0f;
        float b1 = (u1 < p) ? 1.0f : 0.0f;
        float b2 = (u2 < p) ? 1.0f : 0.0f;
        float b3 = (u3 < p) ? 1.0f : 0.0f;

        int li0 = linear_index;
        int li1 = linear_index + stride;
        int li2 = linear_index + 2 * stride;
        int li3 = linear_index + 3 * stride;
        if (li0 < numel) out[li0] = b0;
        if (li1 < numel) out[li1] = b1;
        if (li2 < numel) out[li2] = b2;
        if (li3 < numel) out[li3] = b3;

        curand4_calls += 1;
    }
}

// ---------------------------------------------------------------------------
// `torch.randint(low, high, shape, dtype=torch.int32)` — uniform integers.
//
// PyTorch's random_from_to_kernel (DistributionTemplates.h:287) for range
// < 2^32 uses curand4 (uint4 directly) and the transform
// `uniform_int_from_to<scalar_t>(val, range, base) = (val % range) + base`
// (TransformationHelper.h:41). Output dtype here is I32 (flame-core does not
// have an I64 storage path; matches the int32-cast of torch.randint).
__global__ void flame_randint_torch_i32(int* __restrict__ out,
                                         int numel,
                                         unsigned long long seed,
                                         unsigned int range,
                                         long long base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int unroll = 4;
    int rounded_size = ((numel - 1) / (stride * unroll) + 1) * stride * unroll;
    unsigned int k0 = (unsigned int)(seed & 0xFFFFFFFFu);
    unsigned int k1 = (unsigned int)((seed >> 32) & 0xFFFFFFFFu);
    unsigned int idx_u = (unsigned int)idx;
    unsigned long long curand4_calls = 0;

    for (int linear_index = idx; linear_index < rounded_size; linear_index += stride * unroll) {
        unsigned int c0 = (unsigned int)curand4_calls;
        unsigned int c1 = 0;
        unsigned int c2 = idx_u;
        unsigned int c3 = 0;
        flame_philox_10(&c0, &c1, &c2, &c3, k0, k1);

        // uniform_int_from_to: (val % range) + base, then static_cast<int64_t>,
        // then static_cast<scalar_t>. We narrow to i32 here.
        long long v0 = (long long)(c0 % range) + base;
        long long v1 = (long long)(c1 % range) + base;
        long long v2 = (long long)(c2 % range) + base;
        long long v3 = (long long)(c3 % range) + base;

        int li0 = linear_index;
        int li1 = linear_index + stride;
        int li2 = linear_index + 2 * stride;
        int li3 = linear_index + 3 * stride;
        if (li0 < numel) out[li0] = (int)v0;
        if (li1 < numel) out[li1] = (int)v1;
        if (li2 < numel) out[li2] = (int)v2;
        if (li3 < numel) out[li3] = (int)v3;

        curand4_calls += 1;
    }
}

// ---------------------------------------------------------------------------
// `tensor.uniform_(a, b, generator=g)` — uniform [a, b) with reverse-bound.
//
// Same as flame_rand_torch_f32 but with arbitrary [from, to) range. Kaiming
// and Xavier uniform initialisers both call this with `a = -bound, b = +bound`.
// Matches PyTorch's uniform_kernel (DistributionTemplates.h:458) bit-for-bit.
__global__ void flame_uniform_torch_f32(float* __restrict__ out,
                                         int numel,
                                         unsigned long long seed,
                                         float from,
                                         float to) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int unroll = 4;
    int rounded_size = ((numel - 1) / (stride * unroll) + 1) * stride * unroll;
    unsigned int k0 = (unsigned int)(seed & 0xFFFFFFFFu);
    unsigned int k1 = (unsigned int)((seed >> 32) & 0xFFFFFFFFu);
    unsigned int idx_u = (unsigned int)idx;
    unsigned long long curand4_calls = 0;
    float range = to - from;

    for (int linear_index = idx; linear_index < rounded_size; linear_index += stride * unroll) {
        unsigned int c0 = (unsigned int)curand4_calls;
        unsigned int c1 = 0;
        unsigned int c2 = idx_u;
        unsigned int c3 = 0;
        flame_philox_10(&c0, &c1, &c2, &c3, k0, k1);

        float u0 = flame_curand_uniform_u32(c0);
        float u1 = flame_curand_uniform_u32(c1);
        float u2 = flame_curand_uniform_u32(c2);
        float u3 = flame_curand_uniform_u32(c3);
        float v0 = u0 * range + from;
        float v1 = u1 * range + from;
        float v2 = u2 * range + from;
        float v3 = u3 * range + from;
        // reverse-bound: value == to -> from. See DistributionTemplates.h:474.
        if (v0 == to) v0 = from;
        if (v1 == to) v1 = from;
        if (v2 == to) v2 = from;
        if (v3 == to) v3 = from;

        int li0 = linear_index;
        int li1 = linear_index + stride;
        int li2 = linear_index + 2 * stride;
        int li3 = linear_index + 3 * stride;
        if (li0 < numel) out[li0] = v0;
        if (li1 < numel) out[li1] = v1;
        if (li2 < numel) out[li2] = v2;
        if (li3 < numel) out[li3] = v3;

        curand4_calls += 1;
    }
}

} // extern "C"
"#;

static MOD_ONCE: OnceLock<()> = OnceLock::new();

const KERNEL_NAMES: &[&str] = &[
    "flame_randn_torch_f32",
    "flame_rand_torch_f32",
    "flame_bernoulli_torch_f32",
    "flame_randint_torch_i32",
    "flame_uniform_torch_f32",
];

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
            .map_err(|e| Error::KernelError(format!("torch_compat NVRTC: {e:?}")))?;
        dev.load_ptx(ptx, "flame_rng_torch", KERNEL_NAMES)
            .map_err(|e| Error::KernelError(format!("torch_compat load_ptx: {e:?}")))?;
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

/// Allocate an F32 tensor and fill it with samples bit-identical to
/// `torch.rand(shape, generator=torch.Generator(device='cuda').manual_seed(seed))`.
///
/// Output is in `[0, 1)`. Mirrors PyTorch's `uniform_kernel`
/// (DistributionTemplates.h:458) with `from=0, to=1`: each Philox u32 is
/// mapped through `_curand_uniform` to (0, 1], then any 1.0 value is
/// flipped to 0.0 by the reverse-bound rule.
///
/// Same SM-count caveat as [`randn_torch`].
pub fn rand_torch(
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
            .get_func("flame_rng_torch", "flame_rand_torch_f32")
            .ok_or_else(|| Error::KernelError("flame_rand_torch_f32 missing".into()))?;
        let seed_arg: u64 = seed;
        let numel_arg: i32 = n as i32;
        func.launch(cfg, (slice, numel_arg, seed_arg))
            .map_err(|e| Error::KernelError(format!("flame_rand_torch_f32 launch: {e:?}")))?;
    }
    tensor.storage_mut().bump_version();
    Ok(tensor)
}

/// Allocate an F32 tensor and fill it with samples bit-identical to
/// `torch.empty(shape, dtype=torch.float32, device='cuda').bernoulli_(p,
/// generator=torch.Generator(device='cuda').manual_seed(seed))`.
///
/// Output values are 0.0 or 1.0. Mirrors PyTorch's scalar-`p`
/// `bernoulli_kernel` (DistributionTemplates.h:649) which is
/// `uniform_and_transform` composed with
/// `transformation::bernoulli(rand, p) = (rand < p)` where `rand` is the
/// uniform (0, 1] value from `curand_uniform4`.
///
/// `p` must be in `[0, 1]`. Same SM-count caveat as [`randn_torch`].
pub fn bernoulli_torch(
    seed: u64,
    shape: Shape,
    p: f32,
    device: Arc<CudaDevice>,
) -> Result<Tensor, Error> {
    if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
        return Err(Error::InvalidInput(format!(
            "bernoulli_torch: p must be in [0, 1], got {p}"
        )));
    }
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
            .get_func("flame_rng_torch", "flame_bernoulli_torch_f32")
            .ok_or_else(|| Error::KernelError("flame_bernoulli_torch_f32 missing".into()))?;
        let seed_arg: u64 = seed;
        let numel_arg: i32 = n as i32;
        let p_arg: f32 = p;
        func.launch(cfg, (slice, numel_arg, seed_arg, p_arg))
            .map_err(|e| Error::KernelError(format!("flame_bernoulli_torch_f32 launch: {e:?}")))?;
    }
    tensor.storage_mut().bump_version();
    Ok(tensor)
}

/// Allocate an I32 tensor and fill it with samples that match
/// `torch.randint(low, high, shape, dtype=torch.int32, device='cuda',
/// generator=torch.Generator(device='cuda').manual_seed(seed))`.
///
/// **Constraint**: `high - low` must fit in a `u32` (i.e. range < 2^32).
/// This is the only regime PyTorch's `random_from_to_kernel` uses the
/// uint32 dispatch — and it's the only one flame-core's I32 storage can
/// represent. For ranges ≥ 2^32 you would need an I64 storage path
/// (which flame-core does not currently expose).
///
/// Mirrors `uniform_int_from_to<scalar_t>(val, range, base) = (val %
/// range) + base` (TransformationHelper.h:41).
///
/// Compatibility against `torch.randint` (whose default dtype is `int64`)
/// works because for ranges < 2^32 every sampled value fits in an `i32`
/// and the bit pattern of the i32 cast matches the low 32 bits of the
/// i64. The Python fixture generator casts to int32 before saving for
/// direct bit comparison.
///
/// Same SM-count caveat as [`randn_torch`].
pub fn randint_torch(
    seed: u64,
    low: i64,
    high: i64,
    shape: Shape,
    device: Arc<CudaDevice>,
) -> Result<Tensor, Error> {
    if high <= low {
        return Err(Error::InvalidInput(format!(
            "randint_torch: high ({high}) must be > low ({low})"
        )));
    }
    let range_i64 = high - low;
    if range_i64 > u32::MAX as i64 {
        return Err(Error::InvalidInput(format!(
            "randint_torch: range {range_i64} exceeds 2^32; I64 storage not supported"
        )));
    }
    let range_u32 = range_i64 as u32;
    let n = shape.elem_count();
    let mut tensor = Tensor::zeros_dtype(shape, crate::DType::I32, Arc::clone(&device))?;
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
        let slice = tensor.storage_mut().try_as_mut_slice_i32()?;
        let func = device
            .get_func("flame_rng_torch", "flame_randint_torch_i32")
            .ok_or_else(|| Error::KernelError("flame_randint_torch_i32 missing".into()))?;
        let seed_arg: u64 = seed;
        let numel_arg: i32 = n as i32;
        let range_arg: u32 = range_u32;
        let base_arg: i64 = low;
        func.launch(cfg, (slice, numel_arg, seed_arg, range_arg, base_arg))
            .map_err(|e| Error::KernelError(format!("flame_randint_torch_i32 launch: {e:?}")))?;
    }
    tensor.storage_mut().bump_version();
    Ok(tensor)
}

/// Internal helper: allocate F32 tensor and fill it with uniform[a, b)
/// samples matching `tensor.uniform_(a, b, generator=g)`. Used by the
/// Kaiming and Xavier init functions; not exported because the
/// dedicated init helpers are the intended entry points.
fn uniform_torch(
    seed: u64,
    shape: Shape,
    from: f32,
    to: f32,
    device: Arc<CudaDevice>,
) -> Result<Tensor, Error> {
    if !(from.is_finite() && to.is_finite() && to > from) {
        return Err(Error::InvalidInput(format!(
            "uniform_torch: requires finite from < to, got from={from}, to={to}"
        )));
    }
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
            .get_func("flame_rng_torch", "flame_uniform_torch_f32")
            .ok_or_else(|| Error::KernelError("flame_uniform_torch_f32 missing".into()))?;
        let seed_arg: u64 = seed;
        let numel_arg: i32 = n as i32;
        let from_arg: f32 = from;
        let to_arg: f32 = to;
        func.launch(cfg, (slice, numel_arg, seed_arg, from_arg, to_arg))
            .map_err(|e| Error::KernelError(format!("flame_uniform_torch_f32 launch: {e:?}")))?;
    }
    tensor.storage_mut().bump_version();
    Ok(tensor)
}

/// `gain` for `torch.nn.init.calculate_gain(nonlinearity, a)`.
///
/// Returns `f64` (matching PyTorch's Python-float semantics) — the value is
/// kept at full f64 precision until the final `bound: f32` is computed at
/// the kernel-launch boundary. Casting to `f32` here drops 29 bits of
/// mantissa and breaks bit-exact parity for non-power-of-2 `fan` (e.g.
/// `fan=137, 768, 1280, 3072`), since `gain / sqrt(fan)` would inherit the
/// 1-ulp loss from the gain narrowing.
///
/// Reference: `torch/nn/init.py:calculate_gain` keeps `gain` as Python
/// float (= f64) throughout.
fn kaiming_gain(nonlinearity: &str, a: f64) -> Result<f64, Error> {
    match nonlinearity {
        "leaky_relu" => Ok((2.0_f64 / (1.0 + a * a)).sqrt()),
        "relu" => Ok((2.0_f64).sqrt()),
        "linear" | "sigmoid" | "conv1d" | "conv2d" | "conv3d" | "conv_transpose1d"
        | "conv_transpose2d" | "conv_transpose3d" => Ok(1.0_f64),
        "tanh" => Ok(5.0_f64 / 3.0_f64),
        "selu" => Ok(0.75_f64),
        other => Err(Error::InvalidInput(format!(
            "kaiming_gain: unsupported nonlinearity '{other}'"
        ))),
    }
}

/// Allocate an F32 tensor of the given `shape` and fill it with samples
/// bit-identical to PyTorch's
/// `torch.nn.init.kaiming_uniform_(tensor, a, mode='fan_in',
/// nonlinearity, generator=torch.Generator(device='cuda').manual_seed(seed))`.
///
/// Reference: `torch/nn/init.py:456-518`. The bound is computed as
///   `gain = calculate_gain(nonlinearity, a)`
///   `bound = sqrt(3.0) * gain / sqrt(fan)`
/// where `fan = fan_in` (the canonical default mode) — pass `fan_in` for
/// the typical case, or pass `fan_out` if you need `mode='fan_out'`.
///
/// Returns a fresh tensor (flame-core has no `Tensor::uniform_` in-place
/// at present). Match PyTorch by initialising your weight buffer from
/// this tensor's data.
///
/// Same SM-count caveat as [`randn_torch`].
pub fn kaiming_uniform_torch(
    shape: Shape,
    a: f64,
    fan: usize,
    nonlinearity: &str,
    seed: u64,
    device: Arc<CudaDevice>,
) -> Result<Tensor, Error> {
    if fan == 0 {
        return Err(Error::InvalidInput("kaiming_uniform_torch: fan == 0".into()));
    }
    // Match PyTorch's computation order (`torch/nn/init.py:kaiming_uniform_`):
    //   gain  = calculate_gain(nonlinearity, a)
    //   std   = gain / math.sqrt(fan)
    //   bound = math.sqrt(3.0) * std
    // Every intermediate is Python-float (f64). Only the final `bound` is
    // cast to f32 at the kernel-launch boundary, matching PyTorch's CUDA
    // path which hands f32 (from, to) to its uniform_ kernel.
    let gain = kaiming_gain(nonlinearity, a)?;
    let std = gain / (fan as f64).sqrt();
    let bound = (3.0_f64).sqrt() * std;
    let b = bound as f32;
    uniform_torch(seed, shape, -b, b, device)
}

/// Allocate an F32 tensor of the given `shape` and fill it with samples
/// bit-identical to PyTorch's
/// `torch.nn.init.xavier_uniform_(tensor, gain,
/// generator=torch.Generator(device='cuda').manual_seed(seed))`.
///
/// Reference: `torch/nn/init.py:366-404`. The bound is
///   `std = gain * sqrt(2.0 / (fan_in + fan_out))`
///   `bound = sqrt(3.0) * std`.
///
/// Returns a fresh tensor (flame-core has no `Tensor::uniform_` in-place
/// at present).
///
/// Same SM-count caveat as [`randn_torch`].
pub fn xavier_uniform_torch(
    shape: Shape,
    gain: f64,
    fan_in: usize,
    fan_out: usize,
    seed: u64,
    device: Arc<CudaDevice>,
) -> Result<Tensor, Error> {
    if fan_in == 0 && fan_out == 0 {
        return Err(Error::InvalidInput(
            "xavier_uniform_torch: fan_in + fan_out == 0".into(),
        ));
    }
    // Match PyTorch's computation order (`torch/nn/init.py:xavier_uniform_`):
    //   std   = gain * sqrt(2.0 / (fan_in + fan_out))
    //   bound = sqrt(3.0) * std
    // `gain` is Python-float (f64) — callers passing `math.sqrt(2.0)` etc.
    // must NOT narrow to f32 before this call, or the final `bound` will be
    // off by 1 ulp from PyTorch's. Only the f32 cast at the kernel boundary
    // is allowed to lose precision.
    let std = gain * (2.0_f64 / ((fan_in + fan_out) as f64)).sqrt();
    let bound = (3.0_f64).sqrt() * std;
    let b = bound as f32;
    uniform_torch(seed, shape, -b, b, device)
}
