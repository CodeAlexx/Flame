//! Minimal GPU RNG using xorshift32 kernels (NVRTC). Fills tensors on-device.

use crate::{DType, Error, Shape, Tensor};
use cudarc::{
    driver::{CudaDevice, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const CUDA_SRC: &str = r#"
extern "C" __device__ inline unsigned xorshift32(unsigned x){
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}
extern "C" __global__
void fill_rand_f32(float* __restrict__ y, size_t n, unsigned seed){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        unsigned s = xorshift32(seed ^ (unsigned)i);
        y[i] = (s >> 8) * (1.0f / 16777216.0f);
    }
}
#include <cuda_bf16.h>
extern "C" __global__
#ifdef __CUDA_ARCH__
void fill_rand_bf16(__nv_bfloat16* __restrict__ y, size_t n, unsigned seed){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        unsigned s = xorshift32(seed ^ (unsigned)i);
        float v = (s >> 8) * (1.0f / 16777216.0f);
        y[i] = __float2bfloat16_rn(v);
    }
}
#endif
"#;

static MOD_ONCE: OnceLock<()> = OnceLock::new();
static GLOBAL_SEED: AtomicU64 = AtomicU64::new(0);
static RNG_STATE: OnceLock<Mutex<StdRng>> = OnceLock::new();

/// Access the global RNG used across training and tests.
pub fn global_rng() -> &'static Mutex<StdRng> {
    RNG_STATE.get_or_init(|| {
        let seed = GLOBAL_SEED.load(Ordering::Relaxed);
        Mutex::new(StdRng::seed_from_u64(seed))
    })
}

fn rng_state() -> &'static Mutex<StdRng> {
    global_rng()
}

/// Set the global RNG seed used by host-side sampling and as the default for GPU kernels.
pub fn set_seed(seed: u64) -> Result<(), Error> {
    GLOBAL_SEED.store(seed, Ordering::Relaxed);
    let mut guard = rng_state()
        .lock()
        .map_err(|_| Error::InvalidOperation("RNG mutex poisoned".into()))?;
    *guard = StdRng::seed_from_u64(seed);
    Ok(())
}

pub fn current_seed() -> u64 {
    GLOBAL_SEED.load(Ordering::Relaxed)
}

/// Draw the next u64 from the global RNG stream (deterministic under `set_seed`).
pub fn next_u64() -> u64 {
    let mut guard = rng_state().lock().unwrap_or_else(|e| e.into_inner());
    guard.next_u64()
}

pub fn sample_normal(len: usize, mean: f32, std: f32) -> Result<Vec<f32>, Error> {
    let normal = Normal::new(mean, std)
        .map_err(|e| Error::InvalidInput(format!("invalid normal params: {e:?}")))?;
    let mut guard = rng_state()
        .lock()
        .map_err(|_| Error::InvalidOperation("RNG mutex poisoned".into()))?;
    Ok((0..len).map(|_| normal.sample(&mut *guard)).collect())
}

fn ensure_module(dev: &Arc<CudaDevice>) -> Result<(), Error> {
    if dev.get_func("flame_rng", "fill_rand_f32").is_some() {
        return Ok(());
    }
    if MOD_ONCE.get().is_none() {
        let include_path = std::env::var("CUDA_HOME")
            .map(|p| format!("{}/include", p))
            .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
        let mut opts = CompileOptions::default();
        opts.include_paths.push(include_path);
        let ptx =
            compile_ptx_with_opts(CUDA_SRC, opts).map_err(|e| Error::KernelError(format!("{e:?}")))?;
        #[cfg(feature = "bf16_u16")]
        let symbols: &[&str] = &["fill_rand_f32", "fill_rand_bf16"];
        #[cfg(not(feature = "bf16_u16"))]
        let symbols: &[&str] = &["fill_rand_f32"];
        dev.load_ptx(ptx, "flame_rng", symbols)
            .map_err(|e| Error::KernelError(format!("{e:?}")))?;
        let _ = MOD_ONCE.set(());
    }
    Ok(())
}

/// Fill `tensor` in-place with uniform random numbers using a simple GPU RNG.
pub fn rand_fill_(tensor: &mut Tensor, seed: u32) -> Result<(), Error> {
    let dev = Arc::clone(tensor.device());
    ensure_module(&dev)?;
    let n = tensor.shape().elem_count();
    if n == 0 {
        return Ok(());
    }
    let actual_seed = if seed == 0 {
        current_seed() as u32
    } else {
        seed
    };
    let cfg = LaunchConfig::for_num_elems(n as u32);

    unsafe {
        match tensor.dtype() {
            DType::F32 => {
                let slice = tensor.storage_mut().try_as_mut_slice_f32()?;
                let func = dev
                    .get_func("flame_rng", "fill_rand_f32")
                    .ok_or_else(|| Error::KernelError("fill_rand_f32 missing".into()))?;
                func.launch(cfg, (slice, n as u64, actual_seed))
                    .map_err(|e| Error::KernelError(format!("{e:?}")))?;
            }
            DType::BF16 => {
                #[cfg(feature = "bf16_u16")]
                {
                    let slice = tensor.storage_mut().try_as_mut_slice_u16()?;
                    let func = dev
                        .get_func("flame_rng", "fill_rand_bf16")
                        .ok_or_else(|| Error::KernelError("fill_rand_bf16 missing".into()))?;
                    func.launch(cfg, (slice, n as u64, actual_seed))
                        .map_err(|e| Error::KernelError(format!("{e:?}")))?;
                }
                #[cfg(not(feature = "bf16_u16"))]
                {
                    return Err(Error::Unsupported(
                        "BF16 requires the bf16_u16 feature".into(),
                    ));
                }
            }
            _ => return Err(Error::Unsupported("rand_fill_: dtype not supported".into())),
        }
    }
    Ok(())
}

/// Create a random tensor on the provided device.
pub fn rand_on(
    dev: &Arc<CudaDevice>,
    shape: &[usize],
    dtype: DType,
    seed: u32,
) -> Result<Tensor, Error> {
    let mut tensor = Tensor::zeros_dtype(Shape::from_dims(shape), dtype, Arc::clone(dev))?;
    rand_fill_(
        &mut tensor,
        if seed == 0 {
            current_seed() as u32
        } else {
            seed
        },
    )?;
    Ok(tensor)
}
