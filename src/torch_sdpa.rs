//! Bridge to PyTorch's CUTLASS flash attention via the AOTI C shim.
//!
//! PyTorch 2.8+ ships compiled CUTLASS FlashAttention-2 kernels in
//! `libtorch_cuda.so`.  The AOTI (Ahead-Of-Time Inductor) surface exposes
//! them through plain `extern "C"` functions — no name mangling, no C++
//! types — making them callable from Rust via `libloading::dlsym`.
//!
//! This module lazily dlopen's libtorch at first use and provides
//! `torch_flash_sdpa(q, k, v) -> Result<Tensor>` which is ~4 ms at
//! [1,32,4352,128] BF16 on a 3090 Ti — matching ComfyUI / PyTorch speed.
//!
//! If libtorch is not found or the call fails, callers fall back to
//! flame-core's native (slower) SDPA path. Zero hard dependency.

use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::DevicePtr;
use std::ffi::{c_void, CString};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// AOTI type aliases
// ---------------------------------------------------------------------------

type Handle = *mut c_void; // AtenTensorHandle (opaque)
type AErr = i32; // AOTITorchError (0 = success)

// ---------------------------------------------------------------------------
// Function pointer table
// ---------------------------------------------------------------------------

struct TorchLib {
    _lib: libloading::Library,

    create_tensor_from_blob: unsafe extern "C" fn(
        *mut c_void,    // data
        i64,            // ndim
        *const i64,     // sizes
        *const i64,     // strides
        i64,            // storage_offset
        i32,            // dtype
        i32,            // device_type
        i32,            // device_index
        *mut Handle,    // out
    ) -> AErr,

    scaled_dot_product_flash_attention: unsafe extern "C" fn(
        Handle,         // query
        Handle,         // key
        Handle,         // value
        f64,            // dropout_p
        i32,            // is_causal
        i32,            // return_debug_mask
        *const f64,     // scale (nullable)
        *mut Handle,    // ret0: output
        *mut Handle,    // ret1: logsumexp
        *mut Handle,    // ret2: cum_seq_q
        *mut Handle,    // ret3: cum_seq_k
        *mut i64,       // ret4: max_q
        *mut i64,       // ret5: max_k
        *mut Handle,    // ret6: philox_seed
        *mut Handle,    // ret7: philox_offset
        *mut Handle,    // ret8: debug_attn_mask
    ) -> AErr,

    get_data_ptr: unsafe extern "C" fn(Handle, *mut *mut c_void) -> AErr,
    delete_tensor_object: unsafe extern "C" fn(Handle) -> AErr,
    dtype_bfloat16: unsafe extern "C" fn() -> i32,
    device_type_cuda: unsafe extern "C" fn() -> i32,
}

// Safety: TorchLib only holds function pointers and a library handle.
// All calls are serialized through the caller's thread.
unsafe impl Send for TorchLib {}
unsafe impl Sync for TorchLib {}

static LIB: OnceLock<Option<TorchLib>> = OnceLock::new();

fn find_torch_lib_path() -> Option<String> {
    // Try common locations
    let candidates = [
        // pip install torch (user)
        "/home/alex/.local/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so",
        // pip install torch (system)
        "/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so",
        // venv
        "/home/alex/serenity/venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so",
    ];
    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    // Fallback: ask Python
    let output = std::process::Command::new("python3")
        .args(["-c", "import torch; print(torch.__path__[0] + '/lib/libtorch_cuda.so')"])
        .output()
        .ok()?;
    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if std::path::Path::new(&path).exists() {
            return Some(path);
        }
    }
    None
}

fn load_lib() -> Option<TorchLib> {
    let path = find_torch_lib_path()?;
    log::info!("[torch_sdpa] Loading libtorch from: {path}");

    // Also need to load libtorch_cpu.so and libc10.so first
    let lib_dir = std::path::Path::new(&path).parent()?;

    unsafe {
        // Load dependencies in order (RTLD_GLOBAL so symbols are visible)
        for dep in &["libc10.so", "libc10_cuda.so", "libtorch_cpu.so"] {
            let dep_path = lib_dir.join(dep);
            if dep_path.exists() {
                let cpath = CString::new(dep_path.to_str()?).ok()?;
                let h = libc::dlopen(cpath.as_ptr(), libc::RTLD_LAZY | libc::RTLD_GLOBAL);
                if h.is_null() {
                    let err = std::ffi::CStr::from_ptr(libc::dlerror());
                    log::warn!("[torch_sdpa] Failed to load {dep}: {err:?}");
                    return None;
                }
            }
        }

        // Load libtorch_cuda.so
        let lib = libloading::Library::new(&path).ok().or_else(|| {
            log::warn!("[torch_sdpa] Failed to dlopen {path}");
            None
        })?;

        macro_rules! sym {
            ($name:expr) => {
                *lib.get::<unsafe extern "C" fn()>($name)
                    .ok()
                    .or_else(|| { log::warn!("[torch_sdpa] Missing symbol"); None })?
            };
        }

        // Transmute each symbol to the correct function pointer type.
        // Safety: we trust PyTorch's C ABI to match the header signatures.
        let create_tensor_from_blob = std::mem::transmute(sym!(b"aoti_torch_create_tensor_from_blob\0"));
        let scaled_dot_product_flash_attention = std::mem::transmute(sym!(b"aoti_torch_cuda__scaled_dot_product_flash_attention\0"));
        let get_data_ptr = std::mem::transmute(sym!(b"aoti_torch_get_data_ptr\0"));
        let delete_tensor_object = std::mem::transmute(sym!(b"aoti_torch_delete_tensor_object\0"));
        let dtype_bfloat16 = std::mem::transmute(sym!(b"aoti_torch_dtype_bfloat16\0"));
        let device_type_cuda = std::mem::transmute(sym!(b"aoti_torch_device_type_cuda\0"));

        log::info!("[torch_sdpa] All symbols resolved — PyTorch flash attention available");

        Some(TorchLib {
            _lib: lib,
            create_tensor_from_blob,
            scaled_dot_product_flash_attention,
            get_data_ptr,
            delete_tensor_object,
            dtype_bfloat16,
            device_type_cuda,
        })
    }
}

fn get_lib() -> Option<&'static TorchLib> {
    LIB.get_or_init(|| load_lib()).as_ref()
}

/// Returns true if the PyTorch flash SDPA bridge is available.
pub fn is_available() -> bool {
    get_lib().is_some()
}

// ---------------------------------------------------------------------------
// Handle wrapper for RAII cleanup
// ---------------------------------------------------------------------------

struct TensorHandle {
    h: Handle,
    lib: &'static TorchLib,
}

impl Drop for TensorHandle {
    fn drop(&mut self) {
        if !self.h.is_null() {
            unsafe { (self.lib.delete_tensor_object)(self.h); }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run PyTorch's CUTLASS flash attention.
///
/// Q/K/V: `[B, H, S, D]` BF16, contiguous.
/// Returns: `[B, H, S, D]` BF16.
///
/// Returns `Err(Unsupported(...))` if libtorch is not available.
pub fn torch_flash_sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let lib = get_lib().ok_or_else(|| {
        Error::Unsupported("PyTorch flash SDPA not available (libtorch_cuda.so not found)".into())
    })?;

    let q_dims = q.shape().dims();
    if q_dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "torch_flash_sdpa expects [B,H,S,D], got {:?}", q_dims
        )));
    }
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        return Err(Error::InvalidInput("torch_flash_sdpa requires BF16".into()));
    }

    let (b, h, sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let k_dims = k.shape().dims();
    let sk = k_dims[2];

    unsafe {
        let bf16_dtype = (lib.dtype_bfloat16)();
        let cuda_device = (lib.device_type_cuda)();

        // Wrap flame-core tensors as AOTI handles (non-owning views)
        let wrap = |t: &Tensor, dims: &[usize]| -> Result<TensorHandle> {
            let ptr = t.as_device_ptr_bf16("torch_sdpa")? as *mut c_void;
            let sizes: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
            // Compute contiguous strides
            let mut strides = vec![0i64; dims.len()];
            if !dims.is_empty() {
                strides[dims.len() - 1] = 1;
                for i in (0..dims.len() - 1).rev() {
                    strides[i] = strides[i + 1] * sizes[i + 1];
                }
            }
            let mut handle: Handle = std::ptr::null_mut();
            let err = (lib.create_tensor_from_blob)(
                ptr,
                dims.len() as i64,
                sizes.as_ptr(),
                strides.as_ptr(),
                0, // storage_offset
                bf16_dtype,
                cuda_device,
                0, // device_index
                &mut handle,
            );
            if err != 0 || handle.is_null() {
                return Err(Error::Cuda(format!(
                    "aoti_torch_create_tensor_from_blob failed: err={err}"
                )));
            }
            Ok(TensorHandle { h: handle, lib })
        };

        let q_h = wrap(q, q_dims)?;
        let k_h = wrap(k, k_dims)?;
        let v_h = wrap(v, v.shape().dims())?;

        // Compute scale
        let scale = 1.0f64 / (d as f64).sqrt();

        // Call flash attention
        let mut out_h: Handle = std::ptr::null_mut();
        let mut lse_h: Handle = std::ptr::null_mut();
        let mut csq_h: Handle = std::ptr::null_mut();
        let mut csk_h: Handle = std::ptr::null_mut();
        let mut max_q: i64 = 0;
        let mut max_k: i64 = 0;
        let mut seed_h: Handle = std::ptr::null_mut();
        let mut offset_h: Handle = std::ptr::null_mut();
        let mut debug_h: Handle = std::ptr::null_mut();

        let err = (lib.scaled_dot_product_flash_attention)(
            q_h.h, k_h.h, v_h.h,
            0.0,    // dropout_p
            0,      // is_causal = false
            0,      // return_debug_mask = false
            &scale, // scale
            &mut out_h,
            &mut lse_h,
            &mut csq_h,
            &mut csk_h,
            &mut max_q,
            &mut max_k,
            &mut seed_h,
            &mut offset_h,
            &mut debug_h,
        );

        if err != 0 {
            return Err(Error::Cuda(format!(
                "aoti_torch_cuda__scaled_dot_product_flash_attention failed: err={err}"
            )));
        }

        // Wrap output handle for RAII
        let out_handle = TensorHandle { h: out_h, lib };

        // Get output data pointer
        let mut out_ptr: *mut c_void = std::ptr::null_mut();
        let err = (lib.get_data_ptr)(out_handle.h, &mut out_ptr);
        if err != 0 || out_ptr.is_null() {
            return Err(Error::Cuda("aoti_torch_get_data_ptr failed".into()));
        }

        // Copy output to a flame-core tensor (owned by our allocator)
        let out_shape = Shape::from_dims(&[b, h, sq, d]);
        let out_elems = b * h * sq * d;
        let out_bytes = out_elems * 2; // BF16 = 2 bytes

        let out_tensor = Tensor::empty_dtype(out_shape, DType::BF16, q.device().clone())?;
        let dst_ptr = out_tensor.as_device_ptr_bf16("torch_sdpa_out")? as *mut c_void;

        // cudaMemcpy device-to-device (raw driver call)
        let status = cudarc::driver::sys::lib().cuMemcpy(
            dst_ptr as u64,
            out_ptr as u64,
            out_bytes,
        );
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(Error::Cuda(format!("cuMemcpy failed: {status:?}")));
        }

        // Clean up PyTorch handles (output, lse, etc.)
        // out_handle drops via RAII
        // Clean up other non-null handles
        for h in [lse_h, csq_h, csk_h, seed_h, offset_h, debug_h] {
            if !h.is_null() {
                (lib.delete_tensor_object)(h);
            }
        }

        Ok(out_tensor)
    }
}
