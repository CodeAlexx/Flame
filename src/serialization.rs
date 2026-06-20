use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::CudaDevice;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

/// Convert FP8 E4M3 byte to f32.
/// E4M3: 1 sign, 4 exponent (bias=7), 3 mantissa.
#[inline]
fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF;
    let mant = bits & 0x7;
    if exp == 0 && mant == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    if exp == 0xF && mant == 0x7 {
        return f32::NAN;
    }
    let (e, m) = if exp == 0 {
        (-6i32, mant as f32 / 8.0)
    } else {
        (exp as i32 - 7, 1.0 + mant as f32 / 8.0)
    };
    let mag = m * (2.0f32).powi(e);
    if sign == 1 {
        -mag
    } else {
        mag
    }
}

/// Format for saving tensors
#[derive(Debug)]
pub enum SerializationFormat {
    /// Native binary format (fast, but not portable)
    Binary,
    /// SafeTensors format (compatible with Hugging Face)
    SafeTensors,
}

/// Save a single tensor to a file
pub fn save_tensor(tensor: &Tensor, path: &Path, format: SerializationFormat) -> Result<()> {
    match format {
        SerializationFormat::Binary => save_tensor_binary(tensor, path),
        SerializationFormat::SafeTensors => save_tensor_safetensors(tensor, path),
    }
}

/// Load a single tensor from a file
pub fn load_tensor(
    path: &Path,
    device: Arc<CudaDevice>,
    format: SerializationFormat,
) -> Result<Tensor> {
    match format {
        SerializationFormat::Binary => load_tensor_binary(path, device),
        SerializationFormat::SafeTensors => load_tensor_safetensors(path, device),
    }
}

/// Save multiple tensors to a file
pub fn save_tensors(
    tensors: &HashMap<String, Tensor>,
    path: &Path,
    format: SerializationFormat,
) -> Result<()> {
    match format {
        SerializationFormat::Binary => save_tensors_binary(tensors, path),
        SerializationFormat::SafeTensors => save_tensors_safetensors(tensors, path, None),
    }
}

/// SafeTensors-only save with extra header metadata (string→string).
/// Embedded under the standard `__metadata__` header key (per safetensors
/// spec). Loaders that ignore the key see a normal weights file.
pub fn save_tensors_with_metadata(
    tensors: &HashMap<String, Tensor>,
    extra_metadata: &HashMap<String, String>,
    path: &Path,
) -> Result<()> {
    save_tensors_safetensors(tensors, path, Some(extra_metadata))
}

/// SafeTensors-only load that also returns the `__metadata__` map.
pub fn load_tensors_with_metadata(
    path: &Path,
    device: Arc<CudaDevice>,
) -> Result<(HashMap<String, Tensor>, HashMap<String, String>)> {
    load_tensors_safetensors_with_metadata(path, device)
}

/// Load multiple tensors from a file
pub fn load_tensors(
    path: &Path,
    device: Arc<CudaDevice>,
    format: SerializationFormat,
) -> Result<HashMap<String, Tensor>> {
    match format {
        SerializationFormat::Binary => load_tensors_binary(path, device),
        SerializationFormat::SafeTensors => load_tensors_safetensors(path, device),
    }
}

// Binary format implementation
fn save_tensor_binary(tensor: &Tensor, path: &Path) -> Result<()> {
    let file =
        File::create(path).map_err(|e| Error::Io(format!("Failed to create file: {:?}", e)))?;
    let mut writer = BufWriter::new(file);

    // Write magic number
    writer
        .write_all(b"FLMT")
        .map_err(|e| Error::Io(e.to_string()))?;

    // Write version (little-endian)
    writer
        .write_all(&1u32.to_le_bytes())
        .map_err(|e| Error::Io(e.to_string()))?;

    // Write shape
    let dims = tensor.shape().dims();
    writer
        .write_all(&(dims.len() as u32).to_le_bytes())
        .map_err(|e| Error::Io(e.to_string()))?;
    for &dim in dims {
        writer
            .write_all(&(dim as u64).to_le_bytes())
            .map_err(|e| Error::Io(e.to_string()))?;
    }

    // Write data
    let data = tensor.to_vec()?;
    for &value in &data {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|e| Error::Io(e.to_string()))?;
    }

    writer.flush().map_err(|e| Error::Io(e.to_string()))?;
    Ok(())
}

fn load_tensor_binary(path: &Path, device: Arc<CudaDevice>) -> Result<Tensor> {
    let file = File::open(path).map_err(|e| Error::Io(format!("Failed to open file: {:?}", e)))?;
    let mut reader = BufReader::new(file);

    // Read and verify magic number
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|e| Error::Io(e.to_string()))?;
    if &magic != b"FLMT" {
        return Err(Error::InvalidOperation("Invalid file format".to_string()));
    }

    // Read version
    let mut version = [0u8; 4];
    reader
        .read_exact(&mut version)
        .map_err(|e| Error::Io(e.to_string()))?;
    let version = u32::from_le_bytes(version);
    if version != 1 {
        return Err(Error::InvalidOperation(format!(
            "Unsupported version: {}",
            version
        )));
    }

    // Read shape
    let mut ndims_bytes = [0u8; 4];
    reader
        .read_exact(&mut ndims_bytes)
        .map_err(|e| Error::Io(e.to_string()))?;
    let ndims = u32::from_le_bytes(ndims_bytes) as usize;

    let mut dims = vec![0usize; ndims];
    for dim in dims.iter_mut() {
        let mut dim_bytes = [0u8; 8];
        reader
            .read_exact(&mut dim_bytes)
            .map_err(|e| Error::Io(e.to_string()))?;
        *dim = u64::from_le_bytes(dim_bytes) as usize;
    }

    let shape = Shape::from_dims(&dims);
    let numel = shape.elem_count();

    // Read data
    let mut data = vec![0.0f32; numel];
    for value in data.iter_mut() {
        let mut value_bytes = [0u8; 4];
        reader
            .read_exact(&mut value_bytes)
            .map_err(|e| Error::Io(e.to_string()))?;
        *value = f32::from_le_bytes(value_bytes);
    }

    Tensor::from_vec(data, shape, device)
}

fn save_tensors_binary(tensors: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    let file =
        File::create(path).map_err(|e| Error::Io(format!("Failed to create file: {:?}", e)))?;
    let mut writer = BufWriter::new(file);

    // Write magic number
    writer
        .write_all(b"FLMM")
        .map_err(|e| Error::Io(e.to_string()))?;

    // Write version (little-endian)
    writer
        .write_all(&1u32.to_le_bytes())
        .map_err(|e| Error::Io(e.to_string()))?;

    // Write number of tensors
    writer
        .write_all(&(tensors.len() as u32).to_le_bytes())
        .map_err(|e| Error::Io(e.to_string()))?;

    // Write each tensor
    for (name, tensor) in tensors {
        // Write name length and name
        let name_bytes = name.as_bytes();
        writer
            .write_all(&(name_bytes.len() as u32).to_le_bytes())
            .map_err(|e| Error::Io(e.to_string()))?;
        writer
            .write_all(name_bytes)
            .map_err(|e| Error::Io(e.to_string()))?;

        // Write shape
        let dims = tensor.shape().dims();
        writer
            .write_all(&(dims.len() as u32).to_le_bytes())
            .map_err(|e| Error::Io(e.to_string()))?;
        for &dim in dims {
            writer
                .write_all(&(dim as u64).to_le_bytes())
                .map_err(|e| Error::Io(e.to_string()))?;
        }

        // Write data
        let data = tensor.to_vec()?;
        for value in data {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|e| Error::Io(e.to_string()))?;
        }
    }

    writer.flush().map_err(|e| Error::Io(e.to_string()))?;
    Ok(())
}

fn load_tensors_binary(path: &Path, device: Arc<CudaDevice>) -> Result<HashMap<String, Tensor>> {
    let file = File::open(path).map_err(|e| Error::Io(format!("Failed to open file: {:?}", e)))?;
    let mut reader = BufReader::new(file);

    // Read and verify magic number
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|e| Error::Io(e.to_string()))?;
    if &magic != b"FLMM" {
        return Err(Error::InvalidOperation("Invalid file format".to_string()));
    }

    // Read version
    let mut version = [0u8; 4];
    reader
        .read_exact(&mut version)
        .map_err(|e| Error::Io(e.to_string()))?;
    let version = u32::from_le_bytes(version);
    if version != 1 {
        return Err(Error::InvalidOperation(format!(
            "Unsupported version: {}",
            version
        )));
    }

    // Read number of tensors
    let mut num_tensors_bytes = [0u8; 4];
    reader
        .read_exact(&mut num_tensors_bytes)
        .map_err(|e| Error::Io(e.to_string()))?;
    let num_tensors = u32::from_le_bytes(num_tensors_bytes) as usize;

    let mut tensors = HashMap::new();

    for _ in 0..num_tensors {
        // Read name
        let mut name_len_bytes = [0u8; 4];
        reader
            .read_exact(&mut name_len_bytes)
            .map_err(|e| Error::Io(e.to_string()))?;
        let name_len = u32::from_le_bytes(name_len_bytes) as usize;

        let mut name_bytes = vec![0u8; name_len];
        reader
            .read_exact(&mut name_bytes)
            .map_err(|e| Error::Io(e.to_string()))?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| Error::InvalidOperation(format!("Invalid UTF-8 in name: {:?}", e)))?;

        // Read shape
        let mut ndims_bytes = [0u8; 4];
        reader
            .read_exact(&mut ndims_bytes)
            .map_err(|e| Error::Io(e.to_string()))?;
        let ndims = u32::from_le_bytes(ndims_bytes) as usize;

        let mut dims = vec![0usize; ndims];
        for dim in dims.iter_mut() {
            let mut dim_bytes = [0u8; 8];
            reader
                .read_exact(&mut dim_bytes)
                .map_err(|e| Error::Io(e.to_string()))?;
            *dim = u64::from_le_bytes(dim_bytes) as usize;
        }

        let shape = Shape::from_dims(&dims);
        let numel = shape.elem_count();

        // Read data
        let mut data = vec![0.0f32; numel];
        for value in data.iter_mut() {
            let mut value_bytes = [0u8; 4];
            reader
                .read_exact(&mut value_bytes)
                .map_err(|e| Error::Io(e.to_string()))?;
            *value = f32::from_le_bytes(value_bytes);
        }

        let tensor = Tensor::from_vec(data, shape, device.clone())?;
        tensors.insert(name, tensor);
    }

    Ok(tensors)
}

// SafeTensors format implementation
fn save_tensor_safetensors(tensor: &Tensor, path: &Path) -> Result<()> {
    let mut tensors = HashMap::new();
    tensors.insert("tensor".to_string(), tensor);
    save_tensors_safetensors(&tensors, path, None)
}

fn load_tensor_safetensors(path: &Path, device: Arc<CudaDevice>) -> Result<Tensor> {
    let tensors = load_tensors_safetensors(path, device)?;
    Ok(tensors
        .get("tensor")
        .ok_or_else(|| Error::InvalidOperation("No 'tensor' key found".to_string()))?
        .clone())
}

fn save_tensors_safetensors<T: AsRef<Tensor>>(
    tensors: &HashMap<String, T>,
    path: &Path,
    extra_metadata: Option<&HashMap<String, String>>,
) -> Result<()> {
    use serde_json::{json, Value};

    // Atomic write: stage to `{path}.tmp`, then `rename` to `path` once the
    // full payload + flush succeeds. On the same filesystem `rename` is an
    // atomic directory-entry swap, so a Ctrl-C / crash mid-write leaves
    // either the old file (if any) or no final file — never a truncated
    // half-byte `.safetensors` that downstream loaders / `--skip-existing`
    // caches would treat as valid. Fixes the corruption window flagged in
    // the HiDream-O1 M2 skeptic review for every `prepare_*` binary at once.
    let tmp_path = {
        let mut s = path.as_os_str().to_owned();
        s.push(".tmp");
        std::path::PathBuf::from(s)
    };
    let file = File::create(&tmp_path)
        .map_err(|e| Error::Io(format!("Failed to create file '{}': {:?}", tmp_path.display(), e)))?;
    let mut writer = BufWriter::new(file);

    // Create metadata
    let mut metadata = serde_json::Map::new();
    if let Some(extra) = extra_metadata {
        if !extra.is_empty() {
            let mut m = serde_json::Map::new();
            for (k, v) in extra {
                m.insert(k.clone(), Value::String(v.clone()));
            }
            metadata.insert("__metadata__".into(), Value::Object(m));
        }
    }
    let mut offset = 0u64;

    // Collect tensor info. Honor the tensor's dtype (F16/BF16 stay 2-byte; F32
    // stays 4-byte) instead of always upcasting to F32 — so an F16/BF16 LoRA
    // saves at reference size, not 2x. `to_vec()` upcasts to f32; we round back
    // to the native dtype for the bytes (lossless: the tensor already held that
    // precision).
    let mut tensor_data: Vec<Vec<u8>> = Vec::new();
    for (name, tensor) in tensors {
        let tensor = tensor.as_ref();
        let f32_data = tensor.to_vec()?;
        let shape = tensor.shape().dims();

        let (dtype_str, bytes): (&str, Vec<u8>) = match tensor.dtype() {
            DType::F16 => (
                "F16",
                f32_data
                    .iter()
                    .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                    .collect(),
            ),
            DType::BF16 => (
                "BF16",
                f32_data
                    .iter()
                    .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
                    .collect(),
            ),
            _ => (
                "F32",
                f32_data.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            ),
        };

        let end = offset + bytes.len() as u64;
        let tensor_info = json!({
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, end],
        });
        metadata.insert(name.clone(), tensor_info);
        tensor_data.push(bytes);
        offset = end;
    }

    // Convert metadata to JSON
    let metadata_json = serde_json::to_string(&Value::Object(metadata))
        .map_err(|e| Error::Io(format!("Failed to serialize metadata: {:?}", e)))?;
    let metadata_bytes = metadata_json.as_bytes();

    // Write header size (8 bytes, little-endian)
    let header_size = metadata_bytes.len() as u64;
    writer
        .write_all(&header_size.to_le_bytes())
        .map_err(|e| Error::Io(e.to_string()))?;

    // Write metadata
    writer
        .write_all(metadata_bytes)
        .map_err(|e| Error::Io(e.to_string()))?;

    // Write tensor data (already serialized to the native-dtype byte layout).
    for bytes in tensor_data {
        writer
            .write_all(&bytes)
            .map_err(|e| Error::Io(e.to_string()))?;
    }

    writer.flush().map_err(|e| Error::Io(e.to_string()))?;
    // Drop the BufWriter (and the underlying File) before renaming so all
    // bytes are committed to the OS page cache.
    drop(writer);
    std::fs::rename(&tmp_path, path).map_err(|e| {
        // Best-effort cleanup of the staged file if rename fails.
        let _ = std::fs::remove_file(&tmp_path);
        Error::Io(format!(
            "atomic rename '{}' -> '{}': {e}",
            tmp_path.display(),
            path.display()
        ))
    })?;
    Ok(())
}

fn load_tensors_safetensors(
    path: &Path,
    device: Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let (tensors, _meta) = load_tensors_safetensors_with_metadata(path, device)?;
    Ok(tensors)
}

/// Re-read just the `__metadata__` (string→string) safetensors entry.
///
/// `eri_safetensors::MmapFile` indexes only data tensors and skips the
/// `__metadata__` key. When the loader needs that entry (e.g. for trainer
/// resume to read `train_dtype`, `seed`, etc.), we re-open the file just
/// to slice the header. Cheap: the header is at most 100 MB and the
/// kernel page cache already has it warm from the mmap probe.
fn read_safetensors_extra_metadata(path: &Path) -> Result<HashMap<String, String>> {
    let mut file = File::open(path).map_err(|e| Error::Io(format!("open: {e}")))?;
    let mut header_size_bytes = [0u8; 8];
    file.read_exact(&mut header_size_bytes)
        .map_err(|e| Error::Io(e.to_string()))?;
    let header_size = u64::from_le_bytes(header_size_bytes) as usize;
    let mut header_bytes = vec![0u8; header_size];
    file.read_exact(&mut header_bytes)
        .map_err(|e| Error::Io(e.to_string()))?;
    let metadata: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| Error::Io(format!("parse header: {e}")))?;
    Ok(metadata
        .get("__metadata__")
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default())
}

/// Re-read the full safetensors header object (so the FP8 scale-lookup
/// path can resolve `*_scale` / `*.scale_weight` companion entries).
fn read_safetensors_full_header(path: &Path) -> Result<serde_json::Value> {
    let mut file = File::open(path).map_err(|e| Error::Io(format!("open: {e}")))?;
    let mut header_size_bytes = [0u8; 8];
    file.read_exact(&mut header_size_bytes)
        .map_err(|e| Error::Io(e.to_string()))?;
    let header_size = u64::from_le_bytes(header_size_bytes) as usize;
    let mut header_bytes = vec![0u8; header_size];
    file.read_exact(&mut header_bytes)
        .map_err(|e| Error::Io(e.to_string()))?;
    serde_json::from_slice(&header_bytes).map_err(|e| Error::Io(format!("parse header: {e}")))
}

/// Decode all (or filtered) tensors out of a mmap'd safetensors file into
/// device memory. Shared between `load_tensors_safetensors_with_metadata`
/// (filter = always-true) and `load_file_filtered` (caller-supplied filter).
///
/// Each tensor's bytes come from the mmap region (paged in on access);
/// after the device copy the page can be dropped by the OS.
///
/// `header_for_scales` is the parsed safetensors header — only required
/// for the FP8 path which needs to resolve `*_scale` / `*.scale_weight`
/// companion entries.
fn decode_tensors_from_mmap(
    mmap: &eri_safetensors::MmapFile,
    header_for_scales: Option<&serde_json::Value>,
    device: &Arc<CudaDevice>,
    mut filter: impl FnMut(&str) -> bool,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();
    for (name, info) in &mmap.tensors {
        if !filter(name) {
            continue;
        }
        // Skip unsupported dtypes (I64, I32, BOOL, etc.)
        if !matches!(info.dtype.as_str(), "F32" | "BF16" | "F16" | "F8_E4M3") {
            continue;
        }
        let bytes = mmap
            .tensor_bytes(name)
            .ok_or_else(|| Error::InvalidInput(format!("mmap missing tensor '{name}'")))?;
        let shape = info.shape.clone();

        let tensor = match info.dtype.as_str() {
            "BF16" => {
                // BF16: 2 bytes per element — load raw u16 directly into BF16 tensor (no f32 intermediate)
                let num_elems = bytes.len() / 2;
                let mut bf16_u16 = vec![0u16; num_elems];
                for (value, chunk) in bf16_u16.iter_mut().zip(bytes.chunks_exact(2)) {
                    *value = u16::from_le_bytes([chunk[0], chunk[1]]);
                }
                let s = Shape::from_dims(&shape);
                let mut tensor = Tensor::zeros_dtype(s, DType::BF16, device.clone())?;
                tensor.copy_from_bf16_slice(&bf16_u16)?;
                tensor
            }
            "F8_E4M3" => {
                // FP8 E4M3: 1 byte per element. Dequant with per-tensor OR per-row
                // scale — per-tensor `weight_scale` is [1]; per-row (e.g. Ideogram-4)
                // is [out_rows], one scale per output row. Reading only scale[0]
                // mis-scales every row but the first → garbage weights.
                //   LTX-2:         `foo.weight_scale` (suffix on the key)
                //   Comfy-scaled:  `foo.scale_weight` (replaces `.weight` with `.scale_weight`)
                let scales: Vec<f32> = if let Some(header) = header_for_scales {
                    let lookup_scale = |key: &str| -> Option<Vec<f32>> {
                        header.get(key)?;
                        let bytes = mmap.tensor_bytes(key)?;
                        if bytes.len() >= 4 {
                            Some(
                                bytes
                                    .chunks_exact(4)
                                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect(),
                            )
                        } else {
                            None
                        }
                    };
                    lookup_scale(&format!("{name}_scale")).or_else(|| {
                        name.strip_suffix(".weight")
                            .and_then(|base| lookup_scale(&format!("{base}.scale_weight")))
                    })
                } else {
                    None
                }
                .unwrap_or_else(|| vec![1.0]);
                let num_elems = bytes.len();
                // per-row when the scale length matches the leading (out) dim.
                let out_dim = shape.first().copied().unwrap_or(1);
                let per_row = out_dim > 1 && scales.len() == out_dim && num_elems % out_dim == 0;
                let in_dim = if per_row { num_elems / out_dim } else { 1 };
                let mut bf16_u16 = vec![0u16; num_elems];
                for (i, (value, &byte)) in bf16_u16.iter_mut().zip(bytes.iter()).enumerate() {
                    let scale = if per_row { scales[i / in_dim] } else { scales[0] };
                    let f = fp8_e4m3_to_f32(byte) * scale;
                    *value = half::bf16::from_f32(f).to_bits();
                }
                let mut tensor =
                    Tensor::zeros_dtype(Shape::from_dims(&shape), DType::BF16, device.clone())?;
                tensor.copy_from_bf16_slice(&bf16_u16)?;
                tensor
            }
            "F16" => {
                // F16: 2 bytes per element — convert to F32 then upload
                let num_elems = bytes.len() / 2;
                let mut f32_data = vec![0.0f32; num_elems];
                for (value, chunk) in f32_data.iter_mut().zip(bytes.chunks_exact(2)) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    *value = half::f16::from_bits(bits).to_f32();
                }
                Tensor::from_vec(f32_data, Shape::from_dims(&shape), device.clone())?
            }
            _ => {
                // F32: 4 bytes per element
                let num_floats = bytes.len() / 4;
                let mut data = vec![0.0f32; num_floats];
                for (value, chunk) in data.iter_mut().zip(bytes.chunks_exact(4)) {
                    *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Tensor::from_vec(data, Shape::from_dims(&shape), device.clone())?
            }
        };
        tensors.insert(name.clone(), tensor);
    }
    Ok(tensors)
}

fn load_tensors_safetensors_with_metadata(
    path: &Path,
    device: Arc<CudaDevice>,
) -> Result<(HashMap<String, Tensor>, HashMap<String, String>)> {
    // Pre-2026-05-10 this path used `read_to_end` — the entire safetensors
    // file (often 10–20+ GB for diffusion DiTs) was pulled into the heap
    // before per-tensor copy. Migrated to `eri-safetensors`'s MAP_NORESERVE
    // mmap: pages are paged in on-demand and freed by the OS without swap.
    // The data segment is never materialized as a single heap allocation.
    use eri_safetensors::MmapFile;

    let mmap = MmapFile::open_path(path).map_err(|e| {
        Error::Io(format!(
            "eri-safetensors mmap open '{}': {e}",
            path.display()
        ))
    })?;

    // Re-parse the header for `__metadata__` (mmap layer indexes data
    // tensors only). Cheap: header is at most 100 MB and the kernel has
    // already pulled it into the page cache via the mmap probe.
    let extra_metadata = read_safetensors_extra_metadata(path).unwrap_or_default();
    let metadata_obj_for_scale = read_safetensors_full_header(path).ok();

    // Allocate tensors using mmap'd byte slices. No bulk heap copy — each
    // tensor copies its own slice into device memory, and the mmap is
    // dropped at function exit.
    let tensors =
        decode_tensors_from_mmap(&mmap, metadata_obj_for_scale.as_ref(), &device, |_| true)?;

    Ok((tensors, extra_metadata))
}

// Convenience methods for Tensor
impl Tensor {
    /// Save this tensor to a file
    pub fn save(&self, path: &Path) -> Result<()> {
        save_tensor(self, path, SerializationFormat::Binary)
    }

    /// Load a tensor from a file
    pub fn load(path: &Path, device: Arc<CudaDevice>) -> Result<Self> {
        load_tensor(path, device, SerializationFormat::Binary)
    }
}

/// Load a safetensors file (convenience function)
pub fn load_file<P: AsRef<Path>>(
    path: P,
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    load_tensors(
        path.as_ref(),
        device.clone(),
        SerializationFormat::SafeTensors,
    )
}

/// Load only matching keys from a safetensors file (for block-level offloading).
///
/// Uses memory-mapping — only the selected tensors' bytes are paged in from
/// disk. The full file is NOT read into RAM, so this works for 40GB+ files.
///
/// Backed by `eri-safetensors`'s MAP_NORESERVE mmap (same path as
/// `load_tensors_safetensors_with_metadata` post-2026-05-10).
pub fn load_file_filtered<P, F>(
    path: P,
    device: &Arc<CudaDevice>,
    filter: F,
) -> Result<HashMap<String, Tensor>>
where
    P: AsRef<Path>,
    F: Fn(&str) -> bool,
{
    let mmap = eri_safetensors::MmapFile::open_path(path.as_ref()).map_err(|e| {
        Error::Io(format!(
            "eri-safetensors mmap open '{}': {e}",
            path.as_ref().display()
        ))
    })?;
    // FP8 scale lookup needs the parsed safetensors header (the mmap layer
    // skips `__metadata__` and only indexes data tensors). Re-read the
    // header bytes once — they're already paged in by the mmap probe.
    let header_for_scales = read_safetensors_full_header(path.as_ref()).ok();
    decode_tensors_from_mmap(&mmap, header_for_scales.as_ref(), device, filter)
}

/// Save a safetensors file (convenience function)  
pub fn save_file<P: AsRef<Path>>(tensors: &HashMap<String, Tensor>, path: P) -> Result<()> {
    save_tensors(tensors, path.as_ref(), SerializationFormat::SafeTensors)
}
