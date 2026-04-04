use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::CudaDevice;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

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
        SerializationFormat::SafeTensors => save_tensors_safetensors(tensors, path),
    }
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
    save_tensors_safetensors(&tensors, path)
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
) -> Result<()> {
    use serde_json::{json, Value};

    let file =
        File::create(path).map_err(|e| Error::Io(format!("Failed to create file: {:?}", e)))?;
    let mut writer = BufWriter::new(file);

    // Create metadata
    let mut metadata = serde_json::Map::new();
    let mut offset = 0u64;

    // Collect tensor info
    let mut tensor_data = Vec::new();
    for (name, tensor) in tensors {
        let tensor = tensor.as_ref();
        let data = tensor.to_vec()?;
        let shape = tensor.shape().dims();

        let tensor_info = json!({
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + (data.len() * 4) as u64],
        });

        let data_len = data.len();
        metadata.insert(name.clone(), tensor_info);
        tensor_data.push(data);
        offset += (data_len * 4) as u64;
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

    // Write tensor data
    for data in tensor_data {
        for &value in &data {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|e| Error::Io(e.to_string()))?;
        }
    }

    writer.flush().map_err(|e| Error::Io(e.to_string()))?;
    Ok(())
}

fn load_tensors_safetensors(
    path: &Path,
    device: Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    use serde_json::Value;

    let file = File::open(path).map_err(|e| Error::Io(format!("Failed to open file: {:?}", e)))?;
    let mut reader = BufReader::new(file);

    // Read header size
    let mut header_size_bytes = [0u8; 8];
    reader
        .read_exact(&mut header_size_bytes)
        .map_err(|e| Error::Io(e.to_string()))?;
    let header_size = u64::from_le_bytes(header_size_bytes) as usize;

    // Read metadata
    let mut metadata_bytes = vec![0u8; header_size];
    reader
        .read_exact(&mut metadata_bytes)
        .map_err(|e| Error::Io(e.to_string()))?;

    let metadata: Value = serde_json::from_slice(&metadata_bytes)
        .map_err(|e| Error::Io(format!("Failed to parse metadata: {:?}", e)))?;

    let metadata_obj = metadata
        .as_object()
        .ok_or_else(|| Error::InvalidInput("Invalid metadata format".to_string()))?;

    // Read all remaining data
    let mut all_data = Vec::new();
    reader
        .read_to_end(&mut all_data)
        .map_err(|e| Error::Io(e.to_string()))?;

    let mut tensors = HashMap::new();

    for (name, info) in metadata_obj {
        // Skip the __metadata__ key (safetensors global metadata, not a tensor)
        if name == "__metadata__" {
            continue;
        }
        let shape = info["shape"]
            .as_array()
            .ok_or_else(|| Error::InvalidInput("Missing shape".to_string()))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| Error::InvalidInput("invalid shape entry".into()))
                    .map(|u| u as usize)
            })
            .collect::<Result<Vec<_>>>()?;

        let offsets = info["data_offsets"]
            .as_array()
            .ok_or_else(|| Error::InvalidInput("Missing data_offsets".to_string()))?;
        let start = offsets
            .first()
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::InvalidInput("invalid start offset".into()))?
            as usize;
        let end = offsets
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::InvalidInput("invalid end offset".into()))?
            as usize;

        // Detect dtype from metadata
        let dtype_str = info["dtype"]
            .as_str()
            .unwrap_or("F32");

        // Skip unsupported dtypes (I64, I32, BOOL, etc.)
        if !matches!(dtype_str, "F32" | "BF16" | "F16") {
            continue;
        }

        let tensor = match dtype_str {
            "BF16" => {
                // BF16: 2 bytes per element — load raw u16 directly into BF16 tensor (no f32 intermediate)
                let num_elems = (end - start) / 2;
                let mut bf16_u16 = vec![0u16; num_elems];
                for (value, chunk) in bf16_u16.iter_mut().zip(all_data[start..end].chunks_exact(2)) {
                    *value = u16::from_le_bytes([chunk[0], chunk[1]]);
                }
                let s = Shape::from_dims(&shape);
                let mut tensor = Tensor::zeros_dtype(s, DType::BF16, device.clone())?;
                tensor.copy_from_bf16_slice(&bf16_u16)?;
                tensor
            }
            "F16" => {
                // F16: 2 bytes per element — convert to F32 then upload
                let num_elems = (end - start) / 2;
                let mut f32_data = vec![0.0f32; num_elems];
                for (value, chunk) in f32_data.iter_mut().zip(all_data[start..end].chunks_exact(2)) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    *value = half::f16::from_bits(bits).to_f32();
                }
                Tensor::from_vec(f32_data, Shape::from_dims(&shape), device.clone())?
            }
            _ => {
                // F32: 4 bytes per element
                let num_floats = (end - start) / 4;
                let mut data = vec![0.0f32; num_floats];
                for (value, chunk) in data.iter_mut().zip(all_data[start..end].chunks_exact(4)) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    *value = f32::from_le_bytes(bytes);
                }
                Tensor::from_vec(data, Shape::from_dims(&shape), device.clone())?
            }
        };
        tensors.insert(name.clone(), tensor);
    }

    Ok(tensors)
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
pub fn load_file_filtered<P, F>(
    path: P,
    device: &Arc<CudaDevice>,
    filter: F,
) -> Result<HashMap<String, Tensor>>
where
    P: AsRef<Path>,
    F: Fn(&str) -> bool,
{
    use serde_json::Value;

    let file = File::open(path.as_ref())
        .map_err(|e| Error::Io(format!("Failed to open file: {:?}", e)))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| Error::Io(format!("Failed to mmap: {:?}", e)))?;

    if mmap.len() < 8 {
        return Err(Error::Io("File too small for safetensors".into()));
    }

    // Parse header
    let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_size;
    let data_start = header_end; // tensor data begins right after header

    let metadata: Value = serde_json::from_slice(&mmap[8..header_end])
        .map_err(|e| Error::Io(format!("Failed to parse metadata: {:?}", e)))?;
    let metadata_obj = metadata.as_object()
        .ok_or_else(|| Error::InvalidInput("Invalid metadata format".to_string()))?;

    let mut tensors = HashMap::new();

    for (name, info) in metadata_obj {
        if name == "__metadata__" || !filter(name) {
            continue;
        }

        let shape = info["shape"].as_array()
            .ok_or_else(|| Error::InvalidInput("Missing shape".to_string()))?
            .iter()
            .map(|v| v.as_u64().ok_or_else(|| Error::InvalidInput("invalid shape".into())).map(|u| u as usize))
            .collect::<Result<Vec<_>>>()?;

        let offsets = info["data_offsets"].as_array()
            .ok_or_else(|| Error::InvalidInput("Missing data_offsets".to_string()))?;
        let start = data_start + offsets.first().and_then(|v| v.as_u64())
            .ok_or_else(|| Error::InvalidInput("invalid start".into()))? as usize;
        let end = data_start + offsets.get(1).and_then(|v| v.as_u64())
            .ok_or_else(|| Error::InvalidInput("invalid end".into()))? as usize;

        let dtype_str = info["dtype"].as_str().unwrap_or("F32");
        if !matches!(dtype_str, "F32" | "BF16" | "F16") {
            continue;
        }

        let data = &mmap[start..end];

        let tensor = match dtype_str {
            "BF16" => {
                let num_elems = data.len() / 2;
                let mut bf16_u16 = vec![0u16; num_elems];
                for (value, chunk) in bf16_u16.iter_mut().zip(data.chunks_exact(2)) {
                    *value = u16::from_le_bytes([chunk[0], chunk[1]]);
                }
                let mut tensor = Tensor::zeros_dtype(
                    Shape::from_dims(&shape), DType::BF16, device.clone(),
                )?;
                tensor.copy_from_bf16_slice(&bf16_u16)?;
                tensor
            }
            "F16" => {
                let num_elems = data.len() / 2;
                let mut f32_data = vec![0.0f32; num_elems];
                for (value, chunk) in f32_data.iter_mut().zip(data.chunks_exact(2)) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    *value = half::f16::from_bits(bits).to_f32();
                }
                Tensor::from_vec(f32_data, Shape::from_dims(&shape), device.clone())?
            }
            _ => {
                let num_floats = data.len() / 4;
                let mut f32_data = vec![0.0f32; num_floats];
                for (value, chunk) in f32_data.iter_mut().zip(data.chunks_exact(4)) {
                    *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Tensor::from_vec(f32_data, Shape::from_dims(&shape), device.clone())?
            }
        };
        tensors.insert(name.clone(), tensor);
    }

    Ok(tensors)
}

/// Save a safetensors file (convenience function)  
pub fn save_file<P: AsRef<Path>>(tensors: &HashMap<String, Tensor>, path: P) -> Result<()> {
    save_tensors(tensors, path.as_ref(), SerializationFormat::SafeTensors)
}
