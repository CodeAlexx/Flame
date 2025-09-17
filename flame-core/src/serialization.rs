use crate::{Tensor, Shape, Result, FlameError};
use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use std::io::{Write, Read, BufWriter, BufReader};
use std::sync::Arc;
use cudarc::driver::CudaDevice;

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
pub fn load_tensor(path: &Path, device: Arc<CudaDevice>, format: SerializationFormat) -> Result<Tensor> {
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
    let file = File::create(path)
        .map_err(|e| FlameError::Io(format!("Failed to create file: {}", e)))?;
    let mut writer = BufWriter::new(file);
    
    // Write magic number
    writer.write_all(b"FLMT").map_err(|e| FlameError::Io(e.to_string()))?;
    
    // Write version (little-endian)
    writer.write_all(&1u32.to_le_bytes()).map_err(|e| FlameError::Io(e.to_string()))?;
    
    // Write shape
    let dims = tensor.shape().dims();
    writer.write_all(&(dims.len() as u32).to_le_bytes())
        .map_err(|e| FlameError::Io(e.to_string()))?;
    for &dim in dims {
        writer.write_all(&(dim as u64).to_le_bytes())
            .map_err(|e| FlameError::Io(e.to_string()))?;
    }
    
    // Write data
    let data = tensor.to_vec()?;
    for &value in &data {
        writer.write_all(&value.to_le_bytes())
            .map_err(|e| FlameError::Io(e.to_string()))?;
    }
    
    writer.flush().map_err(|e| FlameError::Io(e.to_string()))?;
    Ok(())
}

fn load_tensor_binary(path: &Path, device: Arc<CudaDevice>) -> Result<Tensor> {
    let file = File::open(path)
        .map_err(|e| FlameError::Io(format!("Failed to open file: {}", e)))?;
    let mut reader = BufReader::new(file);
    
    // Read and verify magic number
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|e| FlameError::Io(e.to_string()))?;
    if &magic != b"FLMT" {
        return Err(FlameError::InvalidOperation("Invalid file format".to_string()));
    }
    
    // Read version
    let mut version = [0u8; 4];
    reader.read_exact(&mut version).map_err(|e| FlameError::Io(e.to_string()))?;
    let version = u32::from_le_bytes(version);
    if version != 1 {
        return Err(FlameError::InvalidOperation(format!("Unsupported version: {}", version)));
    }
    
    // Read shape
    let mut ndims_bytes = [0u8; 4];
    reader.read_exact(&mut ndims_bytes).map_err(|e| FlameError::Io(e.to_string()))?;
    let ndims = u32::from_le_bytes(ndims_bytes) as usize;
    
    let mut dims = vec![0usize; ndims];
    for i in 0..ndims {
        let mut dim_bytes = [0u8; 8];
        reader.read_exact(&mut dim_bytes).map_err(|e| FlameError::Io(e.to_string()))?;
        dims[i] = u64::from_le_bytes(dim_bytes) as usize;
    }
    
    let shape = Shape::from_dims(&dims);
    let numel = shape.elem_count();
    
    // Read data
    let mut data = vec![0.0f32; numel];
    for i in 0..numel {
        let mut value_bytes = [0u8; 4];
        reader.read_exact(&mut value_bytes).map_err(|e| FlameError::Io(e.to_string()))?;
        data[i] = f32::from_le_bytes(value_bytes);
    }
    
    Tensor::from_vec(data, shape, device)
}

fn save_tensors_binary(tensors: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    let file = File::create(path)
        .map_err(|e| FlameError::Io(format!("Failed to create file: {}", e)))?;
    let mut writer = BufWriter::new(file);
    
    // Write magic number
    writer.write_all(b"FLMM").map_err(|e| FlameError::Io(e.to_string()))?;
    
    // Write version (little-endian)
    writer.write_all(&1u32.to_le_bytes()).map_err(|e| FlameError::Io(e.to_string()))?;
    
    // Write number of tensors
    writer.write_all(&(tensors.len() as u32).to_le_bytes())
        .map_err(|e| FlameError::Io(e.to_string()))?;
    
    // Write each tensor
    for (name, tensor) in tensors {
        // Write name length and name
        let name_bytes = name.as_bytes();
        writer.write_all(&(name_bytes.len() as u32).to_le_bytes())
            .map_err(|e| FlameError::Io(e.to_string()))?;
        writer.write_all(name_bytes)
            .map_err(|e| FlameError::Io(e.to_string()))?;
        
        // Write shape
        let dims = tensor.shape().dims();
        writer.write_all(&(dims.len() as u32).to_le_bytes())
            .map_err(|e| FlameError::Io(e.to_string()))?;
        for &dim in dims {
            writer.write_all(&(dim as u64).to_le_bytes())
                .map_err(|e| FlameError::Io(e.to_string()))?;
        }
        
        // Write data
        let data = tensor.to_vec()?;
        for &value in &data {
            writer.write_all(&value.to_le_bytes())
                .map_err(|e| FlameError::Io(e.to_string()))?;
        }
    }
    
    writer.flush().map_err(|e| FlameError::Io(e.to_string()))?;
    Ok(())
}

fn load_tensors_binary(path: &Path, device: Arc<CudaDevice>) -> Result<HashMap<String, Tensor>> {
    let file = File::open(path)
        .map_err(|e| FlameError::Io(format!("Failed to open file: {}", e)))?;
    let mut reader = BufReader::new(file);
    
    // Read and verify magic number
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|e| FlameError::Io(e.to_string()))?;
    if &magic != b"FLMM" {
        return Err(FlameError::InvalidOperation("Invalid file format".to_string()));
    }
    
    // Read version
    let mut version = [0u8; 4];
    reader.read_exact(&mut version).map_err(|e| FlameError::Io(e.to_string()))?;
    let version = u32::from_le_bytes(version);
    if version != 1 {
        return Err(FlameError::InvalidOperation(format!("Unsupported version: {}", version)));
    }
    
    // Read number of tensors
    let mut num_tensors_bytes = [0u8; 4];
    reader.read_exact(&mut num_tensors_bytes).map_err(|e| FlameError::Io(e.to_string()))?;
    let num_tensors = u32::from_le_bytes(num_tensors_bytes) as usize;
    
    let mut tensors = HashMap::new();
    
    for _ in 0..num_tensors {
        // Read name
        let mut name_len_bytes = [0u8; 4];
        reader.read_exact(&mut name_len_bytes).map_err(|e| FlameError::Io(e.to_string()))?;
        let name_len = u32::from_le_bytes(name_len_bytes) as usize;
        
        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes).map_err(|e| FlameError::Io(e.to_string()))?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| FlameError::InvalidOperation(format!("Invalid UTF-8 in name: {}", e)))?;
        
        // Read shape
        let mut ndims_bytes = [0u8; 4];
        reader.read_exact(&mut ndims_bytes).map_err(|e| FlameError::Io(e.to_string()))?;
        let ndims = u32::from_le_bytes(ndims_bytes) as usize;
        
        let mut dims = vec![0usize; ndims];
        for i in 0..ndims {
            let mut dim_bytes = [0u8; 8];
            reader.read_exact(&mut dim_bytes).map_err(|e| FlameError::Io(e.to_string()))?;
            dims[i] = u64::from_le_bytes(dim_bytes) as usize;
        }
        
        let shape = Shape::from_dims(&dims);
        let numel = shape.elem_count();
        
        // Read data
        let mut data = vec![0.0f32; numel];
        for i in 0..numel {
            let mut value_bytes = [0u8; 4];
            reader.read_exact(&mut value_bytes).map_err(|e| FlameError::Io(e.to_string()))?;
            data[i] = f32::from_le_bytes(value_bytes);
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
    Ok(
        tensors.get("tensor")
            .ok_or_else(|| FlameError::InvalidOperation("No 'tensor' key found".to_string()))?
            .clone()
    )
}

fn save_tensors_safetensors<T: AsRef<Tensor>>(tensors: &HashMap<String, T>, path: &Path) -> Result<()> {
    use serde_json::{json, Value};
    
    let file = File::create(path)
        .map_err(|e| FlameError::Io(format!("Failed to create file: {}", e)))?;
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
        .map_err(|e| FlameError::Io(format!("Failed to serialize metadata: {}", e)))?;
    let metadata_bytes = metadata_json.as_bytes();
    
    // Write header size (8 bytes, little-endian)
    let header_size = metadata_bytes.len() as u64;
    writer.write_all(&header_size.to_le_bytes())
        .map_err(|e| FlameError::Io(e.to_string()))?;
    
    // Write metadata
    writer.write_all(metadata_bytes)
        .map_err(|e| FlameError::Io(e.to_string()))?;
    
    // Write tensor data
    for data in tensor_data {
        for &value in &data {
            writer.write_all(&value.to_le_bytes())
                .map_err(|e| FlameError::Io(e.to_string()))?;
        }
    }
    
    writer.flush().map_err(|e| FlameError::Io(e.to_string()))?;
    Ok(())
}

fn load_tensors_safetensors(path: &Path, device: Arc<CudaDevice>) -> Result<HashMap<String, Tensor>> {
    use serde_json::Value;
    
    let file = File::open(path)
        .map_err(|e| FlameError::Io(format!("Failed to open file: {}", e)))?;
    let mut reader = BufReader::new(file);
    
    // Read header size
    let mut header_size_bytes = [0u8; 8];
    reader.read_exact(&mut header_size_bytes)
        .map_err(|e| FlameError::Io(e.to_string()))?;
    let header_size = u64::from_le_bytes(header_size_bytes) as usize;
    
    // Read metadata
    let mut metadata_bytes = vec![0u8; header_size];
    reader.read_exact(&mut metadata_bytes)
        .map_err(|e| FlameError::Io(e.to_string()))?;
    
    let metadata: Value = serde_json::from_slice(&metadata_bytes)
        .map_err(|e| FlameError::Io(format!("Failed to parse metadata: {}", e)))?;
    
    let metadata_obj = metadata.as_object()
        .ok_or_else(|| FlameError::InvalidInput("Invalid metadata format".to_string()))?;
    
    // Read all remaining data
    let mut all_data = Vec::new();
    reader.read_to_end(&mut all_data)
        .map_err(|e| FlameError::Io(e.to_string()))?;
    
    let mut tensors = HashMap::new();
    
    for (name, info) in metadata_obj {
        let shape = info["shape"].as_array()
            .ok_or_else(|| FlameError::InvalidInput("Missing shape".to_string()))?
            .iter()
            .map(|v| v.as_u64().ok_or_else(|| FlameError::InvalidInput("invalid shape entry".into())).map(|u| u as usize))
            .collect::<Result<Vec<_>>>()?;
        
        let offsets = info["data_offsets"].as_array()
            .ok_or_else(|| FlameError::InvalidInput("Missing data_offsets".to_string()))?;
        let start = offsets.get(0).and_then(|v| v.as_u64()).ok_or_else(|| FlameError::InvalidInput("invalid start offset".into()))? as usize;
        let end = offsets.get(1).and_then(|v| v.as_u64()).ok_or_else(|| FlameError::InvalidInput("invalid end offset".into()))? as usize;
        
        // Extract tensor data
        let num_floats = (end - start) / 4;
        let mut data = vec![0.0f32; num_floats];
        
        for i in 0..num_floats {
            let offset = start + i * 4;
            let bytes = [
                all_data[offset],
                all_data[offset + 1],
                all_data[offset + 2],
                all_data[offset + 3],
            ];
            data[i] = f32::from_le_bytes(bytes);
        }
        
        let tensor = Tensor::from_vec(data, Shape::from_dims(&shape), device.clone())?;
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
pub fn load_file<P: AsRef<Path>>(path: P, device: &Arc<CudaDevice>) -> Result<HashMap<String, Tensor>> {
    load_tensors(path.as_ref(), device.clone(), SerializationFormat::SafeTensors)
}

/// Save a safetensors file (convenience function)  
pub fn save_file<P: AsRef<Path>>(tensors: &HashMap<String, Tensor>, path: P) -> Result<()> {
    save_tensors(tensors, path.as_ref(), SerializationFormat::SafeTensors)
}
