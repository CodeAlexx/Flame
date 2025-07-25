# EriDiffusion Data Loading API Documentation

## Overview

EriDiffusion provides comprehensive data loading capabilities for training diffusion models. The data loading system supports multiple dataset formats, bucket-based batching, caching, and various augmentation techniques.

## Core Data Loading Components

### 1. Dataset Trait (`crates/data/src/dataset.rs`)

The base trait that all datasets must implement.

```rust
pub trait Dataset: Send + Sync {
    /// Get dataset length
    fn len(&self) -> usize;
    
    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get item by index
    fn get_item(&self, index: usize) -> anyhow::Result<DatasetItem>;
    
    /// Get metadata
    fn metadata(&self) -> &DatasetMetadata;
    
    /// Get all indices
    fn indices(&self) -> Vec<usize> {
        (0..self.len()).collect()
    }
}
```

**Parameters:**
- `index: usize` - The index of the item to retrieve

**Returns:**
- `Result<DatasetItem>` - The dataset item containing image tensor and caption

### 2. DatasetItem Structure

```rust
#[derive(Debug, Clone)]
pub struct DatasetItem {
    pub image: Tensor,
    pub caption: String,
    pub metadata: HashMap<String, serde_json::Value>,
}
```

**Fields:**
- `image: Tensor` - The image tensor (C, H, W) format
- `caption: String` - The text caption for the image
- `metadata: HashMap<String, Value>` - Additional metadata (resolution, aspect ratio, etc.)

### 3. DataLoader (`crates/data/src/dataloader.rs`)

Main data loading class for batching and parallel loading.

```rust
impl<D: Dataset> DataLoader<D> {
    /// Create new DataLoader
    pub fn new(dataset: D, config: DataLoaderConfig, device: Device) -> Self;
    
    /// Get number of batches
    pub fn len(&self) -> usize;
    
    /// Create iterator
    pub async fn iter(&self) -> DataLoaderIterator<D>;
    
    /// Reset for new epoch
    pub async fn reset(&self);
}
```

**Parameters:**
- `dataset: D` - The dataset to load from
- `config: DataLoaderConfig` - Configuration for the dataloader
- `device: Device` - Target device (CPU or CUDA)

**Configuration:**
```rust
pub struct DataLoaderConfig {
    pub batch_size: usize,         // Batch size (default: 4)
    pub shuffle: bool,              // Whether to shuffle data (default: true)
    pub drop_last: bool,            // Drop incomplete last batch (default: false)
    pub num_workers: usize,         // Number of parallel workers (default: 4)
    pub prefetch_factor: usize,     // Prefetch multiplier (default: 2)
}
```

### 4. DataLoaderBatch

```rust
pub struct DataLoaderBatch {
    pub images: Tensor,
    pub captions: Vec<String>,
    pub masks: Option<Tensor>,
    pub loss_weights: Vec<f32>,
    pub metadata: std::collections::HashMap<String, Vec<serde_json::Value>>,
}

impl DataLoaderBatch {
    pub fn new(
        images: Tensor,
        captions: Vec<String>,
        masks: Option<Tensor>,
        loss_weights: Option<Vec<f32>>,
        metadata: HashMap<String, Vec<Value>>,
    ) -> Self;
    
    pub fn batch_size(&self) -> usize;
    
    pub fn to_device(&self, device: &Device) -> anyhow::Result<Self>;
}
```

**Usage Example:**
```rust
// Create dataset
let dataset = ImageDataset::new("/path/to/images", config)?;

// Create dataloader
let dataloader = DataLoader::new(
    dataset,
    DataLoaderConfig {
        batch_size: 4,
        shuffle: true,
        drop_last: false,
        num_workers: 4,
        prefetch_factor: 2,
    },
    Device::cuda(0)?
);

// Iterate through batches
let mut iter = dataloader.iter().await;
while let Some(batch_result) = iter.next().await {
    let batch = batch_result?;
    // Process batch...
}
```

## Enhanced Data Loading (`src/trainers/enhanced_data_loader.rs`)

### 1. EnhancedDataLoader

Provides advanced features like caption dropout, concept balancing, and aspect ratio bucketing.

```rust
pub struct EnhancedDataLoader {
    batch_size: usize,
    current_step: usize,
}

impl EnhancedDataLoader {
    pub fn new(batch_size: usize) -> Self;
    
    pub fn next_batch(&mut self) -> Result<HashMap<String, Tensor>>;
}
```

**Returns HashMap with:**
- `"pixel_values"` - Image tensors `[batch_size, 3, height, width]`
- `"input_ids"` - Token IDs `[batch_size, max_length]`

### 2. EnhancedCaptionHandler

Handles caption processing with dropout and model-specific unconditional prompts.

```rust
pub struct EnhancedCaptionHandler {
    config: EnhancedDataConfig,
    empty_prompt_content: Option<String>,
    concept_counts: std::collections::HashMap<String, usize>,
    total_samples: usize,
}

impl EnhancedCaptionHandler {
    /// Create new caption handler
    pub fn new(config: EnhancedDataConfig) -> flame_core::Result<Self>;
    
    /// Process caption with proper dropout and model-specific handling
    pub fn process_caption(
        &mut self,
        caption: &str,
        concept: Option<&str>,
        rng: &mut impl Rng,
    ) -> String;
    
    /// Get statistics about concept distribution
    pub fn get_concept_stats(&self) -> HashMap<String, f32>;
    
    /// Helper to detect concepts in captions
    pub fn extract_concept_from_caption(
        caption: &str, 
        trigger_words: &[String]
    ) -> Option<String>;
}
```

**Configuration:**
```rust
pub struct EnhancedDataConfig {
    /// Path to empty prompt file (for dropout)
    pub empty_prompt_file: Option<std::path::PathBuf>,
    
    /// Caption dropout rate (0.0 - 1.0)
    pub caption_dropout_rate: f32,
    
    /// Whether to use empty prompt file for dropout
    pub use_empty_prompt_for_dropout: bool,
    
    /// Duplicate balancing threshold (e.g., 0.1 for 10%)
    pub duplicate_threshold: f32,
    
    /// Duplicate balancing limit (e.g., 0.3 for 30%)
    pub duplicate_limit: f32,
    
    /// Model type for proper unconditional handling
    pub model_type: ModelType,
    
    /// Aspect ratio bucketing configuration
    pub aspect_ratio_buckets: AspectRatioBucketConfig,
}
```

### 3. AspectRatioBucketConfig

Manages resolution bucketing for efficient batch processing.

```rust
pub struct AspectRatioBucketConfig {
    pub enabled: bool,
    pub buckets: Vec<(usize, usize)>,
    pub max_aspect_ratio: f32,
    pub min_aspect_ratio: f32,
}

impl AspectRatioBucketConfig {
    /// Generate standard buckets for a given resolution
    pub fn generate_buckets(resolution: usize, step: usize) -> Vec<(usize, usize)>;
    
    /// Find the best bucket for given image dimensions
    pub fn find_bucket(&self, width: usize, height: usize) -> (usize, usize);
}
```

**Usage Example:**
```rust
// Create enhanced config
let config = EnhancedDataConfig {
    empty_prompt_file: Some(PathBuf::from("empty_prompt.txt")),
    caption_dropout_rate: 0.1,
    use_empty_prompt_for_dropout: true,
    duplicate_threshold: 0.1,
    duplicate_limit: 0.3,
    model_type: ModelType::SDXL,
    aspect_ratio_buckets: AspectRatioBucketConfig::generate_buckets(1024, 64),
};

// Create caption handler
let mut handler = EnhancedCaptionHandler::new(config)?;

// Process caption
let processed = handler.process_caption(
    "a photo of a woman",
    Some("woman"),
    &mut rng
);
```

## Specialized Dataset Implementations

### 1. ImageDataset (`crates/data/src/image_dataset.rs`)

Standard image dataset for loading images from directories.

### 2. WebDataset (`crates/data/src/webdataset.rs`)

Support for WebDataset format (tar archives).

### 3. LatentDataLoader (`crates/data/src/latent_dataloader.rs`)

Loads pre-computed latents for faster training.

### 4. WomanDataset (`crates/data/src/woman_dataset.rs`)

Specialized dataset for the woman training example.

## Data Processing Components

### 1. Bucket Manager (`crates/data/src/bucket.rs`)

Manages aspect ratio bucketing for efficient batching.

### 2. Caption Preprocessor (`crates/data/src/caption_preprocessor.rs`)

Preprocesses and tokenizes captions.

### 3. Cache Manager (`crates/data/src/cache_manager.rs`)

Manages caching of processed data.

### 4. VAE Preprocessor (`crates/data/src/vae_preprocessor.rs`)

Preprocesses images through VAE for latent caching.

## Integration Notes

1. **Device Management**: All data loading functions support both CPU and CUDA devices.
2. **Memory Efficiency**: Use prefetching and parallel workers for optimal performance.
3. **Caption Dropout**: Essential for preventing overfitting on specific prompts.
4. **Aspect Ratio Bucketing**: Reduces padding and improves training efficiency.
5. **Model-Specific Handling**: Different models (SDXL, Flux, SD3.5) have different unconditional prompt requirements.

## Best Practices

1. **Use Bucketing**: Enable aspect ratio bucketing for variable resolution training
2. **Enable Caching**: Cache latents to disk for faster subsequent epochs
3. **Balance Concepts**: Use the 10%/30% rule for duplicate concept balancing
4. **Set Workers**: Use 4-8 workers for optimal data loading performance
5. **Monitor Stats**: Use `get_concept_stats()` to monitor dataset balance