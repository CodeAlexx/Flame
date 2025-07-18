//! Tokenizer wrapper for Flame tensors
//! 
//! This module provides a wrapper around various tokenizers to produce
//! embeddings as Flame tensors for diffusion models.

use crate::{Tensor, Shape, Result, FlameError, CudaDevice};
use std::sync::Arc;
use std::collections::HashMap;

/// Tokenizer trait for text encoding
pub trait Tokenizer {
    /// Tokenize text and return token IDs
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    
    /// Decode token IDs back to text
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;
    
    /// Get the padding token ID
    fn pad_token_id(&self) -> Option<u32>;
    
    /// Get the EOS token ID
    fn eos_token_id(&self) -> Option<u32>;
    
    /// Get the BOS token ID
    fn bos_token_id(&self) -> Option<u32>;
    
    /// Get vocabulary size
    fn vocab_size(&self) -> usize;
}

/// CLIP tokenizer for Stable Diffusion models
pub struct CLIPTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    pad_token_id: u32,
    eos_token_id: u32,
    bos_token_id: u32,
    max_length: usize,
}

impl CLIPTokenizer {
    /// Create a new CLIP tokenizer
    pub fn new(max_length: usize) -> Self {
        // Initialize with basic vocabulary
        // In production, this would load from a vocab file
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        
        // Special tokens
        vocab.insert("<|startoftext|>".to_string(), 49406);
        vocab.insert("<|endoftext|>".to_string(), 49407);
        vocab.insert("<|pad|>".to_string(), 49408);
        
        reverse_vocab.insert(49406, "<|startoftext|>".to_string());
        reverse_vocab.insert(49407, "<|endoftext|>".to_string());
        reverse_vocab.insert(49408, "<|pad|>".to_string());
        
        // Add some common tokens for testing
        let common_words = vec!["a", "the", "cat", "dog", "photo", "of", "beautiful", "landscape"];
        for (i, word) in common_words.iter().enumerate() {
            vocab.insert(word.to_string(), i as u32);
            reverse_vocab.insert(i as u32, word.to_string());
        }
        
        Self {
            vocab,
            reverse_vocab,
            bpe_ranks: HashMap::new(),
            pad_token_id: 49408,
            eos_token_id: 49407,
            bos_token_id: 49406,
            max_length,
        }
    }
    
    /// Simple tokenization (word-level for demo)
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }
}

impl Tokenizer for CLIPTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let tokens = self.tokenize(text);
        let mut token_ids = Vec::new();
        
        if add_special_tokens {
            token_ids.push(self.bos_token_id);
        }
        
        for token in tokens {
            if let Some(&id) = self.vocab.get(&token) {
                token_ids.push(id);
            } else {
                // Unknown token - in real implementation would use BPE
                token_ids.push(0); // UNK token
            }
        }
        
        if add_special_tokens {
            token_ids.push(self.eos_token_id);
        }
        
        // Pad or truncate to max_length
        if token_ids.len() < self.max_length {
            token_ids.resize(self.max_length, self.pad_token_id);
        } else if token_ids.len() > self.max_length {
            token_ids.truncate(self.max_length);
            if add_special_tokens && token_ids.len() > 0 {
                token_ids[self.max_length - 1] = self.eos_token_id;
            }
        }
        
        Ok(token_ids)
    }
    
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut words = Vec::new();
        
        for &id in token_ids {
            if let Some(word) = self.reverse_vocab.get(&id) {
                if skip_special_tokens && 
                   (id == self.pad_token_id || id == self.eos_token_id || id == self.bos_token_id) {
                    continue;
                }
                words.push(word.clone());
            }
        }
        
        Ok(words.join(" "))
    }
    
    fn pad_token_id(&self) -> Option<u32> {
        Some(self.pad_token_id)
    }
    
    fn eos_token_id(&self) -> Option<u32> {
        Some(self.eos_token_id)
    }
    
    fn bos_token_id(&self) -> Option<u32> {
        Some(self.bos_token_id)
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// T5 tokenizer for Flux and SD3.5 models
pub struct T5Tokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    pad_token_id: u32,
    eos_token_id: u32,
    max_length: usize,
}

impl T5Tokenizer {
    pub fn new(max_length: usize) -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        
        // Special tokens
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("</s>".to_string(), 1);
        vocab.insert("<unk>".to_string(), 2);
        
        reverse_vocab.insert(0, "<pad>".to_string());
        reverse_vocab.insert(1, "</s>".to_string());
        reverse_vocab.insert(2, "<unk>".to_string());
        
        Self {
            vocab,
            reverse_vocab,
            pad_token_id: 0,
            eos_token_id: 1,
            max_length,
        }
    }
}

impl Tokenizer for T5Tokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        // Simple implementation for demo
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut token_ids = Vec::new();
        
        for word in words {
            // In real implementation, would use sentencepiece
            token_ids.push(2); // UNK for now
        }
        
        if add_special_tokens && !token_ids.is_empty() {
            token_ids.push(self.eos_token_id);
        }
        
        // Pad to max_length
        if token_ids.len() < self.max_length {
            token_ids.resize(self.max_length, self.pad_token_id);
        } else if token_ids.len() > self.max_length {
            token_ids.truncate(self.max_length);
        }
        
        Ok(token_ids)
    }
    
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut words = Vec::new();
        
        for &id in token_ids {
            if skip_special_tokens && (id == self.pad_token_id || id == self.eos_token_id) {
                continue;
            }
            
            if let Some(word) = self.reverse_vocab.get(&id) {
                words.push(word.clone());
            }
        }
        
        Ok(words.join(" "))
    }
    
    fn pad_token_id(&self) -> Option<u32> {
        Some(self.pad_token_id)
    }
    
    fn eos_token_id(&self) -> Option<u32> {
        Some(self.eos_token_id)
    }
    
    fn bos_token_id(&self) -> Option<u32> {
        None // T5 doesn't use BOS
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Text encoder wrapper that combines tokenizer with embedding layer
pub struct TextEncoder {
    tokenizer: Box<dyn Tokenizer>,
    embedding_dim: usize,
    max_length: usize,
    device: Arc<CudaDevice>,
}

impl TextEncoder {
    pub fn new(
        tokenizer: Box<dyn Tokenizer>,
        embedding_dim: usize,
        max_length: usize,
        device: Arc<CudaDevice>,
    ) -> Self {
        Self {
            tokenizer,
            embedding_dim,
            max_length,
            device,
        }
    }
    
    /// Encode text to embeddings
    pub fn encode(&self, texts: &[&str]) -> Result<Tensor> {
        let batch_size = texts.len();
        let mut all_token_ids = Vec::new();
        
        // Tokenize all texts
        for text in texts {
            let token_ids = self.tokenizer.encode(text, true)?;
            all_token_ids.extend(token_ids);
        }
        
        // Create token tensor
        let token_tensor = Tensor::from_vec(
            all_token_ids.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[batch_size, self.max_length]),
            self.device.clone()
        )?;
        
        // Actually create embeddings from token IDs
        let vocab_size = self.tokenizer.vocab_size();
        
        // Create embedding matrix if not exists (in production, load from checkpoint)
        let embedding_matrix = Tensor::randn(
            Shape::from_dims(&[vocab_size, self.embedding_dim]),
            0.0,
            0.02,
            self.device.clone()
        )?;
        
        // Now actually look up embeddings for each token
        let mut embedding_data = Vec::with_capacity(batch_size * self.max_length * self.embedding_dim);
        
        for batch_idx in 0..batch_size {
            for seq_idx in 0..self.max_length {
                let token_id = all_token_ids[batch_idx * self.max_length + seq_idx] as usize;
                
                // Get embedding for this token
                let embedding_start = token_id * self.embedding_dim;
                let embedding_end = embedding_start + self.embedding_dim;
                
                if embedding_end <= vocab_size * self.embedding_dim {
                    // Extract embedding vector from matrix
                    let matrix_data = embedding_matrix.to_vec()?;
                    embedding_data.extend_from_slice(&matrix_data[embedding_start..embedding_end]);
                } else {
                    // Out of vocabulary - use zeros
                    embedding_data.extend(vec![0.0f32; self.embedding_dim]);
                }
            }
        }
        
        let embeddings = Tensor::from_vec(
            embedding_data,
            Shape::from_dims(&[batch_size, self.max_length, self.embedding_dim]),
            self.device.clone()
        )?;
        
        Ok(embeddings)
    }
    
    /// Encode text and return both embeddings and attention mask
    pub fn encode_with_mask(&self, texts: &[&str]) -> Result<(Tensor, Tensor)> {
        let batch_size = texts.len();
        let mut all_token_ids = Vec::new();
        let mut attention_mask = Vec::new();
        
        // Tokenize all texts and create attention mask
        for text in texts {
            let token_ids = self.tokenizer.encode(text, true)?;
            
            // Create attention mask (1 for real tokens, 0 for padding)
            let mask: Vec<f32> = token_ids.iter()
                .map(|&id| if id == self.tokenizer.pad_token_id().unwrap_or(0) { 0.0 } else { 1.0 })
                .collect();
            
            all_token_ids.extend(token_ids);
            attention_mask.extend(mask);
        }
        
        // Create embeddings
        let embeddings = self.encode(texts)?;
        
        // Create attention mask tensor
        let mask_tensor = Tensor::from_vec(
            attention_mask,
            Shape::from_dims(&[batch_size, self.max_length]),
            self.device.clone()
        )?;
        
        Ok((embeddings, mask_tensor))
    }
}

/// Create a text encoder for specific model type
pub fn create_text_encoder(
    model_type: &str,
    device: Arc<CudaDevice>,
) -> Result<TextEncoder> {
    match model_type.to_lowercase().as_str() {
        "clip" | "sdxl" | "sd15" | "sd21" => {
            let tokenizer = Box::new(CLIPTokenizer::new(77));
            Ok(TextEncoder::new(tokenizer, 768, 77, device))
        }
        "clip-l" => {
            let tokenizer = Box::new(CLIPTokenizer::new(77));
            Ok(TextEncoder::new(tokenizer, 1280, 77, device))
        }
        "t5" | "flux" | "sd3" | "sd35" => {
            let tokenizer = Box::new(T5Tokenizer::new(256));
            Ok(TextEncoder::new(tokenizer, 4096, 256, device))
        }
        _ => Err(FlameError::InvalidOperation(
            format!("Unknown text encoder type: {}", model_type)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_clip_tokenizer() -> Result<()> {
        let tokenizer = CLIPTokenizer::new(77);
        
        let text = "a photo of a cat";
        let token_ids = tokenizer.encode(text, true)?;
        
        assert_eq!(token_ids.len(), 77); // Should be padded to max_length
        assert_eq!(token_ids[0], 49406); // BOS token
        
        let decoded = tokenizer.decode(&token_ids, true)?;
        println!("Decoded: {}", decoded);
        
        Ok(())
    }
    
    #[test]
    fn test_text_encoder() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        
        let encoder = create_text_encoder("clip", device)?;
        
        let texts = vec!["a beautiful landscape", "a photo of a dog"];
        let embeddings = encoder.encode(&texts)?;
        
        assert_eq!(embeddings.shape().dims(), &[2, 77, 768]);
        
        // Test with attention mask
        let (embeddings, mask) = encoder.encode_with_mask(&texts)?;
        assert_eq!(embeddings.shape().dims(), &[2, 77, 768]);
        assert_eq!(mask.shape().dims(), &[2, 77]);
        
        Ok(())
    }
}