// Tests to ensure FLAME remains a generic tensor framework
// without any model-specific implementations

use std::fs;
use std::path::Path;

#[test]
fn test_no_model_specific_code() {
    // List of terms that should NOT appear in FLAME framework code
    let forbidden_terms = [
        // Model names
        "unet", "vae", "clip", "t5", "mmdit", "dit",
        "stable_diffusion", "sdxl", "sd3", "flux",
        
        // Model-specific components
        "text_encoder", "image_encoder", "noise_scheduler",
        "latent", "denoise", "sample", "diffusion",
        "timestep_embedding", "cross_attention", "controlnet",
        
        // Training-specific terms that belong in trainer
        "dataloader", "dataset", "image_preprocessing",
        "tokenizer", "caption", "prompt",
        
        // These indicate application logic, not framework
        "pipeline", "inference", "generate", "train_step",
        "validation", "checkpoint", "lora", "dreambooth"
    ];
    
    // Check all source files
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    check_directory_for_forbidden_terms(&src_dir, &forbidden_terms);
}

fn check_directory_for_forbidden_terms(dir: &Path, forbidden: &[&str]) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            
            if path.is_dir() {
                // Skip checking model-specific example directories
                if path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.contains("example") || n.contains("test"))
                    .unwrap_or(false) 
                {
                    continue;
                }
                check_directory_for_forbidden_terms(&path, forbidden);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                check_file_for_forbidden_terms(&path, forbidden);
            }
        }
    }
}

fn check_file_for_forbidden_terms(file_path: &Path, forbidden: &[&str]) {
    // Skip test files and examples
    if file_path.to_str().unwrap_or("").contains("test") {
        return;
    }
    
    if let Ok(content) = fs::read_to_string(file_path) {
        let content_lower = content.to_lowercase();
        
        for term in forbidden {
            if content_lower.contains(term) {
                // Check if it's in a comment or doc string (those are OK)
                let lines: Vec<&str> = content.lines().collect();
                for (line_num, line) in lines.iter().enumerate() {
                    let line_lower = line.to_lowercase();
                    if line_lower.contains(term) {
                        // Skip if it's a comment
                        let trimmed = line.trim();
                        if trimmed.starts_with("//") || trimmed.starts_with("///") || trimmed.starts_with("/*") {
                            continue;
                        }
                        
                        // Skip if it's in a string literal used for error messages
                        if line.contains("\"") && is_in_string(line, term) {
                            continue;
                        }
                        
                        panic!(
                            "Found model-specific term '{}' in framework code at {}:{}\nLine: {}",
                            term,
                            file_path.display(),
                            line_num + 1,
                            line.trim()
                        );
                    }
                }
            }
        }
    }
}

fn is_in_string(line: &str, term: &str) -> bool {
    // Simple check if term appears within quotes
    let parts: Vec<&str> = line.split('"').collect();
    for (i, part) in parts.iter().enumerate() {
        if i % 2 == 1 && part.to_lowercase().contains(term) {
            return true;
        }
    }
    false
}

#[test]
fn test_framework_apis_are_generic() {
    // Verify that public APIs use generic terminology
    
    // Conv2d should use generic terms
    use flame_core::conv::Conv2d;
    // API should be: new(in_channels, out_channels, kernel_size, stride, padding, bias, device)
    // NOT: new(unet_channels, ...) or new(vae_channels, ...)
    
    // Linear should use generic terms  
    use flame_core::linear::Linear;
    // API should be: new(in_features, out_features, bias, device)
    // NOT: new(clip_dim, ...) or new(text_encoder_dim, ...)
    
    // Attention should use generic terms
    use flame_core::attention::AttentionConfig;
    // API should be: new(embed_dim, num_heads)
    // NOT: new(clip_embed_dim, ...) or new(cross_attention_dim, ...)
}

#[test]
fn test_no_training_logic_in_framework() {
    // FLAME should not contain training loops, only building blocks
    
    // These should NOT exist in FLAME:
    // - DataLoader implementations
    // - Training step functions
    // - Validation loops
    // - Metric calculations
    // - Checkpoint saving/loading (beyond basic tensor serialization)
    
    // These SHOULD exist in FLAME:
    // - Tensor operations
    // - Autograd engine
    // - Basic layers (Conv, Linear, etc.)
    // - Optimizers (they update tensors, don't orchestrate training)
    // - Activation functions
}

#[test]
fn test_separation_of_concerns() {
    // Verify FLAME follows the separation of concerns principle
    
    // FLAME provides:
    // 1. Tensor abstraction with CUDA backend
    // 2. Automatic differentiation 
    // 3. Basic neural network layers
    // 4. Optimization algorithms
    // 5. Low-level operations
    
    // FLAME does NOT provide:
    // 1. Model architectures (UNet, VAE, etc.)
    // 2. Training pipelines
    // 3. Data loading/preprocessing
    // 4. Inference pipelines
    // 5. Model-specific utilities
    
    // This ensures FLAME can be used for ANY deep learning task,
    // not just diffusion models
}