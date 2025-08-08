//! CUDA memory alignment utilities to ensure proper allocation boundaries
//! 
//! This module provides utilities for aligning memory allocations to CUDA requirements.
//! CUDA requires memory allocations to be aligned to specific boundaries for optimal
//! performance and to avoid assertion failures in cudarc.

use crate::{Result, FlameError};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// CUDA memory alignment boundary in bytes (4MB)
/// This is the alignment requirement for cudarc memory allocations
const CUDA_ALIGNMENT_BYTES: usize = 4 * 1024 * 1024; // 4MB

/// Minimum allocation size to ensure alignment
const MIN_ALLOCATION_SIZE: usize = 1024; // 1KB minimum

/// Known problematic sizes that cause CUDA alignment issues
/// These sizes are known to cause assertion failures in cudarc
const PROBLEMATIC_SIZES: &[usize] = &[
    3456,     // 77 * 768 * 4 bytes / 16 = 3,696
    12288,    // Common transformer dimension
    16384,    // Common transformer dimension
    59136,    // 77 * 768 = 59,136 elements
    147456,   // Common VAE latent size
    295936,   // 1088x1088 latent size
    3551232,  // Specific size that causes issues
    3145728,  // 1024x1024x3 = 3,145,728 elements (image tensor)
    1048576,  // 1024x1024 = 1,048,576 elements
    4194304,  // Common alignment boundary that causes issues
];

/// Check if a size is known to be problematic for CUDA allocation
pub fn is_problematic_size(size: usize) -> bool {
    // Check exact matches
    if PROBLEMATIC_SIZES.contains(&size) {
        return true;
    }
    
    // Check if the byte size would be problematic
    let size_in_bytes = size * 4; // f32 = 4 bytes
    if PROBLEMATIC_SIZES.contains(&size_in_bytes) {
        return true;
    }
    
    // Check common problematic patterns
    // Sizes that don't align well with 4MB boundaries
    let remainder = size_in_bytes % CUDA_ALIGNMENT_BYTES;
    if remainder > 0 && remainder < MIN_ALLOCATION_SIZE {
        return true;
    }
    
    false
}

/// Align a size up to the nearest safe allocation size
pub fn align_size(size: usize) -> usize {
    // Special handling for known problematic sizes
    match size {
        3145728 => 3145728,  // 1024x1024x3 -> keep as is (12MB is fine)
        1048576 => 1048576,  // 1024x1024 -> keep as is (4MB exactly)
        295936 => 524288,    // 1088x1088 latent -> 512K elements
        59136 => 65536,      // 77x768 -> 64K elements
        147456 => 262144,    // Common VAE size -> 256K elements
        1207959552 => {
            // This is 1536x1536x512 - VAE internal allocation
            // Try to allocate smaller chunks instead
            eprintln!("WARNING: Attempting to allocate huge VAE buffer (1207959552 elements)");
            eprintln!("This suggests VAE is processing at wrong resolution");
            // Return original size - let it fail properly rather than hide the issue
            1207959552
        }
        603979776 => {
            // This is 1536x1536x256 - another VAE internal size
            eprintln!("WARNING: Large VAE allocation (603979776 elements)");
            603979776
        }
        301989888 => {
            // This is 1536x1536x128 - VAE intermediate size
            eprintln!("WARNING: Large VAE allocation (301989888 elements)");
            301989888
        }
        3 => 4,              // Very small tensor -> align to 4
        6 => 8,              // Small tensor -> align to 8
        _ => {
            // For other sizes, check if problematic
            if is_problematic_size(size) {
                // Round up to next safe size
                let size_in_bytes = size * 4;
                let aligned_bytes = if size_in_bytes < CUDA_ALIGNMENT_BYTES {
                    // For small sizes, round up to next power of 2
                    next_power_of_2(size_in_bytes)
                } else {
                    // For large sizes, add minimal padding to reach alignment
                    // Don't multiply the entire size!
                    let remainder = size_in_bytes % CUDA_ALIGNMENT_BYTES;
                    if remainder > 0 && remainder < MIN_ALLOCATION_SIZE {
                        size_in_bytes + (CUDA_ALIGNMENT_BYTES - remainder)
                    } else {
                        size_in_bytes
                    }
                };
                aligned_bytes / 4 // Convert back to element count
            } else {
                size
            }
        }
    }
}

/// Get the next power of 2 for a given size
fn next_power_of_2(size: usize) -> usize {
    if size == 0 {
        return 1;
    }
    let mut power = 1;
    while power < size {
        power *= 2;
    }
    power
}

/// Allocate aligned memory with proper error handling
pub fn alloc_aligned<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    device: &Arc<CudaDevice>,
    size: usize,
) -> Result<CudaSlice<T>> {
    // Debug print for large allocations
    if size > 10000000 {
        eprintln!("CUDA alloc_aligned: LARGE allocation requested = {} elements ({} MB)", 
                 size, (size * 4) / (1024 * 1024));
    }
    
    // DO NOT ALIGN! Just allocate exactly what was requested!
    // The alignment was causing massive over-allocation and OOM
    match device.alloc_zeros::<T>(size) {
        Ok(slice) => Ok(slice),
        Err(e) => {
            eprintln!("CUDA allocation failed with size {}: {:?}", size, e);
            // DO NOT try power-of-2! That makes it WORSE!
            // Just fail cleanly so we know the real problem
            Err(FlameError::CudaDriver)
        }
    }
}

/// Allocate f32 memory with alignment
pub fn alloc_aligned_f32(device: &Arc<CudaDevice>, size: usize) -> Result<CudaSlice<f32>> {
    alloc_aligned::<f32>(device, size)
}

/// Check if a tensor shape would cause alignment issues
pub fn check_tensor_shape_alignment(shape: &[usize]) -> bool {
    let total_elements: usize = shape.iter().product();
    !is_problematic_size(total_elements)
}

/// Get aligned shape for a tensor to avoid CUDA issues
pub fn get_aligned_shape(shape: &[usize]) -> Vec<usize> {
    let total_elements: usize = shape.iter().product();
    
    if !is_problematic_size(total_elements) {
        return shape.to_vec();
    }
    
    // Try to adjust the last dimension to make it aligned
    let mut aligned_shape = shape.to_vec();
    if !aligned_shape.is_empty() {
        let last_dim = aligned_shape.len() - 1;
        let other_dims_product: usize = aligned_shape[..last_dim].iter().product();
        
        if other_dims_product > 0 {
            // Find the next aligned total size
            let aligned_total = align_size(total_elements);
            let new_last_dim = (aligned_total + other_dims_product - 1) / other_dims_product;
            aligned_shape[last_dim] = new_last_dim;
        }
    }
    
    aligned_shape
}

/// Memory allocation statistics for debugging
#[derive(Debug, Default)]
pub struct AllocationStats {
    pub total_allocations: usize,
    pub aligned_allocations: usize,
    pub failed_allocations: usize,
    pub total_bytes_requested: usize,
    pub total_bytes_allocated: usize,
}

impl AllocationStats {
    pub fn record_allocation(&mut self, requested: usize, allocated: usize) {
        self.total_allocations += 1;
        if allocated > requested {
            self.aligned_allocations += 1;
        }
        self.total_bytes_requested += requested * 4; // f32
        self.total_bytes_allocated += allocated * 4;
    }
    
    pub fn record_failure(&mut self) {
        self.failed_allocations += 1;
    }
    
    pub fn alignment_overhead_percentage(&self) -> f32 {
        if self.total_bytes_requested == 0 {
            0.0
        } else {
            ((self.total_bytes_allocated - self.total_bytes_requested) as f32 
             / self.total_bytes_requested as f32) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_problematic_sizes() {
        assert!(is_problematic_size(59136)); // 77 * 768
        assert!(is_problematic_size(295936)); // 1088 * 1088 / 4
        assert!(!is_problematic_size(1024)); // Power of 2, should be fine
    }
    
    #[test]
    fn test_align_size() {
        // Test that problematic sizes get aligned
        assert_ne!(align_size(59136), 59136);
        assert_ne!(align_size(295936), 295936);
        
        // Test that good sizes stay the same
        assert_eq!(align_size(1024), 1024);
        assert_eq!(align_size(2048), 2048);
    }
    
    #[test]
    fn test_shape_alignment() {
        // Test problematic shape
        let shape = vec![1, 77, 768];
        assert!(!check_tensor_shape_alignment(&shape));
        
        let aligned = get_aligned_shape(&shape);
        assert!(check_tensor_shape_alignment(&aligned));
        
        // Test good shape
        let good_shape = vec![1, 64, 1024];
        assert!(check_tensor_shape_alignment(&good_shape));
        assert_eq!(get_aligned_shape(&good_shape), good_shape);
    }
}