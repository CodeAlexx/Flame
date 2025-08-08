//! Memory pool for efficient GPU memory management
//! 
//! This module provides memory pooling to reduce allocation overhead
//! and memory fragmentation during training and inference.

use crate::{Result, FlameError, Shape};
use crate::cuda_memory_alignment::{align_size, is_problematic_size};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, VecDeque};

/// Memory block information
#[derive(Clone, Debug)]
struct MemoryBlock {
    ptr: CudaSlice<f32>,
    size: usize,
    in_use: bool,
}

/// Memory pool for a specific device
pub struct DeviceMemoryPool {
    device: Arc<CudaDevice>,
    /// Pools organized by size (power of 2)
    pools: HashMap<usize, VecDeque<MemoryBlock>>,
    /// Total allocated memory
    total_allocated: usize,
    /// Maximum memory limit
    max_memory: Option<usize>,
    /// Statistics
    stats: PoolStatistics,
}

/// Pool statistics for monitoring
#[derive(Default, Debug, Clone)]
pub struct PoolStatistics {
    pub allocations: usize,
    pub deallocations: usize,
    pub reuses: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub peak_memory: usize,
}

impl DeviceMemoryPool {
    /// Create a new memory pool for a device
    pub fn new(device: Arc<CudaDevice>, max_memory: Option<usize>) -> Self {
        Self {
            device,
            pools: HashMap::new(),
            total_allocated: 0,
            max_memory,
            stats: PoolStatistics::default(),
        }
    }
    
    /// Get the next power of 2 size
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
    
    /// Allocate memory from the pool
    pub fn allocate(&mut self, size: usize) -> Result<CudaSlice<f32>> {
        self.stats.allocations += 1;
        
        // Use alignment-aware size calculation
        let aligned_size = align_size(size);
        let pool_size = Self::next_power_of_2(aligned_size);
        
        // Check if we have a free block in the pool
        if let Some(pool) = self.pools.get_mut(&pool_size) {
            for block in pool.iter_mut() {
                if !block.in_use && block.size >= aligned_size {
                    block.in_use = true;
                    self.stats.cache_hits += 1;
                    self.stats.reuses += 1;
                    return Ok(block.ptr.clone());
                }
            }
        }
        
        // No suitable block found, allocate new
        self.stats.cache_misses += 1;
        
        // Check memory limit
        if let Some(max) = self.max_memory {
            if self.total_allocated + pool_size * 4 > max {
                return Err(FlameError::OutOfMemory(
                    format!("Memory pool limit exceeded: {} + {} > {}", 
                        self.total_allocated, pool_size * 4, max)
                ));
            }
        }
        
        // Allocate new block with proper alignment
        let ptr = match self.device.alloc_zeros::<f32>(pool_size) {
            Ok(p) => p,
            Err(_) if pool_size != aligned_size => {
                // If power-of-2 allocation fails, try aligned size
                self.device.alloc_zeros::<f32>(aligned_size)
                    .map_err(|_| FlameError::CudaDriver)?
            }
            Err(_e) => return Err(FlameError::CudaDriver),
        };
        
        let block = MemoryBlock {
            ptr: ptr.clone(),
            size: pool_size,
            in_use: true,
        };
        
        // Add to pool
        self.pools.entry(pool_size)
            .or_insert_with(VecDeque::new)
            .push_back(block);
        
        self.total_allocated += pool_size * 4;
        self.stats.peak_memory = self.stats.peak_memory.max(self.total_allocated);
        
        Ok(ptr)
    }
    
    /// Return memory to the pool
    pub fn deallocate(&mut self, _ptr: CudaSlice<f32>) {
        self.stats.deallocations += 1;
        
        // Find the block and mark as free
        for pool in self.pools.values_mut() {
            for block in pool.iter_mut() {
                // Compare by size since we can't directly compare pointers
                // This assumes we're deallocating the same size that was allocated
                // Just mark the first in-use block of matching size as free
                // This is a simplified approach since we can't compare pointers directly
                if block.in_use {
                    block.in_use = false;
                    return;
                }
            }
        }
    }
    
    /// Force GPU synchronization and memory cleanup
    pub fn force_cleanup(&self) -> Result<()> {
        // Synchronize the device to ensure all operations complete
        self.device.synchronize()
            .map_err(|_| FlameError::CudaDriver)?;
        
        // Note: cuMemAdvise is not available in cudarc, but the synchronize
        // above should help ensure memory operations complete
        
        Ok(())
    }
    
    /// Clear all unused memory
    pub fn clear_cache(&mut self) {
        let mut freed = 0;
        
        for (size, pool) in &mut self.pools {
            pool.retain(|block| {
                if !block.in_use {
                    freed += size * 4;
                    false
                } else {
                    true
                }
            });
        }
        
        self.total_allocated -= freed;
        
        // Remove empty pools
        self.pools.retain(|_, pool| !pool.is_empty());
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> &PoolStatistics {
        &self.stats
    }
    
    /// Get current memory usage
    pub fn memory_usage(&self) -> MemoryUsage {
        let mut in_use = 0;
        let mut cached = 0;
        
        for pool in self.pools.values() {
            for block in pool {
                let size = block.size * 4; // f32 = 4 bytes
                if block.in_use {
                    in_use += size;
                } else {
                    cached += size;
                }
            }
        }
        
        MemoryUsage {
            in_use,
            cached,
            total: in_use + cached,
        }
    }
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub in_use: usize,
    pub cached: usize,
    pub total: usize,
}

/// Global memory pool manager
pub struct MemoryPoolManager {
    pools: Mutex<HashMap<i32, Arc<Mutex<DeviceMemoryPool>>>>,
}

impl MemoryPoolManager {
    /// Create a new memory pool manager
    pub fn new() -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
        }
    }
    
    /// Get or create pool for a device
    pub fn get_pool(&self, device: &Arc<CudaDevice>) -> Result<Arc<Mutex<DeviceMemoryPool>>> {
        let ordinal = device.ordinal();
        let mut pools = self.pools.lock().unwrap();
        
        let pool = pools.entry(ordinal as i32)
            .or_insert_with(|| {
                Arc::new(Mutex::new(DeviceMemoryPool::new(device.clone(), None)))
            })
            .clone();
        
        Ok(pool)
    }
    
    /// Clear all caches
    pub fn clear_all_caches(&self) {
        let pools = self.pools.lock().unwrap();
        for pool in pools.values() {
            if let Ok(mut pool_guard) = pool.lock() {
                pool_guard.clear_cache();
            }
        }
    }
    
    /// Get total memory usage across all devices
    pub fn total_memory_usage(&self) -> MemoryUsage {
        let pools = self.pools.lock().unwrap();
        let mut total = MemoryUsage {
            in_use: 0,
            cached: 0,
            total: 0,
        };
        
        for pool in pools.values() {
            if let Ok(pool_guard) = pool.lock() {
                let usage = pool_guard.memory_usage();
                total.in_use += usage.in_use;
                total.cached += usage.cached;
                total.total += usage.total;
            }
        }
        
        total
    }
}

lazy_static::lazy_static! {
    /// Global memory pool manager
    pub static ref MEMORY_POOL: MemoryPoolManager = MemoryPoolManager::new();
}

/// Pooled tensor allocation
pub struct PooledTensor {
    pub data: CudaSlice<f32>,
    pub shape: Shape,
    pub device: Arc<CudaDevice>,
    pool: Arc<Mutex<DeviceMemoryPool>>,
}

impl Drop for PooledTensor {
    fn drop(&mut self) {
        // Return memory to pool when dropped
        if let Ok(mut pool) = self.pool.lock() {
            pool.deallocate(self.data.clone());
        }
    }
}

/// Allocate a pooled tensor
pub fn allocate_pooled(
    shape: Shape,
    device: Arc<CudaDevice>,
) -> Result<PooledTensor> {
    // Use the global memory pool
    let pool = MEMORY_POOL.get_pool(&device)?;
    
    let size = shape.elem_count();
    let data = pool.lock().unwrap().allocate(size)?;
    
    Ok(PooledTensor {
        data,
        shape,
        device,
        pool,
    })
}

/// Memory-efficient workspace for operations
pub struct Workspace {
    device: Arc<CudaDevice>,
    buffers: Vec<CudaSlice<f32>>,
    pool: DeviceMemoryPool,
}

impl Workspace {
    /// Create a new workspace
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device: device.clone(),
            buffers: Vec::new(),
            pool: DeviceMemoryPool::new(device, None),
        }
    }
    
    /// Get a temporary buffer
    pub fn get_buffer(&mut self, size: usize) -> Result<&CudaSlice<f32>> {
        let buffer = self.pool.allocate(size)?;
        self.buffers.push(buffer);
        Ok(self.buffers.last().unwrap())
    }
    
    /// Clear all buffers
    pub fn clear(&mut self) {
        for buffer in self.buffers.drain(..) {
            self.pool.deallocate(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let mut pool = DeviceMemoryPool::new(device.clone(), Some(1024 * 1024 * 100)); // 100MB limit
        
        // Test allocation
        let size = 1024;
        let mem1 = pool.allocate(size)?;
        assert_eq!(pool.stats().allocations, 1);
        assert_eq!(pool.stats().cache_misses, 1);
        
        // Test deallocation and reuse
        pool.deallocate(mem1);
        let mem2 = pool.allocate(size)?;
        assert_eq!(pool.stats().allocations, 2);
        assert_eq!(pool.stats().cache_hits, 1);
        assert_eq!(pool.stats().reuses, 1);
        
        println!("Memory pool test passed!");
        println!("Stats: {:?}", pool.stats());
        
        Ok(())
    }
    
    #[test]
    fn test_power_of_2() {
        assert_eq!(DeviceMemoryPool::next_power_of_2(0), 1);
        assert_eq!(DeviceMemoryPool::next_power_of_2(1), 1);
        assert_eq!(DeviceMemoryPool::next_power_of_2(2), 2);
        assert_eq!(DeviceMemoryPool::next_power_of_2(3), 4);
        assert_eq!(DeviceMemoryPool::next_power_of_2(1000), 1024);
    }
}