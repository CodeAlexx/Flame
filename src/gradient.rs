//! Gradient storage and management
//!
//! This module provides a separate gradient storage system that avoids
//! borrow checker issues and provides a cleaner API.
//!
//! When a compact index is provided (built from the tape before backward),
//! gradients are stored in a flat `Vec<Option<Tensor>>` for O(1) access
//! with excellent cache locality. Otherwise falls back to HashMap.

use crate::autograd::policy::GradStorePolicy;
use crate::tensor::TensorId;
use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::CudaDevice;
use std::collections::HashMap;
use std::sync::Arc;

/// Maps TensorId -> compact sequential index for Vec-based gradient storage.
/// Built once from the tape before backward; all lookups are O(1) via HashMap.
pub struct CompactIndex {
    id_to_idx: HashMap<TensorId, usize>,
    capacity: usize,
}

impl CompactIndex {
    /// Build a compact index from all tensor IDs that appear in the tape.
    /// Assigns sequential indices 0..N to each unique TensorId.
    pub fn from_tensor_ids(ids: impl Iterator<Item = TensorId>) -> Self {
        let mut id_to_idx = HashMap::new();
        for id in ids {
            let next = id_to_idx.len();
            id_to_idx.entry(id).or_insert(next);
        }
        let capacity = id_to_idx.len();
        Self { id_to_idx, capacity }
    }

    #[inline]
    fn get(&self, id: TensorId) -> Option<usize> {
        self.id_to_idx.get(&id).copied()
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Gradient storage - completely separate from tensors.
///
/// Two modes:
/// - **Indexed** (fast path): When a `CompactIndex` is provided, gradients live in
///   a flat `Vec<Option<Tensor>>` — O(1) lookup, no hashing, cache-friendly.
/// - **HashMap** (fallback): Used when no index is available (e.g., debug paths).
pub struct GradientMap {
    /// Fast-path storage: Vec indexed by compact index
    vec_store: Vec<Option<Tensor>>,
    /// Compact index for fast-path (None = use HashMap fallback)
    index: Option<CompactIndex>,
    /// Fallback storage for IDs not in the compact index
    overflow: HashMap<TensorId, Tensor>,
    device: Arc<CudaDevice>,
    policy: GradStorePolicy,
}

impl GradientMap {
    /// Create a new gradient map (HashMap fallback mode)
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            vec_store: Vec::new(),
            index: None,
            overflow: HashMap::new(),
            device,
            policy: GradStorePolicy::default(),
        }
    }

    /// Create a new gradient map with a compact index for Vec-based storage.
    /// This is the fast path used during backward.
    pub fn with_index(device: Arc<CudaDevice>, index: CompactIndex) -> Self {
        let cap = index.capacity();
        Self {
            vec_store: vec![None; cap],
            index: Some(index),
            overflow: HashMap::new(),
            device,
            policy: GradStorePolicy::default(),
        }
    }

    /// Resolve a TensorId to a Vec index (fast path) or None (overflow).
    #[inline]
    fn resolve(&self, id: TensorId) -> Option<usize> {
        self.index.as_ref().and_then(|idx| idx.get(id))
    }

    /// Set gradient to ones (for loss tensor)
    pub fn set_ones(&mut self, id: TensorId, shape: Shape) -> Result<()> {
        // Enforce FP32 gradients
        let ones = Tensor::ones_dtype(shape, DType::F32, self.device.clone())?;
        self.set(id, ones);
        Ok(())
    }

    /// Set gradient directly (used by checkpoint backward)
    pub fn set(&mut self, id: TensorId, grad: Tensor) {
        if let Some(idx) = self.resolve(id) {
            self.vec_store[idx] = Some(grad);
        } else {
            self.overflow.insert(id, grad);
        }
    }

    /// Get gradient for a tensor
    pub fn get(&self, id: TensorId) -> Option<&Tensor> {
        if let Some(idx) = self.resolve(id) {
            self.vec_store[idx].as_ref()
        } else {
            self.overflow.get(&id)
        }
    }

    fn get_fp32(&self, id: TensorId) -> Result<&Tensor> {
        self.get(id).ok_or_else(|| {
            Error::InvalidOperation(format!("Gradient not found for tensor {:?}", id))
        })
    }

    pub fn iter_fp32(&self) -> Result<impl Iterator<Item = (TensorId, &Tensor)> + '_> {
        // Chain Vec entries (with their original TensorId) and overflow entries
        let vec_iter = self
            .index
            .as_ref()
            .into_iter()
            .flat_map(|idx| {
                idx.id_to_idx.iter().filter_map(|(tid, &i)| {
                    self.vec_store[i].as_ref().map(|t| (*tid, t))
                })
            });
        let overflow_iter = self.overflow.iter().map(|(tid, t)| (*tid, t));
        Ok(vec_iter.chain(overflow_iter))
    }

    pub fn get_public_grad(&self, id: TensorId) -> Result<Tensor> {
        match self.policy {
            GradStorePolicy::InternalFP32_PublicBF16 => {
                let g_fp32 = self.get_fp32(id)?;
                let grad = g_fp32.to_dtype(DType::BF16)?;
                if grad.rank() == 4 {
                    // Layout enforcement handled by boundary guards; tensors should already follow NHWC contract.
                }
                Ok(grad)
            }
        }
    }

    pub fn take_public_grads(&self) -> Result<HashMap<TensorId, Tensor>> {
        match self.policy {
            GradStorePolicy::InternalFP32_PublicBF16 => {
                let mut out = HashMap::with_capacity(self.len());
                for (tid, g_fp32) in self.iter_fp32()? {
                    let grad = g_fp32.to_dtype(DType::BF16)?;
                    if grad.rank() == 4 {
                        // Layout enforcement handled by boundary guards; tensors should already follow NHWC contract.
                    }
                    out.insert(tid, grad);
                }
                Ok(out)
            }
        }
    }

    /// Get mutable gradient for a tensor
    pub fn get_mut(&mut self, id: TensorId) -> Option<&mut Tensor> {
        if let Some(idx) = self.resolve(id) {
            self.vec_store[idx].as_mut()
        } else {
            self.overflow.get_mut(&id)
        }
    }

    /// Insert or replace gradient
    pub fn insert(&mut self, id: TensorId, grad: Tensor) -> Result<()> {
        // Enforce FP32 storage for all gradients
        let grad_f32 = if grad.dtype() != DType::F32 {
            grad.to_dtype(DType::F32)?
        } else {
            grad
        };
        self.set(id, grad_f32);
        Ok(())
    }

    /// Check if gradient exists
    pub fn contains(&self, id: TensorId) -> bool {
        if let Some(idx) = self.resolve(id) {
            self.vec_store[idx].is_some()
        } else {
            self.overflow.contains_key(&id)
        }
    }

    /// Accumulate gradient (in-place GPU addition — no temporary tensor allocation)
    pub fn accumulate(&mut self, id: TensorId, grad: Tensor) -> Result<()> {
        // Always upcast incoming gradient to FP32 before accumulation
        let grad = if grad.dtype() != DType::F32 {
            grad.to_dtype(DType::F32)?
        } else {
            grad
        };

        // Try Vec path first
        if let Some(idx) = self.resolve(id) {
            match &mut self.vec_store[idx] {
                Some(existing) => {
                    // Ensure existing buffer is FP32
                    if existing.dtype() != DType::F32 {
                        let up = existing.to_dtype(DType::F32)?;
                        *existing = up;
                    }
                    // In-place add: existing += grad (no new tensor allocation)
                    crate::ops::elt::add_inplace_same_dtype(existing, &grad)?;
                }
                slot @ None => {
                    *slot = Some(grad);
                }
            }
        } else {
            // Overflow / HashMap fallback
            match self.overflow.get_mut(&id) {
                Some(existing) => {
                    if existing.dtype() != DType::F32 {
                        let up = existing.to_dtype(DType::F32)?;
                        *existing = up;
                    }
                    crate::ops::elt::add_inplace_same_dtype(existing, &grad)?;
                }
                None => {
                    self.overflow.insert(id, grad);
                }
            }
        }
        Ok(())
    }

    /// Get or create gradient initialized to zeros
    pub fn get_or_create(&mut self, id: TensorId, shape: Shape) -> Result<&mut Tensor> {
        if let Some(idx) = self.resolve(id) {
            if self.vec_store[idx].is_none() {
                let zeros = Tensor::zeros_dtype(shape, DType::F32, self.device.clone())?;
                self.vec_store[idx] = Some(zeros);
            }
            self.vec_store[idx]
                .as_mut()
                .ok_or_else(|| crate::Error::InvalidOperation("gradient missing after insert".into()))
        } else {
            if !self.overflow.contains_key(&id) {
                let zeros = Tensor::zeros_dtype(shape, DType::F32, self.device.clone())?;
                self.overflow.insert(id, zeros);
            }
            self.overflow
                .get_mut(&id)
                .ok_or_else(|| crate::Error::InvalidOperation("gradient missing after insert".into()))
        }
    }

    /// Take gradient (remove from map)
    pub fn take(&mut self, id: TensorId) -> Option<Tensor> {
        if let Some(idx) = self.resolve(id) {
            self.vec_store[idx].take()
        } else {
            self.overflow.remove(&id)
        }
    }

    /// Clear all gradients
    pub fn clear(&mut self) {
        for slot in &mut self.vec_store {
            *slot = None;
        }
        self.overflow.clear();
    }

    /// Get number of stored gradients
    pub fn len(&self) -> usize {
        let vec_count = self.vec_store.iter().filter(|s| s.is_some()).count();
        vec_count + self.overflow.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vec_store.iter().all(|s| s.is_none()) && self.overflow.is_empty()
    }

    /// Iterate over gradients
    pub fn iter(&self) -> impl Iterator<Item = (TensorId, &Tensor)> + '_ {
        let vec_iter = self
            .index
            .as_ref()
            .into_iter()
            .flat_map(|idx| {
                idx.id_to_idx.iter().filter_map(|(tid, &i)| {
                    self.vec_store[i].as_ref().map(|t| (*tid, t))
                })
            });
        let overflow_iter = self.overflow.iter().map(|(tid, t)| (*tid, t));
        vec_iter.chain(overflow_iter)
    }
}

/// Extension trait for gradient access
pub trait TensorGradExt {
    /// Get gradient for this tensor
    fn grad<'a>(&self, gradients: &'a GradientMap) -> Option<&'a Tensor>;

    /// Get mutable gradient for this tensor
    fn grad_mut<'a>(&self, gradients: &'a mut GradientMap) -> Option<&'a mut Tensor>;

    /// Take gradient for this tensor (removes from map)
    fn take_grad(&self, gradients: &mut GradientMap) -> Option<Tensor>;

    /// Check if gradient exists
    fn has_grad(&self, gradients: &GradientMap) -> bool;
}
