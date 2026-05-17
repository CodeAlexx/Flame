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
        Self {
            id_to_idx,
            capacity,
        }
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
    /// Create a new gradient map (HashMap fallback mode) on the
    /// default v1/v3 `InternalFP32_PublicBF16` policy.
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
    /// This is the fast path used during backward. Uses the default
    /// v1/v3 `InternalFP32_PublicBF16` policy.
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

    /// Construct a GradientMap on the autograd v2
    /// `MatchInsertedDtype` policy (HashMap fallback mode).
    ///
    /// Preserves the dtype of inserted gradients end-to-end. See
    /// [`GradStorePolicy::MatchInsertedDtype`] and
    /// `docs/BF16_GRAD_DECISION.md`. The corresponding `Parameter`
    /// construct is [`crate::Parameter::new_v2`].
    pub fn new_v2(device: Arc<CudaDevice>) -> Self {
        Self {
            vec_store: Vec::new(),
            index: None,
            overflow: HashMap::new(),
            device,
            policy: GradStorePolicy::MatchInsertedDtype,
        }
    }

    /// Construct a GradientMap on the autograd v2
    /// `MatchInsertedDtype` policy, with a compact index for Vec-based
    /// storage (fast-path equivalent of [`GradientMap::with_index`]).
    pub fn with_index_v2(device: Arc<CudaDevice>, index: CompactIndex) -> Self {
        let cap = index.capacity();
        Self {
            vec_store: vec![None; cap],
            index: Some(index),
            overflow: HashMap::new(),
            device,
            policy: GradStorePolicy::MatchInsertedDtype,
        }
    }

    /// Return the active gradient-storage policy.
    pub fn policy(&self) -> GradStorePolicy {
        self.policy
    }

    /// Resolve a TensorId to a Vec index (fast path) or None (overflow).
    #[inline]
    fn resolve(&self, id: TensorId) -> Option<usize> {
        self.index.as_ref().and_then(|idx| idx.get(id))
    }

    /// Set gradient to ones (for loss tensor).
    ///
    /// Always seeds at F32, regardless of policy. The
    /// `MatchInsertedDtype` policy preserves the dtype of *inserted*
    /// gradients but uses F32 for the loss seed when the caller doesn't
    /// supply an explicit dtype — matches existing trainer behavior
    /// (loss is cast to F32 before backward via `loss.to_dtype(F32)`).
    /// v2 callers that want to seed at BF16 should use
    /// [`GradientMap::set_ones_dtype`].
    pub fn set_ones(&mut self, id: TensorId, shape: Shape) -> Result<()> {
        let ones = Tensor::ones_dtype(shape, DType::F32, self.device.clone())?;
        self.set(id, ones);
        Ok(())
    }

    /// Set gradient to ones at an explicit dtype (for loss tensor seed
    /// under `MatchInsertedDtype`). Permitted under both policies; the
    /// `InternalFP32_PublicBF16` policy converts the seed to F32
    /// internally on insert so behavior is unchanged for v3 callers.
    pub fn set_ones_dtype(&mut self, id: TensorId, shape: Shape, dtype: DType) -> Result<()> {
        let ones = Tensor::ones_dtype(shape, dtype, self.device.clone())?;
        match self.policy {
            GradStorePolicy::InternalFP32_PublicBF16 => {
                let ones_f32 = if ones.dtype() == DType::F32 {
                    ones
                } else {
                    ones.to_dtype(DType::F32)?
                };
                self.set(id, ones_f32);
            }
            GradStorePolicy::MatchInsertedDtype => {
                self.set(id, ones);
            }
        }
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
        let vec_iter = self.index.as_ref().into_iter().flat_map(|idx| {
            idx.id_to_idx
                .iter()
                .filter_map(|(tid, &i)| self.vec_store[i].as_ref().map(|t| (*tid, t)))
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
            GradStorePolicy::MatchInsertedDtype => {
                // Native-dtype path. Caller is responsible for handling
                // BF16 grads (Option A of docs/BF16_GRAD_DECISION.md).
                let g = self.get_fp32(id)?; // helper is "get-or-err", not literally fp32
                Ok(g.clone())
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
            GradStorePolicy::MatchInsertedDtype => {
                // Native-dtype path. Each grad is returned in its
                // stored dtype without conversion.
                let mut out = HashMap::with_capacity(self.len());
                for (tid, g) in self.iter_fp32()? {
                    out.insert(tid, g.clone());
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

    /// Insert or replace gradient.
    ///
    /// Under [`GradStorePolicy::InternalFP32_PublicBF16`] (default) the
    /// gradient is upcast to F32 before storage — preserving v1 / v3
    /// behavior. Under [`GradStorePolicy::MatchInsertedDtype`] the
    /// gradient is stored in its native dtype.
    pub fn insert(&mut self, id: TensorId, grad: Tensor) -> Result<()> {
        let to_store = match self.policy {
            GradStorePolicy::InternalFP32_PublicBF16 => {
                if grad.dtype() != DType::F32 {
                    grad.to_dtype(DType::F32)?
                } else {
                    grad
                }
            }
            GradStorePolicy::MatchInsertedDtype => grad,
        };
        self.set(id, to_store);
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

    /// Accumulate gradient (in-place GPU addition — no temporary tensor allocation).
    ///
    /// Policy semantics:
    /// - [`GradStorePolicy::InternalFP32_PublicBF16`]: existing entry is
    ///   raised to F32 on first re-accumulation (deferred upcast from
    ///   the v3 fast path); incoming grad cast to F32 when dtypes
    ///   differ. v1 / v3 behavior, unchanged.
    /// - [`GradStorePolicy::MatchInsertedDtype`]: accumulator stays at
    ///   the storage dtype. If the incoming grad's dtype differs from
    ///   the existing entry's, returns `Err` — Phase 4b contract is
    ///   that the producer (autograd v2 AccumulateGrad) feeds a
    ///   consistent dtype. F32 is permitted as opmath inside the
    ///   `add_inplace_same_dtype` kernel; the result is written back at
    ///   the storage dtype.
    pub fn accumulate(&mut self, id: TensorId, grad: Tensor) -> Result<()> {
        match self.policy {
            GradStorePolicy::InternalFP32_PublicBF16 => self.accumulate_v1(id, grad),
            GradStorePolicy::MatchInsertedDtype => self.accumulate_v2(id, grad),
        }
    }

    /// v1 / v3 path — preserves the legacy "first-grad stored as-is,
    /// upcast on second accumulation" behavior.
    fn accumulate_v1(&mut self, id: TensorId, grad: Tensor) -> Result<()> {
        #[inline]
        fn ensure_f32(t: &mut Tensor) -> Result<()> {
            if t.dtype() != DType::F32 {
                *t = t.to_dtype_no_grad(DType::F32)?;
            }
            Ok(())
        }

        #[inline]
        fn add_to_existing(existing: &mut Tensor, grad: Tensor) -> Result<()> {
            ensure_f32(existing)?;
            if grad.dtype() == existing.dtype() {
                crate::ops::elt::add_inplace_same_dtype(existing, &grad)?;
            } else {
                let grad_f32 = grad.to_dtype_no_grad(DType::F32)?;
                crate::ops::elt::add_inplace_same_dtype(existing, &grad_f32)?;
            }
            Ok(())
        }

        if let Some(idx) = self.resolve(id) {
            match &mut self.vec_store[idx] {
                Some(existing) => add_to_existing(existing, grad)?,
                slot @ None => {
                    *slot = Some(grad);
                }
            }
        } else {
            match self.overflow.get_mut(&id) {
                Some(existing) => add_to_existing(existing, grad)?,
                None => {
                    self.overflow.insert(id, grad);
                }
            }
        }
        Ok(())
    }

    /// v2 path — preserves the stored entry's dtype on accumulation.
    /// Errs on dtype mismatch (the producer is expected to feed a
    /// consistent dtype; see `autograd_v2::accumulator::AccumulateGrad`
    /// which enforces the same contract at `meta.grad` level).
    fn accumulate_v2(&mut self, id: TensorId, grad: Tensor) -> Result<()> {
        #[inline]
        fn add_to_existing(existing: &mut Tensor, grad: Tensor) -> Result<()> {
            if existing.dtype() != grad.dtype() {
                return Err(Error::InvalidOperation(format!(
                    "GradientMap accumulate (MatchInsertedDtype): stored dtype {:?} \
                     does not match incoming {:?}",
                    existing.dtype(),
                    grad.dtype()
                )));
            }
            crate::ops::elt::add_inplace_same_dtype(existing, &grad)?;
            Ok(())
        }

        if let Some(idx) = self.resolve(id) {
            match &mut self.vec_store[idx] {
                Some(existing) => add_to_existing(existing, grad)?,
                slot @ None => {
                    *slot = Some(grad);
                }
            }
        } else {
            match self.overflow.get_mut(&id) {
                Some(existing) => add_to_existing(existing, grad)?,
                None => {
                    self.overflow.insert(id, grad);
                }
            }
        }
        Ok(())
    }

    /// Get or create gradient initialized to F32 zeros.
    ///
    /// Used by the legacy `autograd_simple.rs` path. Both policies
    /// allocate F32 here — under `MatchInsertedDtype` callers wanting a
    /// specific dtype should use [`GradientMap::get_or_create_dtype`].
    pub fn get_or_create(&mut self, id: TensorId, shape: Shape) -> Result<&mut Tensor> {
        self.get_or_create_dtype(id, shape, DType::F32)
    }

    /// Get or create gradient initialized to zeros at an explicit
    /// dtype. v2 / `MatchInsertedDtype` callers can use this to
    /// pre-allocate a BF16 slot. v1 / `InternalFP32_PublicBF16`
    /// callers passing a non-F32 dtype will get an F32 slot (the v3
    /// invariant).
    pub fn get_or_create_dtype(
        &mut self,
        id: TensorId,
        shape: Shape,
        dtype: DType,
    ) -> Result<&mut Tensor> {
        let alloc_dtype = match self.policy {
            GradStorePolicy::InternalFP32_PublicBF16 => DType::F32,
            GradStorePolicy::MatchInsertedDtype => dtype,
        };
        if let Some(idx) = self.resolve(id) {
            if self.vec_store[idx].is_none() {
                let zeros = Tensor::zeros_dtype(shape, alloc_dtype, self.device.clone())?;
                self.vec_store[idx] = Some(zeros);
            }
            self.vec_store[idx].as_mut().ok_or_else(|| {
                crate::Error::InvalidOperation("gradient missing after insert".into())
            })
        } else {
            if !self.overflow.contains_key(&id) {
                let zeros = Tensor::zeros_dtype(shape, alloc_dtype, self.device.clone())?;
                self.overflow.insert(id, zeros);
            }
            self.overflow.get_mut(&id).ok_or_else(|| {
                crate::Error::InvalidOperation("gradient missing after insert".into())
            })
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

    /// Cast every stored gradient to `dtype`. Reassigns each slot via a
    /// fresh `Tensor::to_dtype` allocation (NOT in-place storage mutation);
    /// cheap fast-path when the stored grad already matches `dtype`.
    ///
    /// Used by the Phase 5b `backward_v2` bridge as a single post-loop pass
    /// to realize the BF16-end-to-end storage promise without forcing the
    /// v3 backward kernels to handle BF16 `output_grad`.
    pub fn cast_all_to_dtype(&mut self, dtype: DType) -> Result<()> {
        for slot in self.vec_store.iter_mut() {
            if let Some(g) = slot.as_ref() {
                if g.dtype() != dtype {
                    *slot = Some(g.to_dtype(dtype)?);
                }
            }
        }
        let ids: Vec<TensorId> = self.overflow.keys().copied().collect();
        for id in ids {
            if let Some(g) = self.overflow.get(&id) {
                if g.dtype() != dtype {
                    let casted = g.to_dtype(dtype)?;
                    self.overflow.insert(id, casted);
                }
            }
        }
        Ok(())
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vec_store.iter().all(|s| s.is_none()) && self.overflow.is_empty()
    }

    /// Drain all remaining gradients as (TensorId, Tensor) pairs.
    /// Used by checkpoint backward to return ALL accumulated gradients
    /// (including those for LoRA weights used inside the checkpoint closure).
    pub fn drain_all(&mut self) -> Result<Vec<(TensorId, Tensor)>> {
        let mut result = Vec::new();
        // Drain vec_store entries
        if let Some(idx) = &self.index {
            for (tid, &i) in &idx.id_to_idx {
                if let Some(grad) = self.vec_store[i].take() {
                    result.push((*tid, grad));
                }
            }
        }
        // Drain overflow entries
        for (tid, grad) in self.overflow.drain() {
            result.push((tid, grad));
        }
        Ok(result)
    }

    /// Iterate over gradients
    pub fn iter(&self) -> impl Iterator<Item = (TensorId, &Tensor)> + '_ {
        let vec_iter = self.index.as_ref().into_iter().flat_map(|idx| {
            idx.id_to_idx
                .iter()
                .filter_map(|(tid, &i)| self.vec_store[i].as_ref().map(|t| (*tid, t)))
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
