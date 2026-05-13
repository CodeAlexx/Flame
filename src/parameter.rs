//! Trainable parameters with mutable updates

use crate::tensor_compute::ComputeF32;
use crate::{DType, Error, Result, Shape, Tensor, TensorId};
use cudarc::driver::CudaDevice;
use std::sync::{Arc, Mutex};

/// Gradient dtype policy for a [`Parameter`].
///
/// Two paths live side-by-side during the v2 migration:
///
/// - [`GradDtypePolicy::CastToF32`] (v1 / v3 default): every grad
///   reaching [`Parameter::set_grad`] is upcast to F32. This is the
///   `InternalFP32_PublicBF16` policy that v1 / v3 trainers expect
///   (Klein, Z-Image, Chroma today). The Adam fused-step kernels for
///   `(BF16 param, F32 grad)` are the dominant production path.
///
/// - [`GradDtypePolicy::MatchParamDtype`] (autograd v2, Phase 4a +): the
///   grad is stored in its native dtype. A BF16 param with a BF16 grad
///   keeps the grad as BF16 end-to-end, halving gradient memory and
///   unlocking the `adam_fused_multi_bf16_bf16grad_kernel` /
///   `adam_fused_f32param_bf16grad_kernel` paths that are dead code
///   today. See `docs/BF16_GRAD_DECISION.md`.
///
/// The v3 path is unchanged: [`Parameter::new`] still gives you a
/// `CastToF32` parameter and existing trainers continue to work.
/// Autograd-v2 callers construct via [`Parameter::new_v2`] (Phase 4a)
/// or set the policy explicitly via
/// [`Parameter::set_grad_dtype_policy`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradDtypePolicy {
    /// v1 / v3 behavior: cast every incoming grad to F32.
    CastToF32,
    /// Autograd v2 / Phase 4 BF16-grad path: preserve the grad's native
    /// dtype on `set_grad`. Adam fused kernels for `(BF16 param, BF16
    /// grad)` and `(F32 param, BF16 grad)` are activated through this
    /// policy (see `src/adam.rs`).
    MatchParamDtype,
}

impl Default for GradDtypePolicy {
    fn default() -> Self {
        Self::CastToF32
    }
}

/// A trainable parameter that supports in-place updates
#[derive(Clone)]
pub struct Parameter {
    /// The parameter data (wrapped in mutex for mutability)
    data: Arc<Mutex<Tensor>>,
    /// Current gradient (if any)
    grad: Arc<Mutex<Option<Tensor>>>,
    /// Whether this parameter requires gradients
    requires_grad: bool,
    /// Unique ID for this parameter
    id: TensorId,
    /// Gradient dtype handling — see [`GradDtypePolicy`].
    grad_dtype_policy: GradDtypePolicy,
}

impl Parameter {
    /// Create a new parameter from a tensor.
    ///
    /// Uses the v1 / v3 default [`GradDtypePolicy::CastToF32`].
    /// Autograd-v2 callers wanting BF16-grad preservation should call
    /// [`Parameter::new_v2`] instead.
    pub fn new(tensor: Tensor) -> Self {
        let requires_grad = tensor.requires_grad;
        let id = tensor.id;
        Self {
            data: Arc::new(Mutex::new(tensor)),
            grad: Arc::new(Mutex::new(None)),
            requires_grad,
            id,
            grad_dtype_policy: GradDtypePolicy::CastToF32,
        }
    }

    /// Create a new autograd-v2 parameter from a tensor.
    ///
    /// Same as [`Parameter::new`] but with
    /// [`GradDtypePolicy::MatchParamDtype`] — gradients are stored in
    /// their native dtype (BF16 param → BF16 grad, F32 param → F32
    /// grad). This unlocks the `adam_fused_multi_bf16_bf16grad_kernel`
    /// and `adam_fused_f32param_bf16grad_kernel` paths in `src/adam.rs`,
    /// and is the policy required by `docs/BF16_GRAD_DECISION.md`
    /// Option A.
    ///
    /// `apply_update` likewise preserves param dtype (no F32 upcast of
    /// the data tensor). The fused Adam kernels write back the param
    /// in-place at its native dtype.
    pub fn new_v2(tensor: Tensor) -> Self {
        let mut p = Self::new(tensor);
        p.grad_dtype_policy = GradDtypePolicy::MatchParamDtype;
        p
    }

    /// Create a parameter with specific initialization
    pub fn randn(shape: Shape, mean: f32, std: f32, device: Arc<CudaDevice>) -> Result<Self> {
        let tensor = Tensor::randn(shape, mean, std, device)?.requires_grad_(true);
        Ok(Self::new(tensor))
    }

    /// Create a parameter initialized with zeros
    pub fn zeros(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let tensor = Tensor::zeros(shape, device)?.requires_grad_(true);
        Ok(Self::new(tensor))
    }

    /// Return the current grad-dtype policy.
    pub fn grad_dtype_policy(&self) -> GradDtypePolicy {
        self.grad_dtype_policy
    }

    /// Set the grad-dtype policy. Use this only at construction time —
    /// switching policies mid-training would leave previously cast
    /// grads / Adam state inconsistent.
    pub fn set_grad_dtype_policy(&mut self, policy: GradDtypePolicy) {
        self.grad_dtype_policy = policy;
    }

    /// Get the parameter ID
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Get a clone of the current tensor value
    pub fn tensor(&self) -> Result<Tensor> {
        Ok(self
            .data
            .lock()
            .map_err(|_| Error::Training("parameter data mutex poisoned".into()))?
            .clone())
    }

    /// Get a reference to the tensor (as_tensor compatibility)
    pub fn as_tensor(&self) -> Result<Tensor> {
        self.tensor()
    }

    /// Return the logical dtype of the parameter without cloning device memory.
    pub fn dtype(&self) -> Result<DType> {
        let data_lock = self
            .data
            .lock()
            .map_err(|_| Error::Training("parameter data mutex poisoned".into()))?;
        Ok(data_lock.dtype())
    }

    /// Set the parameter data directly.
    ///
    /// Pinning self.id across in-place updates: optimizers like Adafactor /
    /// Lion / Prodigy / StableAdamW / RAdamScheduleFree compute updates as
    /// new Tensors (via cast/detach) and write them back with set_data. Each
    /// such Tensor has a fresh TensorId. Without this overwrite step,
    /// `Parameter.id` (cached at `new()`) drifts out of sync with the inner
    /// storage's id — every downstream `param.id()` query returns a stale
    /// value that never matches the next forward's recorded op outputs, so
    /// backward stores grads under the new (post-set_data) id and
    /// `grads.get(param.id())` returns None forever. The optimizer then
    /// silently no-ops every subsequent step.
    ///
    /// Fix: rewrite the incoming tensor's id to match `self.id` BEFORE
    /// storing. Parameter.id stays fixed; the next forward records ops with
    /// this same id; grads land at this id; AdamW-style state (m, v) keyed
    /// by param.id stays valid across steps. Verified end-to-end with
    /// Adafactor / RAdamScheduleFree on z-image and u1 trainers.
    pub fn set_data(&self, tensor: Tensor) -> Result<()> {
        let mut data_lock = self
            .data
            .lock()
            .map_err(|_| Error::Training("parameter data mutex poisoned".into()))?;
        let mut t = tensor;
        // Pin self.id across in-place updates (see top of impl block for why).
        t.id = self.id;
        // ALSO pin requires_grad. Optimizers like Adafactor / Lion / Prodigy
        // / StableAdamW / RAdamScheduleFree write back via
        // `set_data(cast_back.detach()?)` — `detach()` strips requires_grad
        // before reaching us. If the stored tensor ends up with
        // requires_grad=false, the next forward's `param.tensor().to_dtype(...)`
        // skips the autograd Cast record (see `tensor.rs::to_dtype`), so
        // backward can't reach this parameter at all. Without this line, every
        // non-AdamW optimizer silently no-ops from step 1 onward.
        t.requires_grad = self.requires_grad;
        *data_lock = t;
        Ok(())
    }

    /// Set gradient for this parameter.
    ///
    /// Behavior depends on [`GradDtypePolicy`]:
    ///
    /// - [`GradDtypePolicy::CastToF32`] (v1 / v3 default): the incoming
    ///   grad is cast to F32 if needed. Existing trainers rely on this.
    /// - [`GradDtypePolicy::MatchParamDtype`] (v2 / Phase 4a): the grad
    ///   is stored in its native dtype. Callers using
    ///   [`Parameter::new_v2`] get this path automatically.
    pub fn set_grad(&self, grad: Tensor) -> Result<()> {
        let mut grad_lock = self
            .grad
            .lock()
            .map_err(|_| Error::Training("parameter grad mutex poisoned".into()))?;
        let grad = match self.grad_dtype_policy {
            GradDtypePolicy::CastToF32 => {
                if grad.dtype() == DType::F32 {
                    grad
                } else {
                    grad.to_dtype(DType::F32)?
                }
            }
            GradDtypePolicy::MatchParamDtype => grad,
        };
        *grad_lock = Some(grad);
        Ok(())
    }

    /// Get current gradient (if any) — always returns a clone.
    ///
    /// Under the v1 / v3 [`GradDtypePolicy::CastToF32`] policy this is
    /// always F32; under [`GradDtypePolicy::MatchParamDtype`] it is
    /// whatever dtype was passed to `set_grad` (typically BF16 for
    /// BF16 params, F32 for F32 params).
    ///
    /// V2 trainers that want to avoid the cast should prefer
    /// [`Parameter::grad_bf16_or_f32`].
    pub fn grad(&self) -> Option<Tensor> {
        if let Ok(grad_lock) = self.grad.lock() {
            grad_lock.as_ref().map(|g| g.clone())
        } else {
            None
        }
    }

    /// Autograd v2 accessor: return the gradient in its native storage
    /// dtype, without casting.
    ///
    /// Functionally identical to [`Parameter::grad`] but the name
    /// signals the v2 contract: the caller is prepared to handle a BF16
    /// grad (Option A of `docs/BF16_GRAD_DECISION.md`). The Adam BF16-
    /// grad classifier arms in `src/adam.rs` look at the grad's actual
    /// dtype, not the policy enum.
    pub fn grad_bf16_or_f32(&self) -> Option<Tensor> {
        self.grad()
    }

    /// Clear gradient
    pub fn zero_grad(&self) {
        if let Ok(mut grad_lock) = self.grad.lock() {
            *grad_lock = None;
        }
    }

    /// Update parameter in-place with gradient descent
    /// param = param - learning_rate * grad
    pub fn update(&self, learning_rate: f32) -> Result<()> {
        let grad_lock = self
            .grad
            .lock()
            .map_err(|_| Error::Training("parameter grad mutex poisoned".into()))?;
        if let Some(grad) = grad_lock.as_ref() {
            let mut data_lock = self
                .data
                .lock()
                .map_err(|_| Error::Training("parameter data mutex poisoned".into()))?;

            // Compute update: param = param - lr * grad
            let update = grad.mul_scalar(learning_rate)?;
            let new_data = data_lock.sub(&update)?;

            // Replace data
            *data_lock = new_data;
        }
        Ok(())
    }

    /// Get mutable access to the raw parameter tensor for in-place optimizers.
    ///
    /// The closure receives `&mut Tensor` while the data mutex is held.
    /// Use this for fused optimizer kernels that modify param in-place.
    pub fn with_data_mut<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Tensor) -> Result<R>,
    {
        let mut data_lock = self
            .data
            .lock()
            .map_err(|_| Error::Training("parameter data mutex poisoned".into()))?;
        f(&mut data_lock)
    }

    /// Apply an arbitrary update tensor.
    ///
    /// Behavior depends on [`GradDtypePolicy`]:
    ///
    /// - [`GradDtypePolicy::CastToF32`] (v1 / v3 default): compute in
    ///   F32 (current behavior — unchanged).
    /// - [`GradDtypePolicy::MatchParamDtype`] (v2): compute in the
    ///   param's native dtype. F32 is permitted only as `opmath_t`
    ///   inside individual ops; the resulting tensor stays at the
    ///   param's dtype. Matches the docs/BF16_GRAD_DECISION.md Option
    ///   A contract.
    pub fn apply_update(&self, update: &Tensor) -> Result<()> {
        let mut data_lock = self
            .data
            .lock()
            .map_err(|_| Error::Training("parameter data mutex poisoned".into()))?;

        match self.grad_dtype_policy {
            GradDtypePolicy::CastToF32 => {
                let compute = ComputeF32::for_input(&data_lock)?;
                let update_f32 = if update.dtype() == DType::F32 {
                    update.clone_result()?
                } else {
                    update.to_dtype(DType::F32)?
                };
                let new_f32 = compute.tensor().sub(&update_f32)?;
                *data_lock = compute.into_output(new_f32)?;
            }
            GradDtypePolicy::MatchParamDtype => {
                // Native-dtype path. Bring the update to the param's
                // dtype (typical case: both are already the same dtype
                // and `to_dtype` short-circuits). Then a straight
                // `data - update` at the param's dtype. F32 may still
                // appear as opmath inside `sub`'s kernel.
                let target_dtype = data_lock.dtype();
                let update_matched = if update.dtype() == target_dtype {
                    update.clone_result()?
                } else {
                    update.to_dtype(target_dtype)?
                };
                let new_t = data_lock.sub(&update_matched)?;
                *data_lock = new_t;
            }
        }
        Ok(())
    }

    /// Get shape of the parameter
    pub fn shape(&self) -> Shape {
        if let Ok(lock) = self.data.lock() {
            lock.shape.clone()
        } else {
            Shape::from_dims(&[])
        }
    }

    /// Check if parameter requires grad
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set requires_grad flag
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        if let Ok(mut data_lock) = self.data.lock() {
            data_lock.requires_grad = requires_grad;
        }
    }
}

/// Collection of parameters for a module
pub struct ParameterDict {
    params: std::collections::HashMap<String, Parameter>,
}

impl ParameterDict {
    pub fn new() -> Self {
        Self {
            params: std::collections::HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, param: Parameter) {
        self.params.insert(name, param);
    }

    pub fn get(&self, name: &str) -> Option<&Parameter> {
        self.params.get(name)
    }

    pub fn parameters(&self) -> Vec<&Parameter> {
        self.params.values().collect()
    }

    pub fn named_parameters(&self) -> impl Iterator<Item = (&String, &Parameter)> {
        self.params.iter()
    }
}

impl Default for ParameterDict {
    fn default() -> Self {
        Self::new()
    }
}
