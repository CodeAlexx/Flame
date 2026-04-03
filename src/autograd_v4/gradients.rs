#![cfg(feature = "autograd_v4")]

use crate::{DType, Result, Tensor, TensorId};
use std::collections::HashMap;

/// Gradient container returned by the v4 autograd engine.
pub struct Gradients {
    pub(crate) map: HashMap<TensorId, Tensor>,
    names: HashMap<TensorId, String>,
}

impl Gradients {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            names: HashMap::new(),
        }
    }

    /// Insert (or accumulate) gradient for a tensor id.
    pub fn accumulate(&mut self, id: TensorId, grad: Tensor) -> Result<()> {
        let mut grad = ensure_fp32(grad)?;

        if let Some(existing) = self.map.remove(&id) {
            let merged = existing.add(&grad)?;
            let merged = ensure_fp32(merged)?;
            self.map.insert(id, merged);
        } else {
            self.map.insert(id, grad);
        }
        Ok(())
    }

    pub fn get(&self, id: &TensorId) -> Option<&Tensor> {
        self.map.get(id)
    }

    pub fn take(&mut self, id: &TensorId) -> Option<Tensor> {
        self.map.remove(id)
    }

    pub fn register_name<S: Into<String>>(&mut self, id: TensorId, name: S) {
        self.names.insert(id, name.into());
    }

    pub fn name_for(&self, id: &TensorId) -> Option<&str> {
        self.names.get(id).map(|s| s.as_str())
    }

    /// Merge another gradient container into this one.
    pub fn merge(&mut self, other: &Gradients) -> Result<()> {
        for (id, grad) in other.map.iter() {
            self.accumulate(*id, grad.clone())?;
        }
        for (id, name) in other.names.iter() {
            self.names.entry(*id).or_insert_with(|| name.clone());
        }
        Ok(())
    }

    /// Return gradients only for tensors marked with names starting with `lora_`.
    pub fn filter_lora_only(&self) -> Gradients {
        let mut filtered = Gradients::new();
        for (id, grad) in self.map.iter() {
            if let Some(name) = self.names.get(id) {
                if name.starts_with("lora_") {
                    let mut clone = ensure_clone_fp32(grad);
                    filtered.map.insert(*id, clone);
                    filtered.names.insert(*id, name.clone());
                }
            }
        }
        filtered
    }

    pub fn ids(&self) -> impl Iterator<Item = &TensorId> {
        self.map.keys()
    }
}

fn ensure_fp32(tensor: Tensor) -> Result<Tensor> {
    let mut tensor = if tensor.dtype() == DType::F32 {
        tensor
    } else {
        tensor.to_dtype(DType::F32)?
    };
    tensor.requires_grad = false;
    Ok(tensor)
}

fn ensure_clone_fp32(tensor: &Tensor) -> Tensor {
    let mut clone = tensor.clone();
    clone.requires_grad = false;
    if clone.dtype() != DType::F32 {
        if let Ok(fp32) = clone.to_dtype(DType::F32) {
            return fp32;
        }
    }
    clone
}
