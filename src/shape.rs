// Origin: adapted from prior art in the Rust DL ecosystem
// Reason: Complex broadcasting logic that's already debugged and tested

use crate::error::{Error, Result};
use smallvec::{smallvec, SmallVec};

/// Inline-storage backing for shape dims. Deep-learning tensors are always
/// 0-6 dimensions, so storing them inline avoids a heap allocation on every
/// `Shape::clone()` (and tensor ops produce thousands of shape clones per
/// training step via `Op::Add { lhs_shape, rhs_shape, .. }` and friends).
pub type ShapeDims = SmallVec<[usize; 6]>;

/// Strides use the same inline-6 SmallVec storage as dims. Returned by
/// `Shape::strides` / `Tensor::strides` / `Tensor::stride` so that every
/// kernel launcher that reads a tensor's strides avoids a heap allocation
/// on every call. Prior to 2026-04-22 these methods returned `Vec<usize>`,
/// costing ~16 ns per call in the default allocator (measured via
/// `benches/strides_alloc.rs`).
pub type Strides = SmallVec<[usize; 6]>;

/// Dimension helper for tensor indexing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D {
    Minus(i32),
}

impl D {
    pub fn to_index(&self, shape: &Shape, _index: usize) -> Result<usize> {
        match self {
            D::Minus(offset) => {
                let rank = shape.rank() as i32;
                let idx = rank + offset;
                if idx < 0 || idx >= rank {
                    return Err(Error::InvalidIndex(format!(
                        "D::Minus({}) out of range for shape {:?}",
                        offset, shape
                    )));
                }
                Ok(idx as usize)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: ShapeDims,
}

impl Shape {
    /// Construct a shape from an owned `Vec<usize>`. Copied into inline
    /// storage when it fits (usually true: rank ≤ 6).
    pub fn new(dims: Vec<usize>) -> Self {
        Self {
            dims: SmallVec::from_vec(dims),
        }
    }

    /// Construct a shape by copying a slice of dims into inline storage.
    pub fn from_dims(dims: &[usize]) -> Self {
        Self {
            dims: SmallVec::from_slice(dims),
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn stride_contiguous(&self) -> Strides {
        let rank = self.rank();
        let mut stride: Strides = smallvec![1; rank];
        let mut cum_prod = 1;
        for (i, &dim) in self.dims.iter().enumerate().rev() {
            stride[i] = cum_prod;
            cum_prod *= dim;
        }
        stride
    }

    /// Get strides for this shape (alias for stride_contiguous)
    pub fn strides(&self) -> Strides {
        self.stride_contiguous()
    }

    // COPIED FROM CANDLE - Critical broadcasting logic
    pub fn broadcast_shape_binary_op(&self, rhs: &Self) -> Result<Shape> {
        // Fast-path: same shape — by far the most common case in training
        // hot loops (residual adds, GEGLU muls, norms, etc.). Skip the
        // dim-matching loop and smallvec allocation entirely.
        if self.dims == rhs.dims {
            return Ok(self.clone());
        }

        let lhs_dims = self.dims();
        let rhs_dims = rhs.dims();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);

        let mut bcast_dims: ShapeDims = smallvec![0; bcast_ndims];
        for i in 0..bcast_ndims {
            let lhs_i = if i < lhs_ndims {
                lhs_dims[lhs_ndims - i - 1]
            } else {
                1
            };
            let rhs_i = if i < rhs_ndims {
                rhs_dims[rhs_ndims - i - 1]
            } else {
                1
            };

            if lhs_i == rhs_i {
                bcast_dims[bcast_ndims - i - 1] = lhs_i;
            } else if lhs_i == 1 {
                bcast_dims[bcast_ndims - i - 1] = rhs_i;
            } else if rhs_i == 1 {
                bcast_dims[bcast_ndims - i - 1] = lhs_i;
            } else {
                return Err(Error::BroadcastIncompatible {
                    lhs: self.clone(),
                    rhs: rhs.clone(),
                });
            }
        }

        Ok(Shape { dims: bcast_dims })
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}
