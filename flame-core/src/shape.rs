// EXTRACTED FROM: candle-core/src/shape.rs
// REASON: Complex broadcasting logic that's already debugged and tested

use crate::error::{FlameError, Result};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn from_dims(dims: &[usize]) -> Self {
        Self { dims: dims.to_vec() }
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

    pub fn stride_contiguous(&self) -> Vec<usize> {
        let mut stride = vec![1; self.rank()];
        let mut cum_prod = 1;
        for (i, &dim) in self.dims.iter().enumerate().rev() {
            stride[i] = cum_prod;
            cum_prod *= dim;
        }
        stride
    }

    // COPIED FROM CANDLE - Critical broadcasting logic
    pub fn broadcast_shape_binary_op(&self, rhs: &Self) -> Result<Shape> {
        let lhs_dims = self.dims();
        let rhs_dims = rhs.dims();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
        
        let mut bcast_dims = vec![0; bcast_ndims];
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
                return Err(FlameError::BroadcastIncompatible {
                    lhs: self.clone(),
                    rhs: rhs.clone(),
                });
            }
        }
        
        Ok(Shape::new(bcast_dims))
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