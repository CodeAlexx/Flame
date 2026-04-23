// TensorIterator port — Phase 1 skeleton (config + base + broadcast + dim_vec).
//
// Reference: `flame-core/docs/TENSORITERATOR_PORT_REFERENCE.md` §2 (port
// mapping) and §10 (file layout). Migration plan:
// `/home/alex/.claude/plans/plan-this-and-fix-encapsulated-hennessy.md`.
//
// Phase 1 scope: fields, accessors, geometry helpers. No CUDA, no op
// migration. `TensorIteratorBase::build_unary_op` / `build_binary_op` are
// stubs returning `Error::NotImplemented` — Phase 3 fills them in.

pub mod base;
pub mod broadcast;
pub mod config;
pub mod dim_vec;

pub use base::{FastSetupType, OperandInfo, OperandSrc, TensorIteratorBase};
pub use broadcast::{
    broadcast_pair, can_use_32bit_indexing, coalesce_dimensions, compute_shape, compute_strides,
    reorder_dimensions, OperandView,
};
pub use config::TensorIteratorConfig;
pub use dim_vec::{contiguous_element_strides, element_strides_to_bytes, DimVec, I64StrideVec, StrideVec};
