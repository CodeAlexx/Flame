// TensorIterator port — Phases 1–3 wired (config + base + broadcast + dim_vec
// + dispatch).
//
// Reference: `flame-core/docs/TENSORITERATOR_PORT_REFERENCE.md` §2 (port
// mapping) and §10 (file layout). Migration plan:
// `/home/alex/.claude/plans/plan-this-and-fix-encapsulated-hennessy.md`.
//
// Phase 1: fields, accessors, geometry helpers.
// Phase 2: CUDA-side `OffsetCalculator<NARGS>` + `launch_gpu_kernel`
//          (`src/cuda/tensor_iterator.cuh`).
// Phase 3: DispatchStub port + real `build_unary_op`/`build_binary_op`
//          builders. No op migrations yet — those start Phase 4.

pub mod base;
pub mod broadcast;
pub mod config;
pub mod dim_vec;
pub mod dispatch;
#[cfg(feature = "cuda")]
pub mod iter_metadata;

pub use base::{FastSetupType, OperandInfo, OperandSrc, TensorIteratorBase};
pub use broadcast::{
    broadcast_pair, can_use_32bit_indexing, coalesce_dimensions, compute_shape, compute_strides,
    reorder_dimensions, OperandView,
};
pub use config::TensorIteratorConfig;
pub use dim_vec::{contiguous_element_strides, element_strides_to_bytes, DimVec, I64StrideVec, StrideVec};
pub use dispatch::{BF16ElementwiseKernel, StubEntry};
#[cfg(feature = "cuda")]
pub use iter_metadata::{IterMetadata, FLAME_MAX_DIMS as ITER_FLAME_MAX_DIMS, MAX_NARGS};
