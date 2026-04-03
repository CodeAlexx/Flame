#![cfg(feature = "autograd_v4")]

mod sdpa;

#[cfg(feature = "sdpa_debug")]
pub use sdpa::register_sdpa_hooks;
pub use sdpa::{sdpa_backward, sdpa_forward, SdpaConfig, SdpaCtx, SdpaHooks, SdpaSave, SdpaStats};

use crate::{DType, Error, Result, Tensor};
use std::collections::HashSet;
use std::sync::Arc;

use super::gradients::Gradients;
use super::graph::{self, GradNode, Op};

/// Attach a backward node to the global v4 tape.
pub fn attach_backward_node(node: Arc<GradNode>) {
    graph::record_node(node);
}

/// Clear the global v4 tape.
pub fn clear_tape() {
    graph::clear();
}

/// Execute backward pass starting from `loss` and return accumulated gradients.
pub fn backward_v4(loss: &Tensor) -> Result<Gradients> {
    if !loss.requires_grad {
        return Err(Error::InvalidOperation(
            "backward_v4 called on tensor that does not require gradients".into(),
        ));
    }

    let reachable: HashSet<_> = graph::reachable_from(loss.id);
    let mut grads = Gradients::new();
    let seed = Tensor::ones_dtype(loss.shape().clone(), DType::F32, loss.device().clone())?;
    grads.accumulate(loss.id, seed)?;

    for node_id in graph::order().into_iter().rev() {
        if !reachable.contains(&node_id) {
            continue;
        }
        let Some(node) = graph::get_node(&node_id) else {
            continue;
        };
        let Some(grad_out) = grads.take(&node_id) else {
            continue;
        };

        match node.op {
            Op::Sdpa {
                ref ctx,
                q_id,
                k_id,
                v_id,
            } => {
                let (d_q, d_k, d_v) = sdpa_backward(ctx, &grad_out)?;

                if node.parents.contains(&q_id) {
                    grads.accumulate(q_id, d_q)?;
                }
                if node.parents.contains(&k_id) {
                    grads.accumulate(k_id, d_k)?;
                }
                if node.parents.contains(&v_id) {
                    grads.accumulate(v_id, d_v)?;
                }
            }
        }
    }

    clear_tape();
    Ok(grads)
}
