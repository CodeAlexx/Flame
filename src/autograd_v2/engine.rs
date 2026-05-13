//! Autograd v2 — `GraphRoot` + `Engine` (Phase 2).
//!
//! The engine is a single-threaded driver that:
//!
//! 1. Walks the DAG from `GraphRoot::outputs` via `next_edges()` and
//!    builds a per-node dependency count.
//! 2. Seeds a ready queue with the output nodes (their grad-output is
//!    supplied by `GraphRoot::grad_outputs` or `ones_like` is the
//!    default).
//! 3. Pops nodes in `(topological_nr desc, sequence_nr desc)` order,
//!    dispatches `GradFn::apply`, and routes each output grad to the
//!    edge in `next_edges[output_slot]`'s `input_nr` slot of the
//!    child's `InputBuffer`. Decrements the child's dependency count;
//!    pushes to the ready queue when it reaches zero.
//! 4. Hook dispatch fires around `apply()` — tensor hooks rewrite the
//!    incoming grads, pre-backward hooks observe them, post-backward
//!    hooks observe the result. The no-hook fast path is a pointer
//!    comparison against `Hooks::empty_ref()`.
//!
//! Phase 2 is **engine-only**. No forward ops are migrated; tests use
//! synthetic test-only `GradFn` impls (see `tests/autograd_v2_engine.rs`).
//!
//! Nested `Engine::execute` (per §clause 12) is supported by treating
//! each call as a fresh local engine — `Engine` carries no state across
//! `execute` calls. A `GradFn::apply` can call `Engine::new().execute(...)`
//! freely; the outer engine resumes when the nested call returns.
//!
//! `create_graph=true` (per §clause 7 / §8): accepted on `GraphRoot`,
//! threaded into `InputBuffer::new(_, create_graph)`. Engine itself
//! does not install any no-grad guard, so Phase 3 forward ops will
//! correctly record into v2's tape during backward.

use std::any::Any;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use super::accumulator::AccumulateGrad;
use super::dispatch::DispatchCtx;
use super::error::AutogradV2Error;
use super::hooks::Hooks;
use super::input_buffer::InputBuffer;
use super::node::{GradFn, NodeId};
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// GraphRoot — backward entry-point builder
// ---------------------------------------------------------------------------

/// The set of outputs to backprop from + per-output upstream gradients +
/// optional list of inputs whose gradients should be returned by
/// `Engine::execute`.
pub struct GraphRoot {
    outputs: Vec<Tensor>,
    grad_outputs: Vec<Option<Tensor>>,
    /// `None` → standard backward (collect grads into leaf `meta.grad`).
    /// `Some(inputs)` → `torch.autograd.grad` semantics: return per-input
    /// grads in the order given, do **not** write to leaf metas.
    inputs: Option<Vec<Tensor>>,
    create_graph: bool,
    retain_graph: bool,
}

impl GraphRoot {
    /// Build a root from a list of outputs. By default each output's
    /// upstream grad is `None` — the engine will materialize a
    /// `ones_like(output)` for any `None` slot at execute time.
    pub fn new(outputs: Vec<Tensor>) -> Self {
        let n = outputs.len();
        Self {
            outputs,
            grad_outputs: vec![None; n],
            inputs: None,
            create_graph: false,
            retain_graph: false,
        }
    }

    /// Supply explicit upstream gradients (one per output). The vector
    /// length must match `outputs.len()`; any `None` entries fall back
    /// to `ones_like(output)`.
    pub fn with_grad_outputs(mut self, g: Vec<Option<Tensor>>) -> Self {
        self.grad_outputs = g;
        self
    }

    /// Request `torch.autograd.grad`-style return: instead of accumulating
    /// into leaf metas, return one `Option<Tensor>` per input in the
    /// order given.
    pub fn with_inputs(mut self, inputs: Vec<Tensor>) -> Self {
        self.inputs = Some(inputs);
        self
    }

    pub fn with_create_graph(mut self, b: bool) -> Self {
        self.create_graph = b;
        self
    }

    pub fn with_retain_graph(mut self, b: bool) -> Self {
        self.retain_graph = b;
        self
    }

    pub fn outputs(&self) -> &[Tensor] {
        &self.outputs
    }
}

// ---------------------------------------------------------------------------
// Ready-queue ordering: (topological_nr desc, sequence_nr desc, node_id desc)
// ---------------------------------------------------------------------------

/// Key for the ready queue. PyTorch orders by `topological_nr` so the
/// node deepest in the DAG (furthest from any leaf) fires first —
/// ensures all gradient contributions flow into a node before it runs.
/// `sequence_nr` and `node_id` break ties deterministically.
#[derive(Clone, Copy, Debug)]
struct ReadyKey {
    topological_nr: u64,
    sequence_nr: u64,
    node_id: u64,
}

impl PartialEq for ReadyKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for ReadyKey {}
impl PartialOrd for ReadyKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ReadyKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; order so larger topological_nr
        // pops first.
        self.topological_nr
            .cmp(&other.topological_nr)
            .then(self.sequence_nr.cmp(&other.sequence_nr))
            .then(self.node_id.cmp(&other.node_id))
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// The single-threaded backward driver. `Engine` carries no state
/// across calls; nested `execute` is just `Engine::new().execute(...)`.
pub struct Engine;

impl Engine {
    pub fn new() -> Self {
        Self
    }

    /// Drive backward from the `GraphRoot`. Returns:
    /// - `Vec<Option<Tensor>>` of length `root.inputs.len()` when
    ///   `with_inputs` was set (per-input grads in order).
    /// - empty vector otherwise (caller reads `meta.grad` off leaves).
    pub fn execute(
        &self,
        root: GraphRoot,
        ctx: &DispatchCtx,
    ) -> Result<Vec<Option<Tensor>>, AutogradV2Error> {
        if root.outputs.len() != root.grad_outputs.len() {
            return Err(AutogradV2Error::OutputGradLenMismatch {
                outputs: root.outputs.len(),
                grad_outputs: root.grad_outputs.len(),
            });
        }

        // -----------------------------------------------------------------
        // Step 1: walk the DAG, build dependency counts.
        // -----------------------------------------------------------------
        //
        // For each reachable GradFn, count the number of incoming edges
        // (the number of times some upstream node will call into it
        // via `add()`). When this hits 0, the node is ready to fire.
        //
        // Note: PyTorch's `compute_dependencies` walks the next_edges
        // graph; we do the same. The walk is duplicate-safe because we
        // track visited nodes by `NodeId`.

        let mut dep_count: HashMap<NodeId, usize> = HashMap::new();
        let mut nodes_by_id: HashMap<NodeId, Arc<dyn GradFn>> = HashMap::new();
        // Seed: collect output `grad_fn`s.
        let mut seed_nodes: Vec<Arc<dyn GradFn>> = Vec::with_capacity(root.outputs.len());
        for (i, out) in root.outputs.iter().enumerate() {
            let gf = grad_fn_of(out);
            match gf {
                None => {
                    return Err(AutogradV2Error::NoGradFnOnOutput { index: i });
                }
                Some(node) => {
                    seed_nodes.push(node.clone());
                    nodes_by_id.entry(node.node_id()).or_insert_with(|| node);
                }
            }
        }

        // BFS over next_edges to fill dep_count.
        let mut visit_stack: Vec<Arc<dyn GradFn>> = seed_nodes.clone();
        let mut seen: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        while let Some(node) = visit_stack.pop() {
            if !seen.insert(node.node_id()) {
                continue;
            }
            nodes_by_id.entry(node.node_id()).or_insert_with(|| node.clone());
            for edge in node.next_edges() {
                if let Some(child) = &edge.function {
                    *dep_count.entry(child.node_id()).or_insert(0) += 1;
                    nodes_by_id
                        .entry(child.node_id())
                        .or_insert_with(|| child.clone());
                    visit_stack.push(child.clone());
                }
            }
        }

        // -----------------------------------------------------------------
        // Step 2: prepare per-node InputBuffers and seed the ready queue
        // with the output nodes.
        // -----------------------------------------------------------------
        //
        // InputBuffers are sized by `num_inputs()` on each node. The
        // grad seed for each output goes into a *virtual* slot — we
        // build a buffer entry for each output node and write the
        // grad-output directly into slot `output_nr` of its grad_fn's
        // input buffer.
        //
        // Wait: that conflates output index vs input index. Let me be
        // careful here. PyTorch's model:
        //   - Each backward node has N inputs (= `num_inputs()`),
        //     each receives a grad from upstream.
        //   - The OUTPUTS of the forward node correspond to the INPUTS
        //     of the backward node. So the engine writes
        //     `grad_outputs[i]` (the i-th forward output's upstream
        //     grad) into the backward node's input slot `i`.
        //   - `Tensor::output_nr` on the forward output tells us which
        //     slot of the *grad_fn* this tensor came out of.
        //
        // So: for each `(out, grad_out_opt)` pair, we look at
        // `out.output_nr`, materialize a buffer for `out.grad_fn`, and
        // write the grad-output into slot `out.output_nr`.

        let mut buffers: HashMap<NodeId, InputBuffer> = HashMap::new();

        for (i, out) in root.outputs.iter().enumerate() {
            let gf = nodes_by_id
                .get(&grad_fn_of(out).unwrap().node_id())
                .expect("seeded above")
                .clone();
            let slot = output_nr_of(out) as usize;

            // Default grad: ones_like(output) when caller passed None.
            let g: Tensor = match root.grad_outputs[i].clone() {
                Some(g) => g,
                None => Tensor::ones_dtype(
                    out.shape().clone(),
                    out.dtype(),
                    ctx.device().cuda_device().clone(),
                )
                .map_err(AutogradV2Error::FlameCore)?,
            };

            let buf = buffers
                .entry(gf.node_id())
                .or_insert_with(|| InputBuffer::new(gf.num_inputs(), root.create_graph));
            buf.add(slot, g, ctx)?;
        }

        // Seed ready queue: every node referenced by an output goes in.
        // (Output nodes have dep_count == 0 because no incoming next_edge
        // walk reached them — they're roots of the backward DAG.)
        let mut ready: BinaryHeap<(ReadyKey, NodeId)> = BinaryHeap::new();
        {
            let mut already_queued: std::collections::HashSet<NodeId> =
                std::collections::HashSet::new();
            for out in &root.outputs {
                let nid = grad_fn_of(out).unwrap().node_id();
                if already_queued.insert(nid) {
                    let node = nodes_by_id.get(&nid).unwrap();
                    ready.push((
                        ReadyKey {
                            topological_nr: node.topological_nr(),
                            sequence_nr: node.sequence_nr(),
                            node_id: nid.0,
                        },
                        nid,
                    ));
                }
            }
        }

        // For `with_inputs` mode: remember which input-tensor maps to which
        // (grad_fn node, input_nr-on-grad_fn or input_nr-on-accumulator).
        // We collect by inspecting each input tensor's grad_fn and
        // output_nr (mirror of how outputs are seeded above).
        let want_input_grads = root.inputs.is_some();
        let input_targets: Vec<(NodeId, usize)> = if let Some(ref inputs) = root.inputs {
            inputs
                .iter()
                .map(|inp| {
                    let gf = grad_fn_of(inp).expect("with_inputs entry has no grad_fn");
                    (gf.node_id(), output_nr_of(inp) as usize)
                })
                .collect()
        } else {
            Vec::new()
        };

        // -----------------------------------------------------------------
        // Step 3: drive the queue.
        // -----------------------------------------------------------------

        while let Some((_, node_id)) = ready.pop() {
            let node = nodes_by_id
                .get(&node_id)
                .expect("ready node missing from nodes_by_id")
                .clone();

            // Materialize the input grads vector for this node from the
            // accumulated buffer. If no buffer entry exists (an output
            // node with no contributions — shouldn't happen because we
            // seeded above, but defensive), the input is None for every
            // slot.
            let input_grads: Vec<Option<Tensor>> = match buffers.remove(&node_id) {
                Some(buf) => buf.take(),
                None => vec![None; node.num_inputs()],
            };

            // Run hooks. Fast path: pointer-compare against the empty
            // sentinel to skip the for-loops entirely in the common case.
            let hooks_ref: &Hooks = node.hooks();
            let no_hooks =
                std::ptr::eq(hooks_ref as *const Hooks, Hooks::empty_ref() as *const Hooks);

            let processed_grads: Vec<Option<Tensor>> = if no_hooks {
                input_grads
            } else {
                // Tensor hooks: applied to every non-None grad.
                let mut out = input_grads;
                if !hooks_ref.tensor_hooks.is_empty() {
                    for slot in out.iter_mut() {
                        if let Some(g) = slot.take() {
                            let mut current = g;
                            for h in &hooks_ref.tensor_hooks {
                                if let Some(replaced) = h(&current) {
                                    current = replaced;
                                }
                            }
                            *slot = Some(current);
                        }
                    }
                }
                // Pre-backward hooks observe the input grads.
                for h in &hooks_ref.pre_backward {
                    h(&out);
                }
                out
            };

            let output_grads = node.apply(processed_grads, ctx)?;

            if output_grads.len() != node.next_edges().len() {
                // PyTorch's contract: apply returns one grad per
                // *next_edge* (one per input). Mismatch indicates a bug
                // in the op impl. AccumulateGrad legitimately returns
                // `Vec::new()` because it has `next_edges() == &[]`, so
                // this check passes when both are empty.
                return Err(AutogradV2Error::ApplyArityMismatch {
                    op: node.name(),
                    expected: node.next_edges().len(),
                    got: output_grads.len(),
                });
            }

            // Post-backward hooks observe the output grads.
            if !no_hooks {
                for h in &hooks_ref.post_backward {
                    h(&output_grads);
                }
            }

            // Route each output grad to the corresponding next_edge.
            for (output_slot, grad) in output_grads.into_iter().enumerate() {
                let edge = &node.next_edges()[output_slot];
                let child = match &edge.function {
                    None => continue, // null edge: drop the grad
                    Some(c) => c.clone(),
                };
                if grad.is_none() {
                    // No grad to forward; we still need to decrement
                    // the child's dep_count so it can fire when all
                    // contributors have reported (whether they had a
                    // grad or not).
                    decrement_and_maybe_enqueue(
                        &child,
                        &mut dep_count,
                        &nodes_by_id,
                        &mut ready,
                    );
                    continue;
                }
                let g = grad.unwrap();

                let buf = buffers
                    .entry(child.node_id())
                    .or_insert_with(|| InputBuffer::new(child.num_inputs(), root.create_graph));
                buf.add(edge.input_nr as usize, g, ctx)?;

                decrement_and_maybe_enqueue(
                    &child,
                    &mut dep_count,
                    &nodes_by_id,
                    &mut ready,
                );
            }

            // Optional: drop saved tensors when retain_graph is false.
            if !root.retain_graph {
                node.release_variables();
            }
        }

        // -----------------------------------------------------------------
        // Step 4: collect input grads if requested.
        // -----------------------------------------------------------------
        if want_input_grads {
            // For each requested input, look at its grad_fn:
            //   - If grad_fn is an AccumulateGrad, the grad was sunk into
            //     the leaf's meta.grad. Read it out.
            //   - Otherwise (non-leaf with_inputs), the grad lived in the
            //     buffer entry for that node at slot input_nr; but we've
            //     already consumed buffers. The supported case in Phase 2
            //     is leaf inputs (the common `grad(loss, [param0, param1])`
            //     usage). Non-leaf input collection is a Phase 3+ concern.
            let mut out = Vec::with_capacity(input_targets.len());
            for (nid, _slot) in input_targets.iter() {
                let node = nodes_by_id.get(nid).expect("with_inputs entry missing");
                // Downcast through GradFn::as_any. AccumulateGrad
                // overrides as_any to return &self, so this is the
                // canonical leaf-grad lookup path.
                if let Some(acc) = node.as_any().downcast_ref::<AccumulateGrad>() {
                    let meta = acc.upgrade_variable();
                    let g = meta.and_then(|m| m.lock().ok().and_then(|mg| mg.grad.clone()));
                    out.push(g);
                } else {
                    // Non-leaf inputs: Phase 3+ concern. Return None.
                    out.push(None);
                }
            }
            Ok(out)
        } else {
            Ok(Vec::new())
        }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

fn decrement_and_maybe_enqueue(
    child: &Arc<dyn GradFn>,
    dep_count: &mut HashMap<NodeId, usize>,
    nodes_by_id: &HashMap<NodeId, Arc<dyn GradFn>>,
    ready: &mut BinaryHeap<(ReadyKey, NodeId)>,
) {
    let nid = child.node_id();
    let entry = dep_count.entry(nid).or_insert(0);
    if *entry > 0 {
        *entry -= 1;
    }
    if *entry == 0 {
        // Push only if not already in queue — BinaryHeap doesn't have
        // a contains() check, but since each child is reached from
        // each parent exactly once and we only enqueue when count hits
        // 0, a node can be pushed at most once.
        //
        // Defense: nodes_by_id is the source of truth for the node;
        // we still push from it to ensure the Arc reference is alive.
        if let Some(arc) = nodes_by_id.get(&nid) {
            ready.push((
                ReadyKey {
                    topological_nr: arc.topological_nr(),
                    sequence_nr: arc.sequence_nr(),
                    node_id: nid.0,
                },
                nid,
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor → AutogradMetaV2 bridge (Phase 2 stop-gap)
// ---------------------------------------------------------------------------
//
// Phase 3 wires `Tensor` to carry an `Option<AutogradMetaRef>` field
// directly. Phase 2 doesn't yet — forward op migration is the Phase 3
// job. To make the engine testable, we expose a thread-local-shaped
// side table that Phase 2 tests use to associate a Tensor with a
// grad_fn at construction time of the test scenario.
//
// This is a TEST-ONLY backdoor; production op recording in Phase 3 will
// replace it with proper per-tensor metadata. The functions are
// `pub(crate)` so the engine module sees them, the test module exports
// them through a re-export under `#[doc(hidden)]`.

use std::sync::Mutex;

/// `(grad_fn, output_nr)` slot in the test-only side table.
type TensorMetaEntry = (Arc<dyn GradFn>, u32);

thread_local! {
    static TENSOR_META: Mutex<HashMap<crate::tensor::TensorId, TensorMetaEntry>> =
        Mutex::new(HashMap::new());
}

/// Register a (grad_fn, output_nr) pair for a given tensor id. Used by
/// Phase 2 tests to construct synthetic DAGs without yet wiring a
/// per-tensor metadata field. Phase 3 op migration removes this in
/// favor of `Tensor::autograd_meta`.
#[doc(hidden)]
pub fn _v2_set_grad_fn(tensor: &Tensor, grad_fn: Arc<dyn GradFn>, output_nr: u32) {
    TENSOR_META.with(|m| {
        m.lock()
            .expect("autograd_v2: TENSOR_META poisoned")
            .insert(tensor.id(), (grad_fn, output_nr));
    });
}

/// Clear all registered associations. Test cleanup.
#[doc(hidden)]
pub fn _v2_clear_tensor_meta() {
    TENSOR_META.with(|m| {
        if let Ok(mut g) = m.lock() {
            g.clear();
        }
    });
}

fn grad_fn_of(t: &Tensor) -> Option<Arc<dyn GradFn>> {
    TENSOR_META.with(|m| {
        m.lock()
            .ok()
            .and_then(|g| g.get(&t.id()).map(|(gf, _)| gf.clone()))
    })
}

fn output_nr_of(t: &Tensor) -> u32 {
    TENSOR_META.with(|m| {
        m.lock()
            .ok()
            .and_then(|g| g.get(&t.id()).map(|(_, nr)| *nr))
            .unwrap_or(0)
    })
}

// `Any` is in the prelude of this module via the import at the top —
// engine's `with_inputs` downcast uses `GradFn::as_any() → &dyn Any` →
// `downcast_ref::<AccumulateGrad>()`. No separate downcast trait is
// needed.

// Silence unused-import warning if downstream callers don't exercise
// `Any` in this module's tests.
#[allow(dead_code)]
fn _assert_any_in_scope(x: &dyn Any) -> &dyn Any {
    x
}
