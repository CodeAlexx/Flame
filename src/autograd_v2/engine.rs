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

/// Phase 3a: read the `grad_fn` slot from a tensor's autograd_meta.
///
/// Replaces the Phase 2 test-only side-table (`TENSOR_META` /
/// `_v2_set_grad_fn`). Now that `Tensor::autograd_meta` is a real
/// field, every recorded op leaves a meta on its outputs and the
/// engine reads through that.
fn grad_fn_of(t: &Tensor) -> Option<Arc<dyn GradFn>> {
    let meta = t.autograd_meta()?;
    let guard = meta.lock().ok()?;
    guard.grad_fn.clone()
}

/// Phase 3a: read the `output_nr` slot from a tensor's autograd_meta.
fn output_nr_of(t: &Tensor) -> u32 {
    match t.autograd_meta() {
        None => 0,
        Some(meta) => meta.lock().map(|m| m.output_nr).unwrap_or(0),
    }
}

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

        // Phase 3a: validate grad_outputs shapes before any DAG work.
        // Bug-fixer flagged that the existing Phase 2 code accepted
        // any shape and silently routed the mismatched grad through
        // the downstream InputBuffer, masking the caller bug. The
        // check is O(outputs) and pre-empts confusing failures later.
        for (i, (out, g_opt)) in root
            .outputs
            .iter()
            .zip(root.grad_outputs.iter())
            .enumerate()
        {
            if let Some(g) = g_opt {
                if g.shape() != out.shape() {
                    return Err(AutogradV2Error::GradOutputShapeMismatch {
                        index: i,
                        out_shape: out.shape().dims().to_vec(),
                        grad_shape: g.shape().dims().to_vec(),
                    });
                }
            }
        }

        // Thread `create_graph` from the GraphRoot into the dispatch ctx
        // so AccumulateGrad::apply / InputBuffer::add / op forward
        // wrappers can pick recording vs in-place behavior off a single
        // flag. Phase 3a default (`ctx.create_graph=false`) keeps the
        // inference-fast in-place path.
        let ctx_local = DispatchCtx {
            stream: ctx.stream.clone(),
            create_graph: root.create_graph,
        };
        let ctx = &ctx_local;

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
            nodes_by_id
                .entry(node.node_id())
                .or_insert_with(|| node.clone());
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
            // Re-fetch via the seed_nodes vec (parallel to outputs). The
            // first pass already validated each output has a grad_fn and
            // returned NoGradFnOnOutput on miss.
            let gf = nodes_by_id
                .get(&seed_nodes[i].node_id())
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

        // Seed ready queue: every output node goes in.
        //
        // BUG-FIX 2026-05-13 (bug-fixer audit): output nodes may ALSO
        // appear as descendants of *other* outputs (consider `outputs =
        // [loss, intermediate]` where `loss` flows through
        // `intermediate`'s grad_fn). In that case `intermediate`'s
        // dep_count was incremented by the BFS walk above. If we seed
        // it at dep_count=0 (ready to fire immediately), and ALSO later
        // decrement when its upstream output runs, the node fires twice
        // — once with the user's grad_outputs[i] in its buffer, once
        // with the upstream's contribution.
        //
        // Fix: the user-supplied grad_outputs IS a contribution. Account
        // for it by initializing dep_count[output] to (BFS_count + 1),
        // then decrementing on seed. Equivalently: add the user grad as
        // if it came from a "virtual root" that has just fired. The
        // implementation below adds 1 to each output's dep_count before
        // the seed phase, then uses decrement_and_maybe_enqueue to push
        // it. Outputs not referenced by other outputs go from
        // (0+1)→0 and enqueue immediately; outputs that ARE referenced
        // wait for their full contribution count.
        let mut ready: BinaryHeap<(ReadyKey, NodeId)> = BinaryHeap::new();
        let mut in_queue: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        {
            // First pass: bump dep_count by 1 for each output occurrence,
            // accounting for the user's grad_outputs as one contribution.
            for sn in &seed_nodes {
                let nid = sn.node_id();
                *dep_count.entry(nid).or_insert(0) += 1;
            }
            // Second pass: for each *unique* output node, decrement and
            // (maybe) enqueue. If a node appears N times in `outputs`,
            // we bumped its dep_count by N; we must decrement N times.
            //
            // PyTorch handles duplicates by counting once per
            // *grad_output*; same here.
            let mut seen_outputs: std::collections::HashSet<NodeId> =
                std::collections::HashSet::new();
            for sn in &seed_nodes {
                let nid = sn.node_id();
                // Always decrement (even for duplicates) since we
                // bumped per-occurrence above.
                let entry = dep_count.entry(nid).or_insert(0);
                if *entry > 0 {
                    *entry -= 1;
                }
                if *entry == 0 && seen_outputs.insert(nid) {
                    let node = nodes_by_id.get(&nid).unwrap();
                    ready.push((
                        ReadyKey {
                            topological_nr: node.topological_nr(),
                            sequence_nr: node.sequence_nr(),
                            node_id: nid.0,
                        },
                        nid,
                    ));
                    in_queue.insert(nid);
                }
            }
        }

        // For `with_inputs` mode: remember which input-tensor maps to which
        // (grad_fn node, output_nr-on-grad_fn).
        //
        // - If the grad_fn is an `AccumulateGrad`, the leaf's grad lives
        //   in `meta.grad` after the engine finishes (sunk by the
        //   accumulator). We read it back via the same `as_any`
        //   downcast path as Phase 2.
        // - Otherwise (non-leaf input), the grad we want is the value
        //   routed INTO that grad_fn's InputBuffer at slot
        //   `output_nr_of(input)`. Phase 3a captures it at apply-time
        //   in `nonleaf_grads` (keyed on `(NodeId, output_nr)`).
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
        // Set of (node_id, output_nr) we want to capture in the
        // ready-queue loop. Phase 3a: bug-fixer flagged that the Phase 2
        // engine returned None for non-leaf inputs. Capture lives at the
        // node's `apply()` entry point — the InputBuffer for that node
        // holds the grad routed into slot `output_nr`.
        let mut nonleaf_capture_keys: std::collections::HashSet<(NodeId, usize)> =
            std::collections::HashSet::new();
        if want_input_grads {
            for (nid, slot) in &input_targets {
                let node = nodes_by_id.get(nid).expect("with_inputs entry missing");
                // Only capture non-leaf inputs. Leaf grads are sunk via
                // AccumulateGrad → meta.grad and read back post-loop.
                if node.as_any().downcast_ref::<AccumulateGrad>().is_none() {
                    nonleaf_capture_keys.insert((*nid, *slot));
                }
            }
        }
        let mut nonleaf_grads: HashMap<(NodeId, usize), Tensor> = HashMap::new();

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

            // Phase 3a non-leaf capture for `with_inputs`. If this
            // node has any (NodeId, slot) keys we care about, snapshot
            // the buffered grads at those slots before they're passed
            // into `apply()`. Cloning here is cheap (single Arc bump on
            // the storage) — this path only fires when the user
            // explicitly asked for non-leaf grads via `with_inputs`.
            if want_input_grads {
                for slot in 0..input_grads.len() {
                    let key = (node_id, slot);
                    if nonleaf_capture_keys.contains(&key) {
                        if let Some(g) = &input_grads[slot] {
                            nonleaf_grads.insert(key, g.clone());
                        }
                    }
                }
            }

            // Run hooks. Fast path: pointer-compare against the empty
            // sentinel to skip the for-loops entirely in the common case.
            let hooks_ref: &Hooks = node.hooks();
            let no_hooks = std::ptr::eq(
                hooks_ref as *const Hooks,
                Hooks::empty_ref() as *const Hooks,
            );

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
                        &mut in_queue,
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
                    &mut in_queue,
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
            //   - If grad_fn is an `AccumulateGrad`, the grad was sunk into
            //     the leaf's `meta.grad`. Read it back via downcast.
            //   - Otherwise (non-leaf), Phase 3a captured the grad at
            //     apply-time into `nonleaf_grads[(node_id, output_nr)]`.
            //     If no entry is present (the node never fired with a
            //     contributing buffer), the requested grad is None.
            let mut out = Vec::with_capacity(input_targets.len());
            for (nid, slot) in input_targets.iter() {
                let node = nodes_by_id.get(nid).expect("with_inputs entry missing");
                if let Some(acc) = node.as_any().downcast_ref::<AccumulateGrad>() {
                    let meta = acc.upgrade_variable();
                    let g = meta.and_then(|m| m.lock().ok().and_then(|mg| mg.grad.clone()));
                    out.push(g);
                } else {
                    // Phase 3a: look up the captured non-leaf grad.
                    let g = nonleaf_grads.get(&(*nid, *slot)).cloned();
                    out.push(g);
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
    in_queue: &mut std::collections::HashSet<NodeId>,
) {
    let nid = child.node_id();
    let entry = dep_count.entry(nid).or_insert(0);
    if *entry > 0 {
        *entry -= 1;
    }
    if *entry == 0 {
        // Dedup against the in_queue set. BUG-FIX 2026-05-13: without
        // this, a node that is BOTH a seed output AND a descendant of
        // another seed output can be pushed twice (once at seed, once
        // when its upstream-output's apply decrements its dep_count
        // to zero). Second pop fires `apply()` on an empty buffer and
        // wastes a call — for ops with side effects (counter bumps,
        // hook fires) that is a correctness bug.
        if !in_queue.insert(nid) {
            return;
        }
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

// Phase 3a: `grad_fn_of` / `output_nr_of` now live at the top of this
// file and read from `Tensor::autograd_meta()`. The Phase 2 test-only
// TENSOR_META side-table has been removed (DELETED).
//
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
