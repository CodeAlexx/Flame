#![cfg(feature = "autograd_v4")]

use crate::TensorId;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::ops::SdpaCtx;

/// Backward operation variants supported by autograd v4.
pub enum Op {
    Sdpa {
        ctx: Arc<SdpaCtx>,
        q_id: TensorId,
        k_id: TensorId,
        v_id: TensorId,
    },
}

/// Node in the backward tape.
pub struct GradNode {
    pub id: TensorId,
    pub parents: SmallVec<[TensorId; 4]>,
    pub op: Op,
}

impl GradNode {
    pub fn new(id: TensorId, parents: SmallVec<[TensorId; 4]>, op: Op) -> Arc<Self> {
        Arc::new(Self { id, parents, op })
    }
}

#[derive(Default)]
pub struct Tape {
    nodes: HashMap<TensorId, Arc<GradNode>>,
    order: Vec<TensorId>,
}

impl Tape {
    fn record(&mut self, node: Arc<GradNode>) {
        self.order.push(node.id);
        self.nodes.insert(node.id, node);
    }

    fn get(&self, id: &TensorId) -> Option<Arc<GradNode>> {
        self.nodes.get(id).cloned()
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.order.clear();
    }

    pub fn order(&self) -> &[TensorId] {
        &self.order
    }
}

thread_local! {
    static TAPE: RefCell<Tape> = RefCell::new(Tape::default());
}

pub fn record_node(node: Arc<GradNode>) {
    TAPE.with(|t| t.borrow_mut().record(node));
}

pub fn get_node(id: &TensorId) -> Option<Arc<GradNode>> {
    TAPE.with(|t| t.borrow().get(id))
}

pub fn clear() {
    TAPE.with(|t| t.borrow_mut().clear());
}

pub fn order() -> Vec<TensorId> {
    TAPE.with(|t| t.borrow().order.clone())
}

pub fn reachable_from(root: TensorId) -> HashSet<TensorId> {
    let mut visited = HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if let Some(node) = get_node(&id) {
            for parent in node.parents.iter() {
                stack.push(*parent);
            }
        }
    }
    visited
}
