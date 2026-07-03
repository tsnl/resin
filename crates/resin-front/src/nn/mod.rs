//! Neural-network helpers: [`layers`], losses, and [`Tree`]-level training ops.

pub mod layers;
pub mod loss;

pub use layers::{classifier_layers, forward, forward_probs, Layer, Linear};
pub use loss::{cross_entropy, mean, softmax};

use crate::dsl::View;
use resin_core::{Tree, TreePath, TreePathElement};

/// Graph-level SGD: builds `p - lr * g` views (same module shape as `params`).
pub fn sgd_tree<M: Tree<Leaf = View>>(params: &M, grads: &M::Map<View>, lr: f32) -> M::Map<View> {
    let lr_v: View = lr.into();
    params.zip(grads, |_path, p, g| p - &(g * &lr_v))
}

/// Host weight initialization strategy for [`initialize_tree`].
#[derive(Debug, Clone)]
pub enum InitMethod {
    /// Kaiming He uniform (fan-in) for ReLU-style nets. `bias` leaves are zero.
    He { seed: Box<[u8]> },
}

/// Initialize every leaf of `tree` to host F32 bytes per `method`.
pub fn initialize_tree<M: Tree<Leaf = View>>(tree: &M, method: InitMethod) -> M::Map<Vec<u8>> {
    match method {
        InitMethod::He { seed } => {
            let mut rng = seed_to_rng(&seed);
            tree.map(|path, view| he_uniform_leaf_bytes(path, view, &mut rng))
        }
    }
}

fn he_uniform_leaf_bytes(path: &TreePath, view: &View, rng: &mut u64) -> Vec<u8> {
    let n = view_leaf_len(view);
    let nbytes = view_leaf_nbytes(view);
    if is_bias_leaf(path) {
        return vec![0u8; nbytes];
    }
    let fan_in = *view.shape().last().unwrap_or(&1) as f32;
    let bound = (6.0 / fan_in).sqrt();
    let mut out = Vec::with_capacity(nbytes);
    for _ in 0..n {
        let u = unit_rng(rng);
        let v = (u * 2.0 - 1.0) * bound;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn seed_to_rng(seed: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in seed {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h | 1
}

fn is_bias_leaf(path: &TreePath) -> bool {
    matches!(
        path.back(),
        Some(TreePathElement::Name(s)) if s.as_ref() == "bias"
    )
}

fn view_leaf_len(view: &View) -> usize {
    view.shape().iter().map(|&d| d as usize).product()
}

fn view_leaf_nbytes(view: &View) -> usize {
    view_leaf_len(view) * view.element_type().nbytes() as usize
}

fn unit_rng(rng: &mut u64) -> f32 {
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*rng >> 11) as f32) * (1.0 / ((1u64 << 53) as f32))
}
