//! Trace-time tensor expressions.
//!
//! A [`Tensor`] is a node in an immutable expression graph: constants,
//! parameters, elementwise ops, matmul, reductions, and a few view ops
//! (broadcast / transpose / squeeze). Building tensors performs no compute —
//! the JIT lowers a traced graph to [`crate::ir`] and runs it on a backend.
//!
//! Everything is f32. Shape errors at trace time are programmer errors and
//! panic, like indexing out of bounds.

mod debug;
pub mod grad;

pub use debug::{debug_print, debug_str, dedent};
pub use grad::{GradError, grad, grad_wrt};

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use crate::ops::{AssocOp, Op, UnaryOp};

/// A node in the expression graph. Cheap to clone; equality and hashing are
/// by node identity, not value.
#[derive(Clone)]
pub struct Tensor {
    inner: Arc<Inner>,
}

struct Inner {
    shape: Box<[usize]>,
    kind: TensorKind,
}

pub enum TensorKind {
    Constant { values: Box<[f32]> },
    /// Placeholder bound to a caller-supplied array at run time.
    Parameter,
    Elementwise { op: Op, args: Vec<Tensor> },
    Matmul { lhs: Tensor, rhs: Tensor },
    /// Keepdims reduction: reduced axes stay with size 1.
    Reduction { op: AssocOp, axes: Box<[usize]>, arg: Tensor },
    /// Explicit broadcast: `axes[i]` is the output axis of input axis `i`.
    /// The node's shape is the broadcast target.
    Broadcast { arg: Tensor, axes: Box<[usize]> },
    Transpose { arg: Tensor },
    /// Drop size-1 axes (NumPy-style squeeze). Turns keepdims reductions
    /// into true scalars for reverse-mode autodiff.
    Squeeze { arg: Tensor, axes: Box<[usize]> },
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}
impl Eq for Tensor {}
impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl Tensor {
    fn new(shape: impl Into<Box<[usize]>>, kind: TensorKind) -> Self {
        Tensor { inner: Arc::new(Inner { shape: shape.into(), kind }) }
    }

    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    pub fn kind(&self) -> &TensorKind {
        &self.inner.kind
    }

    /// Placeholder with a shape but no data; bound by the JIT at call time.
    pub fn parameter(shape: &[usize]) -> Self {
        Tensor::new(shape, TensorKind::Parameter)
    }

    pub fn constant(shape: &[usize], values: &[f32]) -> Self {
        assert_eq!(
            crate::ir::element_count(shape),
            values.len(),
            "constant: shape {shape:?} does not hold {} values",
            values.len()
        );
        Tensor::new(shape, TensorKind::Constant { values: values.into() })
    }

    pub fn full(shape: &[usize], value: f32) -> Self {
        Tensor::new(
            shape,
            TensorKind::Constant {
                values: vec![value; crate::ir::element_count(shape)].into(),
            },
        )
    }

    pub fn scalar(value: f32) -> Self {
        Tensor::full(&[], value)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Tensor::full(shape, 0.0)
    }

    pub fn zeros_like(&self) -> Self {
        Tensor::zeros(self.shape())
    }

    pub fn ones_like(&self) -> Self {
        Tensor::full(self.shape(), 1.0)
    }
}

//
// Elementwise ops (NumPy-style trailing broadcast between operands).
//

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Box<[usize]> {
    let rank = a.len().max(b.len());
    (0..rank)
        .map(|i| {
            let da = a.len().checked_sub(rank - i).map_or(1, |j| a[j]);
            let db = b.len().checked_sub(rank - i).map_or(1, |j| b[j]);
            assert!(
                da == db || da == 1 || db == 1,
                "incompatible elementwise shapes {a:?} vs {b:?}"
            );
            da.max(db)
        })
        .collect()
}

impl Tensor {
    fn unary(&self, op: UnaryOp) -> Self {
        Tensor::new(
            self.shape(),
            TensorKind::Elementwise { op: Op::Unary(op), args: vec![self.clone()] },
        )
    }

    fn binary(&self, op: Op, rhs: &Tensor) -> Self {
        Tensor::new(
            broadcast_shapes(self.shape(), rhs.shape()),
            TensorKind::Elementwise { op, args: vec![self.clone(), rhs.clone()] },
        )
    }

    pub fn exp(&self) -> Self {
        self.unary(UnaryOp::Exp)
    }
    pub fn log(&self) -> Self {
        self.unary(UnaryOp::Log)
    }
    pub fn relu(&self) -> Self {
        self.unary(UnaryOp::Relu)
    }
    pub fn abs(&self) -> Self {
        self.unary(UnaryOp::Abs)
    }
    pub fn sqrt(&self) -> Self {
        self.unary(UnaryOp::Sqrt)
    }
    pub fn sin(&self) -> Self {
        self.unary(UnaryOp::Sin)
    }
    pub fn cos(&self) -> Self {
        self.unary(UnaryOp::Cos)
    }
    pub fn pow(&self, rhs: &Tensor) -> Self {
        self.binary(Op::POW, rhs)
    }
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:expr) => {
        impl $trait for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                self.binary($op, &rhs)
            }
        }
    };
}
impl_binary_op!(Add, add, Op::ADD);
impl_binary_op!(Sub, sub, Op::SUB);
impl_binary_op!(Mul, mul, Op::MUL);
impl_binary_op!(Div, div, Op::DIV);

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.unary(UnaryOp::Neg)
    }
}

//
// Matmul, reductions, and view ops.
//

impl Tensor {
    /// Matrix product over the last two axes; leading axes are batch dims.
    pub fn matmul(&self, rhs: &Tensor) -> Self {
        let (a, b) = (self.shape(), rhs.shape());
        assert!(a.len() >= 2 && b.len() >= 2, "matmul requires rank >= 2");
        assert_eq!(
            a[a.len() - 1],
            b[b.len() - 2],
            "matmul inner dimension mismatch: {a:?} @ {b:?}"
        );
        let mut shape = a[..a.len() - 1].to_vec();
        shape.push(b[b.len() - 1]);
        Tensor::new(shape, TensorKind::Matmul { lhs: self.clone(), rhs: rhs.clone() })
    }

    pub fn transpose(&self) -> Self {
        let shape = self.shape();
        assert_eq!(shape.len(), 2, "transpose requires a rank-2 tensor");
        Tensor::new([shape[1], shape[0]], TensorKind::Transpose { arg: self.clone() })
    }

    /// Broadcast into `target_shape`; `axes[i]` is the output axis that input
    /// axis `i` maps to.
    pub fn broadcast_to(&self, target_shape: &[usize], axes: &[usize]) -> Self {
        assert_eq!(
            axes.len(),
            self.shape().len(),
            "broadcast_to: one output axis per input axis"
        );
        Tensor::new(
            target_shape,
            TensorKind::Broadcast { arg: self.clone(), axes: axes.into() },
        )
    }

    /// Keepdims sum over `axes`.
    pub fn sum_axes(&self, axes: &[usize]) -> Self {
        let axes_set: HashSet<usize> = axes.iter().copied().collect();
        let shape: Box<[usize]> = self
            .shape()
            .iter()
            .enumerate()
            .map(|(axis, &dim)| if axes_set.contains(&axis) { 1 } else { dim })
            .collect();
        Tensor::new(
            shape,
            TensorKind::Reduction { op: AssocOp::Add, axes: axes.into(), arg: self.clone() },
        )
    }

    /// Remove the given size-1 axes. Axes must be unique and in-bounds.
    pub fn squeeze(&self, axes: &[usize]) -> Self {
        let shape = self.shape();
        let mut seen = HashSet::new();
        for &axis in axes {
            assert!(axis < shape.len(), "squeeze axis {axis} out of range");
            assert!(seen.insert(axis), "squeeze axes must be unique, got {axes:?}");
            assert_eq!(shape[axis], 1, "cannot squeeze axis {axis} of size {}", shape[axis]);
        }
        let new_shape: Box<[usize]> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !seen.contains(i))
            .map(|(_, &d)| d)
            .collect();
        Tensor::new(new_shape, TensorKind::Squeeze { arg: self.clone(), axes: axes.into() })
    }

    /// Squeeze every size-1 axis (no-op if none).
    pub fn squeeze_all(&self) -> Self {
        let axes: Vec<usize> = self
            .shape()
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 1)
            .map(|(i, _)| i)
            .collect();
        if axes.is_empty() { self.clone() } else { self.squeeze(&axes) }
    }

    /// Mean over all elements, as a true scalar.
    pub fn mean_all(&self) -> Self {
        let count = crate::ir::element_count(self.shape()) as f32;
        let summed = if self.shape().is_empty() {
            self.clone()
        } else {
            let axes: Vec<usize> = (0..self.shape().len()).collect();
            self.sum_axes(&axes).squeeze_all()
        };
        summed / Tensor::scalar(count)
    }
}

//
// Graph traversal.
//

impl Tensor {
    pub(crate) fn args(&self) -> Vec<Tensor> {
        match self.kind() {
            TensorKind::Constant { .. } | TensorKind::Parameter => vec![],
            TensorKind::Elementwise { args, .. } => args.clone(),
            TensorKind::Matmul { lhs, rhs } => vec![lhs.clone(), rhs.clone()],
            TensorKind::Reduction { arg, .. }
            | TensorKind::Broadcast { arg, .. }
            | TensorKind::Transpose { arg }
            | TensorKind::Squeeze { arg, .. } => vec![arg.clone()],
        }
    }

    /// Post-order traversal of the subgraph reachable from `self` (inputs
    /// before outputs). The graph is an immutable DAG, so no cycle checks.
    pub(crate) fn toposort(&self) -> Vec<Tensor> {
        fn visit(node: &Tensor, order: &mut Vec<Tensor>, visited: &mut HashSet<Tensor>) {
            if !visited.insert(node.clone()) {
                return;
            }
            for arg in node.args() {
                visit(&arg, order, visited);
            }
            order.push(node.clone());
        }
        let mut order = Vec::new();
        visit(self, &mut order, &mut HashSet::new());
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toposort_visits_shared_operands_once() {
        let a = Tensor::parameter(&[]);
        let loss = a.clone() + a.clone();
        assert_eq!(loss.toposort().len(), 2);
    }

    #[test]
    fn sum_axes_is_keepdims() {
        let t = Tensor::parameter(&[8, 10]);
        assert_eq!(t.sum_axes(&[0]).shape(), &[1, 10]);
    }

    #[test]
    fn squeeze_all_after_full_sum_yields_scalar() {
        let t = Tensor::parameter(&[8, 10]);
        let reduced = t.sum_axes(&[0, 1]);
        assert_eq!(reduced.shape(), &[1, 1]);
        assert!(reduced.squeeze_all().shape().is_empty());
    }

    #[test]
    fn binary_ops_broadcast_shapes() {
        let m = Tensor::parameter(&[4, 3]);
        let s = Tensor::scalar(0.5);
        assert_eq!((m.clone() * s).shape(), &[4, 3]);
        let row = Tensor::parameter(&[3]);
        assert_eq!((m + row).shape(), &[4, 3]);
    }

    #[test]
    #[should_panic(expected = "incompatible elementwise shapes")]
    fn binary_ops_reject_incompatible_shapes() {
        let _ = Tensor::parameter(&[2]) + Tensor::parameter(&[3]);
    }

    #[test]
    fn matmul_output_shape() {
        let a = Tensor::parameter(&[2, 4]);
        let b = Tensor::parameter(&[4, 3]);
        assert_eq!(a.matmul(&b).shape(), &[2, 3]);
    }
}
