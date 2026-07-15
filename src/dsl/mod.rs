//! Trace-time tensor expressions.
//!
//! A [`Tensor`] is a node in an immutable expression graph: constants,
//! parameters, elementwise ops, matmul, reductions, remaps (gather/scatter),
//! and view ops (broadcast / permute / unfold / squeeze / index). Building tensors
//! performs no compute — the JIT lowers a traced graph to [`crate::ir`] and
//! runs it on a backend.
//!
//! Element types are f32 (default) and u32. Shape errors at trace time are
//! programmer errors and panic, like indexing out of bounds.

mod debug;
pub mod grad;
pub mod scan;
pub mod sort;

pub use debug::{debug_print, debug_str, dedent};
pub use grad::{GradError, grad, grad_wrt};
pub use scan::{
    cumprod, cumprod_exclusive, cumsum, cumsum_exclusive, scan, scan_exclusive, shift_axis,
};
pub use sort::{argsort_f32, argsort_u32, float_sort_key};

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Range, Shl, Shr, Sub};
use std::sync::Arc;

use crate::ir::{Accessor, dense_stride, element_count as shape_elems};
use crate::ops::{AssocOp, ElementType, Op, UnaryOp};

/// A node in the expression graph. Cheap to clone; equality and hashing are
/// by node identity, not value.
#[derive(Clone)]
pub struct Tensor {
    inner: Arc<Inner>,
}

struct Inner {
    shape: Box<[usize]>,
    element_type: ElementType,
    kind: TensorKind,
}

/// Host constant payload (typed).
#[derive(Clone, Debug, PartialEq)]
pub enum ConstantData {
    F32(Box<[f32]>),
    U32(Box<[u32]>),
}

/// Accumulation rule for [`Remap::ScatterRows`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScatterOp {
    /// Overwrite the target row (unordered for duplicate indices).
    Write,
    /// Accumulate into the target row.
    Add,
}

/// Static index key element: keep a size-1 axis (`Single`) or take a slice.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IndexKeyElement {
    Single(usize),
    Slice(Range<usize>),
}

/// Materializing data-movement; lowers 1:1 to [`crate::ir::Kernel::Remap`].
///
/// Dynamic row remaps use U32 index arrays. Static affine remaps use an
/// [`Accessor`] map (pad/embed/crop) — dual gather/scatter share one map.
#[derive(Clone)]
pub enum Remap {
    /// `out[i, tail…] = source[indices[i], tail…]` (OOR clamps).
    GatherRows { source: Tensor, indices: Tensor },
    /// `out[indices[i], tail…] ⊕= source[i, tail…]` into a cleared target
    /// of `target_rows` rows (OOR drops).
    ScatterRows {
        source: Tensor,
        indices: Tensor,
        op: ScatterOp,
        target_rows: usize,
    },
    /// `out[map(c)] = source[c]` into zeros of `target_shape` (pad/embed).
    /// `map` addresses the **output**; `map.shape` equals `source.shape`.
    ScatterView {
        source: Tensor,
        map: Accessor,
        target_shape: Box<[usize]>,
    },
    /// `out[c] = source[map(c)]` with OOR linear indices reading as zero.
    /// `map` addresses the **source**; `map.shape` is the output shape.
    GatherView { source: Tensor, map: Accessor },
}

pub enum TensorKind {
    Constant {
        values: ConstantData,
    },
    /// Placeholder bound to a caller-supplied array at run time.
    Parameter,
    Elementwise {
        op: Op,
        args: Vec<Tensor>,
    },
    Matmul {
        lhs: Tensor,
        rhs: Tensor,
    },
    /// Keepdims reduction: reduced axes stay with size 1.
    Reduction {
        op: AssocOp,
        axes: Box<[usize]>,
        arg: Tensor,
    },
    /// Explicit broadcast: `axes[i]` is the output axis of input axis `i`.
    /// The node's shape is the broadcast target.
    Broadcast {
        arg: Tensor,
        axes: Box<[usize]>,
    },
    /// Axis reorder: pure stride permute at lower time.
    Permute {
        arg: Tensor,
        axes: Box<[usize]>,
    },
    /// Sliding windows along one axis; trailing window dim appended.
    Unfold {
        arg: Tensor,
        axis: usize,
        size: usize,
        step: usize,
    },
    /// Drop size-1 axes (NumPy-style squeeze). Turns keepdims reductions
    /// into true scalars for reverse-mode autodiff.
    Squeeze {
        arg: Tensor,
        axes: Box<[usize]>,
    },
    /// Static slice/select; pure accessor view at lower time.
    Index {
        arg: Tensor,
        key: Box<[IndexKeyElement]>,
    },
    /// Materializing gather/scatter (see [`Remap`]).
    Remap(Remap),
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
    fn new(shape: impl Into<Box<[usize]>>, element_type: ElementType, kind: TensorKind) -> Self {
        Tensor {
            inner: Arc::new(Inner {
                shape: shape.into(),
                element_type,
                kind,
            }),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    pub fn element_type(&self) -> ElementType {
        self.inner.element_type
    }

    pub fn kind(&self) -> &TensorKind {
        &self.inner.kind
    }

    /// Placeholder with a shape but no data; bound by the JIT at call time.
    pub fn parameter(shape: &[usize]) -> Self {
        Tensor::parameter_typed(shape, ElementType::F32)
    }

    pub fn parameter_typed(shape: &[usize], element_type: ElementType) -> Self {
        Tensor::new(shape, element_type, TensorKind::Parameter)
    }

    pub fn constant(shape: &[usize], values: &[f32]) -> Self {
        assert_eq!(
            crate::ir::element_count(shape),
            values.len(),
            "constant: shape {shape:?} does not hold {} values",
            values.len()
        );
        Tensor::new(
            shape,
            ElementType::F32,
            TensorKind::Constant {
                values: ConstantData::F32(values.into()),
            },
        )
    }

    pub fn constant_f32(shape: &[usize], values: &[f32]) -> Self {
        Tensor::constant(shape, values)
    }

    pub fn constant_u32(shape: &[usize], values: &[u32]) -> Self {
        assert_eq!(
            crate::ir::element_count(shape),
            values.len(),
            "constant_u32: shape {shape:?} does not hold {} values",
            values.len()
        );
        Tensor::new(
            shape,
            ElementType::U32,
            TensorKind::Constant {
                values: ConstantData::U32(values.into()),
            },
        )
    }

    /// `[0, 1, …, n-1]` as a U32 constant.
    pub fn iota(n: usize) -> Self {
        let values: Vec<u32> = (0..n as u32).collect();
        Tensor::constant_u32(&[n], &values)
    }

    pub fn full(shape: &[usize], value: f32) -> Self {
        Tensor::new(
            shape,
            ElementType::F32,
            TensorKind::Constant {
                values: ConstantData::F32(vec![value; crate::ir::element_count(shape)].into()),
            },
        )
    }

    pub fn full_u32(shape: &[usize], value: u32) -> Self {
        Tensor::new(
            shape,
            ElementType::U32,
            TensorKind::Constant {
                values: ConstantData::U32(vec![value; crate::ir::element_count(shape)].into()),
            },
        )
    }

    pub fn scalar(value: f32) -> Self {
        Tensor::full(&[], value)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Tensor::full(shape, 0.0)
    }

    pub fn zeros_typed(shape: &[usize], element_type: ElementType) -> Self {
        match element_type {
            ElementType::F32 => Tensor::full(shape, 0.0),
            ElementType::U32 => Tensor::full_u32(shape, 0),
        }
    }

    pub fn zeros_like(&self) -> Self {
        Tensor::zeros_typed(self.shape(), self.element_type())
    }

    pub fn ones_like(&self) -> Self {
        match self.element_type() {
            ElementType::F32 => Tensor::full(self.shape(), 1.0),
            ElementType::U32 => Tensor::full_u32(self.shape(), 1),
        }
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
            self.element_type(),
            TensorKind::Elementwise {
                op: Op::Unary(op),
                args: vec![self.clone()],
            },
        )
    }

    fn binary(&self, op: Op, rhs: &Tensor) -> Self {
        assert_eq!(
            self.element_type(),
            rhs.element_type(),
            "elementwise operand element types must match"
        );
        Tensor::new(
            broadcast_shapes(self.shape(), rhs.shape()),
            self.element_type(),
            TensorKind::Elementwise {
                op,
                args: vec![self.clone(), rhs.clone()],
            },
        )
    }

    /// Unary that produces a different element type (cast / bitcast).
    fn type_change(&self, op: Op, element_type: ElementType) -> Self {
        Tensor::new(
            self.shape(),
            element_type,
            TensorKind::Elementwise {
                op,
                args: vec![self.clone()],
            },
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
    pub fn floor(&self) -> Self {
        self.unary(UnaryOp::Floor)
    }
    pub fn ceil(&self) -> Self {
        self.unary(UnaryOp::Ceil)
    }
    pub fn pow(&self, rhs: &Tensor) -> Self {
        self.binary(Op::POW, rhs)
    }
    pub fn minimum(&self, rhs: &Tensor) -> Self {
        self.binary(Op::MIN, rhs)
    }
    pub fn maximum(&self, rhs: &Tensor) -> Self {
        self.binary(Op::MAX, rhs)
    }

    pub fn cmp_eq(&self, rhs: &Tensor) -> Self {
        self.binary(Op::CMP_EQ, rhs)
    }
    pub fn cmp_ne(&self, rhs: &Tensor) -> Self {
        self.binary(Op::CMP_NE, rhs)
    }
    pub fn cmp_lt(&self, rhs: &Tensor) -> Self {
        self.binary(Op::CMP_LT, rhs)
    }
    pub fn cmp_le(&self, rhs: &Tensor) -> Self {
        self.binary(Op::CMP_LE, rhs)
    }
    pub fn cmp_gt(&self, rhs: &Tensor) -> Self {
        self.binary(Op::CMP_GT, rhs)
    }
    pub fn cmp_ge(&self, rhs: &Tensor) -> Self {
        self.binary(Op::CMP_GE, rhs)
    }

    /// Numeric value conversion; no-op when the type already matches.
    pub fn cast(&self, element_type: ElementType) -> Self {
        if self.element_type() == element_type {
            return self.clone();
        }
        self.type_change(Op::cast(element_type), element_type)
    }

    /// Reinterpret raw bits as `element_type` (same width).
    pub fn bitcast(&self, element_type: ElementType) -> Self {
        if self.element_type() == element_type {
            return self.clone();
        }
        assert_eq!(
            self.element_type().nbytes(),
            element_type.nbytes(),
            "bitcast requires equal element widths"
        );
        self.type_change(Op::bitcast(element_type), element_type)
    }

    /// `self` is a 0/1 mask: `mask * on_true + (1 - mask) * on_false`.
    ///
    /// Composed from arithmetic so it works for both f32 and u32 and needs no
    /// dedicated kernel. No gradient flows through the mask (compare→zero).
    pub fn select(&self, on_true: &Self, on_false: &Self) -> Self {
        let one = self.ones_like();
        self.clone() * on_true.clone() + (one - self.clone()) * on_false.clone()
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
impl_binary_op!(BitAnd, bitand, Op::BAND);
impl_binary_op!(BitOr, bitor, Op::BOR);
impl_binary_op!(BitXor, bitxor, Op::BXOR);
impl_binary_op!(Shl, shl, Op::SHL);
impl_binary_op!(Shr, shr, Op::SHR);

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.unary(UnaryOp::Neg)
    }
}

//
// Matmul, reductions, views, gather/scatter.
//

impl Tensor {
    /// Matrix product over the last two axes; leading axes are batch dims.
    /// Float tensors only (not integers); both operands must share an etype.
    pub fn matmul(&self, rhs: &Tensor) -> Self {
        let (a, b) = (self.shape(), rhs.shape());
        assert!(a.len() >= 2 && b.len() >= 2, "matmul requires rank >= 2");
        assert_eq!(
            a[a.len() - 1],
            b[b.len() - 2],
            "matmul inner dimension mismatch: {a:?} @ {b:?}"
        );
        assert!(
            self.element_type().is_float() && rhs.element_type().is_float(),
            "matmul requires float operands, got {:?} and {:?}",
            self.element_type(),
            rhs.element_type()
        );
        assert_eq!(
            self.element_type(),
            rhs.element_type(),
            "matmul operand element types must match"
        );
        let mut shape = a[..a.len() - 1].to_vec();
        shape.push(b[b.len() - 1]);
        Tensor::new(
            shape,
            self.element_type(),
            TensorKind::Matmul {
                lhs: self.clone(),
                rhs: rhs.clone(),
            },
        )
    }

    /// Reorder axes (NumPy / PyTorch): `out.shape[i] = self.shape[axes[i]]`.
    /// Pure view — lowers to a stride permute.
    pub fn permute(&self, axes: &[usize]) -> Self {
        let rank = self.shape().len();
        assert_eq!(
            axes.len(),
            rank,
            "permute: expected {rank} axes, got {}",
            axes.len()
        );
        let mut seen = vec![false; rank];
        for &a in axes {
            assert!(a < rank, "permute axis {a} out of range for rank {rank}");
            assert!(
                !seen[a],
                "permute axes must be a permutation, got {axes:?}"
            );
            seen[a] = true;
        }
        let shape: Box<[usize]> = axes.iter().map(|&a| self.shape()[a]).collect();
        Tensor::new(
            shape,
            self.element_type(),
            TensorKind::Permute {
                arg: self.clone(),
                axes: axes.into(),
            },
        )
    }

    /// Matrix transpose: `permute([1, 0])` on a rank-2 tensor.
    pub fn transpose(&self) -> Self {
        assert_eq!(
            self.shape().len(),
            2,
            "transpose requires a rank-2 tensor"
        );
        self.permute(&[1, 0])
    }

    /// Sliding windows along `axis`. Replaces that axis with the number of
    /// windows and appends a trailing axis of length `size`:
    ///
    /// `out[..., i, ..., j] = self[..., i * step + j, ...]`
    /// with `n_win = (dim - size) / step + 1` (remainder dropped).
    ///
    /// Pure view — composed windowing for conv is elementwise mul + reduce
    /// over the window axes, not im2col.
    pub fn unfold(&self, axis: usize, size: usize, step: usize) -> Self {
        let shape = self.shape();
        assert!(axis < shape.len(), "unfold axis {axis} out of range");
        assert!(size >= 1, "unfold size must be >= 1");
        assert!(step >= 1, "unfold step must be >= 1");
        let dim = shape[axis];
        assert!(
            size <= dim,
            "unfold size {size} exceeds axis {axis} dim {dim}"
        );
        let n_win = (dim - size) / step + 1;
        let mut out_shape = shape.to_vec();
        out_shape[axis] = n_win;
        out_shape.push(size);
        Tensor::new(
            out_shape,
            self.element_type(),
            TensorKind::Unfold {
                arg: self.clone(),
                axis,
                size,
                step,
            },
        )
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
            self.element_type(),
            TensorKind::Broadcast {
                arg: self.clone(),
                axes: axes.into(),
            },
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
            self.element_type(),
            TensorKind::Reduction {
                op: AssocOp::Add,
                axes: axes.into(),
                arg: self.clone(),
            },
        )
    }

    /// Remove the given size-1 axes. Axes must be unique and in-bounds.
    pub fn squeeze(&self, axes: &[usize]) -> Self {
        let shape = self.shape();
        let mut seen = HashSet::new();
        for &axis in axes {
            assert!(axis < shape.len(), "squeeze axis {axis} out of range");
            assert!(
                seen.insert(axis),
                "squeeze axes must be unique, got {axes:?}"
            );
            assert_eq!(
                shape[axis], 1,
                "cannot squeeze axis {axis} of size {}",
                shape[axis]
            );
        }
        let new_shape: Box<[usize]> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !seen.contains(i))
            .map(|(_, &d)| d)
            .collect();
        Tensor::new(
            new_shape,
            self.element_type(),
            TensorKind::Squeeze {
                arg: self.clone(),
                axes: axes.into(),
            },
        )
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
        if axes.is_empty() {
            self.clone()
        } else {
            self.squeeze(&axes)
        }
    }

    /// Mean over all elements, as a true scalar. Float tensors only
    /// (integer average is not defined here).
    ///
    /// The divisor is currently an f32 constant; that matches today's only
    /// float etype. When f16/f64 land, the count should be a same-etype scalar.
    pub fn mean_all(&self) -> Self {
        assert!(
            self.element_type().is_float(),
            "mean_all requires a float tensor, got {:?}",
            self.element_type()
        );
        let count = crate::ir::element_count(self.shape()) as f32;
        let summed = if self.shape().is_empty() {
            self.clone()
        } else {
            let axes: Vec<usize> = (0..self.shape().len()).collect();
            self.sum_axes(&axes).squeeze_all()
        };
        summed / Tensor::scalar(count)
    }

    /// Slice/select by static key (`Single` keeps a size-1 axis). Lowers to a
    /// pure accessor view — no kernel.
    pub fn index(&self, key: &[IndexKeyElement]) -> Self {
        Self::new_index(self.clone(), key)
    }

    pub(crate) fn new_index(arg: Tensor, key: &[IndexKeyElement]) -> Self {
        assert_eq!(
            key.len(),
            arg.shape().len(),
            "index key must cover every axis"
        );
        for (axis, element) in key.iter().enumerate() {
            let dim = arg.shape()[axis];
            match element {
                IndexKeyElement::Single(i) => {
                    assert!(
                        *i < dim,
                        "index {i} out of range for axis {axis} (dim {dim})"
                    )
                }
                IndexKeyElement::Slice(range) => assert!(
                    range.start <= range.end && range.end <= dim,
                    "slice {range:?} out of range for axis {axis} (dim {dim})"
                ),
            }
        }
        let shape = index_output_shape(key);
        Tensor::new(
            shape,
            arg.element_type(),
            TensorKind::Index {
                arg,
                key: key.into(),
            },
        )
    }

    /// Materializing data-movement node (see [`Remap`]).
    pub fn remap(kind: Remap) -> Self {
        let (shape, element_type) = match &kind {
            Remap::GatherRows { source, indices } => {
                assert_eq!(
                    indices.element_type(),
                    ElementType::U32,
                    "gather_rows indices must be U32"
                );
                assert_eq!(
                    indices.shape().len(),
                    1,
                    "gather_rows indices must be rank 1"
                );
                assert!(
                    !source.shape().is_empty(),
                    "gather_rows source must have rank >= 1"
                );
                let mut shape = source.shape().to_vec();
                shape[0] = indices.shape()[0];
                (shape.into_boxed_slice(), source.element_type())
            }
            Remap::ScatterRows {
                source,
                indices,
                target_rows,
                ..
            } => {
                assert_eq!(
                    indices.element_type(),
                    ElementType::U32,
                    "scatter_rows indices must be U32"
                );
                assert_eq!(
                    indices.shape().len(),
                    1,
                    "scatter_rows indices must be rank 1"
                );
                assert!(
                    !source.shape().is_empty(),
                    "scatter_rows source must have rank >= 1"
                );
                assert_eq!(
                    indices.shape()[0],
                    source.shape()[0],
                    "scatter_rows: one index per source row"
                );
                let mut shape = source.shape().to_vec();
                shape[0] = *target_rows;
                (shape.into_boxed_slice(), source.element_type())
            }
            Remap::ScatterView {
                source,
                map,
                target_shape,
            } => {
                assert_eq!(
                    map.shape.as_ref(),
                    source.shape(),
                    "scatter_view: map shape {:?} must match source {:?}",
                    map.shape,
                    source.shape()
                );
                assert_eq!(
                    map.rank(),
                    target_shape.len(),
                    "scatter_view: map rank must match target rank"
                );
                assert!(
                    shape_elems(target_shape) == 0 || map.max_index() < shape_elems(target_shape),
                    "scatter_view: map reaches past target {:?}",
                    target_shape
                );
                (target_shape.clone(), source.element_type())
            }
            Remap::GatherView { source, map } => {
                assert_eq!(
                    map.rank(),
                    source.shape().len(),
                    "gather_view: map rank must match source rank"
                );
                (map.shape.clone(), source.element_type())
            }
        };
        Tensor::new(shape, element_type, TensorKind::Remap(kind))
    }

    /// Embed `self` into a zero-filled `target_shape` through affine `map`
    /// (`map.shape == self.shape`; `map` addresses the target).
    pub fn scatter_view(&self, map: Accessor, target_shape: &[usize]) -> Self {
        Tensor::remap(Remap::ScatterView {
            source: self.clone(),
            map,
            target_shape: target_shape.into(),
        })
    }

    /// Embed `self` into a zero-filled `target_shape` at the region described
    /// by `key` (the inverse of [`Tensor::index`]).
    pub fn scatter_index(&self, target_shape: &[usize], key: &[IndexKeyElement]) -> Self {
        self.scatter_view(region_map(target_shape, key), target_shape)
    }

    /// Materializing gather: `out[c] = self[decode(map(c))]`, where `map`
    /// yields a dense-logical linear index into `self`'s shape (OOR → zero).
    /// `map.shape` is the output shape.
    pub fn gather_view(&self, map: Accessor) -> Self {
        Tensor::remap(Remap::GatherView {
            source: self.clone(),
            map,
        })
    }

    /// Zero-pad each axis by `(before, after)`. Implemented as scatter into a
    /// larger canvas (affine embed).
    pub fn pad(&self, padding: &[(usize, usize)]) -> Self {
        let rank = self.shape().len();
        assert_eq!(
            padding.len(),
            rank,
            "pad: one (before, after) pair per axis, got {} for rank {rank}",
            padding.len()
        );
        let mut target_shape = Vec::with_capacity(rank);
        for (axis, &(before, after)) in padding.iter().enumerate() {
            target_shape.push(before + self.shape()[axis] + after);
        }
        let target_stride = dense_stride(&target_shape);
        let mut offset = 0;
        for (axis, &(before, _)) in padding.iter().enumerate() {
            offset += before * target_stride[axis];
        }
        let map = Accessor {
            offset,
            shape: self.shape().into(),
            stride: target_stride,
        };
        self.scatter_view(map, &target_shape)
    }

    /// `out[i, tail…] = self[indices[i], tail…]` (indices: rank-1 U32).
    pub fn gather_rows(&self, indices: &Tensor) -> Self {
        Tensor::remap(Remap::GatherRows {
            source: self.clone(),
            indices: indices.clone(),
        })
    }

    /// `out[indices[i], tail…] ⊕= self[i, tail…]` into a zero-filled
    /// `[target_len, tail…]` tensor (indices: rank-1 U32, one per row of `self`).
    pub fn scatter_rows(&self, indices: &Tensor, target_len: usize, op: ScatterOp) -> Self {
        Tensor::remap(Remap::ScatterRows {
            source: self.clone(),
            indices: indices.clone(),
            op,
            target_rows: target_len,
        })
    }
}

/// Dense-region accessor for a static index key over `target_shape`.
fn region_map(target_shape: &[usize], key: &[IndexKeyElement]) -> Accessor {
    assert_eq!(
        key.len(),
        target_shape.len(),
        "region key must cover every target axis"
    );
    let stride = dense_stride(target_shape);
    let mut offset = 0;
    let mut shape = Vec::with_capacity(key.len());
    for (axis, element) in key.iter().enumerate() {
        let dim = target_shape[axis];
        let (start, len) = match element {
            IndexKeyElement::Single(i) => {
                assert!(*i < dim, "index {i} out of range for axis {axis} (dim {dim})");
                (*i, 1usize)
            }
            IndexKeyElement::Slice(range) => {
                assert!(
                    range.start <= range.end && range.end <= dim,
                    "slice {range:?} out of range for axis {axis} (dim {dim})"
                );
                (range.start, range.end - range.start)
            }
        };
        offset += start * stride[axis];
        shape.push(len);
    }
    Accessor {
        offset,
        shape: shape.into(),
        stride,
    }
}

fn index_output_shape(key: &[IndexKeyElement]) -> Box<[usize]> {
    key.iter()
        .map(|element| match element {
            IndexKeyElement::Single(_) => 1,
            IndexKeyElement::Slice(range) => range.end - range.start,
        })
        .collect()
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
            | TensorKind::Permute { arg, .. }
            | TensorKind::Unfold { arg, .. }
            | TensorKind::Squeeze { arg, .. }
            | TensorKind::Index { arg, .. } => vec![arg.clone()],
            TensorKind::Remap(remap) => match remap {
                Remap::GatherRows { source, indices }
                | Remap::ScatterRows {
                    source, indices, ..
                } => {
                    vec![source.clone(), indices.clone()]
                }
                Remap::ScatterView { source, .. } | Remap::GatherView { source, .. } => {
                    vec![source.clone()]
                }
            },
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

    #[test]
    fn permute_reorders_shape() {
        let t = Tensor::parameter(&[2, 3, 4]);
        assert_eq!(t.permute(&[2, 0, 1]).shape(), &[4, 2, 3]);
        let m = Tensor::parameter(&[2, 3]);
        assert_eq!(m.transpose().shape(), &[3, 2]);
    }

    #[test]
    fn unfold_appends_window_axis() {
        let t = Tensor::parameter(&[2, 8]);
        // (8 - 3) / 2 + 1 = 3 windows
        assert_eq!(t.unfold(1, 3, 2).shape(), &[2, 3, 3]);
    }

    #[test]
    fn u32_constant_carries_type() {
        let t = Tensor::constant_u32(&[3], &[1, 2, 3]);
        assert_eq!(t.element_type(), ElementType::U32);
        assert_eq!(t.shape(), &[3]);
    }
}
