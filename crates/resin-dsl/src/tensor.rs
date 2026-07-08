use super::*;
use arrayvec::ArrayVec;
use paste::paste;
use std::{
    collections::HashSet,
    hash::Hash,
    ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Range, Rem, Shl, Shr, Sub},
    sync::Arc,
};

//
// Tensor
//

#[derive(Clone)]
pub struct Tensor {
    inner: Arc<TensorInner>,
}
impl PartialEq for Tensor {
    // Tensor uses pointer equality to determine if two tensors are equal.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}
impl Eq for Tensor {}
impl Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

pub struct TensorInner {
    element_type: ElementType,
    shape: Box<[usize]>,
    kind: TensorKind,
}

pub enum TensorKind {
    Constant {
        bytes: Box<[u8]>,
    },
    Parameter,
    Elementwise {
        operator: ElementOperator,
        args: ElementwiseArgs,
    },
    Reduction {
        operator: ElementOperator,
        axes: Box<[usize]>,
        arg: Tensor,
    },
    Index {
        arg: Tensor,
        key: Box<[IndexKeyElement]>,
    },
    /// Row gather along axis 0: `out[i, tail…] = source[indices[i], tail…]`.
    /// Out-of-range indices clamp to the last row.
    Gather {
        source: Tensor,
        indices: Tensor,
    },
    /// Row scatter along axis 0 into a zero-initialized target:
    /// `out[indices[i], tail…] ⊕= source[i, tail…]`. Out-of-range indices are
    /// dropped. With [`ScatterOp::Write`], duplicate indices are unordered
    /// (last-writer-wins nondeterministically on GPU).
    Scatter {
        source: Tensor,
        indices: Tensor,
        op: ScatterOp,
    },
    Broadcast {
        arg: Tensor,
        target_shape: Box<[usize]>,
        axes: Box<[usize]>,
    },
    ScatterIndex {
        source: Tensor,
        key: Box<[IndexKeyElement]>,
        target_shape: Box<[usize]>,
    },
    Transpose {
        arg: Tensor,
    },
    /// Drop size-1 axes (NumPy-style squeeze). Required to turn keepdims
    /// reductions into true scalars for reverse-mode autodiff.
    Squeeze {
        arg: Tensor,
        axes: Box<[usize]>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    F32,
    U32,
}
impl ElementType {
    pub fn nbytes(&self) -> usize {
        match self {
            ElementType::F32 | ElementType::U32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ElementOperator {
    // Unary
    Neg,
    Log,
    Exp,
    Relu,
    Abs,
    Sqrt,
    Floor,
    Ceil,
    /// Numeric value conversion to the node's element type.
    Cast,
    /// Bit reinterpretation as the node's element type.
    Bitcast,

    // Binary
    Pow,
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Min,
    Max,
    Matmul,

    // Comparisons: 0/1 mask in the operand element type.
    CmpEq,
    CmpNe,
    CmpLt,
    CmpLe,
    CmpGt,
    CmpGe,

    // Bitwise (U32 only).
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Accumulation rule for [`TensorKind::Scatter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScatterOp {
    /// Overwrite the target row (unordered for duplicate indices).
    Write,
    /// Accumulate into the target row.
    Add,
}

type ElementwiseArgs = ArrayVec<Tensor, MAX_ELEMENTWISE_ARGS>;
const MAX_ELEMENTWISE_ARGS: usize = 2;

#[derive(Clone)]
pub enum IndexKeyElement {
    Single(usize),
    Slice(Range<usize>),
}

//
// Methods
//

impl From<Arc<TensorInner>> for Tensor {
    fn from(inner: Arc<TensorInner>) -> Self {
        Tensor { inner }
    }
}
impl From<TensorInner> for Tensor {
    fn from(inner: TensorInner) -> Self {
        Tensor::from(Arc::new(inner))
    }
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }
    pub fn kind(&self) -> &TensorKind {
        &self.inner.kind
    }
    pub fn element_type(&self) -> ElementType {
        self.inner.element_type
    }
}

impl Tensor {
    pub fn zeros(shape: &[usize], element_type: ElementType) -> Self {
        let nbytes = shape.iter().product::<usize>() * element_type.nbytes();
        Tensor::from(TensorInner {
            element_type,
            shape: shape.into(),
            kind: TensorKind::Constant {
                bytes: vec![0u8; nbytes].into_boxed_slice(),
            },
        })
    }
    pub fn zeros_like(&self) -> Self {
        Tensor::zeros(self.shape(), self.element_type())
    }
    pub fn ones_like(&self) -> Self {
        Self::full(self.shape(), 1.0, self.element_type())
    }
    pub(crate) fn full_like(&self, value: f32) -> Self {
        Self::full(self.shape(), value, self.element_type())
    }
    pub fn constant_f32(shape: &[usize], values: &[f32]) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            values.len(),
            "constant_f32: shape/values length mismatch"
        );
        let mut data = Vec::with_capacity(values.len() * ElementType::F32.nbytes());
        for &value in values {
            data.extend_from_slice(&f32::to_le_bytes(value));
        }
        Tensor::from(TensorInner {
            element_type: ElementType::F32,
            shape: shape.into(),
            kind: TensorKind::Constant {
                bytes: data.into_boxed_slice(),
            },
        })
    }

    pub fn constant_u32(shape: &[usize], values: &[u32]) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            values.len(),
            "constant_u32: shape/values length mismatch"
        );
        let mut data = Vec::with_capacity(values.len() * ElementType::U32.nbytes());
        for &value in values {
            data.extend_from_slice(&u32::to_le_bytes(value));
        }
        Tensor::from(TensorInner {
            element_type: ElementType::U32,
            shape: shape.into(),
            kind: TensorKind::Constant {
                bytes: data.into_boxed_slice(),
            },
        })
    }

    /// `[0, 1, …, n-1]` as a U32 constant.
    pub fn iota(n: usize) -> Self {
        let values: Vec<u32> = (0..n as u32).collect();
        Tensor::constant_u32(&[n], &values)
    }

    pub fn full(shape: &[usize], value: f32, element_type: ElementType) -> Self {
        let count = shape.iter().product::<usize>();
        let bytes = match element_type {
            ElementType::F32 => f32::to_le_bytes(value),
            ElementType::U32 => u32::to_le_bytes(value as u32),
        };
        let mut data = Vec::with_capacity(count * element_type.nbytes());
        for _ in 0..count {
            data.extend_from_slice(&bytes);
        }
        Tensor::from(TensorInner {
            element_type,
            shape: shape.into(),
            kind: TensorKind::Constant {
                bytes: data.into_boxed_slice(),
            },
        })
    }

    pub fn full_u32(shape: &[usize], value: u32) -> Self {
        let count = shape.iter().product::<usize>();
        let mut data = Vec::with_capacity(count * ElementType::U32.nbytes());
        for _ in 0..count {
            data.extend_from_slice(&u32::to_le_bytes(value));
        }
        Tensor::from(TensorInner {
            element_type: ElementType::U32,
            shape: shape.into(),
            kind: TensorKind::Constant {
                bytes: data.into_boxed_slice(),
            },
        })
    }
}

impl Tensor {
    #[inline]
    fn new_elementwise<const N: usize>(
        &self,
        operator: ElementOperator,
        extra_args: [Tensor; N],
    ) -> Self {
        let mut args = arrvec![self.clone()];
        args.extend(extra_args);
        for arg in args.iter().skip(1) {
            assert_eq!(
                arg.element_type(),
                self.element_type(),
                "elementwise {operator:?}: operand element types must match"
            );
        }

        let shape = if operator == ElementOperator::Matmul {
            matmul_output_shape(self.shape(), args[1].shape())
        } else {
            elementwise_output_shape(&args, operator)
        };

        Tensor::from(TensorInner {
            element_type: self.inner.element_type,
            shape,
            kind: TensorKind::Elementwise { operator, args },
        })
    }

    /// Unary elementwise node whose element type differs from its operand's.
    #[inline]
    fn new_type_change(&self, operator: ElementOperator, element_type: ElementType) -> Self {
        Tensor::from(TensorInner {
            element_type,
            shape: self.inner.shape.clone(),
            kind: TensorKind::Elementwise {
                operator,
                args: arrvec![self.clone()],
            },
        })
    }
}

/// NumPy trailing-align broadcast join of all operand shapes. Lowering
/// expands each operand's accessor to this shape (pitch-0 views, no copies).
fn elementwise_output_shape(args: &[Tensor], operator: ElementOperator) -> Box<[usize]> {
    let rank = args
        .iter()
        .map(|arg| arg.shape().len())
        .max()
        .unwrap_or(0);
    let mut shape = vec![1usize; rank];
    for arg in args {
        let arg_shape = arg.shape();
        let pad = rank - arg_shape.len();
        for (i, &dim) in arg_shape.iter().enumerate() {
            let idx = pad + i;
            if shape[idx] == 1 {
                shape[idx] = dim;
            } else {
                assert!(
                    dim == 1 || dim == shape[idx],
                    "elementwise {operator:?}: incompatible operand shapes \
                     ({arg_shape:?} vs joined {shape:?})",
                );
            }
        }
    }
    shape.into()
}

fn matmul_output_shape(lhs: &[usize], rhs: &[usize]) -> Box<[usize]> {
    assert!(lhs.len() >= 2 && rhs.len() >= 2, "matmul requires rank >= 2");
    assert_eq!(
        lhs[lhs.len() - 1],
        rhs[rhs.len() - 2],
        "matmul inner dimension mismatch: {} != {}",
        lhs[lhs.len() - 1],
        rhs[rhs.len() - 2],
    );
    let mut out = lhs[..lhs.len() - 1].to_vec();
    out.push(rhs[rhs.len() - 1]);
    out.into()
}

macro_rules! impl_unary_elementwise {
    ($trait:ty, $operator:expr, $method_name:ident) => {
        impl $trait for Tensor {
            type Output = Self;
            fn $method_name(self) -> Self::Output {
                self.new_elementwise($operator, [])
            }
        }
    };
    ($trait:ty, $operator:expr) => {
        paste! {
            impl_unary_elementwise!($trait, $operator, [<$trait:snake:lower>]);
        }
    };
}
impl_unary_elementwise!(Neg, ElementOperator::Neg);

macro_rules! impl_binary_elementwise {
    ($trait:ty, $operator:expr, $method_name:ident) => {
        impl $trait for Tensor {
            type Output = Self;
            fn $method_name(self, rhs: Self) -> Self::Output {
                self.new_elementwise($operator, [rhs])
            }
        }
    };
    ($trait:ty, $operator:expr) => {
        paste! {
            impl_binary_elementwise!($trait, $operator, [<$trait:snake:lower>]);
        }
    };
}
impl_binary_elementwise!(Mul, ElementOperator::Mul);
impl_binary_elementwise!(Div, ElementOperator::Div);
impl_binary_elementwise!(Rem, ElementOperator::Rem);
impl_binary_elementwise!(Add, ElementOperator::Add);
impl_binary_elementwise!(Sub, ElementOperator::Sub);
impl_binary_elementwise!(BitAnd, ElementOperator::BitAnd, bitand);
impl_binary_elementwise!(BitOr, ElementOperator::BitOr, bitor);
impl_binary_elementwise!(BitXor, ElementOperator::BitXor, bitxor);
impl_binary_elementwise!(Shl, ElementOperator::Shl, shl);
impl_binary_elementwise!(Shr, ElementOperator::Shr, shr);

impl Tensor {
    pub fn log(&self) -> Self {
        self.new_elementwise(ElementOperator::Log, [])
    }
    pub fn exp(&self) -> Self {
        self.new_elementwise(ElementOperator::Exp, [])
    }
    pub fn relu(&self) -> Self {
        self.new_elementwise(ElementOperator::Relu, [])
    }
    pub fn abs(&self) -> Self {
        self.new_elementwise(ElementOperator::Abs, [])
    }
    pub fn sqrt(&self) -> Self {
        self.new_elementwise(ElementOperator::Sqrt, [])
    }
    pub fn floor(&self) -> Self {
        self.new_elementwise(ElementOperator::Floor, [])
    }
    pub fn ceil(&self) -> Self {
        self.new_elementwise(ElementOperator::Ceil, [])
    }
    pub fn pow(&self, rhs: &Self) -> Self {
        self.new_elementwise(ElementOperator::Pow, [rhs.clone()])
    }
    pub fn minimum(&self, rhs: &Self) -> Self {
        self.new_elementwise(ElementOperator::Min, [rhs.clone()])
    }
    pub fn maximum(&self, rhs: &Self) -> Self {
        self.new_elementwise(ElementOperator::Max, [rhs.clone()])
    }

    pub fn cmp_eq(&self, rhs: &Self) -> Self {
        self.new_elementwise(ElementOperator::CmpEq, [rhs.clone()])
    }
    pub fn cmp_ne(&self, rhs: &Self) -> Self {
        self.new_elementwise(ElementOperator::CmpNe, [rhs.clone()])
    }
    pub fn cmp_lt(&self, rhs: &Self) -> Self {
        self.new_elementwise(ElementOperator::CmpLt, [rhs.clone()])
    }
    pub fn cmp_le(&self, rhs: &Self) -> Self {
        self.new_elementwise(ElementOperator::CmpLe, [rhs.clone()])
    }
    pub fn cmp_gt(&self, rhs: &Self) -> Self {
        self.new_elementwise(ElementOperator::CmpGt, [rhs.clone()])
    }
    pub fn cmp_ge(&self, rhs: &Self) -> Self {
        self.new_elementwise(ElementOperator::CmpGe, [rhs.clone()])
    }

    /// Numeric value conversion (`f32 ↔ u32`); no-op when the type matches.
    pub fn cast(&self, element_type: ElementType) -> Self {
        if self.element_type() == element_type {
            return self.clone();
        }
        self.new_type_change(ElementOperator::Cast, element_type)
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
        self.new_type_change(ElementOperator::Bitcast, element_type)
    }

    /// `self` is a 0/1 mask: `mask * on_true + (1 - mask) * on_false`.
    ///
    /// Composed from arithmetic ops, so it works for both F32 and U32 and
    /// needs no dedicated kernel. No gradient flows through the mask.
    pub fn select(&self, on_true: &Self, on_false: &Self) -> Self {
        let one = match self.element_type() {
            ElementType::F32 => Self::full(self.shape(), 1.0, ElementType::F32),
            ElementType::U32 => Self::full_u32(self.shape(), 1),
        };
        self.clone() * on_true.clone() + (one - self.clone()) * on_false.clone()
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
        self.new_elementwise(ElementOperator::Matmul, [rhs.clone()])
    }
    pub fn transpose(&self) -> Self {
        let shape = self.shape();
        assert_eq!(shape.len(), 2, "transpose requires a rank-2 tensor");
        Tensor::from(TensorInner {
            element_type: self.element_type(),
            shape: Box::from([shape[1], shape[0]]),
            kind: TensorKind::Transpose { arg: self.clone() },
        })
    }

    pub fn broadcast_to(&self, target_shape: &[usize], axes: &[usize]) -> Self {
        Tensor::from(TensorInner {
            element_type: self.element_type(),
            shape: target_shape.into(),
            kind: TensorKind::Broadcast {
                arg: self.clone(),
                target_shape: target_shape.into(),
                axes: axes.into(),
            },
        })
    }

    /// Embed `self` into a zero-filled `target_shape` tensor at the region
    /// described by `key` (the inverse of [`Tensor::index`]).
    pub fn scatter_index(&self, target_shape: &[usize], key: &[IndexKeyElement]) -> Self {
        assert_eq!(
            key.len(),
            target_shape.len(),
            "scatter_index key must cover every target axis"
        );
        let region_shape = index_output_shape(target_shape, key);
        assert_eq!(
            region_shape.as_ref(),
            self.shape(),
            "scatter_index source shape must match the key region"
        );
        for (axis, element) in key.iter().enumerate() {
            let dim = target_shape[axis];
            match element {
                IndexKeyElement::Single(i) => {
                    assert!(*i < dim, "index {i} out of range for axis {axis} (dim {dim})")
                }
                IndexKeyElement::Slice(range) => assert!(
                    range.start <= range.end && range.end <= dim,
                    "slice {range:?} out of range for axis {axis} (dim {dim})"
                ),
            }
        }
        Tensor::from(TensorInner {
            element_type: self.element_type(),
            shape: target_shape.into(),
            kind: TensorKind::ScatterIndex {
                source: self.clone(),
                key: key.into(),
                target_shape: target_shape.into(),
            },
        })
    }

    /// `out[i, tail…] = self[indices[i], tail…]` (indices: rank-1 U32).
    pub fn gather_rows(&self, indices: &Tensor) -> Self {
        assert_eq!(
            indices.element_type(),
            ElementType::U32,
            "gather_rows indices must be U32"
        );
        assert_eq!(indices.shape().len(), 1, "gather_rows indices must be rank 1");
        assert!(!self.shape().is_empty(), "gather_rows source must have rank >= 1");
        let mut shape = self.shape().to_vec();
        shape[0] = indices.shape()[0];
        Tensor::from(TensorInner {
            element_type: self.element_type(),
            shape: shape.into(),
            kind: TensorKind::Gather {
                source: self.clone(),
                indices: indices.clone(),
            },
        })
    }

    /// `out[indices[i], tail…] ⊕= self[i, tail…]` into a zero-filled
    /// `[target_len, tail…]` tensor (indices: rank-1 U32, one per row of `self`).
    pub fn scatter_rows(&self, indices: &Tensor, target_len: usize, op: ScatterOp) -> Self {
        assert_eq!(
            indices.element_type(),
            ElementType::U32,
            "scatter_rows indices must be U32"
        );
        assert_eq!(indices.shape().len(), 1, "scatter_rows indices must be rank 1");
        assert!(!self.shape().is_empty(), "scatter_rows source must have rank >= 1");
        assert_eq!(
            indices.shape()[0],
            self.shape()[0],
            "scatter_rows: one index per source row"
        );
        let mut shape = self.shape().to_vec();
        shape[0] = target_len;
        Tensor::from(TensorInner {
            element_type: self.element_type(),
            shape: shape.into(),
            kind: TensorKind::Scatter {
                source: self.clone(),
                indices: indices.clone(),
                op,
            },
        })
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
                    assert!(*i < dim, "index {i} out of range for axis {axis} (dim {dim})")
                }
                IndexKeyElement::Slice(range) => assert!(
                    range.start <= range.end && range.end <= dim,
                    "slice {range:?} out of range for axis {axis} (dim {dim})"
                ),
            }
        }
        let shape = index_output_shape(arg.shape(), key);
        Tensor::from(TensorInner {
            element_type: arg.element_type(),
            shape,
            kind: TensorKind::Index {
                arg,
                key: key.into(),
            },
        })
    }

    pub fn sum_axes(&self, axes: &[usize]) -> Self {
        let shape = reduction_output_shape(self.shape(), axes);
        Tensor::from(TensorInner {
            element_type: self.element_type(),
            shape,
            kind: TensorKind::Reduction {
                operator: ElementOperator::Add,
                axes: axes.into(),
                arg: self.clone(),
            },
        })
    }

    /// Remove the given size-1 axes. Axes must be unique and in-bounds.
    pub fn squeeze(&self, axes: &[usize]) -> Self {
        let shape = self.shape();
        let mut seen = HashSet::new();
        for &axis in axes {
            assert!(
                axis < shape.len(),
                "squeeze axis {axis} out of range for rank {}",
                shape.len()
            );
            assert!(
                seen.insert(axis),
                "squeeze axes must be unique, got {axes:?}"
            );
            assert_eq!(
                shape[axis], 1,
                "cannot squeeze axis {axis} with size {}",
                shape[axis]
            );
        }
        let new_shape: Box<[usize]> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !seen.contains(i))
            .map(|(_, &d)| d)
            .collect();
        Tensor::from(TensorInner {
            element_type: self.element_type(),
            shape: new_shape,
            kind: TensorKind::Squeeze {
                arg: self.clone(),
                axes: axes.into(),
            },
        })
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
}

fn index_output_shape(_arg_shape: &[usize], key: &[IndexKeyElement]) -> Box<[usize]> {
    key.iter()
        .map(|element| match element {
            IndexKeyElement::Single(_) => 1,
            IndexKeyElement::Slice(range) => range.end - range.start,
        })
        .collect()
}

fn reduction_output_shape(arg_shape: &[usize], axes: &[usize]) -> Box<[usize]> {
    let mut axes: HashSet<usize> = axes.iter().copied().collect();
    arg_shape
        .iter()
        .enumerate()
        .map(|(axis, &dim)| if axes.remove(&axis) { 1 } else { dim })
        .collect()
}

//
// Toposort graph
//

impl Tensor {
    pub(crate) fn data_dependencies(&self) -> Vec<Tensor> {
        match &self.inner.kind {
            TensorKind::Constant { .. } => Vec::default(),
            TensorKind::Parameter => Vec::default(),
            TensorKind::Elementwise { args, .. } => Vec::from(args.as_slice()),
            TensorKind::Reduction { arg, .. } => vec![arg.clone()],
            TensorKind::Index { arg, .. } => vec![arg.clone()],
            TensorKind::Gather { source, indices } => {
                vec![source.clone(), indices.clone()]
            }
            TensorKind::Scatter {
                source, indices, ..
            } => {
                vec![source.clone(), indices.clone()]
            }
            TensorKind::Broadcast { arg, .. } => vec![arg.clone()],
            TensorKind::ScatterIndex { source, .. } => vec![source.clone()],
            TensorKind::Transpose { arg } => vec![arg.clone()],
            TensorKind::Squeeze { arg, .. } => vec![arg.clone()],
        }
    }
    /// Post-order traversal of the subgraph reachable from `self` (inputs before outputs).
    pub(crate) fn toposort(&self) -> Result<Vec<Tensor>, CyclicGraph> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        Self::collect_postorder(self, &mut order, &mut visited, &mut stack)?;
        Ok(order)
    }

    fn collect_postorder(
        node: &Tensor,
        order: &mut Vec<Tensor>,
        visited: &mut HashSet<Tensor>,
        stack: &mut HashSet<Tensor>,
    ) -> Result<(), CyclicGraph> {
        if visited.contains(node) {
            return Ok(());
        }
        if !stack.insert(node.clone()) {
            return Err(CyclicGraph);
        }
        for dep in node.data_dependencies() {
            Self::collect_postorder(&dep, order, visited, stack)?;
        }
        stack.remove(node);
        visited.insert(node.clone());
        order.push(node.clone());
        Ok(())
    }
}

impl Tensor {
    /// Trace-time parameter placeholder (shape and dtype only; no buffer).
    ///
    /// Created by the JIT when lifting concrete arrays into a DSL graph.
    pub fn parameter(shape: &[usize], element_type: ElementType) -> Self {
        Tensor::from(TensorInner {
            element_type,
            shape: shape.into(),
            kind: TensorKind::Parameter,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toposort_allows_shared_operands() {
        let a = Tensor::parameter(&[], ElementType::F32);
        let loss = a.clone() + a.clone();
        assert!(loss.toposort().is_ok());
    }

    #[test]
    fn zeros_respects_element_type() {
        let tensor = Tensor::zeros(&[2, 3], ElementType::F32);
        assert_eq!(tensor.element_type(), ElementType::F32);
    }

    #[test]
    fn sum_axes_is_keepdims() {
        let t = Tensor::parameter(&[8, 10], ElementType::F32);
        let s = t.sum_axes(&[0]);
        assert_eq!(s.shape(), &[1, 10]);
    }

    #[test]
    fn squeeze_all_after_full_sum_yields_scalar() {
        // Regression: mean_all used `while !shape.is_empty() { sum_axes([0]) }`,
        // which never terminates under keepdims (shape stays [1, …]).
        let t = Tensor::parameter(&[8, 10], ElementType::F32);
        let axes: Vec<usize> = (0..t.shape().len()).collect();
        let reduced = t.sum_axes(&axes);
        assert_eq!(reduced.shape(), &[1, 1]);
        let scalar = reduced.squeeze_all();
        assert!(scalar.shape().is_empty(), "got {:?}", scalar.shape());
    }
}

/// The tensor dependency graph contains a cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CyclicGraph;
