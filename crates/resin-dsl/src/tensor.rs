use super::*;
use arrayvec::ArrayVec;
use paste::paste;
use std::{
    collections::HashSet,
    hash::Hash,
    ops::{Add, Div, Mul, Neg, Range, Rem, Sub},
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
    Remap {
        key: Tensor,
        source: Tensor,
        direction: RemapDirection,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ElementType {
    F32,
}
impl ElementType {
    pub fn nbytes(&self) -> usize {
        match self {
            ElementType::F32 => 4,
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

    // Binary
    Pow,
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Matmul,
}

type ElementwiseArgs = ArrayVec<Tensor, MAX_ELEMENTWISE_ARGS>;
const MAX_ELEMENTWISE_ARGS: usize = 2;

#[derive(Clone)]
pub enum IndexKeyElement {
    Single(usize),
    Slice(Range<usize>),
}

#[derive(Clone, Copy)]
pub enum RemapDirection {
    Gather,  // output := source[key]
    Scatter, // output[key] := source
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

    pub fn full(shape: &[usize], value: f32, element_type: ElementType) -> Self {
        let count = shape.iter().product::<usize>();
        let bytes = f32::to_le_bytes(value);
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

        let shape = if operator == ElementOperator::Matmul {
            matmul_output_shape(self.shape(), args[1].shape())
        } else {
            self.inner.shape.clone()
        };

        Tensor::from(TensorInner {
            element_type: self.inner.element_type,
            shape,
            kind: TensorKind::Elementwise { operator, args },
        })
    }
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
    pub fn pow(&self, rhs: &Self) -> Self {
        self.new_elementwise(ElementOperator::Pow, [rhs.clone()])
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

    pub(crate) fn scatter_index(&self, target_shape: &[usize], key: &[IndexKeyElement]) -> Self {
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

    pub(crate) fn new_remap(
        key: Tensor,
        source: Tensor,
        direction: RemapDirection,
        shape: &[usize],
    ) -> Self {
        Tensor::from(TensorInner {
            element_type: source.element_type(),
            shape: shape.into(),
            kind: TensorKind::Remap {
                key,
                source,
                direction,
            },
        })
    }

    pub(crate) fn new_index(arg: Tensor, key: &[IndexKeyElement]) -> Self {
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
            TensorKind::Remap { key, source, .. } => {
                vec![key.clone(), source.clone()]
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
