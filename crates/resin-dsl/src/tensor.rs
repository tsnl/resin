use super::*;
use arrayvec::ArrayVec;
use paste::paste;
use std::{
    collections::{HashSet, VecDeque},
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
}

#[derive(Clone, Copy)]
pub enum ElementType {
    F32,
}

#[derive(Clone, Copy)]
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

pub enum IndexKeyElement {
    Single(usize),
    Slice(Range<usize>),
}

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
    #[inline]
    fn new_elementwise<const N: usize>(
        &self,
        operator: ElementOperator,
        extra_args: [Tensor; N],
    ) -> Self {
        let mut args = arrvec![self.clone()];
        args.extend(extra_args);

        Tensor::from(TensorInner {
            element_type: self.inner.element_type,
            shape: self.inner.shape.clone(),
            kind: TensorKind::Elementwise { operator, args },
        })
    }
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
}

//
// Toposort graph
//

impl Tensor {
    fn data_dependencies(&self) -> Vec<Tensor> {
        match &self.inner.kind {
            TensorKind::Constant { .. } => Vec::default(),
            TensorKind::Parameter => Vec::default(),
            TensorKind::Elementwise { args, .. } => Vec::from(args.as_slice()),
            TensorKind::Reduction { arg, .. } => vec![arg.clone()],
            TensorKind::Index { arg, .. } => vec![arg.clone()],
            TensorKind::Remap { key, source, .. } => {
                vec![key.clone(), source.clone()]
            }
        }
    }
    pub fn detect_cyclic_dependencies(roots: &[Tensor]) -> bool {
        let mut visited = HashSet::with_capacity(roots.len());
        let mut queue = VecDeque::from(roots.to_vec());
        while let Some(tensor) = queue.pop_front() {
            let newly_inserted = visited.insert(tensor.clone());
            if !newly_inserted {
                // Same tensor was already visited, so we have a cycle.
                return true;
            }
            for dep in tensor.data_dependencies() {
                queue.push_back(dep);
            }
        }
        false
    }
}
