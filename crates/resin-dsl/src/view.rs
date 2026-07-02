use crate::node::{
    ElementwiseNodeKind, MatmulNodeKind, Node, NodeKind, ReductionNodeKind, RemapGatherInfo,
    RemapInfo, RemapNodeKind,
};
use crate::node_ref::NodeRef;
use resin_core::{
    shape_join, Accessor, ElementOperator, ElementType, IntoIndexKey, Leaf, Tree, TreePath, F4, U4,
};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Shl, Shr, Sub};
/// Public tensor handle: identity-keyed node plus logical accessor.
#[derive(Clone)]
pub struct View {
    pub node_ref: NodeRef,
    pub accessor: Accessor,
}

/// A lone [`View`] is a single-leaf param tree (empty path). Prefer wrapping with
/// [`resin_core::Named`] at compile roots so buffer names are non-empty.
impl Tree for View {
    type Leaf = View;
    type Map<U> = Leaf<U>;

    fn flatten(&self) -> impl Iterator<Item = (TreePath, &View)> + '_ {
        std::iter::once((resin_core::path_empty(), self))
    }

    fn consume_unflatten<I>(stream: &mut resin_core::TreeLeaves<View, I>) -> Self
    where
        I: Iterator<Item = (resin_core::TreePath, View)>,
    {
        match stream.pop() {
            None => panic!("Tree::unflatten for View: missing leaf"),
            Some((path, view)) => {
                assert!(
                    path.is_empty(),
                    "Tree::unflatten for View: unexpected path {path:?}"
                );
                view
            }
        }
    }
}

/// Rank-0 F4 constant (Python bare `const(1.0)`). Prefer `1.0f32.into()` / `View::from(1.0)`.
impl From<f32> for View {
    fn from(value: f32) -> Self {
        crate::const_bytes([], F4, value.to_le_bytes())
    }
}

/// Rank-0 U4 constant. Prefer `1u32.into()` / `View::from(1u32)`.
impl From<u32> for View {
    fn from(value: u32) -> Self {
        crate::const_bytes([], U4, value.to_le_bytes())
    }
}

impl View {
    pub fn identity(node_ref: NodeRef) -> Self {
        let accessor = Accessor::new_c_contiguous(node_ref.shape.clone(), 0);
        Self { node_ref, accessor }
    }

    pub fn shape(&self) -> &[u32] {
        &self.accessor.shape
    }

    pub fn pitch(&self) -> &[u32] {
        &self.accessor.pitch
    }

    pub fn offset(&self) -> u32 {
        self.accessor.offset
    }

    pub fn element_type(&self) -> ElementType {
        self.node_ref.element_type
    }

    pub fn rank(&self) -> usize {
        self.accessor.rank()
    }

    pub fn is_identity(&self) -> bool {
        self.accessor
            .is_dense_c_contiguous(self.node_ref.shape.as_ref())
    }

    pub fn broadcast(&self, leading: &[u32]) -> Self {
        Self {
            node_ref: self.node_ref.clone(),
            accessor: self.accessor.broadcast(leading),
        }
    }

    pub fn permute(&self, permutation: &[usize]) -> Result<Self, String> {
        Ok(Self {
            node_ref: self.node_ref.clone(),
            accessor: self.accessor.permute(permutation)?,
        })
    }

    pub fn transpose(&self) -> Result<Self, String> {
        Ok(Self {
            node_ref: self.node_ref.clone(),
            accessor: self.accessor.transpose()?,
        })
    }

    pub fn squeeze(&self, axes: &[usize]) -> Result<Self, String> {
        Ok(Self {
            node_ref: self.node_ref.clone(),
            accessor: self.accessor.squeeze(axes)?,
        })
    }

    /// Python `View.__getitem__`: ints / `Range`s / tuples (up to 5 axes).
    ///
    /// Examples: `v.index(0)?`, `v.index(1..3)?`, `v.index((0, ..))?`,
    /// `v.index((1, 1..3))?`.
    pub fn index(&self, key: impl IntoIndexKey) -> Result<Self, String> {
        let key = key.into_index_key();
        Ok(Self {
            node_ref: self.node_ref.clone(),
            accessor: self.accessor.narrow(&key)?,
        })
    }

    fn elementwise(args: Vec<View>, op: ElementOperator) -> Result<Self, String> {
        let arity = op.arity();
        assert_eq!(args.len(), arity);
        let shape: Box<[u32]> = args[0].shape().into();
        let element_type = args[0].element_type();
        Ok(Self::identity(NodeRef::new(Node {
            shape,
            element_type,
            args,
            kind: NodeKind::Elementwise(ElementwiseNodeKind { op }),
        })))
    }

    pub fn elementwise_unary(&self, op: ElementOperator) -> Self {
        debug_assert_eq!(op.arity(), 1);
        Self::elementwise(vec![self.clone()], op).expect("unary elementwise")
    }

    pub fn elementwise_binary(&self, other: &Self, op: ElementOperator) -> Result<Self, String> {
        debug_assert_eq!(op.arity(), 2);
        let (a, b) = join_for_elementwise(self, other)?;
        Self::elementwise(vec![a, b], op)
    }

    pub fn relu(&self) -> Self {
        self.elementwise_unary(ElementOperator::Relu)
    }

    pub fn exp(&self) -> Self {
        self.elementwise_unary(ElementOperator::Exp)
    }

    pub fn log(&self) -> Self {
        self.elementwise_unary(ElementOperator::Log)
    }

    pub fn sqrt(&self) -> Self {
        self.elementwise_unary(ElementOperator::Sqrt)
    }

    pub fn sin(&self) -> Self {
        self.elementwise_unary(ElementOperator::Sin)
    }

    pub fn cos(&self) -> Self {
        self.elementwise_unary(ElementOperator::Cos)
    }

    pub fn max_elem(&self, other: &Self) -> Result<Self, String> {
        self.elementwise_binary(other, ElementOperator::Max)
    }

    pub fn min_elem(&self, other: &Self) -> Result<Self, String> {
        self.elementwise_binary(other, ElementOperator::Min)
    }

    pub fn gt(&self, other: &Self) -> Result<Self, String> {
        self.elementwise_binary(other, ElementOperator::Gt)
    }

    pub fn lt(&self, other: &Self) -> Result<Self, String> {
        self.elementwise_binary(other, ElementOperator::Lt)
    }

    pub fn eq(&self, other: &Self) -> Result<Self, String> {
        self.elementwise_binary(other, ElementOperator::Eq)
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        let (a, b) = join_shapes_for_matmul(self, other)?;
        let (a, b) = join_etypes(&a, &b)?;
        let mut shape = a.shape()[..a.rank() - 1].to_vec();
        shape.push(b.shape()[b.rank() - 1]);
        Ok(Self::identity(NodeRef::new(Node {
            shape: shape.into_boxed_slice(),
            element_type: a.element_type(),
            args: vec![a, b],
            kind: NodeKind::Matmul(MatmulNodeKind),
        })))
    }

    pub fn reduce(&self, axes: &[usize], op: ElementOperator) -> Result<Self, String> {
        assert!(op.is_binary_assoc(), "reduce requires associative op");
        let mut out_shape = self.shape().to_vec();
        for &axis in axes {
            if axis >= self.rank() {
                return Err(format!("reduce axis {axis} out of range"));
            }
            out_shape[axis] = 1;
        }
        Ok(Self::identity(NodeRef::new(Node {
            shape: out_shape.into_boxed_slice(),
            element_type: self.element_type(),
            args: vec![self.clone()],
            kind: NodeKind::Reduction(ReductionNodeKind {
                op,
                axes: axes.iter().map(|&a| a as u32).collect(),
            }),
        })))
    }

    pub fn sum(&self, axes: Option<&[usize]>) -> Result<Self, String> {
        let all: Vec<usize> = (0..self.rank()).collect();
        let axes = axes.unwrap_or(&all);
        self.reduce(axes, ElementOperator::Add)
    }

    pub fn remap(
        source: &Self,
        info: RemapInfo,
        out_shape: impl Into<Box<[u32]>>,
        element_type: ElementType,
        extra_args: impl IntoIterator<Item = Self>,
    ) -> Self {
        let mut args = vec![source.clone()];
        args.extend(extra_args);
        Self::identity(NodeRef::new(Node {
            shape: out_shape.into(),
            element_type,
            args,
            kind: NodeKind::Remap(RemapNodeKind { info }),
        }))
    }

    pub fn copy(&self, element_type: Option<ElementType>) -> Self {
        let out = element_type.unwrap_or(self.element_type());
        if out == self.element_type() && self.is_identity() {
            return self.clone();
        }
        Self::remap(
            self,
            RemapInfo::Gather(RemapGatherInfo {
                accessor: None,
                source_shape: None,
            }),
            self.shape(),
            out,
            [],
        )
    }

    pub(crate) fn with_accessor(&self, accessor: Accessor) -> Self {
        Self {
            node_ref: self.node_ref.clone(),
            accessor,
        }
    }

    /// Topological order of nodes reachable from these roots (deps before dependents).
    pub fn toposort(roots: &[Self]) -> Vec<NodeRef> {
        let mut order = Vec::new();
        let mut visiting = std::collections::HashSet::<NodeRef>::new();
        let mut visited = std::collections::HashSet::<NodeRef>::new();

        fn visit(
            node: &NodeRef,
            visiting: &mut std::collections::HashSet<NodeRef>,
            visited: &mut std::collections::HashSet<NodeRef>,
            order: &mut Vec<NodeRef>,
        ) {
            if visited.contains(node) {
                return;
            }
            assert!(visiting.insert(node.clone()), "cycle in computation graph");
            for arg in &node.args {
                visit(&arg.node_ref, visiting, visited, order);
            }
            visiting.remove(node);
            visited.insert(node.clone());
            order.push(node.clone());
        }

        for root in roots {
            visit(&root.node_ref, &mut visiting, &mut visited, &mut order);
        }
        order
    }
}

fn join_etypes(a: &View, b: &View) -> Result<(View, View), String> {
    let res = a.element_type().join(b.element_type())?;
    let a = if a.element_type() != res {
        a.copy(Some(res))
    } else {
        a.clone()
    };
    let b = if b.element_type() != res {
        b.copy(Some(res))
    } else {
        b.clone()
    };
    Ok((a, b))
}

fn join_for_elementwise(a: &View, b: &View) -> Result<(View, View), String> {
    let (a, b) = join_etypes(a, b)?;
    let join = shape_join(a.shape(), a.pitch(), b.shape(), b.pitch())?;
    Ok((
        a.with_accessor(Accessor::new(a.offset(), join.shape.clone(), join.pitch1)),
        b.with_accessor(Accessor::new(b.offset(), join.shape, join.pitch2)),
    ))
}

fn join_shapes_for_matmul(a: &View, b: &View) -> Result<(View, View), String> {
    if a.rank() < 2 || b.rank() < 2 {
        return Err(format!(
            "shapes {:?} and {:?} are not compatible for matmul",
            a.shape(),
            b.shape()
        ));
    }
    let k_a = a.shape()[a.rank() - 1];
    let k_b = b.shape()[b.rank() - 2];
    if k_a != k_b {
        return Err(format!("matmul inner dim mismatch: {k_a} vs {k_b}"));
    }
    let batch_a = &a.shape()[..a.rank() - 2];
    let batch_b = &b.shape()[..b.rank() - 2];
    if batch_a != batch_b {
        return Err(format!(
            "matmul batch dims mismatch: {batch_a:?} vs {batch_b:?}"
        ));
    }
    Ok((a.clone(), b.clone()))
}

macro_rules! impl_bin_op {
    ($trait:ident, $method:ident, $op:expr) => {
        impl $trait for View {
            type Output = View;
            fn $method(self, rhs: View) -> View {
                self.elementwise_binary(&rhs, $op)
                    .unwrap_or_else(|e| panic!("{}: {e}", stringify!($trait)))
            }
        }

        impl $trait for &View {
            type Output = View;
            fn $method(self, rhs: &View) -> View {
                self.elementwise_binary(rhs, $op)
                    .unwrap_or_else(|e| panic!("{}: {e}", stringify!($trait)))
            }
        }
    };
}

impl_bin_op!(Add, add, ElementOperator::Add);
impl_bin_op!(Sub, sub, ElementOperator::Sub);
impl_bin_op!(Mul, mul, ElementOperator::Mul);
impl_bin_op!(Div, div, ElementOperator::Div);
impl_bin_op!(BitAnd, bitand, ElementOperator::Band);
impl_bin_op!(BitOr, bitor, ElementOperator::Bor);
impl_bin_op!(BitXor, bitxor, ElementOperator::Bxor);
impl_bin_op!(Shl, shl, ElementOperator::Shl);
impl_bin_op!(Shr, shr, ElementOperator::Shr);

impl Neg for View {
    type Output = View;
    fn neg(self) -> View {
        self.elementwise_unary(ElementOperator::Neg)
    }
}

impl Neg for &View {
    type Output = View;
    fn neg(self) -> View {
        self.elementwise_unary(ElementOperator::Neg)
    }
}

impl Not for View {
    type Output = View;
    fn not(self) -> View {
        self.elementwise_unary(ElementOperator::Not)
    }
}

impl Not for &View {
    type Output = View;
    fn not(self) -> View {
        self.elementwise_unary(ElementOperator::Not)
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::param;
    use resin_core::F4;

    #[test]
    fn identity_and_add_allocate_distinct_nodes() {
        let a = param([2, 3], F4);
        let b = param([2, 3], F4);
        let c = &a + &b;
        assert!(a.node_ref != c.node_ref);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.element_type(), F4);
    }

    #[test]
    fn same_param_cloned_shares_node() {
        let a = param([4], F4);
        let b = a.clone();
        assert_eq!(a.node_ref, b.node_ref);
    }

    #[test]
    fn matmul_shapes() {
        let a = param([2, 3], F4);
        let b = param([3, 5], F4);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 5]);
    }
}
