use crate::node::{Node, NodeKind, RemapGatherInfo, RemapInfo};
use resin_core::{
    etype_join, shape_join, Accessor, BinaryAssocElementOperator, BinaryBitwiseOperator,
    BinaryCompareOperator, BinaryElementOperator, ElementOperator, ElementType,
    UnaryElementOperator,
};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Shl, Shr, Sub};
use std::sync::Arc;

/// Public tensor handle: identity-keyed node plus logical accessor.
#[derive(Clone)]
pub struct View {
    /// Shared graph node (identity = allocation). Public for `resin-ir` lowering.
    pub node: Arc<Node>,
    pub accessor: Accessor,
}

impl View {
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        Arc::ptr_eq(&a.node, &b.node)
    }

    pub fn node_key(&self) -> *const Node {
        Arc::as_ptr(&self.node)
    }

    pub fn identity(node: Arc<Node>) -> Self {
        let accessor = Accessor::dense(node.shape.clone(), 0);
        Self { node, accessor }
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

    pub fn etype(&self) -> ElementType {
        self.node.etype
    }

    pub fn rank(&self) -> usize {
        self.accessor.rank()
    }

    pub fn is_identity(&self) -> bool {
        self.accessor
            .is_dense_c_contiguous(self.node.shape.as_ref())
    }

    pub fn broadcast(&self, leading: &[u32]) -> Self {
        Self {
            node: Arc::clone(&self.node),
            accessor: self.accessor.broadcast(leading),
        }
    }

    pub fn permute(&self, permutation: &[usize]) -> Result<Self, String> {
        Ok(Self {
            node: Arc::clone(&self.node),
            accessor: self.accessor.permute(permutation)?,
        })
    }

    pub fn transpose(&self) -> Result<Self, String> {
        Ok(Self {
            node: Arc::clone(&self.node),
            accessor: self.accessor.transpose()?,
        })
    }

    pub fn squeeze(&self, axes: &[usize]) -> Result<Self, String> {
        Ok(Self {
            node: Arc::clone(&self.node),
            accessor: self.accessor.squeeze(axes)?,
        })
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        let (a, b) = join_shapes_for_matmul(self, other)?;
        let (a, b) = join_etypes(&a, &b)?;
        let mut shape = a.shape()[..a.rank() - 1].to_vec();
        shape.push(b.shape()[b.rank() - 1]);
        Ok(Self::identity(Arc::new(Node {
            shape: shape.into_boxed_slice(),
            etype: a.etype(),
            args: vec![a, b],
            kind: NodeKind::Matmul,
        })))
    }

    pub fn reduce(
        &self,
        axes: &[usize],
        op: BinaryAssocElementOperator,
    ) -> Result<Self, String> {
        let mut out_shape = self.shape().to_vec();
        for &axis in axes {
            if axis >= self.rank() {
                return Err(format!("reduce axis {axis} out of range"));
            }
            out_shape[axis] = 1;
        }
        Ok(Self::identity(Arc::new(Node {
            shape: out_shape.into_boxed_slice(),
            etype: self.etype(),
            args: vec![self.clone()],
            kind: NodeKind::Reduction {
                op,
                axes: axes.iter().map(|&a| a as u32).collect(),
            },
        })))
    }

    pub fn sum(&self, axes: Option<&[usize]>) -> Result<Self, String> {
        let all: Vec<usize> = (0..self.rank()).collect();
        let axes = axes.unwrap_or(&all);
        self.reduce(axes, BinaryAssocElementOperator::Add)
    }

    pub fn elementwise_unary(&self, op: UnaryElementOperator) -> Self {
        Self::identity(Arc::new(Node {
            shape: self.shape().into(),
            etype: self.etype(),
            args: vec![self.clone()],
            kind: NodeKind::Elementwise {
                op: ElementOperator::Unary(op),
            },
        }))
    }

    pub fn exp(&self) -> Self {
        self.elementwise_unary(UnaryElementOperator::Exp)
    }

    pub fn log(&self) -> Self {
        self.elementwise_unary(UnaryElementOperator::Log)
    }

    pub fn sqrt(&self) -> Self {
        self.elementwise_unary(UnaryElementOperator::Sqrt)
    }

    pub fn sin(&self) -> Self {
        self.elementwise_unary(UnaryElementOperator::Sin)
    }

    pub fn cos(&self) -> Self {
        self.elementwise_unary(UnaryElementOperator::Cos)
    }

    pub fn max_elem(&self, other: &Self) -> Result<Self, String> {
        self.elementwise_binary(
            other,
            BinaryElementOperator::Assoc(BinaryAssocElementOperator::Max),
        )
    }

    pub fn min_elem(&self, other: &Self) -> Result<Self, String> {
        self.elementwise_binary(
            other,
            BinaryElementOperator::Assoc(BinaryAssocElementOperator::Min),
        )
    }

    pub fn gt(&self, other: &Self) -> Result<Self, String> {
        self.elementwise_compare(other, BinaryCompareOperator::Gt)
    }

    pub fn lt(&self, other: &Self) -> Result<Self, String> {
        self.elementwise_compare(other, BinaryCompareOperator::Lt)
    }

    pub fn eq(&self, other: &Self) -> Result<Self, String> {
        self.elementwise_compare(other, BinaryCompareOperator::Eq)
    }

    /// Topological order of nodes reachable from these roots (deps before dependents).
    pub fn toposort(roots: &[Self]) -> Vec<Arc<Node>> {
        let mut order = Vec::new();
        let mut visiting = std::collections::HashSet::<*const Node>::new();
        let mut visited = std::collections::HashSet::<*const Node>::new();

        fn visit(
            node: &Arc<Node>,
            visiting: &mut std::collections::HashSet<*const Node>,
            visited: &mut std::collections::HashSet<*const Node>,
            order: &mut Vec<Arc<Node>>,
        ) {
            let key = Arc::as_ptr(node);
            if visited.contains(&key) {
                return;
            }
            assert!(
                visiting.insert(key),
                "cycle in computation graph"
            );
            for arg in &node.args {
                visit(&arg.node, visiting, visited, order);
            }
            visiting.remove(&key);
            visited.insert(key);
            order.push(Arc::clone(node));
        }

        for root in roots {
            visit(&root.node, &mut visiting, &mut visited, &mut order);
        }
        order
    }

    pub fn elementwise_binary(
        &self,
        other: &Self,
        op: BinaryElementOperator,
    ) -> Result<Self, String> {
        let (a, b) = join_for_elementwise(self, other)?;
        Ok(Self::identity(Arc::new(Node {
            shape: a.shape().into(),
            etype: a.etype(),
            args: vec![a, b],
            kind: NodeKind::Elementwise {
                op: ElementOperator::Binary(op),
            },
        })))
    }

    pub fn elementwise_compare(
        &self,
        other: &Self,
        op: BinaryCompareOperator,
    ) -> Result<Self, String> {
        let (a, b) = join_for_elementwise(self, other)?;
        Ok(Self::identity(Arc::new(Node {
            shape: a.shape().into(),
            etype: a.etype(),
            args: vec![a, b],
            kind: NodeKind::Elementwise {
                op: ElementOperator::Compare(op),
            },
        })))
    }

    pub fn elementwise_bitwise(
        &self,
        other: &Self,
        op: BinaryBitwiseOperator,
    ) -> Result<Self, String> {
        let (a, b) = join_for_elementwise(self, other)?;
        Ok(Self::identity(Arc::new(Node {
            shape: a.shape().into(),
            etype: a.etype(),
            args: vec![a, b],
            kind: NodeKind::Elementwise {
                op: ElementOperator::Bitwise(op),
            },
        })))
    }

    pub fn remap(
        source: &Self,
        info: RemapInfo,
        out_shape: impl Into<Box<[u32]>>,
        etype: ElementType,
        extra_args: impl IntoIterator<Item = Self>,
    ) -> Self {
        let mut args = vec![source.clone()];
        args.extend(extra_args);
        Self::identity(Arc::new(Node {
            shape: out_shape.into(),
            etype,
            args,
            kind: NodeKind::Remap { info },
        }))
    }

    pub fn copy(&self, etype: Option<ElementType>) -> Self {
        let out_etype = etype.unwrap_or(self.etype());
        if out_etype == self.etype() && self.is_identity() {
            return self.clone();
        }
        Self::remap(
            self,
            RemapInfo::Gather(RemapGatherInfo {
                accessor: None,
                source_shape: None,
            }),
            self.shape(),
            out_etype,
            [],
        )
    }

    pub(crate) fn with_accessor(&self, accessor: Accessor) -> Self {
        Self {
            node: Arc::clone(&self.node),
            accessor,
        }
    }
}

/// Named parameter leaf (host-writable buffer).
pub fn param(shape: impl Into<Box<[u32]>>, etype: ElementType, name: impl Into<Arc<str>>) -> View {
    let shape = shape.into();
    View::identity(Arc::new(Node {
        shape,
        etype,
        args: vec![],
        kind: NodeKind::Param { name: name.into() },
    }))
}

/// Constant buffer from raw little-endian bytes.
pub fn const_bytes(
    shape: impl Into<Box<[u32]>>,
    etype: ElementType,
    init: impl Into<Box<[u8]>>,
) -> View {
    let shape = shape.into();
    let init = init.into();
    let expected = shape
        .iter()
        .map(|&d| u64::from(d))
        .product::<u64>()
        * u64::from(etype.nbytes());
    assert_eq!(
        init.len() as u64,
        expected,
        "const init byte length mismatch"
    );
    View::identity(Arc::new(Node {
        shape,
        etype,
        args: vec![],
        kind: NodeKind::Const { init },
    }))
}

fn join_etypes(a: &View, b: &View) -> Result<(View, View), String> {
    let res = etype_join(a.etype(), b.etype())?;
    let a = if a.etype() != res {
        a.copy(Some(res))
    } else {
        a.clone()
    };
    let b = if b.etype() != res {
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
        a.with_accessor(Accessor::new(
            a.offset(),
            join.shape.clone(),
            join.pitch1,
        )),
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

macro_rules! impl_bin_arith {
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

impl_bin_arith!(
    Add,
    add,
    BinaryElementOperator::Assoc(BinaryAssocElementOperator::Add)
);
impl_bin_arith!(Sub, sub, BinaryElementOperator::Sub);
impl_bin_arith!(
    Mul,
    mul,
    BinaryElementOperator::Assoc(BinaryAssocElementOperator::Mul)
);
impl_bin_arith!(Div, div, BinaryElementOperator::Div);

macro_rules! impl_bin_bit {
    ($trait:ident, $method:ident, $op:expr) => {
        impl $trait for View {
            type Output = View;
            fn $method(self, rhs: View) -> View {
                self.elementwise_bitwise(&rhs, $op)
                    .unwrap_or_else(|e| panic!("{}: {e}", stringify!($trait)))
            }
        }

        impl $trait for &View {
            type Output = View;
            fn $method(self, rhs: &View) -> View {
                self.elementwise_bitwise(rhs, $op)
                    .unwrap_or_else(|e| panic!("{}: {e}", stringify!($trait)))
            }
        }
    };
}

impl_bin_bit!(BitAnd, bitand, BinaryBitwiseOperator::Band);
impl_bin_bit!(BitOr, bitor, BinaryBitwiseOperator::Bor);
impl_bin_bit!(BitXor, bitxor, BinaryBitwiseOperator::Bxor);
impl_bin_bit!(Shl, shl, BinaryBitwiseOperator::Shl);
impl_bin_bit!(Shr, shr, BinaryBitwiseOperator::Shr);

impl Neg for View {
    type Output = View;
    fn neg(self) -> View {
        self.elementwise_unary(UnaryElementOperator::Neg)
    }
}

impl Neg for &View {
    type Output = View;
    fn neg(self) -> View {
        self.elementwise_unary(UnaryElementOperator::Neg)
    }
}

impl Not for View {
    type Output = View;
    fn not(self) -> View {
        self.elementwise_unary(UnaryElementOperator::Not)
    }
}

impl Not for &View {
    type Output = View;
    fn not(self) -> View {
        self.elementwise_unary(UnaryElementOperator::Not)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use resin_core::F4;

    #[test]
    fn identity_and_add_allocate_distinct_nodes() {
        let a = param([2, 3], F4, "a");
        let b = param([2, 3], F4, "b");
        let c = &a + &b;
        assert!(!View::ptr_eq(&a, &c));
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.etype(), F4);
    }

    #[test]
    fn same_param_cloned_shares_node() {
        let a = param([4], F4, "x");
        let b = a.clone();
        assert!(View::ptr_eq(&a, &b));
    }

    #[test]
    fn matmul_shapes() {
        let a = param([2, 3], F4, "a");
        let b = param([3, 5], F4, "b");
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 5]);
    }
}
