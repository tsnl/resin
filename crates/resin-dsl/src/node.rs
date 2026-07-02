use resin_core::{Accessor, ElementOperator, ElementType};
use serde::{Deserialize, Serialize};

use crate::view::View;

/// Shared node payload. **Do not** derive structural `PartialEq` / `Eq` / `Ord` /
/// `Hash` — graph identity is allocation identity, exposed via [`crate::NodeRef`].
pub struct Node {
    pub shape: Box<[u32]>,
    pub element_type: ElementType,
    /// Operand views (empty for `Const` / `Param`).
    pub args: Vec<View>,
    pub kind: NodeKind,
}

/// Kind-specific payload only; operands live on [`Node::args`].
#[derive(Clone)]
pub enum NodeKind {
    Const(ConstNodeKind),
    Param,
    Elementwise(ElementwiseNodeKind),
    Matmul(MatmulNodeKind),
    Reduction(ReductionNodeKind),
    Remap(RemapNodeKind),
}

#[derive(Clone)]
pub struct ConstNodeKind {
    pub init: Box<[u8]>,
}

#[derive(Clone)]
pub struct ElementwiseNodeKind {
    pub op: ElementOperator,
}

#[derive(Clone)]
pub struct MatmulNodeKind;

#[derive(Clone)]
pub struct ReductionNodeKind {
    /// Must be an associative combiner (`Add` / `Mul` / `Max` / `Min`).
    pub op: ElementOperator,
    pub axes: Box<[u32]>,
}

#[derive(Clone)]
pub struct RemapNodeKind {
    pub info: RemapInfo,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemapInfo {
    Scatter(RemapScatterInfo),
    Gather(RemapGatherInfo),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemapScatterInfo {
    pub accessor: Option<Accessor>,
    /// When set, accumulate with this associative op (atomic on GPU).
    pub operator: Option<ElementOperator>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemapGatherInfo {
    pub accessor: Option<Accessor>,
    pub source_shape: Option<Box<[u32]>>,
}
