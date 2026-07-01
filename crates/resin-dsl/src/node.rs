use crate::view::View;
use resin_core::{
    BinaryAssocElementOperator, ElementOperator, ElementType, Accessor,
};
use std::sync::Arc;

/// Shared node payload; identity is the `Arc` allocation.
pub struct Node {
    pub shape: Box<[u32]>,
    pub etype: ElementType,
    /// Operand views (empty for `Const` / `Param`).
    pub args: Vec<View>,
    pub kind: NodeKind,
}

/// Kind-specific payload only; operands live on [`Node::args`].
pub enum NodeKind {
    Const { init: Box<[u8]> },
    Param { name: Arc<str> },
    Elementwise { op: ElementOperator },
    Matmul,
    Reduction {
        op: BinaryAssocElementOperator,
        axes: Box<[u32]>,
    },
    Remap { info: RemapInfo },
}

#[derive(Debug, Clone)]
pub enum RemapInfo {
    Scatter(RemapScatterInfo),
    Gather(RemapGatherInfo),
}

#[derive(Debug, Clone)]
pub struct RemapScatterInfo {
    pub accessor: Option<Accessor>,
    pub operator: Option<BinaryAssocElementOperator>,
}

#[derive(Debug, Clone)]
pub struct RemapGatherInfo {
    pub accessor: Option<Accessor>,
    pub source_shape: Option<Box<[u32]>>,
}

