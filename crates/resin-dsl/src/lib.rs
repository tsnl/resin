//! Resin computation-graph frontend (`View` / `Node`).

mod node;
mod view;

pub use node::{Node, NodeKind, RemapGatherInfo, RemapInfo, RemapScatterInfo};
pub use resin_core::{
    BinaryAssocElementOperator, BinaryBitwiseOperator, BinaryCompareOperator,
    BinaryElementOperator, ElementOperator, ElementType, UnaryElementOperator, Accessor, F2, F4,
    U4,
};
pub use view::{const_bytes, param, View};

// End-user code works with `View`. `Node` / `NodeKind` are public for `resin-ir` lowering.
