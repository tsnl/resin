//! Resin computation-graph frontend (`View` / `Node`).

mod debug_print;
mod node;
mod node_ref;
mod prelude;
mod view;

pub use debug_print::{debug_print, refcount};
pub use node::{
    ConstNodeKind, ElementwiseNodeKind, MatmulNodeKind, Node, NodeKind, ReductionNodeKind,
    RemapGatherInfo, RemapInfo, RemapNodeKind, RemapScatterInfo,
};
pub use node_ref::NodeRef;
pub use prelude::{const_bytes, constant, full, full_u32, ones, param, zeros, ConstData};
pub use resin_core::{
    Accessor, AxisIndex, ElementOperator, ElementType, IntoAxisIndex, IntoIndexKey, F2, F4, U4,
};
pub use view::View;
