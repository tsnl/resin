//! Shared host and tensor metadata types for resin.

mod accessor;
mod etype;
mod param_tree;

pub use accessor::{c_contiguous_pitch_for_shape, shape_join, Accessor, ShapeJoin};
pub use etype::{
    etype, etype_join, etype_join_kind, etype_kind, etype_nbytes, BinaryAssocElementOperator,
    BinaryBitwiseOperator, BinaryCompareOperator, BinaryElementOperator, ElementKind,
    ElementOperator, ElementType, UnaryElementOperator, F2, F4, U4,
};
pub use param_tree::ParamTree;
