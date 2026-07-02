//! Shared host and tensor metadata types for resin.

mod accessor;
mod element;
mod param_tree;

pub use accessor::{
    c_contiguous_pitch_for_shape, shape_join, Accessor, AxisIndex, IntoAxisIndex, IntoIndexKey,
    ShapeJoin,
};
pub use element::{ElementKind, ElementOperator, ElementType, F2, F4, U4};
pub use param_tree::{
    format_param_path, join_param_path, ParamTree, ParamTreePath, ParamTreePathElement,
};
