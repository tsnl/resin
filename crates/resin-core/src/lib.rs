//! Shared host and tensor metadata types for resin.

mod accessor;
mod element;

pub use accessor::{
    c_contiguous_pitch_for_shape, shape_join, Accessor, AxisIndex, IntoAxisIndex, IntoIndexKey,
    ShapeJoin,
};
pub use element::{ElementKind, ElementOperator, ElementType, F2, F4, U4};
pub use resin_tree::{
    cmp_path, format_param_path, join_param_path, named, path_drop_first, path_empty, path_name,
    Empty, Leaf, Named, Tree, TreeLeaves, TreePath, TreePathElement,
};
