//! Shared host and tensor metadata types for resin.

mod accessor;
mod element;
pub mod tree;

pub use accessor::{
    c_contiguous_pitch_for_shape, shape_join, Accessor, AxisIndex, IntoAxisIndex, IntoIndexKey,
    ShapeJoin,
};
pub use element::{ElementKind, ElementOperator, ElementType, F2, F4, U4};
pub use tree::{
    cmp_path, format_param_path, join_param_path, named, path_drop_first, path_empty, path_name,
    prepend_name, prepend_name_index, take_named_children, take_named_indexed_children,
    take_next_leaf, take_optional_leaf, Empty, Leaf, Named, Tree, TreeLeaves, TreePath,
    TreePathElement,
};
