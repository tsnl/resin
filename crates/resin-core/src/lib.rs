mod accessor;
mod element_type;
mod macros;
mod tree;

pub use accessor::{
    c_contiguous_pitch_for_shape, shape_join, Accessor, ShapeError, ShapeJoin,
};
pub use element_type::{
    element_type, element_type_join, element_type_join_kind, element_type_kind,
    element_type_nbytes, BinaryAssocElementOperator, BinaryBitwiseOperator,
    BinaryCompareOperator, BinaryElementOperator, ElementKind, ElementOperator, ElementType,
    ElementTypeError, UnaryElementOperator, F2, F4, U4,
};

// Export basics:
pub use tree::Tree;

// Re-export "vendored" dependencies:
pub use arrayvec;
pub use paste;

// Export common collections
pub use hashbrown::{HashMap, HashSet, hash_map};
