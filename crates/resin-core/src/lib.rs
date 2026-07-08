mod accessor;
mod etype;
mod macros;
mod tree;

pub use accessor::{
    c_contiguous_pitch_for_shape, shape_join, Accessor, ShapeError, ShapeJoin,
};
pub use etype::{
    etype, etype_join, etype_join_kind, etype_kind, etype_nbytes, BinaryAssocElementOperator,
    BinaryBitwiseOperator, BinaryCompareOperator, BinaryElementOperator, ElementKind,
    ElementOperator, ElementType, EtypeError, UnaryElementOperator, F2, F4, U4,
};

// Export basics:
pub use tree::Tree;

// Re-export "vendored" dependencies:
pub use arrayvec;
pub use paste;

// Export common collections
pub use hashbrown::{HashMap, HashSet, hash_map};
