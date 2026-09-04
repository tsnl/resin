//! Resolved monomorphic types used by the initial IR.

use std::sync::Arc;

use crate::util::define_id;

define_id! {
    /// A nominal type definition in the module type table.
    pub struct TypeId(usize);
}

/// A nominal type and the representation computed from its defining RHS.
///
/// IR generation reserves the [`TypeId`] and installs its source name before
/// evaluating `body`, which permits productive recursive definitions. The
/// completed module is still required to have a finite representation: every
/// representation cycle must cross an indirection such as [`Ty::Pointer`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeDef {
    pub name: Arc<str>,
    pub body: Ty,
}

/// A named field in a structural record type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordField {
    pub name: Arc<str>,
    pub ty: Ty,
}

/// A target-independent value type.
///
/// Additional ownership, resource, and nominal forms can be added without
/// changing the stack-machine control-flow model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ty {
    /// The type of compile-time type values.
    Type,
    Unit,
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    /// A reference to a module-level nominal type definition.
    Defined {
        definition: TypeId,
    },
    Pointer {
        pointee: Box<Ty>,
    },
    /// A pointer and length to a runtime-sized sequence of `element`.
    ///
    /// Like [`Ty::Pointer`], a span has a fixed-size representation and does
    /// not contain its elements inline.
    Span {
        element: Box<Ty>,
    },
    Array {
        element: Box<Ty>,
        length: usize,
    },
    Record {
        fields: Vec<RecordField>,
    },
    Function {
        params: Vec<Ty>,
        result: Box<Ty>,
    },
}

impl Ty {
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::Int16
                | Self::Int32
                | Self::Int64
                | Self::UInt8
                | Self::UInt16
                | Self::UInt32
                | Self::UInt64
        )
    }
}
