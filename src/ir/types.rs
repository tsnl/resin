//! Resolved monomorphic types used by the initial IR.

use std::sync::Arc;

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
    Pointer { pointee: Box<Ty> },
    Array { element: Box<Ty>, length: usize },
    Record { fields: Vec<RecordField> },
    Function { params: Vec<Ty>, result: Box<Ty> },
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
