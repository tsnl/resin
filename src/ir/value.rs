//! Concrete IR values.

use std::sync::Arc;

use crate::util::define_id;

use super::Ty;

define_id! {
    /// Local allocation index, relative to the containing function.
    pub struct LocalId(usize);

    pub struct GlobalId(usize);

    /// Closure-display index, relative to the containing function.
    pub struct NonLocalId(usize);

    pub struct FunctionId(usize);

    /// Block index, relative to the containing function.
    pub struct BlockId(usize);
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Type { ty: Ty },
    Unit,
    Bool { value: bool },
    Int8 { value: i8 },
    Int16 { value: i16 },
    Int32 { value: i32 },
    Int64 { value: i64 },
    UInt8 { value: u8 },
    UInt16 { value: u16 },
    UInt32 { value: u32 },
    UInt64 { value: u64 },
    Float32 { value: f32 },
    Float64 { value: f64 },
    StaticAddress { address: StaticAddressValue },
    DynamicAddress { address: usize },
    Array { value: ArrayValue },
    Record { value: RecordValue },
    Closure { value: ClosureValue },
}

/// A storage root plus type-directed child indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaticAddressValue {
    Local {
        local: LocalId,
        path: Vec<usize>,
    },
    Global {
        global: GlobalId,
        path: Vec<usize>,
    },
    NonLocal {
        nonlocal: NonLocalId,
        path: Vec<usize>,
    },
}

/// The fields of a record in type-definition order.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordValue {
    pub fields: Vec<RecordFieldValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordFieldValue {
    pub name: Arc<str>,
    pub value: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayValue {
    pub element_ty: Ty,
    pub elements: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClosureValue {
    pub function: FunctionId,
    pub display: Vec<Value>,
}
