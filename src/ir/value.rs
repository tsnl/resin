//! Concrete values used by compile-time evaluation and the reference VM.

use std::sync::Arc;

use crate::util::define_id;

use super::Ty;

define_id! {
    /// A local allocation owned by one function.
    ///
    /// `LocalId`s are interpreted in the context of their containing function.
    pub struct LocalId(usize);

    /// A module-level allocation.
    pub struct GlobalId(usize);

    /// An entry in one function's closure display.
    ///
    /// `NonLocalId`s are interpreted in the context of their containing function.
    pub struct NonLocalId(usize);

    /// A function in the module.
    pub struct FunctionId(usize);

    /// A basic block owned by one function.
    ///
    /// `BlockId`s are interpreted in the context of their containing function.
    pub struct BlockId(usize);
}

/// A value on the reference VM's operand stack.
///
/// Backend emitters use their own corresponding value type so that dynamic
/// values can be represented by native backend handles instead.
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

/// An address rooted in storage known statically by the compiler.
///
/// The path contains type-directed child indices. It begins empty when a local
/// or global address is pushed and grows as access instructions are evaluated.
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

/// A named field in a concrete record value.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordFieldValue {
    pub name: Arc<str>,
    pub value: Value,
}

/// The elements of an array in index order.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayValue {
    pub element_ty: Ty,
    pub elements: Vec<Value>,
}

/// A function paired with its captured environment.
#[derive(Debug, Clone, PartialEq)]
pub struct ClosureValue {
    pub function: FunctionId,
    pub display: Vec<Value>,
}
