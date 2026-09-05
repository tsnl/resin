//! Target-independent, typed intermediate representation.

use std::sync::Arc;

pub mod instr;
pub mod types;
pub mod value;

pub mod generate;
pub mod print;
pub mod typer;
pub mod verify;

pub use generate::{GenerateError, GenerateErrorKind, generate};
pub use instr::{BasicBlock, BlockId, Foreign, Function, Instr, Local, StackEffect, Terminator};
pub use print::format_module;
pub use typer::{
    BuiltinCall, Conv, Converted, FieldAccess, TypeError, TypeErrorKind, TyperContext,
};
pub use types::{RecordField, Ty, TypeDef, TypeId};
pub use value::{
    ArrayValue, FunctionId, GlobalId, LocalId, RecordFieldValue, RecordValue, StaticAddressValue,
    Value,
};
pub use verify::{VerifyError, VerifyErrorKind, VerifyLocation, verify};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Global {
    pub name: Arc<str>,
    pub ty: Ty,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Module {
    /// Nominal definitions, indexed by [`TypeId`].
    pub types: Vec<TypeDef>,
    /// Value definitions, indexed by [`GlobalId`].
    pub globals: Vec<Global>,
    pub functions: Vec<Function>,
}
