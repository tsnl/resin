//! Target-independent, typed intermediate representation.

use std::sync::Arc;

pub mod generate;
pub mod instr;
pub mod instr_verifier;
pub mod print;
pub mod scope;
pub mod typer;
pub mod types;
pub mod value;

pub use generate::{GenerateError, GenerateErrorKind, generate};
pub use instr::{BasicBlock, Function, Instr, Local, NonLocal, Terminator};
pub use instr_verifier::{StackEffect, VerifyError, VerifyErrorKind, VerifyLocation, verify};
pub use print::format_module;
pub use typer::{
    BuiltinCall, Conv, Converted, FieldAccess, TypeError, TypeErrorKind, TyperContext,
};
pub use types::{RecordField, Ty, TypeDef, TypeId};
pub use value::{
    ArrayValue, BlockId, ClosureValue, FunctionId, GlobalId, LocalId, NonLocalId, RecordFieldValue,
    RecordValue, StaticAddressValue, Value,
};

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
