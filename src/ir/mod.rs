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
pub use typer::{BuiltinCall, Conv, Converted, FieldAccess, TypeError, TypeErrorKind, Typer};
pub use types::{RecordField, Ty, TypeDef, TypeId};
pub use value::{
    ArrayValue, BlockId, ClosureValue, FunctionId, GlobalId, LocalId, NonLocalId, RecordFieldValue,
    RecordValue, StaticAddressValue, Value,
};

/// A module-level allocation and its resolved type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Global {
    pub name: Arc<str>,
    pub ty: Ty,
}

/// A complete typed IR module.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Module {
    /// Nominal definitions, indexed by [`TypeId`].
    ///
    /// The generator reserves entries before evaluating definition RHSs so a
    /// definition can refer to itself and to the other definitions currently
    /// in scope.
    pub types: Vec<TypeDef>,
    /// Value definitions, indexed by [`GlobalId`].
    ///
    /// Like type definitions, global identities are reserved and installed in
    /// scope before their initializers are lowered. Eager initializer cycles
    /// are rejected during generation; delayed references from function bodies
    /// remain ordinary [`Instr::GlobalAddress`] operations.
    pub globals: Vec<Global>,
    pub functions: Vec<Function>,
}
