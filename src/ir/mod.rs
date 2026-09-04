//! Target-independent, typed intermediate representation.

pub mod instr;
pub mod instr_verifier;
pub mod types;
pub mod value;

pub use instr::{BasicBlock, Function, Instr, Local, NonLocal, Terminator};
pub use instr_verifier::{StackEffect, VerifyError, VerifyErrorKind, verify};
pub use types::{RecordField, Ty};
pub use value::{
    ArrayValue, BlockId, ClosureValue, FunctionId, GlobalId, LocalId, NonLocalId, RecordFieldValue,
    RecordValue, StaticAddressValue, Value,
};

/// A module-level allocation and its resolved type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Global {
    pub ty: Ty,
}

/// A complete typed IR module.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Module {
    pub globals: Vec<Global>,
    pub functions: Vec<Function>,
}
