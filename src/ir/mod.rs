//! Target-independent, typed intermediate representation.

use std::{collections::BTreeMap, sync::Arc};

pub mod instr;
pub mod layout;
pub(crate) mod literal;
pub mod shader;
pub mod types;
pub mod value;

pub mod generate;
pub mod print;
pub mod typer;
pub mod verify;

pub use generate::{GenerateError, GenerateErrorKind, generate, generate_program};
pub(crate) use generate::{analyze_program, analyze_recovering};
pub use instr::{BasicBlock, BlockId, Foreign, Function, Instr, Local, StackEffect, Terminator};
pub use print::format_module;
pub use typer::{
    BuiltinCall, Conv, Converted, FieldAccess, TypeError, TypeErrorKind, TyperContext,
};
pub use types::{Case, RecordField, Ty, TypeDef, TypeId};
pub use value::{
    ArrayValue, FunctionId, LocalId, RecordFieldValue, RecordValue, StaticAddressValue, Value,
};
pub use verify::{VerifyError, VerifyErrorKind, VerifyLocation, verify};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Module {
    /// Functions exported by the entry source file.
    pub entries: BTreeMap<Arc<str>, FunctionId>,
    /// Nominal definitions, indexed by [`TypeId`].
    pub types: Vec<TypeDef>,
    pub functions: Vec<Function>,
    /// Decorated shader candidates and whether their static artifact is requested.
    pub shaders: BTreeMap<FunctionId, shader::ShaderEntry>,
    /// Optional source origins; direct IR clients may leave this empty.
    pub origins: SourceMap,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SourceMap {
    pub sources: BTreeMap<std::path::PathBuf, Arc<str>>,
    pub functions: BTreeMap<FunctionId, crate::ast::SourceLocation>,
    pub instructions: BTreeMap<(FunctionId, BlockId, usize), crate::ast::SourceLocation>,
}
