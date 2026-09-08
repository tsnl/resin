//! Low-level language: typed stack instructions and explicit control flow.
//! `lower` translates HIR into this representation.
use resin_common::{diagnostic, source, types, util};
use resin_hir as hir;

mod language;
pub mod lower;
pub mod print;
// Re-export only the shared vocabulary used by the LIR language.
pub use language::*;
pub use print::format_module;
pub use resin_common::types::{
    ArrayValue, Case, Foreign, FunctionId, LocalId, RecordField, RecordFieldValue, RecordValue,
    StaticAddressValue, Ty, TypeDef, TypeId, TypeTable, Value,
};
