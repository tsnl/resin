//! High-level language: resolved, typed expressions with structured control flow.
//!
//! Read `language` for the pass contract and `lower` for AST → HIR. Published
//! trees contain no source scopes, inference variables, or stack instructions.
use resin_ast as ast;
use resin_common::{diagnostic, source, types, util};
use resin_cst as cst;

pub mod analysis;
mod language;
pub mod lower;
pub mod print;
pub use language::*;
pub use lower::analyze_program;
pub use lower::{generate, generate_program};
pub use print::format_module;
