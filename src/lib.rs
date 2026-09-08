//! Compiler driver, tooling, and public access to the phase crates.
#![doc = include_str!("../doc/architecture.md")]
pub use resin_ast as ast;
pub use resin_codegen as codegen;
pub use resin_codegen::{c, glsl};
pub use resin_common::{diagnostic, source, types};
pub use resin_cst as cst;
pub use resin_cst::print as formatting;
pub use resin_hir as hir;
pub use resin_lir as lir;
pub use resin_lir_verifier as lir_verifier;
pub mod analysis;
pub mod cli;
pub mod compiler;
pub mod toolchain;
