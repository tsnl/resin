//! Verified LIR becomes a C or GLSL source tree, then text.
//! Target trees contain target syntax and can be printed without compiler state.
//! Implementation modules are private; clients use the operations below.
//!
//! ```compile_fail,E0603
//! use resin_codegen::c;
//! ```
//!
//! ```compile_fail,E0603
//! use resin_codegen::glsl;
//! ```
use resin_common::{source, types};
use resin_lir as lir;
use resin_lir_verifier as lir_verifier;
mod c;
mod error;
mod glsl;
mod layout;
mod numeric;

pub use resin_common::types::shader::Stage;

#[derive(Debug)]
pub struct Error(pub String);

/// C11 translation unit and control flow.
#[derive(Debug, Clone)]
pub struct CModule {
    pub includes: Vec<String>,
    pub assertions: Vec<(String, String)>,
    pub declarations: String,
    pub data: Vec<Vec<u32>>,
    pub prototypes: Vec<String>,
    pub lifecycle: String,
    pub functions: Vec<CFunction>,
    pub entry: CFunction,
}

#[derive(Debug, Clone)]
pub struct CFunction {
    pub signature: String,
    pub body: CBody,
}

#[derive(Debug, Clone)]
pub enum CBody {
    Inline(String),
    Blocks {
        locals: String,
        entry: usize,
        blocks: Vec<CBlock>,
    },
}

#[derive(Debug, Clone)]
pub struct CBlock {
    pub label: usize,
    pub statements: String,
    pub exit: CExit,
}

#[derive(Debug, Clone)]
pub enum CExit {
    Return(String),
    Jump(CEdge),
    Branch {
        condition: String,
        then: CEdge,
        els: CEdge,
    },
    Unreachable,
}

#[derive(Debug, Clone)]
pub struct CEdge {
    pub target: usize,
    pub values: Vec<CEdgeValue>,
}

#[derive(Debug, Clone)]
pub struct CEdgeValue {
    pub ty: String,
    pub value: String,
    pub live: Option<String>,
}

/// GLSL translation unit and control flow.
#[derive(Debug, Clone)]
pub struct GlslModule {
    pub extensions: String,
    pub declarations: String,
    pub globals: String,
    pub functions: Vec<GlslFunction>,
    pub entry: String,
}

#[derive(Debug, Clone)]
pub struct GlslFunction {
    pub signature: String,
    pub locals: String,
    pub entry: usize,
    pub blocks: Vec<GlslBlock>,
    pub default_result: String,
}

#[derive(Debug, Clone)]
pub struct GlslBlock {
    pub label: usize,
    pub statements: String,
    pub exit: GlslExit,
}

#[derive(Debug, Clone)]
pub enum GlslExit {
    Return(String),
    Jump(GlslEdge),
    Branch {
        condition: String,
        then: GlslEdge,
        els: GlslEdge,
    },
    Unreachable,
}

#[derive(Debug, Clone)]
pub struct GlslEdge {
    pub target: usize,
    pub values: Vec<GlslEdgeValue>,
}

#[derive(Debug, Clone)]
pub struct GlslEdgeValue {
    pub slot: usize,
    pub ty: String,
    pub value: String,
}

/// Compiled shader words embedded in a C translation unit.
pub struct Shader {
    pub function: lir::FunctionId,
    pub stage: Stage,
    pub words: Vec<u32>,
}

/// Lower verified LIR and its compiled shaders to a C translation unit.
pub fn generate_c(
    checked: lir_verifier::Verified<'_>,
    entry: &str,
    shaders: &[Shader],
) -> Result<CModule, Error> {
    c::lower::generate(checked, entry, shaders)
}

/// Lower one shader function from verified LIR to a GLSL translation unit.
pub fn generate_glsl(
    checked: lir_verifier::Verified<'_>,
    entry: lir::FunctionId,
    stage: Stage,
) -> Result<GlslModule, Error> {
    glsl::lower::generate(checked, entry, stage)
}

/// Render a completed C tree; no upstream representation is required.
pub fn print_c(module: &CModule) -> String {
    c::print::module(module)
}

/// Render a completed GLSL tree; no upstream representation is required.
pub fn print_glsl(module: &GlslModule) -> String {
    glsl::print::module(module)
}

/// Verify and emit C for a module that does not embed shaders.
pub fn emit_c(module: &lir::Module, entry: &str) -> Result<String, Error> {
    emit_c_with_shaders(module, entry, &[])
}

/// Verify and emit C with the supplied compiled shaders.
pub fn emit_c_with_shaders(
    module: &lir::Module,
    entry: &str,
    shaders: &[Shader],
) -> Result<String, Error> {
    lir_verifier::with_verified(module, |checked| {
        generate_c(checked, entry, shaders).map(|module| print_c(&module))
    })
}

/// Verify and emit GLSL for an exported shader function.
pub fn emit_glsl(module: &lir::Module, entry: &str, stage: Stage) -> Result<String, Error> {
    let function = module.entries.get(entry).ok_or_else(|| {
        Error(format!(
            "expected an exported shader function named {entry:?}"
        ))
    })?;
    emit_glsl_function(module, *function, stage)
}

/// Verify and emit GLSL for a resolved shader function.
pub fn emit_glsl_function(
    module: &lir::Module,
    entry: lir::FunctionId,
    stage: Stage,
) -> Result<String, Error> {
    lir_verifier::with_verified(module, |checked| {
        generate_glsl(checked, entry, stage).map(|module| print_glsl(&module))
    })
}
