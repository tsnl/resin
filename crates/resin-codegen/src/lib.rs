//! Generate completed target artifacts from verified LIR without running external tools.
//! [`generate_native`] produces immutable host object bytes and [`generate_spirv`]
//! produces shader bytes. Both consume complete captured inputs and retain no
//! mutable compiler state or filesystem paths. Applications own native tools.
//! Target languages and lowering are private.
//!
//! ```compile_fail,E0432
//! use resin_codegen::c;
//! ```
//!
//! ```compile_fail,E0603
//! use resin_codegen::spirv;
//! ```
//!
//! ```compile_fail,E0603
//! use resin_codegen::cranelift;
//! ```
//!
//! ```compile_fail,E0432
//! use resin_codegen::{CModule, SpirvModule};
//! ```

use resin_executor::{Cancellation, Execution};
use resin_types::prelude::*;
use std::{collections::BTreeMap, sync::Arc};

mod cranelift;
mod error;
mod layout;
mod spirv;

//
// Native object and shader generation
//

/// Optimization performed during Cranelift translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NativeOptimization {
    None,
    Speed,
}

/// Completed host object code. Clones share immutable bytes; linking is separate.
#[derive(Debug, Clone)]
pub struct NativeObject {
    bytes: Arc<[u8]>,
}

impl NativeObject {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Retain the completed object without copying its bytes.
    pub fn shared_bytes(&self) -> Arc<[u8]> {
        self.bytes.clone()
    }
}

/// Completed external bindings and shader binaries consumed by host generation.
/// The application validates C declarations, resolves their link symbols, and
/// optimizes shaders before constructing these inputs.
/// `foreign` must bind every host foreign function present in the verified module;
/// `shaders` must contain every shader marked `embedded` by LIR construction.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeInputs {
    pub foreign: BTreeMap<FunctionId, Arc<str>>,
    pub shaders: BTreeMap<FunctionId, Arc<[u8]>>,
    /// Optional runtime ABI symbol substitutions, useful for native test fixtures.
    pub runtime: BTreeMap<Arc<str>, Arc<str>>,
}

/// Emit a native object on a bounded worker without running external tools.
/// Verified host functions use the platform baseline ISA and C ABI entry point.
/// Aggregate layout, ownership, runtime operations, and foreign calls are lowered
/// directly; linking the returned object and its referenced runtime remains separate.
///
/// ```compile_fail,E0308
/// let module = std::sync::Arc::new(resin_lir::Module::default());
/// let _ = resin_codegen::generate_native(
///     module, "main".into(), resin_codegen::NativeOptimization::None,
///     std::sync::Arc::new(resin_codegen::NativeInputs::default()),
///     &resin_executor::Execution::default(), &resin_executor::Cancellation::new(),
/// );
/// ```
pub async fn generate_native(
    checked: Arc<resin_lir::VerifiedModule>,
    entry: String,
    optimization: NativeOptimization,
    inputs: Arc<NativeInputs>,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<NativeObject, GenerationError> {
    execution
        .run(cancellation, move |cancellation| {
            cranelift::generate(checked.view(), &entry, optimization, &inputs, cancellation)
        })
        .await?
}

/// Emit one declared shader as unoptimized SPIR-V without filesystem access.
pub async fn generate_spirv(
    checked: Arc<resin_lir::VerifiedModule>,
    function: FunctionId,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<Arc<[u8]>, GenerationError> {
    execution
        .run(cancellation, move |cancellation| {
            cancellation.check()?;
            let shader = checked
                .view()
                .module()
                .shaders
                .get(&function)
                .ok_or_else(|| Error("requested function is not a shader".into()))?;
            let stage = shader.stage.parse().map_err(Error)?;
            let bytes = spirv::generate(checked.view(), function, stage)?;
            cancellation.check()?;
            Ok(bytes.into())
        })
        .await?
}

#[derive(Debug)]
pub enum GenerationError {
    Execution { error: resin_executor::Error },
    Codegen { error: Error },
}

impl From<resin_executor::Error> for GenerationError {
    fn from(error: resin_executor::Error) -> Self {
        Self::Execution { error }
    }
}

impl From<Error> for GenerationError {
    fn from(error: Error) -> Self {
        Self::Codegen { error }
    }
}

impl From<std::io::Error> for GenerationError {
    fn from(error: std::io::Error) -> Self {
        Self::Codegen {
            error: error.into(),
        }
    }
}

impl std::fmt::Display for GenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execution { error } => error.fmt(f),
            Self::Codegen { error } => error.fmt(f),
        }
    }
}

impl std::error::Error for GenerationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Execution { error } => Some(error),
            Self::Codegen { error } => Some(error),
        }
    }
}

#[derive(Debug)]
pub struct Error(String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}
