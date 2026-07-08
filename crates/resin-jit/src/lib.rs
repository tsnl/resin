//! JAX-style JIT: bind trace-time functions to a backend and call with concrete tensors.
//!
//! User code should be backend-agnostic: PyTree leaves are [`Jit::Tensor`], e.g.
//! `struct Inputs<J: Jit> { a: J::Tensor, b: J::Tensor }`.
//!
//! Backends live under [`backends`], enabled via crate features.

mod cache;
mod error;
mod jit;
mod jitted;
mod lower;
mod pipeline;
mod tensor;

#[cfg(any(feature = "cpu", feature = "wgpu"))]
pub mod backends;

pub use error::{CompileError, JitError, RunError};
pub use jit::Jit;
pub use jitted::JittedFn;
pub use tensor::ConcreteTensor;

/// DSL → IR lowering exposed for backend tests (not a stable API).
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn lower_for_tests<PT, ST>(
    params: &PT,
    sinks: &ST,
) -> Result<
    resin_ir::IrProgram<PT::Mapped<resin_ir::BufferRef>, ST::Mapped<resin_ir::BufferViewRef>>,
    CompileError,
>
where
    PT: resin_core::Tree<resin_dsl::Tensor>,
    ST: resin_core::Tree<resin_dsl::Tensor>,
{
    lower::lower_program(params, sinks)
}