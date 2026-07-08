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