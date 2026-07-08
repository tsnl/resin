//! WebGPU backend ([`WgpuJit`]): naive IR → WGSL lowering + dispatch.
//!
//! Optimizations belong in the IR layer. This backend emits one compute shader
//! per IR kernel and executes the queue as written.

mod codegen;
mod error;
mod jit;
mod lower;
mod program;
mod runtime;
mod tensor;

pub use error::{WgpuLowerError, WgpuRuntimeError};
pub use jit::WgpuJit;
pub use program::WgpuProgram;
pub use tensor::WgpuTensor;

/// Whether a GPU adapter is available (for tests / graceful skip).
pub fn shared_context_available() -> bool {
    runtime::shared_context().is_ok()
}
