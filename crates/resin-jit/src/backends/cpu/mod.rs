//! CPU interpreter backend ([`CpuJit`]).

mod error;
mod exec;
mod hw;
mod jit;
mod program;
mod tensor;

pub use error::CpuLowerError;
pub use jit::CpuJit;
pub use tensor::CpuTensor;