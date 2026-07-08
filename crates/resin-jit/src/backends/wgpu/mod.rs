//! WebGPU backend ([`WgpuJit`]).

mod error;
mod jit;
mod program;
mod tensor;

pub use error::WgpuLowerError;
pub use jit::WgpuJit;
pub use tensor::WgpuTensor;