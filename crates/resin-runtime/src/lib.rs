pub mod interp;
pub mod program;

pub use interp::{request_default_device, WgpuInterp, WgpuInterpError};
pub use program::{
    ScalarType, WgpuAccessorSpec, WgpuBufferSpec, WgpuBufferViewSpec, WgpuComputePipelineSpec,
    WgpuDispatch, WgpuPipelineSpec, WgpuProgram,
};