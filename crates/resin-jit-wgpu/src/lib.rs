pub mod codegen;
pub mod interp;
pub mod lower;
pub mod program;

pub use codegen::{dispatch_size_for_kernel, emit_wgsl_for_kernel, WgslKernelConfig};
pub use interp::{
    create_interp, parse_backend, request_default_device, BufferId, Interp, InterpBackend,
    InterpConfig, InterpError, ProgramId, WgpuInterp, WgpuInterpError,
};
pub use lower::{build_wgpu_program, param_buffer_index};
pub use program::*;