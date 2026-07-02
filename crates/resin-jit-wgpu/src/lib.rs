pub mod codegen;
pub mod compile;
pub mod interp;
pub mod lower;
pub mod program;

pub use codegen::{dispatch_size_for_kernel, emit_wgsl_for_kernel, WgslKernelConfig};
pub use compile::{compile_program, compile_with_tree, CompiledProgram, ParamEntry};
pub use interp::{
    create_interp, AdmitProgram, BufferId, Interp, InterpConfig, InterpError, ProgramId,
    WgpuInterp, WgpuInterpError,
};
pub use lower::{build_wgpu_program, param_buffer_index};
pub use program::*;
