//! WGPU interpreter and WGSL lowering for resin IR.

mod codegen;
mod interp;
mod lower;
mod program;

pub use codegen::{dispatch_size_for_kernel, emit_wgsl_for_kernel, WgslKernelConfig};
pub use interp::{
    create_interp, AdmitProgram, BufferId, Interp, InterpConfig, InterpError, ProgramId,
    WgpuInterp, WgpuInterpError,
};
pub use lower::build_wgpu_program;
pub use program::WgpuProgram;
