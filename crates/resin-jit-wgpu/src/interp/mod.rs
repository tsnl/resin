mod wgpu;

use rmpv::Value as MsgpackValue;
use thiserror::Error;

use crate::program::WgpuProgram;

pub use wgpu::{request_default_device, WgpuInterp, WgpuInterpError};

/// Admitted program handle returned by [`Interp::admit_program`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgramId(pub usize);

/// Buffer index within a program's buffer table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpBackend {
    Wgpu,
}

/// Runtime configuration for interpreter construction.
pub type InterpConfig = MsgpackValue;

#[derive(Debug, Error)]
pub enum InterpError {
    #[error("unsupported backend: {0}")]
    UnsupportedBackend(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("program error: {0}")]
    Program(String),
    #[error("buffer map failed")]
    BufferMapFailed,
    #[error("backend error: {0}")]
    Backend(String),
}

/// Device-agnostic interpreter interface.
///
/// Object-safe so backends such as WGPU and future Vulkan implementations can be
/// used behind `Box<dyn Interp>`.
pub trait Interp: Send + Sync {
    fn program_count(&self) -> usize;

    /// Admit a program by value (no serialization on the execution path).
    fn admit_program(&mut self, program: WgpuProgram) -> Result<ProgramId, InterpError>;

    fn run(&self, program_id: ProgramId) -> Result<(), InterpError>;

    fn write_buffer(
        &self,
        program_id: ProgramId,
        buffer_id: BufferId,
        data: &[u8],
    ) -> Result<(), InterpError>;

    fn read_buffer(
        &self,
        program_id: ProgramId,
        buffer_id: BufferId,
    ) -> Result<Vec<u8>, InterpError>;

    fn copy_buffer_to_buffer(
        &self,
        src_program_id: ProgramId,
        src_buffer_id: BufferId,
        dst_program_id: ProgramId,
        dst_buffer_id: BufferId,
    ) -> Result<(), InterpError>;
}

pub fn create_interp(
    backend: InterpBackend,
    config: InterpConfig,
) -> Result<Box<dyn Interp>, InterpError> {
    match backend {
        InterpBackend::Wgpu => Ok(Box::new(WgpuInterp::from_config(&config)?)),
    }
}

pub fn parse_backend(name: &str) -> Result<InterpBackend, InterpError> {
    match name {
        "wgpu" => Ok(InterpBackend::Wgpu),
        other => Err(InterpError::UnsupportedBackend(other.to_string())),
    }
}