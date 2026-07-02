mod wgpu;

use thiserror::Error;

pub use wgpu::{request_default_device, WgpuInterp, WgpuInterpError};

/// Admitted program handle returned by [`AdmitProgram::admit_program`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgramId(pub usize);

/// Buffer index within a program's buffer table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub usize);

/// WGPU interpreter configuration (this crate only supports WGPU today).
#[derive(Debug, Clone, Default)]
pub struct InterpConfig {
    /// When set, request an adapter whose name contains this substring.
    /// When `None`, use the default high-performance (or fallback) adapter.
    pub device_name: Option<String>,
}

impl InterpConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_device_name(device_name: impl Into<String>) -> Self {
        Self {
            device_name: Some(device_name.into()),
        }
    }
}

#[derive(Debug, Error)]
pub enum InterpError {
    #[error("config error: {0}")]
    Config(String),
    #[error("program error: {0}")]
    Program(String),
    #[error("buffer map failed")]
    BufferMapFailed,
    #[error("backend error: {0}")]
    Backend(String),
}

/// Backend-agnostic interpreter API for **admitted** programs.
///
/// Host code should prefer this trait for training/eval loops so it stays
/// independent of the concrete backend. Program admission is backend-specific
/// via [`AdmitProgram`].
///
/// The program artifact lives **inside** the interpreter after admit; use
/// [`Interp::param`] / [`Interp::sink`] instead of cloning the artifact on the host.
pub trait Interp: Send + Sync {
    fn program_count(&self) -> usize;

    /// Buffer id for a named parameter (stable name from IR registration).
    fn param(&self, program_id: ProgramId, name: &str) -> Result<BufferId, InterpError>;

    /// Buffer id for a named sink's storage (the underlying buffer, not a view index).
    fn sink(&self, program_id: ProgramId, name: &str) -> Result<BufferId, InterpError>;

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

    /// Write a named parameter buffer.
    fn write_param(
        &self,
        program_id: ProgramId,
        name: &str,
        data: &[u8],
    ) -> Result<(), InterpError> {
        let id = self.param(program_id, name)?;
        self.write_buffer(program_id, id, data)
    }

    /// Read a named sink's buffer bytes.
    fn read_sink(&self, program_id: ProgramId, name: &str) -> Result<Vec<u8>, InterpError> {
        let id = self.sink(program_id, name)?;
        self.read_buffer(program_id, id)
    }
}

/// Admit a backend-specific program artifact by value (owned by the interpreter).
///
/// Separated from [`Interp`] so admission can be typed per backend while
/// post-admit code only depends on [`Interp`].
pub trait AdmitProgram<P>: Interp {
    fn admit_program(&mut self, program: P) -> Result<ProgramId, InterpError>;
}

/// Create the default WGPU interpreter for this crate.
pub fn create_interp(config: InterpConfig) -> Result<WgpuInterp, InterpError> {
    WgpuInterp::from_config(&config)
}
