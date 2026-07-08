#[derive(Debug, thiserror::Error)]
pub enum WgpuLowerError {
    #[error("IR program validation failed: {0}")]
    InvalidIr(#[from] resin_ir::IrError),
    #[error("unsupported kernel for WGSL lowering: {0}")]
    UnsupportedKernel(&'static str),
    #[error("WGSL lowering error: {0}")]
    Message(String),
}

#[derive(Debug, thiserror::Error)]
pub enum WgpuRuntimeError {
    #[error("no suitable GPU adapter found")]
    NoAdapter,
    #[error("request device failed: {0}")]
    RequestDevice(String),
    #[error("shader/pipeline error: {0}")]
    Pipeline(String),
    #[error("buffer map failed")]
    BufferMapFailed,
    #[error("runtime error: {0}")]
    Message(String),
}
