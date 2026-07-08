#[derive(Debug, thiserror::Error)]
pub enum WgpuLowerError {
    #[error("IR program validation failed: {0}")]
    InvalidIr(#[from] resin_ir::IrError),
    #[error("unsupported kernel for WGSL lowering: {0}")]
    UnsupportedKernel(&'static str),
    #[error("WGSL lowering error: {0}")]
    Message(String),
}

impl From<crate::backends::wgsl::WgslError> for WgpuLowerError {
    fn from(err: crate::backends::wgsl::WgslError) -> Self {
        use crate::backends::wgsl::WgslError;
        match err {
            WgslError::UnsupportedKernel(kernel) => WgpuLowerError::UnsupportedKernel(kernel),
            WgslError::Message(msg) => WgpuLowerError::Message(msg),
        }
    }
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
