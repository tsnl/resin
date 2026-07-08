#[derive(Debug, thiserror::Error)]
pub enum WgpuLowerError {
    #[error("IR program validation failed: {0}")]
    InvalidIr(#[from] resin_ir::IrError),
}