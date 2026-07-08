#[derive(Debug, thiserror::Error)]
pub enum CpuLowerError {
    #[error("IR program validation failed: {0}")]
    InvalidIr(#[from] resin_ir::IrError),
}