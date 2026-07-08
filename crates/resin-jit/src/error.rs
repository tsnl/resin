use resin_dsl::{ElementOperator, TensorKind};

#[derive(Debug, thiserror::Error)]
pub enum JitError {
    #[error(transparent)]
    Compile(#[from] CompileError),
    #[error(transparent)]
    Run(#[from] RunError),
}

#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("cyclic tensor graph")]
    CyclicGraph,
    #[error("shape dimension does not fit in u32: {0}")]
    ShapeOverflow(usize),
    #[error("unsupported tensor kind: {0}")]
    UnsupportedTensorKind(&'static str),
    #[error("unsupported element operator: {0:?}")]
    UnsupportedOperator(ElementOperator),
    #[error("accessor error: {0}")]
    Accessor(#[from] resin_core::ShapeError),
    #[error("IR validation failed: {0}")]
    Ir(#[from] resin_ir::IrError),
    #[error("backend lowering failed: {0}")]
    Lower(Box<dyn std::error::Error + Send + Sync>),
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("program execution is not implemented yet")]
    NotImplemented,
    #[error("param leaf count does not match compiled program")]
    ParamLeafMismatch,
    #[error("sink leaf count does not match compiled program")]
    SinkLeafMismatch,
    #[error("buffer init size mismatch: expected {expected}, got {got}")]
    BufferInitSize { expected: usize, got: usize },
    #[error("buffer size mismatch: expected {expected}, got {got}")]
    BufferSizeMismatch { expected: usize, got: usize },
    #[error("buffer read out of range at index {index}")]
    BufferReadOutOfRange { index: usize },
    #[error("unsupported kernel: {0}")]
    UnsupportedKernel(&'static str),
    #[error("unsupported op: {0}")]
    UnsupportedOp(String),
    #[error("matmul expected 2 args, got {got}")]
    MatmulArgCount { got: usize },
    #[error("reduction expected 1 arg, got {got}")]
    ReductionArgCount { got: usize },
    #[error("rpn arg index {arg} out of range")]
    RpnArgOutOfRange { arg: u32 },
    #[error("rpn evaluation produced an empty stack")]
    RpnEmptyStack,
    #[error("accessor rank mismatch: coords={coords}, shape={shape}")]
    AccessorRank { coords: usize, shape: usize },
    #[error("wgpu: {0}")]
    Wgpu(String),
}

pub(crate) fn tensor_kind_name(kind: &TensorKind) -> &'static str {
    match kind {
        TensorKind::Constant { .. } => "constant",
        TensorKind::Parameter => "parameter",
        TensorKind::Elementwise { .. } => "elementwise",
        TensorKind::Reduction { .. } => "reduction",
        TensorKind::Index { .. } => "index",
        TensorKind::Gather { .. } => "gather",
        TensorKind::Scatter { .. } => "scatter",
        TensorKind::Broadcast { .. } => "broadcast",
        TensorKind::ScatterIndex { .. } => "scatter_index",
        TensorKind::Transpose { .. } => "transpose",
        TensorKind::Squeeze { .. } => "squeeze",
    }
}