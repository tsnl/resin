//! Typed validation errors for the middle-end IR.

#[derive(Debug, thiserror::Error)]
pub enum IrError {
    // --- Elementwise RPN ---
    #[error("elementwise arg shape {arg:?} != kernel shape {kernel:?}")]
    ElementwiseArgShape { arg: Box<[u32]>, kernel: Box<[u32]> },

    #[error("elementwise arg_etypes length {etypes} != arg_accessors length {accessors}")]
    ElementwiseArgEtypesLen { etypes: usize, accessors: usize },

    // --- Matmul ---
    #[error("matmul requires exactly two arguments, got {got}")]
    MatmulArgCount { got: usize },

    #[error("matmul arguments must be at least rank 2")]
    MatmulRankTooLow,

    #[error("matmul inner dimension mismatch: {lhs} != {rhs}")]
    MatmulInnerDim { lhs: u32, rhs: u32 },

    #[error("matmul batch dimensions mismatch")]
    MatmulBatchDims,

    #[error("matmul rank mismatch")]
    MatmulRankMismatch,

    #[error("matmul output shape {output:?} != expected {expected:?}")]
    MatmulOutputShape { output: Box<[u32]>, expected: Box<[u32]> },

    // --- Reduction ---
    #[error("reduction requires exactly one argument, got {got}")]
    ReductionArgCount { got: usize },

    #[error("reduction input/output rank mismatch: input {input}, output {output}")]
    ReductionRankMismatch { input: usize, output: usize },

    #[error("reduction axis {axis} out of range for rank {rank}")]
    ReductionAxisOutOfRange { axis: u32, rank: usize },

    #[error("reduction output axis {axis} must be size 1, got {size}")]
    ReductionAxisNotUnit { axis: u32, size: u32 },

    #[error("reduction preserved axis {axis}: output {output} != input {input}")]
    ReductionPreservedAxis { axis: usize, output: u32, input: u32 },

    // --- Remap (shared) ---
    #[error("remap arg_etypes length {etypes} != arg_accessors length {accessors}")]
    RemapArgEtypesLen { etypes: usize, accessors: usize },

    // --- Remap: scatter ---
    #[error("scatter with accessor requires one argument, got {got}")]
    ScatterAccessorArgCount { got: usize },

    #[error("scatter accessor shape {accessor:?} != source shape {source_shape:?}")]
    ScatterAccessorShape {
        accessor: Box<[u32]>,
        source_shape: Box<[u32]>,
    },

    #[error("scatter without accessor requires two arguments, got {got}")]
    ScatterArgCount { got: usize },

    #[error("scatter indices trailing dim {trailing} must match output rank {output_rank}")]
    ScatterIndicesTrailing { trailing: u32, output_rank: usize },

    #[error("scatter indices/source batch shape mismatch")]
    ScatterBatchShape,

    // --- Remap: gather ---
    #[error("gather with implicit source requires one argument, got {got}")]
    GatherImplicitArgCount { got: usize },

    #[error("gather with accessor requires two arguments, got {got}")]
    GatherAccessorArgCount { got: usize },

    #[error("gather indices trailing dim {trailing} must match accessor rank {accessor_rank}")]
    GatherIndicesTrailing { trailing: u32, accessor_rank: usize },

    #[error("gather indices/source batch shape mismatch")]
    GatherBatchShape,

    #[error("gather accessor shape {accessor:?} != source_shape {source_shape:?}")]
    GatherAccessorSourceShape {
        accessor: Box<[u32]>,
        source_shape: Box<[u32]>,
    },

    #[error(
        "invalid RemapGatherInfo: accessor present={has_accessor}, source_shape present={has_source_shape}"
    )]
    InvalidGatherInfo {
        has_accessor: bool,
        has_source_shape: bool,
    },

    // --- Program ---
    #[error("dispatch arg view index {index} out of range (len {len})")]
    DispatchArgViewOutOfRange { index: usize, len: usize },

    #[error("dispatch output buffer index {index} out of range (len {len})")]
    DispatchOutputBufferOutOfRange { index: usize, len: usize },

    #[error("{kind} index {index} out of range (len {len})")]
    TreeIndexOutOfRange {
        kind: &'static str,
        index: usize,
        len: usize,
    },

    #[error("buffer_view[{view}] buffer_index {index} out of range (len {len})")]
    BufferViewBufferOutOfRange {
        view: usize,
        index: usize,
        len: usize,
    },
}
