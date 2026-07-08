//! Typed validation errors for the middle-end IR.

#[derive(Debug, thiserror::Error)]
pub enum IrError {
    // --- Elementwise RPN ---
    #[error("elementwise arg shape {arg:?} != kernel shape {kernel:?}")]
    ElementwiseArgShape { arg: Box<[u32]>, kernel: Box<[u32]> },

    #[error(
        "elementwise arg_element_types length {element_types} != arg_accessors length {accessors}"
    )]
    ElementwiseArgElementTypesLen {
        element_types: usize,
        accessors: usize,
    },

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
    #[error(
        "remap arg_element_types length {element_types} != arg_accessors length {accessors}"
    )]
    RemapArgElementTypesLen {
        element_types: usize,
        accessors: usize,
    },

    #[error("remap ({info}) requires {expected} argument(s), got {got}")]
    RemapArgCount {
        info: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("remap indices must be rank-1 U4, got shape {shape:?} element type {element_type:?}")]
    RemapIndices {
        shape: Box<[u32]>,
        element_type: resin_core::ElementType,
    },

    #[error("remap indices length {indices} != source rows {rows}")]
    RemapIndicesLen { indices: u32, rows: u32 },

    #[error("remap source/output row shapes differ: source {source_shape:?}, output {output:?}")]
    RemapRowShape {
        source_shape: Box<[u32]>,
        output: Box<[u32]>,
    },

    #[error("remap gather output rows {output} != indices length {indices}")]
    RemapGatherRows { output: u32, indices: u32 },

    #[error("remap source must have rank >= 1")]
    RemapSourceRankZero,

    #[error("scatter_view accessor shape {accessor:?} != source shape {source_shape:?}")]
    ScatterViewShape {
        accessor: Box<[u32]>,
        source_shape: Box<[u32]>,
    },

    #[error("remap scatter kernels must clear their output before dispatch")]
    RemapScatterMustClear,

    #[error("remap scatter operator {operator:?} not supported")]
    RemapScatterOperator {
        operator: resin_core::BinaryAssocElementOperator,
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
