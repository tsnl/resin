use resin_core::{
    Accessor, BinaryAssocElementOperator, ElementType,
};
use resin_dsl::RemapInfo;
use serde::{Deserialize, Serialize};

use crate::rpn::ElementRpnExpr;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrProgram {
    /// Stable param name → buffer index in `buffers`.
    pub param_buffers: Vec<(String, usize)>,
    pub sinks: Vec<(String, usize)>,
    pub queue: Vec<IrDispatch>,
    pub buffers: Vec<IrBuffer>,
    pub buffer_views: Vec<IrBufferView>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrBuffer {
    pub shape: Box<[u32]>,
    pub etype: ElementType,
    pub init: Option<Box<[u8]>>,
    pub readonly: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrBufferView {
    pub buffer_index: usize,
    pub accessor: Accessor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrDispatch {
    pub kernel: IrKernel,
    pub arg_view_indices: Vec<usize>,
    pub output_buffer_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrKernel {
    ElementwiseRpn(IrElementwiseRpnKernel),
    Matmul(IrMatmulKernel),
    Reduction(IrReductionKernel),
    Remap(IrRemapKernel),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrElementwiseRpnKernel {
    pub arg_accessors: Vec<Accessor>,
    pub arg_etypes: Vec<ElementType>,
    pub etype: ElementType,
    pub shape: Box<[u32]>,
    pub rpn_expr: ElementRpnExpr,
    pub clear_output_before_dispatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrMatmulKernel {
    pub arg_accessors: Vec<Accessor>,
    pub etype: ElementType,
    pub shape: Box<[u32]>,
    pub clear_output_before_dispatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrReductionKernel {
    pub arg_accessors: Vec<Accessor>,
    pub etype: ElementType,
    pub shape: Box<[u32]>,
    pub operator: BinaryAssocElementOperator,
    pub axes: Box<[u32]>,
    pub clear_output_before_dispatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrRemapKernel {
    pub arg_accessors: Vec<Accessor>,
    pub arg_etypes: Vec<ElementType>,
    pub etype: ElementType,
    pub shape: Box<[u32]>,
    pub info: RemapInfo,
    pub clear_output_before_dispatch: bool,
}
