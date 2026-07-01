use resin_core::{Accessor, BinaryAssocElementOperator, ElementType};
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

impl IrKernel {
    pub fn shape(&self) -> &[u32] {
        match self {
            Self::ElementwiseRpn(k) => &k.shape,
            Self::Matmul(k) => &k.shape,
            Self::Reduction(k) => &k.shape,
            Self::Remap(k) => &k.shape,
        }
    }

    pub fn etype(&self) -> ElementType {
        match self {
            Self::ElementwiseRpn(k) => k.etype,
            Self::Matmul(k) => k.etype,
            Self::Reduction(k) => k.etype,
            Self::Remap(k) => k.etype,
        }
    }

    pub fn arg_accessors(&self) -> &[Accessor] {
        match self {
            Self::ElementwiseRpn(k) => &k.arg_accessors,
            Self::Matmul(k) => &k.arg_accessors,
            Self::Reduction(k) => &k.arg_accessors,
            Self::Remap(k) => &k.arg_accessors,
        }
    }

    pub fn clear_output_before_dispatch(&self) -> bool {
        match self {
            Self::ElementwiseRpn(k) => k.clear_output_before_dispatch,
            Self::Matmul(k) => k.clear_output_before_dispatch,
            Self::Reduction(k) => k.clear_output_before_dispatch,
            Self::Remap(k) => k.clear_output_before_dispatch,
        }
    }

    pub fn operand_etypes(&self) -> Vec<ElementType> {
        match self {
            Self::ElementwiseRpn(k) => k.arg_etypes.clone(),
            Self::Matmul(k) => vec![k.etype, k.etype],
            Self::Reduction(k) => vec![k.etype],
            Self::Remap(k) => k.arg_etypes.clone(),
        }
    }
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
