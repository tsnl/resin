//! Intermediate representation: DSL graph → index-based program.

mod program;
mod rpn;

pub use program::{
    IrBuffer, IrBufferView, IrDispatch, IrElementwiseRpnKernel, IrKernel, IrMatmulKernel,
    IrProgram, IrReductionKernel, IrRemapKernel,
};
pub use rpn::{ElementRpnExpr, RpnAtom};
