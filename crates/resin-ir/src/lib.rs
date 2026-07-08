//! Intermediate representation: optimized kernel queue over device buffers.

mod optimize;
mod program;
mod refs;
mod remap;
mod rpn;

pub use optimize::optimize;
pub use program::{
    IrBuffer, IrBufferView, IrDispatch, IrElementwiseRpnKernel, IrKernel, IrMatmulKernel,
    IrProgram, IrReductionKernel, IrRemapKernel,
};
pub use refs::{BufferRef, BufferViewRef};
pub use remap::{RemapGatherInfo, RemapInfo, RemapScatterInfo};
pub use rpn::{ElementRpnExpr, RpnAtom};