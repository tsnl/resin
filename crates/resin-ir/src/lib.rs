//! Intermediate representation: optimized kernel queue over device buffers.

mod error;
mod hw;
mod optimize;
mod program;
mod refs;
mod remap;
mod rpn;

pub use error::IrError;
pub use hw::{IrRasterizeKernel, IrTraceRaysKernel};
pub use optimize::optimize;
pub use program::{
    IrBuffer, IrBufferView, IrDispatch, IrElementwiseRpnKernel, IrKernel, IrMatmulKernel,
    IrProgram, IrReductionKernel, IrRemapKernel,
};
pub use refs::{BufferRef, BufferViewRef};
pub use remap::RemapInfo;
pub use rpn::{ElementRpnExpr, RpnAtom};
