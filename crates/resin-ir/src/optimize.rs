//! IR→IR optimization passes.

use resin_core::Tree;

use crate::program::IrProgram;
use crate::refs::{BufferRef, BufferViewRef};

/// Run middle-end optimization passes on `program`.
///
/// Today this is a no-op placeholder for kernel fusion, dead-buffer elimination,
/// constant folding, and scheduling.
pub fn optimize<P: Tree<BufferRef>, S: Tree<BufferViewRef>>(
    program: IrProgram<P, S>,
) -> IrProgram<P, S> {
    program
}