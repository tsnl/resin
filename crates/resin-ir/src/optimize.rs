//! IR→IR optimization passes.
//!
//! Ported from the Python `resin.ir.ir_opt` module notes.
//!
//! ---
//!
//! # Ideas for optimizations
//!
//! ## Kernel fusion and scheduling
//!
//! - **Elementwise RPN fusion**  
//!   If we have a sequence of elementwise operations, we can fuse them into a
//!   single kernel.
//!
//! - **Matmul + RPN epilogue fused kernel**  
//!   Support an RPN scalar epilogue after each matmul. Possibly also RPN
//!   prologue for each argument. But epilogue first.
//!
//! - **Tiled ElementTypes for performance**  
//!   Extend `ElementType` with tile types (e.g. `"mat4x4_f4"`) so block matmul
//!   can use cooperative matrix ops and fuse more cleanly with surrounding
//!   kernels. High-level matmul decomposes into elementwise tiled ops,
//!   reductions, and stride-tricks — easier to schedule than a dedicated
//!   matmul kernel plus epilogue.
//!
//!   **No explicit high-level matmul (long term)**  
//!   Matmul can be expressed using elementwise tiled matmul operations,
//!   reduction, and stride-tricks. This makes scheduling and kernel fusion much
//!   easier. Instead of Matmul + epilogue, we can have reduction + epilogue or
//!   prologue + reduction + epilogue with matmul an elementwise operation on
//!   matrix tiles.
//!
//! ## Constant folding
//!
//! - Constant folding
//! - Constant inlining into RPN expressions, matmul, scatter, gather
//! - Dead code elimination
//!
//! ## Big-picture ideas
//!
//! - **Big symbolic scalar graph**  
//!   A totally different approach: imagine each value in a matrix is a single
//!   scalar symbolic variable. We effectively get a huge scalar formula. How
//!   can we group, schedule, and allot memory for these scalar formulae, using
//!   the fewest kernels possible? This throws out the user's conception of what
//!   is a tensor. Pretty radical, needs the most thought.
//!
//! ## Emitter-side optimizations
//!
//! - **Use SPV extensions** like `SPV_NV_tensor_addressing` to more efficiently
//!   access tensor elements up to 5D.  
//!   <https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_tensor_addressing.html>

use resin_core::Tree;

use crate::program::IrProgram;
use crate::refs::{BufferRef, BufferViewRef};

/// Run middle-end optimization passes on `program`.
///
/// Today this is a no-op placeholder for kernel fusion, dead-buffer elimination,
/// constant folding, and scheduling (see module docs).
pub fn optimize<P: Tree<BufferRef>, S: Tree<BufferViewRef>>(
    program: IrProgram<P, S>,
) -> IrProgram<P, S> {
    // TODO: kernel fusion, dead-buffer elimination, etc.
    program
}
