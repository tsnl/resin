//! IR→IR optimization passes.
//!
//! Today this is a no-op placeholder. Planned passes, roughly in order:
//!
//! - **Elementwise RPN fusion** — a chain of elementwise dispatches whose
//!   intermediate buffers have no other readers can be spliced into a single
//!   RPN expression, to a fixed point.
//! - **Tiled element types** — block matmul as an *elementwise* operation on
//!   16×16 tiles plus a reduction, so it schedules and fuses like everything
//!   else (and can lower to cooperative-matrix WGSL). No dedicated matmul
//!   kernel long-term.
//! - **Matmul epilogues** — fuse a trailing RPN expression into a matmul
//!   (subsumed by the tiled representation once that lands).
//! - **Constant folding and dead-dispatch elimination.**

use super::Program;

/// Run middle-end optimization passes on `program`.
pub fn optimize(program: Program) -> Program {
    program
}
