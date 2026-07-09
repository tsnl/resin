//! IR→IR optimization passes.
//!
//! Implemented:
//!
//! - **Elementwise fusion** ([`fuse_elementwise`]) — producer→consumer chains
//!   of elementwise dispatches collapse into single kernels, to a fixed
//!   point. See [`kernel_fusion`].
//!
//! Planned:
//!
//! - **Tiled element types** — block matmul as an *elementwise* operation on
//!   16×16 tiles plus a reduction, so it schedules and fuses like everything
//!   else (and can lower to cooperative-matrix WGSL). No dedicated matmul
//!   kernel long-term.
//! - **Matmul epilogues** — fuse a trailing expression into a matmul
//!   (subsumed by the tiled representation once that lands).
//! - **Constant folding and dead-dispatch elimination.**

mod kernel_fusion;

pub use kernel_fusion::fuse_elementwise;

use super::Program;

/// Run middle-end optimization passes on `program`.
pub fn optimize(program: Program) -> Program {
    fuse_elementwise(program)
}
