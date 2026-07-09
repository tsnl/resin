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
//!   tile-shaped atoms plus a reduction, so it schedules and fuses like
//!   everything else (and can lower to cooperative-matrix WGSL).
//! - **Matmul epilogues** — fuse a trailing expression into a matmul
//!   (subsumed by the tiled representation once that lands).
//! - **Constant folding and dead-dispatch elimination.**

mod kernel_fusion;

pub use kernel_fusion::fuse_elementwise;

use super::Program;

/// Which IR optimization passes to run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OptPasses {
    /// No IR passes — identity (for A/B benchmarks).
    None,
    /// Elementwise fusion only.
    Fuse,
    /// Full pipeline (currently fusion only).
    #[default]
    All,
}

/// Run the default middle-end pipeline ([`OptPasses::All`]).
pub fn optimize(program: Program) -> Program {
    optimize_with(program, OptPasses::All)
}

/// Run a selected set of middle-end passes.
pub fn optimize_with(program: Program, passes: OptPasses) -> Program {
    match passes {
        OptPasses::None => program,
        OptPasses::Fuse | OptPasses::All => fuse_elementwise(program),
    }
}
