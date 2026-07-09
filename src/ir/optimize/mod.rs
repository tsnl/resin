//! IR→IR **optimization** passes (optional, may be skipped for A/B).
//!
//! Implemented:
//!
//! - **Elementwise fusion** ([`fuse_elementwise`]) — producer→consumer chains
//!   of elementwise dispatches collapse into single kernels, to a fixed
//!   point. See [`kernel_fusion`].
//!
//! Storage layout (arena packing, dead-buffer elim) is **not** an opt pass —
//! see [`crate::ir::layout::prepare_for_backend`], which always runs after
//! this stage and before JIT.
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

/// Which IR optimization passes to run (layout is separate — always applied).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OptPasses {
    /// No optimization passes.
    None,
    /// Elementwise fusion only.
    Fuse,
    /// Full optimization pipeline (currently fusion only).
    #[default]
    All,
}

/// Run the default middle-end optimization pipeline ([`OptPasses::All`]).
///
/// Does **not** pack arenas; call [`crate::ir::layout::prepare_for_backend`]
/// after this before backend lower.
pub fn optimize(program: Program) -> Program {
    optimize_with(program, OptPasses::All)
}

/// Run a selected set of middle-end optimization passes.
pub fn optimize_with(program: Program, passes: OptPasses) -> Program {
    match passes {
        OptPasses::None => program,
        OptPasses::Fuse | OptPasses::All => fuse_elementwise(program),
    }
}
