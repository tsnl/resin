//! Final IR layout for backends — not an optimization pass.
//!
//! Pipeline position (always runs, even when opt is off):
//!
//! ```text
//! lower → optimize? → prepare_for_backend → jit
//!                      └─ eliminate_dead
//!                      └─ pack_arenas
//! ```
//!
//! Densify sinks first so strided outputs get a dense buffer, then dead-elim so
//! packing only sees live buffers/views. Packing is required for the WGSL
//! binding model: one heap per `(dtype, atomic)` key (plain vs RMW targets),
//! regardless of fusion.

mod arena_pack;
mod dead_elim;
mod densify_sinks;

pub use arena_pack::pack_arenas;
pub use dead_elim::eliminate_dead;
pub use densify_sinks::densify_sinks;

use super::Program;

/// Densify strided sinks, tree-shake, then pack into per-dtype arenas. Call
/// after all optimization passes and before any backend lower/invoke. Every
/// sink view is dense afterwards, so backends copy each out as one block.
pub fn prepare_for_backend(program: Program) -> Program {
    pack_arenas(eliminate_dead(densify_sinks(program)))
}
