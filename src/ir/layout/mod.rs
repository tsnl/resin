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
//! Sink densification happens in [`crate::ir::lower`], not here, so optimize
//! can see identity densify copies. Layout tree-shakes then packs: packing is
//! required for the WGSL binding model (one heap per `(dtype, atomic)` key).

mod arena_pack;
mod dead_elim;

pub use arena_pack::pack_arenas;
pub use dead_elim::eliminate_dead;

use super::Program;

/// Tree-shake, then pack into per-dtype arenas. Call after all optimization
/// passes and before any backend lower/invoke. Sinks are already dense from
/// [`crate::ir::lower`]; backends copy each out as one block.
pub fn prepare_for_backend(program: Program) -> Program {
    pack_arenas(eliminate_dead(program))
}
