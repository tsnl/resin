//! JIT execution backends, enabled via crate features.

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "wgpu")]
pub mod wgpu;
