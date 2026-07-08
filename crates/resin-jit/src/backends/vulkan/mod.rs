//! Native Vulkan backend ([`VulkanJit`]): the same naive WGSL kernels as the
//! wgpu backend, translated to SPIR-V through naga and driven with `ash`.
//!
//! Rationale (see `docs/hw-nodes.md`): one shared kernel emitter, two
//! transports. wgpu remains the portable path; this backend talks to Vulkan
//! directly for fixed-function access (`VK_KHR_ray_query` acceleration
//! structures, render passes) and native deployment. Requires a Vulkan 1.2
//! driver; ray tracing is used opportunistically when
//! `VK_KHR_acceleration_structure` + `VK_KHR_ray_query` are present.

mod error;
mod jit;
mod lower;
mod program;
mod raster;
mod runtime;
mod tensor;
mod trace;

pub use error::{VulkanLowerError, VulkanRuntimeError};
pub use jit::VulkanJit;
pub use program::VulkanProgram;
pub use tensor::VulkanTensor;

/// Whether a Vulkan device is available (for tests / graceful skip).
pub fn shared_context_available() -> bool {
    runtime::shared_context().is_ok()
}

/// Whether the shared Vulkan device supports hardware ray queries.
pub fn ray_query_available() -> bool {
    runtime::shared_context()
        .map(|ctx| ctx.ray_query)
        .unwrap_or(false)
}
