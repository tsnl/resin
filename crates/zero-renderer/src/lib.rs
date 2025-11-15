use ash::vk;
use std::sync::Arc;

pub struct Renderer {
    instance: Arc<zero_gpu::GpuManager>,
    surface: vk::SurfaceKHR,
    device: vk::Device,
}
pub struct RendererCreateInfo {
    instance: Arc<zero_gpu::GpuManager>,
    surface: vk::SurfaceKHR,
    device: vk::Device,
}
impl Renderer {
    pub fn create(create_info: RendererCreateInfo) -> Self {
        Renderer {
            instance: create_info.instance,
            surface: create_info.surface,
            device: create_info.device,
        }
    }
}
