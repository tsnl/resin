use zero_prelude::*;

pub struct RenderManager {
    config: RenderManagerConfig,
    gpu_manager: Arc<zero_gpu::GpuManager>,
}
pub struct RenderManagerConfig {}
impl RenderManager {
    pub fn create(config: RenderManagerConfig, gpu_manager: Arc<zero_gpu::GpuManager>) -> Self {
        RenderManager {
            config,
            gpu_manager,
        }
    }
}
