use ash::vk;

use zero_gpu::GpuContext;

pub struct Engine {
    gpu_context: Option<Arc<GpuContext>>,
    render_context: Option<Arc<RenderContext>>,
    window_context: Option<Arc<WindowContext>>,
    physics_context: Option<Arc<PhysicsContext>>,
}

pub struct WindowContext {
    window: Arc<zero_window::Window>,
    gpu_surface: Arc<vk::SurfaceKHR>,
}

pub struct PhysicsContext {
    // TODO
}
