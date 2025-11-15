use ash::vk;

pub struct Engine {
    gpu_context: Option<Arc<GpuContext>>,
    renderer: Option<Arc<Renderer>>,
    window_context: Option<Arc<WindowContext>>,
    physics_context: Option<Arc<PhysicsContext>>,
}

pub struct Renderer {}

pub struct WindowContext {
    window: Arc<zero_window::Window>,
    gpu_surface: Arc<vk::SurfaceKHR>,
}

pub struct PhysicsContext {
    // TODO
}
