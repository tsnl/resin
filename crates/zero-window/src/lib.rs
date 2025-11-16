use zero_prelude::*;

use zero_gpu::GpuManager;

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

pub struct WindowManager {
    config: WindowManagerConfig,
    glfw: RwLock<glfw::Glfw>,
    gpu_manager: OnceLock<Arc<GpuManager>>,
    windows: Mutex<Vec<Weak<Window>>>,
}
pub struct WindowManagerConfig {}
impl WindowManager {
    pub fn create(config: WindowManagerConfig) -> Arc<WindowManager> {
        let glfw = RwLock::new(glfw::init(glfw::fail_on_errors).unwrap());
        let gpu_manager = OnceLock::new();
        let windows = Mutex::default();
        Arc::new(WindowManager {
            config,
            glfw,
            gpu_manager,
            windows,
        })
    }
    pub fn create_window(self: &Arc<Self>, window_config: WindowConfig) -> Arc<Window> {
        let window = Window::create(window_config, self.clone());
        self.windows.lock().push(Arc::downgrade(&window));
        window
    }
    pub fn set_gpu_manager(&self, gpu_manager: Arc<GpuManager>) {
        let result = self.gpu_manager.set(gpu_manager);
        if let Err(_) = result {
            panic!("Failed to set GPU manager: was it already set?");
        };
    }
    pub fn get_vulkan_required_instance_extensions(&self) -> Vec<CString> {
        self.glfw
            .read()
            .get_required_instance_extensions()
            .unwrap()
            .into_iter()
            .map(|it| CString::new(it).unwrap())
            .collect()
    }
    fn glfw(&self) -> &RwLock<glfw::Glfw> {
        &self.glfw
    }
    pub fn gpu_manager(&self) -> &Arc<GpuManager> {
        &self.gpu_manager.get().expect("GPU manager not yet set.")
    }
    pub fn update(&self) {
        let mut glfw = self.glfw().write();
        glfw.poll_events();

        let mut window_vec_lock = self.windows.lock();
        let window_vec_mut: &mut Vec<_> = window_vec_lock.as_mut();
        let new_window_vec: Vec<_> = std::mem::take(window_vec_mut)
            .into_iter()
            .filter_map(|window| {
                if let Some(window) = window.upgrade() {
                    window.update();
                    Some(Arc::downgrade(&window))
                } else {
                    None
                }
            })
            .collect();
        _ = std::mem::replace(window_vec_lock.as_mut(), new_window_vec);
    }
}

pub struct Window {
    manager: Arc<WindowManager>,
    glfw_window: RwLock<glfw::PWindow>,
    glfw_events: glfw::GlfwReceiver<(f64, glfw::WindowEvent)>,
    vulkan_surface: vk::SurfaceKHR,
}
pub struct WindowConfig {
    pub width: u32,
    pub height: u32,
    pub title: String,
}
impl Window {
    fn create(config: WindowConfig, manager: Arc<WindowManager>) -> Arc<Self> {
        let (glfw_window, glfw_events, vulkan_surface) = {
            let mut glfw = manager.glfw.write();

            glfw.window_hint(glfw::WindowHint::Visible(false));
            glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
            let (glfw_window, glfw_events) = glfw
                .create_window(
                    config.width,
                    config.height,
                    &config.title,
                    glfw::WindowMode::Windowed,
                )
                .expect("Failed to create GLFW window");

            let vk_surface = manager.gpu_manager().create_surface(
                glfw_window.raw_display_handle().unwrap(),
                glfw_window.raw_window_handle().unwrap(),
            );

            let glfw_window = RwLock::new(glfw_window);
            (glfw_window, glfw_events, vk_surface)
        };
        Arc::new(Self {
            manager,
            glfw_window,
            glfw_events,
            vulkan_surface,
        })
    }
    fn update(&self) {
        // TODO: process input
    }
    pub fn set_visible(&self, visible: bool) {
        let mut glfw_window = self.glfw_window.write();
        if visible {
            glfw_window.show();
        } else {
            glfw_window.hide();
        }
    }
    pub fn should_close(&self) -> bool {
        let glfw_window = self.glfw_window.read();
        glfw_window.should_close()
    }
    pub fn vulkan_surface(&self) -> vk::SurfaceKHR {
        self.vulkan_surface
    }
}
