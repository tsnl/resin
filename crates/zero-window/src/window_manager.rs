use super::*;

pub struct WindowManager {
    config: WindowManagerConfig,
    glfw: RwLock<glfw::Glfw>,
    gpu_manager: OnceLock<Arc<GpuManager>>,
    windows: Mutex<Vec<Weak<Window>>>,
}
pub struct WindowManagerConfig {
    // todo
}
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
    pub fn glfw(&self) -> &RwLock<glfw::Glfw> {
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
