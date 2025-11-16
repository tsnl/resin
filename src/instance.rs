use super::*;

pub struct Instance {
    gpu_manager: Arc<GpuManager>,
    window_manager: Option<Arc<WindowManager>>,
    render_manager: Option<Arc<RenderManager>>,
}
#[derive(Clone)]
pub struct InstanceConfig {
    pub require_window_support: bool,
    pub require_render_support: bool,
    pub debug_mode: bool,
}
impl Instance {
    pub fn new(config: InstanceConfig) -> Self {
        let window_manager = if config.require_window_support {
            Some(Self::create_window_manager(&config))
        } else {
            None
        };

        let gpu_manager = Self::create_gpu_manager(&config, window_manager.as_ref());

        if let Some(window_manager) = window_manager.as_ref() {
            window_manager.set_gpu_manager(gpu_manager.clone());
        }

        let render_manager = if config.require_render_support {
            Some(Self::create_render_manager(&config, gpu_manager.clone()))
        } else {
            None
        };

        Instance {
            gpu_manager,
            window_manager,
            render_manager,
        }
    }
    fn create_window_manager(_: &InstanceConfig) -> Arc<WindowManager> {
        let window_manager_config = WindowManagerConfig {};
        WindowManager::create(window_manager_config)
    }
    fn create_gpu_manager(
        config: &InstanceConfig,
        window_manager: Option<&Arc<WindowManager>>,
    ) -> Arc<GpuManager> {
        let mut extensions = Vec::default();
        let mut layers = Vec::default();
        let mut require_surface_support = false;

        if config.require_window_support {
            let window_manager = window_manager.unwrap();
            extensions.extend(
                window_manager
                    .get_vulkan_required_instance_extensions()
                    .into_iter(),
            );
            require_surface_support = true;
        }
        if config.debug_mode {
            extensions.push(CString::new("VK_EXT_debug_utils".to_string()).unwrap());
            extensions.push(CString::new("VK_EXT_debug_report".to_string()).unwrap());
            layers.push(CString::new("VK_LAYER_KHRONOS_validation".to_string()).unwrap());
        }

        let gpu_manager_config = GpuManagerConfig {
            extensions,
            layers,
            require_surface_support,
        };
        GpuManager::create(gpu_manager_config)
    }
    fn create_render_manager(
        config: &InstanceConfig,
        gpu_manager: Arc<GpuManager>,
    ) -> Arc<RenderManager> {
        todo!("Implement create_render_manager")
    }
}
impl Instance {
    pub fn gpu_manager(&self) -> &Arc<GpuManager> {
        &self.gpu_manager
    }
    pub fn window_manager(&self) -> &Arc<WindowManager> {
        self.window_manager.as_ref().unwrap()
    }
    pub fn render_manager(&self) -> &Arc<RenderManager> {
        self.render_manager.as_ref().unwrap()
    }
}
impl Instance {
    pub fn run(&self) {
        todo!()
    }
}
