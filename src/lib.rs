use zero_prelude::*;

use zero_gpu::{GpuManager, GpuManagerConfig};
use zero_render::{RenderManager, RenderManagerConfig};
use zero_window::{WindowManager, WindowManagerConfig};

pub struct Engine {
    gpu_manager: Arc<GpuManager>,
    window_manager: Option<Arc<WindowManager>>,
    render_manager: Option<Arc<RenderManager>>,
}
#[repr(C)]
#[derive(Clone)]
pub struct Config {
    pub require_window_support: bool,
    pub require_render_support: bool,
    pub debug_mode: bool,
}
impl Engine {
    pub fn new(config: Config) -> Self {
        let window_manager = if config.require_window_support {
            Some(Self::create_window_manager(&config))
        } else {
            None
        };

        let gpu_manager = Self::create_gpu_manager(&config, window_manager.as_ref());

        if let Some(window_manager) = window_manager.as_ref() {
            window_manager.set_gpu_manager(gpu_manager.clone());
        }

        Engine {
            gpu_manager,
            window_manager,
            render_manager: None,
        }
    }
    fn create_window_manager(_: &Config) -> Arc<WindowManager> {
        let window_manager_config = WindowManagerConfig {};
        WindowManager::create(window_manager_config)
    }
    fn create_gpu_manager(
        config: &Config,
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
    pub fn run(&self) {
        todo!()
    }
}

#[unsafe(no_mangle)]
extern "C" fn config(
    require_window_support: bool,
    require_render_support: bool,
    debug_mode: bool,
) -> Config {
    Config {
        require_window_support,
        require_render_support,
        debug_mode,
    }
}

#[unsafe(no_mangle)]
extern "C" fn init(config: Config) -> *mut Engine {
    let engine = Engine::new(config);
    Box::into_raw(Box::new(engine))
}

#[unsafe(no_mangle)]
extern "C" fn run(engine: *mut Engine) {
    let engine = unsafe { Box::from_raw(engine) };
    engine.run();
}
