use super::*;

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

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
    pub fn create(config: WindowConfig, manager: Arc<WindowManager>) -> Arc<Self> {
        let (glfw_window, glfw_events, vulkan_surface) = {
            let mut glfw = manager.glfw().write();

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
    pub fn update(&self) {
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
