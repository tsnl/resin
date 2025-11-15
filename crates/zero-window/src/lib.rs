use glfw::Context;
use raw_window_handle::HasWindowHandle;
use std::sync::Arc;
use zero_renderer::GpuWindow;

pub struct Window {
    glfw: glfw::Glfw,
    glfw_window: glfw::PWindow,
    glfw_events: glfw::GlfwReceiver<(f64, glfw::WindowEvent)>,
}
pub struct WindowConfig {
    width: u32,
    height: u32,
    title: String,
}
impl Window {
    pub fn new(config: WindowConfig) -> Window {
        let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();

        glfw.window_hint(glfw::WindowHint::Visible(false));
        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
        let (mut glfw_window, glfw_events) = glfw
            .create_window(
                config.width,
                config.height,
                &config.title,
                glfw::WindowMode::Windowed,
            )
            .expect("Failed to create GLFW window");

        Window {
            glfw,
            glfw_window,
            glfw_events,
        }
    }
    pub fn run(&mut self) {
        self.glfw_window.show();
        while !self.glfw_window.should_close() {
            self.glfw.poll_events();
        }
        eprintln!("Application closed");
    }
}
impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: "Unnamed App -- Built with Zero".to_string(),
        }
    }
}
