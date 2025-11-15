use glfw::Context;

pub struct Renderer {
    glfw_instance: glfw::Glfw,
    glfw_window: glfw::PWindow,
}
pub struct RendererConfig {
    window_width: u32,
    window_height: u32,
    window_title: String,
}
impl Renderer {
    pub fn new(config: RendererConfig) -> Renderer {
        let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
        glfw.window_hint(glfw::WindowHint::Visible(false));
        let (mut window, events) = glfw
            .create_window(
                config.window_width,
                config.window_height,
                &config.window_title,
                glfw::WindowMode::Windowed,
            )
            .expect("Failed to create GLFW window");

        Renderer {
            glfw_instance: glfw,
            glfw_window: window,
        }
    }
    pub fn run(&mut self) {
        self.glfw_window.make_current();
        self.glfw_window.show();
        while !self.glfw_window.should_close() {
            self.glfw_instance.poll_events();
            self.glfw_window.swap_buffers();
        }
        eprintln!("Application closed");
    }
}
impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            window_width: 800,
            window_height: 600,
            window_title: "Unnamed App -- Built with Zero".to_string(),
        }
    }
}
