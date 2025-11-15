pub struct Renderer {
    glfw_instance: glfw::Glfw,
    glfw_window: glfw::Window,
}
pub struct RendererConfig {
    window_width: u32,
    window_height: u32,
    window_title: String,
}
impl Renderer {
    pub fn new(config: RendererConfig) -> Renderer {
        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
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
}
