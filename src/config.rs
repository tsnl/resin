#[derive(Clone)]
pub struct Config {
    pub tab_spaces: u16,
    pub debug_ast: bool,
    pub output_dir: String,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            tab_spaces: 4,
            debug_ast: true,
            output_dir: "resin-out".to_string(),
        }
    }
}
