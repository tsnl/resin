#[derive(Clone)]
pub struct Config {
    pub tab_spaces: u16,
    pub debug_ast: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            tab_spaces: 4,
            debug_ast: false,
        }
    }
}
