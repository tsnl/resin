pub struct Config {
    pub tab_spaces: u16,
}
impl Default for Config {
    fn default() -> Self {
        Self { tab_spaces: 4 }
    }
}
