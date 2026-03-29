use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Config {
    pub entry_point: PathBuf,
    pub output_dir: PathBuf,
    pub tab_spaces: u16,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            entry_point: PathBuf::from("main.resin"),
            output_dir: PathBuf::from("resin-build"),
            tab_spaces: 2,
        }
    }
}
