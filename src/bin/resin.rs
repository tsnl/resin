use std::path::PathBuf;

use clap::Parser;
use resin::*;

#[derive(Parser)]
#[command(name = "resin", about = "The Resin compiler")]
struct CliConfig {
    /// Entry point file for compilation
    entry_point: PathBuf,

    /// Output directory for generated artifacts
    #[arg(short, long)]
    output_dir: Option<PathBuf>,
}
impl CliConfig {
    fn into_config(self) -> Config {
        let mut config = Config {
            entry_point: self.entry_point,
            ..Config::default()
        };
        if let Some(output_dir) = self.output_dir {
            config.output_dir = output_dir;
        }
        config
    }
}

fn main() {
    let config = CliConfig::parse().into_config();
    let source = Source::new_file(&config.entry_point, &config);
    match parser::parse_file(&source) {
        Ok(ast) => eprintln!("{ast:#?}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
