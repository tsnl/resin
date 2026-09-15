//! Command-line argument parsing and mode dispatch.
mod args;
mod embed;
mod format;
mod inputs;
mod interpreter;
mod request;
mod source;

use args::Mode;
use resin_toolchain::Environment;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Run the command-line invocation and exit with its status.
pub fn main() -> ! {
    log::set_logger(&Warnings).expect("CLI owns logging");
    log::set_max_level(log::LevelFilter::Warn);
    let ec = match try_main() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            1
        }
    };
    std::process::exit(ec);
}

struct Warnings;

impl log::Log for Warnings {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}: {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

fn try_main() -> Result<i32> {
    let env = Environment::capture()?;
    let args = args::parse(std::env::args_os(), &env)?;
    dispatch_by_mode(args)
}

fn dispatch_by_mode(mode: Mode) -> Result<i32> {
    match mode {
        Mode::Interpreter { request, args } => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(interpreter::run(&request, &args)),
        Mode::Embed {
            input,
            output,
            symbol,
        } => embed::run(&input, &output, &symbol),
        Mode::Formatter { paths, check } => format::run(&paths, check),
        Mode::LanguageServer {
            directory,
            library_root,
        } => resin_lsp::serve(directory, library_root),
    }
}
