//! Command-line argument parsing and mode dispatch.
mod args;
mod embed;
mod format;
mod interpreter;
mod request;
mod source;

use args::Mode;
use resin_toolchain::Environment;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Run the command-line invocation and exit with its status.
pub fn main() -> ! {
    let ec = match try_main() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            1
        }
    };
    std::process::exit(ec);
}

fn try_main() -> Result<i32> {
    let env = Environment::capture()?;
    let args = args::parse(std::env::args_os(), &env)?;
    dispatch_by_mode(args)
}

fn dispatch_by_mode(mode: Mode) -> Result<i32> {
    match mode {
        Mode::Interpreter { request, args } => interpreter::run(&request, &args),
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
