//! Command-line argument parsing and mode dispatch.
use crate::backend;

mod args;
mod format;
mod inspect;
mod output;
mod source;

use args::Mode;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Run the command-line invocation and exit with its status.
pub fn main() -> ! {
    std::process::exit(match args::parse().and_then(run) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            1
        }
    });
}

fn run(mode: Mode) -> Result<i32> {
    match &mode {
        Mode::Compiler(request) | Mode::Interpreter(request) => {
            let executable = backend::compile(request)?;
            if matches!(mode, Mode::Interpreter(_)) {
                return Ok(executable.run()?);
            }
            Ok(0)
        }
        Mode::Codegen { request, target } => {
            let bytes = backend::generate(request, *target)?;
            output::write(&bytes, request.destination.as_deref())
        }
        Mode::Inspector {
            input,
            output,
            destination,
        } => inspect::run(input, *output, destination.as_deref()),
        Mode::Formatter { paths, check } => format::run(paths, *check),
    }
}
