use resin::backend;

#[path = "resin/args.rs"]
mod args;
#[path = "resin/format.rs"]
mod format;
#[path = "resin/inspect.rs"]
mod inspect;
#[path = "resin/output.rs"]
mod output;
#[path = "resin/source.rs"]
mod source;

use args::Mode;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() {
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
