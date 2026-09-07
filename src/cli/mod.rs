//! Command-line argument parsing and mode dispatch.
use crate::{backend, compiler::Session};

mod args;
mod environment;
pub use environment::Environment;
mod format;
mod inspect;
mod output;
mod source;

use args::{Invocation, Mode};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Run the command-line invocation and exit with its status.
pub fn main() -> ! {
    std::process::exit(
        match Environment::capture()
            .map_err(Into::into)
            .and_then(|environment| args::parse(std::env::args_os(), &environment))
            .and_then(run)
        {
            Ok(code) => code,
            Err(error) => {
                eprintln!("{error}");
                1
            }
        },
    );
}

fn run(Invocation { mode, stdlib }: Invocation) -> Result<i32> {
    match &mode {
        Mode::Compiler(request) | Mode::Interpreter(request) => {
            let mut session = Session::new(stdlib);
            let artifact = session.compile(request)?;
            if matches!(mode, Mode::Interpreter(_)) {
                return Ok(artifact.run()?);
            }
            match artifact {
                backend::Artifact::Executable(_) => Ok(0),
                backend::Artifact::Bytes(bytes) => output::write(&bytes, request.destination()),
            }
        }
        Mode::Inspector {
            input,
            output,
            destination,
        } => inspect::run(
            &mut Session::new(stdlib),
            input,
            *output,
            destination.as_deref(),
        ),
        Mode::Formatter { paths, check } => format::run(paths, *check),
    }
}
