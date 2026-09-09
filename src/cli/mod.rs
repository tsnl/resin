//! Command-line argument parsing and mode dispatch.
use resin_compiler::Session;

mod args;
use resin_toolchain::Environment;
mod format;
mod source;

use args::{Invocation, Mode};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

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
        Mode::Compiler(request) | Mode::Interpreter { request, .. } => {
            let mut session = Session::new(stdlib);
            let executable = session.compile(request)?;
            if let Mode::Interpreter { args, .. } = &mode {
                return Ok(executable.run_with_args(args)?);
            }
            Ok(0)
        }
        Mode::Formatter { paths, check } => format::run(paths, *check),
        Mode::LanguageServer { directory } => {
            std::env::set_current_dir(directory)?;
            resin_lsp::serve(stdlib)
        }
    }
}
