//! Command-line argument parsing and mode dispatch.
use resin_compiler::Compiler;

mod args;
mod embed;
use resin_toolchain::Environment;
mod format;
mod request;
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

fn run(Invocation { mode, library_root }: Invocation) -> Result<i32> {
    match &mode {
        Mode::Compiler(request) | Mode::Interpreter { request, .. } => {
            let executable = compile(request, library_root)?;
            if let Mode::Interpreter { args, .. } = &mode {
                return Ok(executable.run_with_args(args)?);
            }
            Ok(0)
        }
        Mode::Embed {
            input,
            output,
            symbol,
        } => embed::run(input, output, symbol),
        Mode::Formatter { paths, check } => format::run(paths, *check),
        Mode::LanguageServer { directory } => {
            std::env::set_current_dir(directory)?;
            resin_lsp::serve(library_root)
        }
    }
}

fn compile(
    request: &request::Request,
    library_root: std::path::PathBuf,
) -> Result<resin_toolchain::Executable> {
    let mut loader = resin_source::Loader::new(library_root);
    let source = loader.load_file(&request.input.path)?;
    let compilation = Compiler::new().compile(source, &mut loader);
    let directory = tempfile::TempDir::new_in(&request.options.temporary)?;
    let project = resin_codegen::generate(
        compilation.verified()?,
        Some(&request.input.entry),
        directory.path(),
    )?;
    let built = request.options.tools.build(
        project.directory(),
        project.name(),
        &request.input.entry,
        request.options.profile,
    )?;
    let executable = built.executable(
        project
            .program()
            .expect("host output")
            .file_name()
            .expect("program filename"),
    )?;
    if let Some(output) = &request.destination {
        executable.copy_to(output)?;
    }
    Ok(executable)
}
