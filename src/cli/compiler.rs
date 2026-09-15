//! Host program compilation for interpreter mode.
use super::{Result, request::Request};
use resin_compiler::Compiler;
use std::ffi::OsString;

pub(super) fn run(request: &Request, args: &[OsString]) -> Result<i32> {
    let executable = compile(request)?;
    if request.destination.is_none() {
        return Ok(executable.run_with_args(args)?);
    }
    Ok(0)
}

fn compile(request: &Request) -> Result<resin_toolchain::Executable> {
    let mut loader = resin_source::Loader::new(request.library_root.clone());
    let source = loader.load_file(&request.input.path)?;
    let compilation = Compiler::new().compile(source, &mut loader, &request.targets);
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
