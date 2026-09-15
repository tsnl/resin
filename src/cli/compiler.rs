//! Host program compilation for compiler mode.
use super::{Result, request::Request};
use std::ffi::OsString;

pub(super) fn run(request: &Request, args: &[OsString]) -> Result<i32> {
    let executable = Compiler::compile(request)?;
    if let Some(output) = &request.destination {
        executable.copy_to(output)?;
        Ok(0)
    } else {
        Ok(executable.run_with_args(args)?)
    }
}

struct Compiler;

impl Compiler {
    fn compile(request: &Request) -> Result<resin_toolchain::Executable> {
        let compilation = Self::lower(request)?;
        let project = Self::generate(&compilation, request)?;
        Self::build(&project, request)
    }

    fn lower(request: &Request) -> Result<std::sync::Arc<resin_frontend::Compilation>> {
        let mut loader = resin_source::Loader::new(request.library_root.clone());
        let source = loader.load_file(&request.input.path)?;
        Ok(resin_frontend::Frontend::new().compile(
            source,
            &mut loader,
            &[resin_frontend::Target::Host {
                entry: request.input.entry.clone().into(),
            }],
        ))
    }

    fn generate(
        compilation: &resin_frontend::Compilation,
        request: &Request,
    ) -> Result<resin_codegen::GeneratedProject> {
        let directory = request
            .options
            .tools
            .generated(compilation.source().name(), &request.input.entry);
        Ok(resin_codegen::generate(
            compilation.verified()?,
            Some(&request.input.entry),
            &directory,
        )?)
    }

    fn build(
        project: &resin_codegen::GeneratedProject,
        request: &Request,
    ) -> Result<resin_toolchain::Executable> {
        let built = request.options.tools.build(
            project.directory(),
            project.name(),
            &request.input.entry,
            request.options.profile,
        )?;
        Ok(built.executable(
            project
                .program()
                .expect("host output")
                .file_name()
                .expect("program filename"),
        )?)
    }
}
