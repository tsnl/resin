//! Host program compilation for interpreter mode.
use super::{Result, request::Request};
use resin_frontend::{FrontendOutput, Target};
use std::ffi::OsString;

pub(super) fn run(request: &Request, args: &[OsString]) -> Result<i32> {
    let executable = Interpreter::compile(request)?;
    if let Some(output) = &request.destination {
        executable.copy_to(output)?;
        Ok(0)
    } else {
        Ok(executable.run_with_args(args)?)
    }
}

struct Interpreter;

impl Interpreter {
    fn compile(request: &Request) -> Result<resin_toolchain::Executable> {
        let output = Self::lower(request)?;
        let project = Self::generate(&output, request)?;
        Self::build(&project, request)
    }

    fn lower(request: &Request) -> Result<std::sync::Arc<FrontendOutput>> {
        let mut loader = resin_source::Loader::new(request.library_root.clone());
        let source = loader.load_file(&request.input.path)?;
        Ok(resin_frontend::Frontend::new().analyze(source, &mut loader))
    }

    fn generate(
        output: &FrontendOutput,
        request: &Request,
    ) -> Result<resin_codegen::GeneratedProject> {
        let directory = request
            .options
            .tools
            .generated(output.source().name(), &request.input.entry);
        let lir = output
            .instantiate(&[Target::Host {
                entry: request.input.entry.clone().into(),
            }])
            .map_err(instantiate_error)?;
        Ok(resin_codegen::generate(
            lir.view(),
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

fn instantiate_error(
    errors: Vec<resin_source::SourceError>,
) -> Box<dyn std::error::Error + Send + Sync> {
    errors
        .into_iter()
        .map(|error| error.to_string())
        .collect::<Vec<_>>()
        .join("\n")
        .into()
}
