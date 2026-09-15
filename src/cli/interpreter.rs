//! Host program compilation for interpreter mode.
use super::{Result, request::Request};
use resin_hir::Hir;
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

    fn lower(request: &Request) -> Result<Hir> {
        let mut loader = resin_source::Loader::new(request.library_root.clone());
        let source = loader.load_file(&request.input.path)?;
        Ok(Hir::build(source, &mut loader, None))
    }

    fn generate(output: &Hir, request: &Request) -> Result<resin_codegen::GeneratedProject> {
        let directory = request
            .options
            .tools
            .generated(output.source().name(), &request.input.entry);
        let hir = output.hir().map_err(|error| build_lir_error(vec![error]))?;
        let entry =
            resin_lir::Entry::exported(hir, request.input.entry.clone(), resin_lir::Profile::Host)
                .map_err(|error| build_lir_error(vec![source_error(output, error)]))?;
        let lir = resin_lir::build_lir(hir, &[entry], &resin_lir::LoweringOptions::default())
            .map_err(|errors| {
                build_lir_error(
                    errors
                        .into_iter()
                        .map(|error| source_error(output, error))
                        .collect(),
                )
            })?;
        let lir = resin_lir::VerifiedModule::new(lir).map_err(|error| {
            build_lir_error(vec![resin_source::SourceError::new(
                output.source().clone(),
                None,
                format!("invalid LIR: {error}"),
            )])
        })?;
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

fn build_lir_error(
    errors: Vec<resin_source::SourceError>,
) -> Box<dyn std::error::Error + Send + Sync> {
    errors
        .into_iter()
        .map(|error| error.to_string())
        .collect::<Vec<_>>()
        .join("\n")
        .into()
}

fn source_error(output: &Hir, error: resin_lir::Error) -> resin_source::SourceError {
    let source = error
        .source
        .clone()
        .unwrap_or_else(|| output.source().clone());
    resin_source::SourceError::new(source, Some(error.span), error.to_string())
}
