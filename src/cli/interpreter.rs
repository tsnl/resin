//! Host program compilation for interpreter mode.
use super::{Result, inputs, request::Request};
use resin_cache::Cache;
use resin_executor::{Cancellation, Execution};
use resin_hir::Hir;
use resin_source::{Source, SourceId};
use std::{collections::BTreeMap, ffi::OsString, sync::Arc};

pub(super) async fn run(request: &Request, args: &[OsString]) -> Result<i32> {
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let result = run_request(request, args, &execution, &cancellation).await;
    cancellation.cancel();
    execution.wait_idle().await;
    result
}

async fn run_request(
    request: &Request,
    args: &[OsString],
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<i32> {
    let (output, origins) = analyze(request, execution, cancellation).await?;
    let hir = output
        .hir()
        .map_err(|error| display_error(&error, &origins))?;
    let entry =
        resin_lir::Entry::exported(hir, request.input.entry.clone(), resin_lir::Profile::Host)
            .map_err(|error| display_error(&source_error(&output, error), &origins))?;
    let lir = resin_lir::build_lir(
        hir.clone(),
        vec![entry],
        resin_lir::LoweringOptions::default(),
        execution,
        cancellation,
    )
    .await
    .map_err(|error| lowering_error(&output, error, &origins))?;
    let lir = resin_lir::VerifiedModule::build(lir, execution, cancellation).await?;
    let project = resin_codegen::generate(
        Arc::new(lir),
        Some(request.input.entry.clone()),
        &request.options.temporary,
        execution,
        cancellation,
    )
    .await?;
    let built = request
        .options
        .tools
        .build(
            project.directory(),
            &request.input.path.to_string_lossy(),
            &request.input.entry,
            request.options.profile,
            execution,
            cancellation,
        )
        .await?;
    let executable = built.executable(
        project
            .program()
            .expect("host output")
            .file_name()
            .expect("program filename"),
    )?;
    if let Some(output) = &request.destination {
        executable.copy_to(output, execution, cancellation).await?;
        Ok(0)
    } else {
        Ok(executable
            .run_with_args(args, execution, cancellation)
            .await?)
    }
}

async fn analyze(
    request: &Request,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<(Hir, BTreeMap<SourceId, Source>)> {
    let mut loader = resin_source::Loader::new(request.library_root.clone());
    let source = loader
        .load_file_async(&request.input.path, execution, cancellation)
        .await?;
    let inputs = inputs::capture(
        source,
        &mut loader,
        &Cache::new(4096),
        execution,
        cancellation,
    )
    .await?;
    let documents = Cache::new(4096)
        .update(
            inputs.graph.sources().cloned(),
            |source| {
                let syntax = inputs.syntax.get(&source).expect("captured syntax").clone();
                async move {
                    let parsed =
                        resin_ast::build_ast(syntax.clone(), execution, cancellation).await?;
                    Ok::<_, resin_executor::Error>(Arc::new(resin_ast::ModuleDocument {
                        source,
                        syntax,
                        file: Arc::new(parsed.file),
                        errors: parsed.errors,
                    }))
                }
            },
            execution,
            cancellation,
        )
        .await?;
    let selected = inputs
        .graph
        .sources()
        .map(|source| {
            (
                source.clone(),
                documents
                    .get(source)
                    .expect("requested AST retained")
                    .clone(),
            )
        })
        .collect();
    let mut program =
        resin_ast::build_program(inputs.graph, selected, execution, cancellation).await?;
    inputs::acquisition_diagnostics(&mut program, inputs.diagnostics);
    Ok((
        Hir::build(Arc::new(program), execution, cancellation).await?,
        inputs.origins,
    ))
}

fn lowering_error(
    output: &Hir,
    error: resin_lir::BuildError,
    origins: &BTreeMap<SourceId, Source>,
) -> Box<dyn std::error::Error + Send + Sync> {
    match error {
        resin_lir::BuildError::Execution { error } => error.into(),
        resin_lir::BuildError::Diagnostics { errors } => errors
            .into_iter()
            .map(|error| display_error(&source_error(output, error), origins))
            .collect::<Vec<_>>()
            .join("\n")
            .into(),
    }
}

fn source_error(output: &Hir, error: resin_lir::Error) -> resin_source::SourceError {
    let source = error
        .source
        .clone()
        .unwrap_or_else(|| output.source().clone());
    resin_source::SourceError::new(source, Some(error.span), error.to_string())
}

fn display_error(
    error: &resin_source::SourceError,
    origins: &BTreeMap<SourceId, Source>,
) -> String {
    let source = origins.get(&error.source.id()).unwrap_or(&error.source);
    let mut message = format!(
        "{}: {}",
        source_position(source, error.span),
        error.diagnostic
    );
    for note in &error.related {
        let source = origins
            .get(&note.location.source.id())
            .unwrap_or(&note.location.source);
        message.push_str(&format!(
            "\n  {}: {}",
            source_position(source, Some(note.location.span)),
            note.message
        ));
    }
    message
}

fn source_position(source: &Source, span: Option<resin_source::Span>) -> String {
    let Some(span) = span else {
        return source.name().into();
    };
    let mut start = span.start.min(source.text().len());
    while !source.text().is_char_boundary(start) {
        start -= 1;
    }
    let prefix = &source.text()[..start];
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let column = prefix.rsplit('\n').next().unwrap().chars().count() + 1;
    format!("{}:{line}:{column}", source.name())
}
