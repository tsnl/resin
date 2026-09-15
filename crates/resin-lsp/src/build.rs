//! The editor's explicit disk build request. Pass order and cache selection live here.
use crate::{
    caches::{Caches, GenerationKey},
    inputs, publication,
};
use lsp_types::Uri;
use resin_executor::{Cancellation, Execution};
use resin_source::{Loader, Source, SourceError, SourceId};
use resin_toolchain::{CProfile, Toolchain};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(test)]
mod tests;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BuildRequest {
    pub uri: Uri,
    #[serde(default = "default_entry")]
    pub entry: String,
    pub destination: PathBuf,
    #[serde(default)]
    pub profile: BuildProfile,
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum BuildProfile {
    Debug,
    #[default]
    Release,
}

fn default_entry() -> String {
    "main".into()
}

pub(crate) async fn run(
    request: BuildRequest,
    library_root: &Path,
    tools: &Toolchain,
    temporary: &Path,
    caches: &Caches,
    execution: &Execution,
    cancellation: &Cancellation,
) -> crate::Result<Value> {
    let (path, destination) = execution
        .run(cancellation, {
            let request = request.clone();
            move |_| validate_paths(&request)
        })
        .await??;
    let mut loader = Loader::new(library_root.to_path_buf());
    let source = loader
        .load_file_async(&path, execution, cancellation)
        .await?;
    let captured = inputs::capture(source, &mut loader, caches, execution, cancellation).await?;
    drop(loader);
    if captured.paths.values().any(|source| source == &destination) {
        return Err("build output would overwrite a source in the import graph".into());
    }

    let documents = publication::select(
        &caches.ast,
        captured.graph.sources().cloned().collect(),
        |source| {
            let syntax = captured.syntax[&source].clone();
            async move {
                let parsed = resin_ast::build_ast(syntax.clone(), execution, cancellation).await?;
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
    let graph = captured.graph;
    let mut program =
        resin_ast::build_program(graph.clone(), documents, execution, cancellation).await?;
    let hir = if captured.diagnostics.is_empty() {
        let program = Arc::new(program);
        publication::select(
            &caches.hir,
            vec![graph.clone()],
            |_| {
                let program = program.clone();
                async move {
                    resin_hir::Hir::build(program, execution, cancellation)
                        .await
                        .map(Arc::new)
                }
            },
            execution,
            cancellation,
        )
        .await?
        .remove(&graph)
        .expect("selected HIR")
    } else {
        inputs::acquisition_diagnostics(&mut program, captured.diagnostics);
        Arc::new(resin_hir::Hir::build(Arc::new(program), execution, cancellation).await?)
    };
    let module = hir
        .hir()
        .map_err(|error| diagnostic(&error, &captured.origins))?
        .clone();
    let entry =
        resin_lir::Entry::exported(&module, request.entry.clone(), resin_lir::Profile::Host)
            .map_err(|error| error.to_string())?;
    let key = resin_lir::LirKey::new(
        Arc::new(graph),
        [entry],
        resin_lir::LoweringOptions::default(),
    );
    let verified = publication::select(
        &caches.verified,
        vec![key.clone()],
        |key| {
            let module = module.clone();
            async move {
                let lir = resin_lir::build_lir(
                    module,
                    key.entries().to_vec(),
                    key.options().clone(),
                    execution,
                    cancellation,
                )
                .await
                .map_err(|error| error.to_string())?;
                resin_lir::VerifiedModule::build(lir, execution, cancellation)
                    .await
                    .map(Arc::new)
                    .map_err(|error| error.to_string())
            }
        },
        execution,
        cancellation,
    )
    .await
    .map_err(|error| error.to_string())?
    .remove(&key)
    .expect("selected verified LIR");
    let generated_key = GenerationKey {
        lir: key,
        host_entry: Some(request.entry.clone()),
    };
    let generated = publication::select(
        &caches.generated,
        vec![generated_key.clone()],
        |key| {
            let verified = verified.clone();
            async move {
                resin_codegen::generate(
                    verified,
                    key.host_entry,
                    temporary,
                    execution,
                    cancellation,
                )
                .await
                .map(Arc::new)
            }
        },
        execution,
        cancellation,
    )
    .await?
    .remove(&generated_key)
    .expect("selected generation");
    let profile = match request.profile {
        BuildProfile::Debug => CProfile::Debug,
        BuildProfile::Release => CProfile::Release,
    };
    // Native input validation runs on every build. LIR alone is not a native cache key.
    let built = tools
        .build(
            generated.directory(),
            &path.to_string_lossy(),
            &request.entry,
            profile,
            execution,
            cancellation,
        )
        .await?;
    let executable = built.executable(
        generated
            .program()
            .expect("host program")
            .file_name()
            .expect("program filename"),
    )?;
    executable
        .copy_to(&destination, execution, cancellation)
        .await?;
    let output = url::Url::from_file_path(&destination)
        .map_err(|_| "cannot represent output as a file URI")?;
    Ok(json!({"outputUri": output.as_str()}))
}

fn validate_paths(request: &BuildRequest) -> crate::Result<(PathBuf, PathBuf)> {
    let path = url::Url::parse(request.uri.as_str())?
        .to_file_path()
        .map_err(|_| "build URI must identify a local file")?;
    let path = resin_source::normalize_path(&path)?;
    if request.entry.is_empty() || request.destination.as_os_str().is_empty() {
        return Err("build entry and destination must be nonempty".into());
    }
    let destination = path
        .parent()
        .expect("absolute source parent")
        .join(&request.destination);
    if destination.is_dir() || destination.file_name().is_none() {
        return Err("build destination must be an explicit output filename".into());
    }
    let destination = resin_source::normalize_path(&destination)?;
    if destination == path {
        return Err("build output would overwrite the source file".into());
    }
    Ok((path, destination))
}

fn diagnostic(error: &SourceError, origins: &BTreeMap<SourceId, Source>) -> String {
    let source = origins.get(&error.source.id()).unwrap_or(&error.source);
    SourceError::new(source.clone(), error.span, error.diagnostic.to_string()).to_string()
}
