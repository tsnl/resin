//! Test-fixture entry points. Production compiler passes are asynchronous and never
//! acquire source files; these helpers keep synchronous language assertions concise.
#![allow(dead_code)]
use resin_executor::{Cancellation, Execution};
use resin_source::{ImportBinding, Loader, Source, SourceGraph};
use std::{
    collections::BTreeMap,
    future::Future,
    path::Path,
    sync::{Arc, OnceLock},
};

pub fn block_on<F: Future>(future: F) -> F::Output {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap()
        })
        .block_on(future)
}

pub fn execution() -> &'static Execution {
    static EXECUTION: OnceLock<Execution> = OnceLock::new();
    EXECUTION.get_or_init(Execution::default)
}

pub fn cst(text: impl Into<String>, previous: Option<&resin_cst::Document>) -> resin_cst::Document {
    block_on(resin_cst::build_cst(
        text.into(),
        previous,
        execution(),
        &Cancellation::new(),
    ))
    .unwrap()
}

pub fn ast(document: &resin_cst::Document) -> resin_ast::Parsed {
    block_on(resin_ast::build_ast(
        Arc::new(document.clone()),
        execution(),
        &Cancellation::new(),
    ))
    .unwrap()
}

pub fn check_hir(program: &resin_ast::Program) -> resin_hir::CheckedProgram {
    block_on(resin_hir::build_hir(
        Arc::new(program.clone()),
        execution(),
        &Cancellation::new(),
    ))
    .unwrap()
}

pub fn lower(
    module: &resin_hir::Module,
    entries: &[resin_lir::Entry],
    options: &resin_lir::LoweringOptions,
) -> Result<resin_lir::Module, Vec<resin_lir::Error>> {
    block_on(resin_lir::build_lir(
        Arc::new(module.clone()),
        entries.to_vec(),
        options.clone(),
        execution(),
        &Cancellation::new(),
    ))
    .map_err(|error| match error {
        resin_lir::BuildError::Diagnostics { errors } => errors,
        error => panic!("test execution failed: {error}"),
    })
}

pub fn generate(
    module: resin_lir::Verified<'_>,
    entry: Option<&str>,
    parent: &Path,
) -> Result<resin_codegen::GeneratedProject, resin_codegen::GenerationError> {
    let module = resin_lir::VerifiedModule::new(module.module().clone()).unwrap();
    block_on(resin_codegen::generate(
        Arc::new(module),
        entry.map(str::to_owned),
        std::sync::Arc::new(resin_codegen::NativeHeaders::default()),
        parent,
        execution(),
        &Cancellation::new(),
    ))
}

pub fn analyze(
    source: Source,
    loader: &mut Loader,
    previous: Option<&resin_hir::Hir>,
) -> resin_hir::Hir {
    block_on(async {
        let cancellation = Cancellation::new();
        let built = Arc::new(capture(source, loader, &cancellation).await);
        // Filesystem errors are request-specific; they do not form reusable cache values.
        if !built.diagnostics.is_empty() {
            return resin_hir::Hir::build(built, execution(), &cancellation)
                .await
                .unwrap();
        }
        let mut cache = resin_cache::Cache::new(2);
        if let Some(previous) = previous.filter(|previous| previous.inputs().diagnostics.is_empty())
        {
            cache = cache
                .update(
                    [previous.inputs().inputs.clone()],
                    |_| async { Ok::<_, resin_executor::Error>(Arc::new(previous.clone())) },
                    execution(),
                    &cancellation,
                )
                .await
                .unwrap();
        }
        let key = built.inputs.clone();
        let cache = cache
            .update(
                [key.clone()],
                |_| async {
                    resin_hir::Hir::build(built.clone(), execution(), &cancellation)
                        .await
                        .map(Arc::new)
                },
                execution(),
                &cancellation,
            )
            .await
            .unwrap();
        cache.get(&key).unwrap().as_ref().clone()
    })
}

async fn capture(
    source: Source,
    loader: &mut Loader,
    cancellation: &Cancellation,
) -> resin_ast::BuiltProgram {
    let mut sources = BTreeMap::from([(source.id(), source.clone())]);
    let mut pending = vec![source.clone()];
    let mut bindings = Vec::new();
    let mut documents = BTreeMap::new();
    let mut errors = Vec::new();
    while let Some(current) = pending.pop() {
        let syntax = Arc::new(
            resin_cst::build_cst(current.text().to_owned(), None, execution(), cancellation)
                .await
                .unwrap(),
        );
        let parsed = resin_ast::build_ast(syntax.clone(), execution(), cancellation)
            .await
            .unwrap();
        for import in &parsed.file.imports {
            match loader
                .load_import_async(&current, &import.val, execution(), cancellation)
                .await
            {
                Ok(imported) => {
                    let id = imported.id();
                    if let std::collections::btree_map::Entry::Vacant(entry) =
                        sources.entry(id.clone())
                    {
                        entry.insert(imported.clone());
                        pending.push(imported);
                    }
                    bindings.push(ImportBinding {
                        source: current.id(),
                        reference: import.val.clone(),
                        target: id,
                    });
                }
                Err(error) => errors.push(resin_source::SourceError::new(
                    current.clone(),
                    Some(import.span),
                    error.to_string(),
                )),
            }
        }
        documents.insert(
            current.clone(),
            Arc::new(resin_ast::ModuleDocument {
                source: current,
                syntax,
                file: Arc::new(parsed.file),
                errors: parsed.errors,
            }),
        );
    }
    let graph = SourceGraph::new(source, sources.into_values(), bindings).unwrap();
    let mut built = resin_ast::build_program(graph, documents, execution(), cancellation)
        .await
        .unwrap();
    for mut error in errors {
        if let Some(existing) = built.diagnostics.iter_mut().find(|existing| {
            existing.source == error.source
                && existing.span == error.span
                && existing.diagnostic.starts_with("unresolved source import:")
        }) {
            let prefix = existing
                .message
                .strip_suffix(&existing.diagnostic)
                .unwrap_or("");
            error.message = format!("{prefix}{}", error.diagnostic).into();
            error.related.extend(existing.related.iter().cloned());
            *existing = error;
        } else {
            built.diagnostics.push(error);
        }
    }
    built
}

pub fn build(
    tools: &resin_toolchain::Toolchain,
    project: &Path,
    name: &str,
    entry: &str,
    profile: resin_toolchain::CProfile,
) -> Result<resin_toolchain::BuiltProject, resin_toolchain::Error> {
    block_on(tools.build(
        project,
        name,
        entry,
        profile,
        execution(),
        &Cancellation::new(),
    ))
}

pub fn run(executable: &resin_toolchain::Executable) -> Result<i32, resin_toolchain::Error> {
    block_on(executable.run(execution(), &Cancellation::new()))
}

pub fn copy(
    executable: &resin_toolchain::Executable,
    output: &Path,
) -> Result<(), resin_toolchain::Error> {
    block_on(executable.copy_to(output, execution(), &Cancellation::new()))
}
