//! One frozen editor root: acquire its graph, then translate explicit compiler phases.
use crate::{
    caches::Caches,
    inputs, publication,
    worker::{EditorSnapshot, Freshness, RegisteredEditor},
};
use lsp_types::Uri;
use resin_executor::{Cancellation, Execution};
use resin_hir::Hir;
use resin_source::SourceId;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RootTicket {
    pub uri: Uri,
    pub epoch: u64,
    pub generation: u64,
}

pub(super) struct RootAnalysis {
    pub ticket: RootTicket,
    pub freshness: Freshness,
    pub hir: Arc<Hir>,
    pub uris: BTreeMap<SourceId, Uri>,
}

pub(super) async fn analyze_root(
    ticket: RootTicket,
    registered: Arc<RegisteredEditor>,
    caches: Arc<Caches>,
    execution: Execution,
    cancellation: Cancellation,
) -> crate::Result<RootAnalysis> {
    let (source, mut loader) = execution
        .run(&cancellation, {
            let registered = registered.clone();
            let uri = ticket.uri.clone();
            move |_| {
                (
                    registered.documents[uri.as_str()].clone(),
                    registered.loader.supplied_snapshot(),
                )
            }
        })
        .await?;
    let captured = inputs::capture(source, &mut loader, &caches, &execution, &cancellation).await?;
    drop(loader);
    let (captured, freshness, uris) = execution
        .run(&cancellation, {
            let ticket = ticket.clone();
            move |_| {
                let (freshness, uris) = presentation(&ticket, &captured, &registered);
                (captured, freshness, uris)
            }
        })
        .await?;
    drop(captured.origins);
    drop(captured.paths);

    let documents = publication::select(
        &caches.ast,
        captured.graph.sources().cloned().collect(),
        |source| {
            let syntax = captured.syntax[&source].clone();
            let execution = &execution;
            let cancellation = &cancellation;
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
        &execution,
        &cancellation,
    )
    .await?;
    drop(captured.syntax);
    let graph = captured.graph;
    let mut program =
        resin_ast::build_program(graph.clone(), documents, &execution, &cancellation).await?;
    let hir = if captured.diagnostics.is_empty() {
        let program = Arc::new(program);
        publication::select(
            &caches.hir,
            vec![graph.clone()],
            |_| {
                let program = program.clone();
                let execution = &execution;
                let cancellation = &cancellation;
                async move {
                    Hir::build(program, execution, cancellation)
                        .await
                        .map(Arc::new)
                }
            },
            &execution,
            &cancellation,
        )
        .await?
        .remove(&graph)
        .expect("selected HIR")
    } else {
        inputs::acquisition_diagnostics(&mut program, captured.diagnostics);
        Arc::new(Hir::build(Arc::new(program), &execution, &cancellation).await?)
    };
    cancellation.check()?;
    Ok(RootAnalysis {
        ticket,
        freshness,
        hir,
        uris,
    })
}

fn presentation(
    ticket: &RootTicket,
    captured: &inputs::Inputs,
    registered: &RegisteredEditor,
) -> (Freshness, BTreeMap<SourceId, Uri>) {
    let mut freshness = Freshness {
        document: Some((
            ticket.uri.clone(),
            registered.editor.documents[ticket.uri.as_str()].stamp,
        )),
        dependencies: Vec::new(),
        registrations: Some(registered.editor.registrations),
        disk_revision: Some(registered.editor.disk_revision),
    };
    let mut uris = BTreeMap::new();
    for (logical, source) in &captured.origins {
        if let Some(uri) = captured.paths.get(logical).and_then(|path| file_uri(path)) {
            uris.insert(logical.clone(), uri);
        }
        for (uri, supplied) in &registered.documents {
            if supplied.id() == source.id() {
                uris.insert(
                    logical.clone(),
                    registered.editor.documents[uri].uri.clone(),
                );
                if uri.as_str() != ticket.uri.as_str() {
                    freshness.dependencies.push((
                        registered.editor.documents[uri].uri.clone(),
                        registered.editor.documents[uri].stamp,
                    ));
                }
            }
        }
    }
    uris.insert(captured.graph.entry().id(), ticket.uri.clone());
    (freshness, uris)
}

fn file_uri(path: &std::path::Path) -> Option<Uri> {
    url::Url::from_file_path(path).ok()?.as_str().parse().ok()
}

pub(super) fn root_is_requested(ticket: &RootTicket, editor: &EditorSnapshot) -> bool {
    editor
        .documents
        .get(ticket.uri.as_str())
        .is_some_and(|document| document.stamp.epoch == ticket.epoch)
}
