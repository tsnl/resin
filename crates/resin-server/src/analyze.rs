//! Analyze exact immutable inputs; pass order is deliberately visible here.
use crate::{
    Server,
    http::failure,
    inputs::{self, internal},
    publication,
};
use resin_executor::Cancellation;
use resin_protocol::*;
use resin_source::{Source, SourceLocation};
use std::{
    collections::BTreeSet,
    sync::{Arc, atomic::Ordering},
};

pub(crate) async fn run(
    server: &Server,
    request: AnalyzeRequest,
    cancellation: &Cancellation,
) -> Result<AnalyzeResponse, Failure> {
    let frozen = inputs::capture(server, request.inputs, cancellation).await?;
    let documents = publication::select(
        &server.caches.ast,
        frozen.graph.sources().cloned().collect(),
        |source| {
            let syntax = frozen.syntax[&source].clone();
            async move {
                server.caches.ast_builds.fetch_add(1, Ordering::Relaxed);
                let parsed =
                    resin_ast::build_ast(syntax.clone(), &server.execution, cancellation).await?;
                Ok::<_, resin_executor::Error>(Arc::new(resin_ast::ModuleDocument {
                    source,
                    syntax,
                    file: Arc::new(parsed.file),
                    errors: parsed.errors,
                }))
            }
        },
        &server.execution,
        cancellation,
    )
    .await
    .map_err(internal)?;
    let graph = frozen.graph;
    let mut program =
        resin_ast::build_program(graph.clone(), documents, &server.execution, cancellation)
            .await
            .map_err(internal)?;
    let hir = if frozen.acquisition.is_empty() {
        let program = Arc::new(program);
        publication::select(
            &server.caches.hir,
            vec![graph.clone()],
            |_| {
                let program = program.clone();
                async move {
                    server.caches.hir_builds.fetch_add(1, Ordering::Relaxed);
                    resin_hir::Hir::build(program, &server.execution, cancellation)
                        .await
                        .map(Arc::new)
                }
            },
            &server.execution,
            cancellation,
        )
        .await
        .map_err(internal)?
        .remove(&graph)
        .expect("selected HIR")
    } else {
        inputs::acquisition_diagnostics(&mut program, frozen.acquisition);
        Arc::new(
            resin_hir::Hir::build(Arc::new(program), &server.execution, cancellation)
                .await
                .map_err(internal)?,
        )
    };
    server
        .execution
        .run(cancellation, move |_| {
            let diagnostics = diagnostics(&hir);
            let results = request
                .queries
                .into_iter()
                .map(|query| query_result(&hir, query))
                .collect::<Result<Vec<_>, _>>()?;
            let managed_sources = managed_sources(&hir, &diagnostics, &results);
            Ok(AnalyzeResponse {
                revision: request.revision,
                input: frozen.input,
                diagnostics,
                results,
                managed_sources,
            })
        })
        .await
        .map_err(internal)?
}

pub(crate) fn diagnostics(hir: &resin_hir::Hir) -> Vec<Diagnostic> {
    hir.diagnostics()
        .iter()
        .map(|diagnostic| Diagnostic {
            severity: Severity::Error,
            message: diagnostic.message.clone(),
            span: Some(location(&diagnostic.location)),
            related: diagnostic
                .related
                .iter()
                .map(|note| RelatedDiagnostic {
                    message: note.message.clone(),
                    span: location(&note.location),
                })
                .collect(),
        })
        .collect()
}

pub(crate) fn managed_sources(
    hir: &resin_hir::Hir,
    diagnostics: &[Diagnostic],
    results: &[QueryResult],
) -> Vec<SourceFile> {
    let mut selected = BTreeSet::new();
    for diagnostic in diagnostics {
        for span in diagnostic
            .span
            .iter()
            .chain(diagnostic.related.iter().map(|note| &note.span))
        {
            selected.insert(span.source.as_str());
        }
    }
    for result in results {
        match result {
            QueryResult::Hover { span, .. } | QueryResult::Definition { span } => {
                if let Some(span) = span {
                    selected.insert(span.source.as_str());
                }
            }
            QueryResult::Completion { items } => {
                for item in items {
                    selected.insert(item.replace.source.as_str());
                }
            }
        }
    }
    hir.sources()
        .filter(|source| source.name().starts_with("$/") && selected.contains(source.name()))
        .map(|source| SourceFile {
            name: source.name().into(),
            text: source.text().into(),
        })
        .collect()
}
fn location(location: &SourceLocation) -> Span {
    span(&location.source, location.span)
}
fn span(source: &Source, span: resin_source::Span) -> Span {
    Span {
        source: source.name().into(),
        start: span.start as u64,
        end: span.end as u64,
    }
}

fn query_result(hir: &resin_hir::Hir, query: Query) -> Result<QueryResult, Failure> {
    let position = match &query {
        Query::Hover { position }
        | Query::Definition { position }
        | Query::Completion { position } => position,
    };
    let source = hir
        .sources()
        .find(|source| source.name() == position.source)
        .ok_or_else(|| {
            failure(
                ErrorCode::InvalidRequest,
                "query source is not in the selected graph",
            )
        })?;
    inputs::validate_span(source, position.offset, position.offset)?;
    let offset = position.offset as usize;
    Ok(match query {
        Query::Hover { .. } => {
            let hover = hir.hover(source, offset);
            QueryResult::Hover {
                markdown: hover
                    .as_ref()
                    .map(|hover| format!("```resin\n{}\n```", hover.text)),
                span: hover.map(|hover| span(source, hover.span)),
            }
        }
        Query::Definition { .. } => QueryResult::Definition {
            span: hir.definition(source, offset).as_ref().map(location),
        },
        Query::Completion { .. } => QueryResult::Completion {
            items: hir
                .completions(source, offset)
                .into_iter()
                .map(|item| CompletionItem {
                    label: item.name.clone(),
                    detail: Some(item.detail),
                    insert_text: item.name,
                    replace: span(source, item.replace),
                    kind: match item.kind {
                        resin_hir::DefinitionKind::Function => CompletionKind::Function,
                        resin_hir::DefinitionKind::Variable
                        | resin_hir::DefinitionKind::Parameter => CompletionKind::Variable,
                        resin_hir::DefinitionKind::Type => CompletionKind::Type,
                        resin_hir::DefinitionKind::Keyword => CompletionKind::Keyword,
                        resin_hir::DefinitionKind::Field => CompletionKind::Field,
                    },
                })
                .collect(),
        },
    })
}
