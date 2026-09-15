//! Render editor queries and diagnostics on bounded workers against immutable inputs.
use crate::{
    analysis::RootAnalysis,
    text::Text,
    worker::{Freshness, OpenDocument, PreparedDiagnostics, PreparedReply, QueryKind, RequestJob},
};
use lsp_server::{ErrorCode, Notification, Response};
use resin_executor::{Cancellation, Execution};
use resin_hir::DefinitionKind;
use resin_source::{Source, SourceLocation, Span};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

type QueryResult = Result<Value, (ErrorCode, String)>;

pub(super) async fn prepare_query(
    job: RequestJob,
    analysis: Arc<RootAnalysis>,
    execution: &Execution,
) -> PreparedReply {
    let RequestJob {
        serial,
        id,
        kind,
        cancellation,
        admission,
        ..
    } = job;
    let freshness = analysis.freshness.clone();
    let result = execution
        .run(&cancellation, move |_| Renderer::new(&analysis).query(kind))
        .await;
    PreparedReply {
        serial,
        response: response(id, result),
        freshness,
        admission,
    }
}

pub(super) async fn prepare_format(
    job: RequestJob,
    document: Arc<OpenDocument>,
    execution: &Execution,
) -> PreparedReply {
    let freshness = Freshness::document(&document);
    let result = execution
        .run(&job.cancellation, move |_| {
            let edits = resin_cst::format_source(&document.text)
                .map(|formatted| formatting_edits(&document.text, &formatted));
            serde_json::to_value(edits).map_err(internal)
        })
        .await;
    PreparedReply {
        serial: job.serial,
        response: response(job.id, result),
        freshness,
        admission: job.admission,
    }
}

fn response(
    id: lsp_server::RequestId,
    result: Result<QueryResult, resin_executor::Error>,
) -> Response {
    match result {
        // The worker already built JSON; avoid serializing a large completion list again.
        Ok(Ok(value)) => Response {
            id,
            result: Some(value),
            error: None,
        },
        Ok(Err((code, message))) => Response::new_err(id, code as i32, message),
        Err(error) => Response::new_err(
            id,
            if error == resin_executor::Error::Cancelled {
                ErrorCode::RequestCanceled as i32
            } else {
                ErrorCode::InternalError as i32
            },
            error.to_string(),
        ),
    }
}

fn internal(error: serde_json::Error) -> (ErrorCode, String) {
    (ErrorCode::InternalError, error.to_string())
}

pub(super) async fn prepare_diagnostics(
    revision: u64,
    roots: Vec<Arc<RootAnalysis>>,
    execution: &Execution,
    cancellation: &Cancellation,
) -> crate::Result<PreparedDiagnostics> {
    Ok(execution
        .run(cancellation, move |cancellation| {
            let mut diagnostics = BTreeMap::<String, Vec<lsp_types::Diagnostic>>::new();
            let mut versions = BTreeMap::new();
            for root in roots {
                cancellation.check()?;
                diagnostics
                    .entry(root.ticket.uri.as_str().to_owned())
                    .or_default();
                for (uri, stamp) in root
                    .freshness
                    .document
                    .iter()
                    .chain(&root.freshness.dependencies)
                {
                    versions.insert(uri.as_str().to_owned(), stamp.version);
                }
                let renderer = Renderer::new(&root);
                for diagnostic in root.hir.diagnostics() {
                    let Some(location) = renderer.location(&diagnostic.location) else {
                        continue;
                    };
                    let related = diagnostic
                        .related
                        .iter()
                        .filter_map(|note| {
                            Some(lsp_types::DiagnosticRelatedInformation {
                                location: renderer.location(&note.location)?,
                                message: note.message.clone(),
                            })
                        })
                        .collect::<Vec<_>>();
                    let diagnostic = lsp_types::Diagnostic {
                        range: location.range,
                        severity: Some(lsp_types::DiagnosticSeverity::ERROR),
                        source: Some("resin".into()),
                        message: diagnostic.message.clone(),
                        related_information: (!related.is_empty()).then_some(related),
                        ..Default::default()
                    };
                    let current = diagnostics
                        .entry(location.uri.as_str().to_owned())
                        .or_default();
                    if !current.contains(&diagnostic) {
                        current.push(diagnostic);
                    }
                }
            }
            let uris: BTreeSet<_> = diagnostics.keys().cloned().collect();
            let notifications = diagnostics
                .into_iter()
                .map(|(uri, diagnostics)| {
                    let version = versions.get(&uri).copied();
                    Notification::new(
                        "textDocument/publishDiagnostics".into(),
                        lsp_types::PublishDiagnosticsParams {
                            uri: uri.parse().expect("retained valid URI"),
                            diagnostics,
                            version,
                        },
                    )
                })
                .collect();
            Ok::<_, resin_executor::Error>(PreparedDiagnostics {
                revision,
                notifications,
                uris,
            })
        })
        .await??)
}

struct Renderer<'a> {
    analysis: &'a RootAnalysis,
    texts: BTreeMap<Source, Text>,
}

impl<'a> Renderer<'a> {
    fn new(analysis: &'a RootAnalysis) -> Self {
        Self {
            analysis,
            texts: analysis
                .hir
                .sources()
                .map(|source| (source.clone(), Text::new(source.text())))
                .collect(),
        }
    }

    fn location(&self, location: &SourceLocation) -> Option<lsp_types::Location> {
        Some(lsp_types::Location {
            uri: self.analysis.uris.get(&location.source.id())?.clone(),
            range: self
                .texts
                .get(&location.source)
                .map(|text| text.range(location.span))
                .unwrap_or_default(),
        })
    }

    fn query(&self, kind: QueryKind) -> QueryResult {
        let position = match &kind {
            QueryKind::Hover { position, .. }
            | QueryKind::Definition { position, .. }
            | QueryKind::Completion { position, .. } => *position,
            _ => {
                return Err((
                    ErrorCode::InvalidRequest,
                    "query requires semantic input".into(),
                ));
            }
        };
        let source = self.analysis.hir.source();
        let Some(text) = self.texts.get(source) else {
            return Ok(Value::Null);
        };
        let offset = text.offset(position).ok_or_else(|| {
            (
                ErrorCode::InvalidParams,
                "position is outside the document or splits a UTF-16 character".into(),
            )
        })?;
        match kind {
            QueryKind::Hover { .. } => {
                serde_json::to_value(self.analysis.hir.hover(source, offset).map(|hover| {
                    lsp_types::Hover {
                        contents: lsp_types::HoverContents::Markup(lsp_types::MarkupContent {
                            kind: lsp_types::MarkupKind::Markdown,
                            value: format!("```resin\n{}\n```", hover.text),
                        }),
                        range: Some(text.range(hover.span)),
                    }
                }))
                .map_err(internal)
            }
            QueryKind::Definition { .. } => serde_json::to_value(
                self.analysis
                    .hir
                    .definition(source, offset)
                    .and_then(|location| self.location(&location)),
            )
            .map_err(internal),
            QueryKind::Completion { .. } => {
                let items = self
                    .analysis
                    .hir
                    .completions(source, offset)
                    .into_iter()
                    .enumerate()
                    .map(|(index, item)| lsp_types::CompletionItem {
                        label: item.name.clone(),
                        sort_text: Some(format!("{index:010}")),
                        detail: Some(item.detail),
                        kind: Some(match item.kind {
                            DefinitionKind::Function => lsp_types::CompletionItemKind::FUNCTION,
                            DefinitionKind::Type => lsp_types::CompletionItemKind::CLASS,
                            DefinitionKind::Keyword => lsp_types::CompletionItemKind::KEYWORD,
                            DefinitionKind::Field => lsp_types::CompletionItemKind::FIELD,
                            DefinitionKind::Variable | DefinitionKind::Parameter => {
                                lsp_types::CompletionItemKind::VARIABLE
                            }
                        }),
                        text_edit: Some(lsp_types::CompletionTextEdit::Edit(lsp_types::TextEdit {
                            range: text.range(item.replace),
                            new_text: item.name,
                        })),
                        ..Default::default()
                    })
                    .collect();
                serde_json::to_value(lsp_types::CompletionList {
                    is_incomplete: false,
                    items,
                })
                .map_err(internal)
            }
            _ => unreachable!(),
        }
    }
}

/// Trim unchanged text around a replacement, preserving UTF-8 and CRLF boundaries.
fn formatting_edits(source: &str, formatted: &str) -> Vec<lsp_types::TextEdit> {
    if source == formatted {
        return Vec::new();
    }
    let mut start = source
        .bytes()
        .zip(formatted.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    while !source.is_char_boundary(start)
        || !formatted.is_char_boundary(start)
        || (start > 0
            && source.as_bytes().get(start - 1) == Some(&b'\r')
            && source.as_bytes().get(start) == Some(&b'\n'))
    {
        start -= 1;
    }
    let suffix = source[start..]
        .bytes()
        .rev()
        .zip(formatted[start..].bytes().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let mut end = source.len() - suffix;
    let mut new_end = formatted.len() - suffix;
    while !source.is_char_boundary(end)
        || !formatted.is_char_boundary(new_end)
        || (end > 0
            && source.as_bytes().get(end - 1) == Some(&b'\r')
            && source.as_bytes().get(end) == Some(&b'\n'))
    {
        end += 1;
        new_end += 1;
    }
    vec![lsp_types::TextEdit {
        range: Text::new(source).range(Span { start, end }),
        new_text: formatted[start..new_end].into(),
    }]
}
