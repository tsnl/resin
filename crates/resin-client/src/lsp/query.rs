//! Send semantic queries and render byte spans against their captured source texts.
use super::{
    analysis::RootAnalysis,
    text::Text,
    worker::{Freshness, OpenDocument, PreparedDiagnostics, PreparedReply, QueryKind, RequestJob},
};
use lsp_server::{ErrorCode, Notification, Response};
use resin_executor::{Cancellation, Execution};
use resin_protocol::{Query, QueryPosition};
use resin_source::Span;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

type QueryResult = Result<Value, (ErrorCode, String)>;

enum RemoteError {
    ManagedSnapshotUnavailable,
    Reply { code: ErrorCode, message: String },
}

impl From<(ErrorCode, String)> for RemoteError {
    fn from((code, message): (ErrorCode, String)) -> Self {
        Self::Reply { code, message }
    }
}

pub(super) async fn prepare_query(
    job: RequestJob,
    mut analysis: Arc<RootAnalysis>,
    registered: Arc<super::worker::RegisteredEditor>,
    config: Arc<super::worker::WorkerConfig>,
    execution: &Execution,
) -> PreparedReply {
    let mut result = Err((
        ErrorCode::InternalError,
        "managed snapshot changed again during query".into(),
    ));
    for attempt in 0..2 {
        match remote_query(&job, analysis.clone(), &config, execution).await {
            Ok(value) => {
                result = Ok(value);
                break;
            }
            Err(RemoteError::ManagedSnapshotUnavailable) if attempt == 0 => {
                match super::analysis::analyze_root(
                    analysis.ticket.clone(),
                    registered.clone(),
                    config.clone(),
                    Some(analysis.clone()),
                    execution.clone(),
                    job.cancellation.clone(),
                )
                .await
                {
                    Ok(refreshed) => analysis = Arc::new(refreshed),
                    Err(error) => {
                        result = Err((
                            if job.cancellation.is_cancelled() {
                                ErrorCode::RequestCanceled
                            } else {
                                ErrorCode::InternalError
                            },
                            error.to_string(),
                        ));
                        break;
                    }
                }
            }
            Err(RemoteError::ManagedSnapshotUnavailable) => break,
            Err(RemoteError::Reply { code, message }) => {
                result = Err((code, message));
                break;
            }
        }
    }
    PreparedReply {
        serial: job.serial,
        response: match result {
            Ok(value) => Response {
                id: job.id,
                result: Some(value),
                error: None,
            },
            Err((code, message)) => Response::new_err(job.id, code as i32, message),
        },
        freshness: analysis.freshness.clone(),
        admission: job.admission,
    }
}

async fn remote_query(
    job: &RequestJob,
    analysis: Arc<RootAnalysis>,
    config: &super::worker::WorkerConfig,
    execution: &Execution,
) -> Result<Value, RemoteError> {
    let position = match job.kind {
        QueryKind::Hover { position, .. }
        | QueryKind::Definition { position, .. }
        | QueryKind::Completion { position, .. } => position,
        _ => {
            return Err((
                ErrorCode::InvalidRequest,
                "query requires semantic input".into(),
            )
                .into());
        }
    };
    let text = Text::new(&analysis.texts[&analysis.captured.inputs.entry]);
    let offset = text.offset(position).ok_or_else(|| {
        (
            ErrorCode::InvalidParams,
            "position is outside the document or splits a UTF-16 character".into(),
        )
    })?;
    let position = QueryPosition {
        source: analysis.captured.inputs.entry.clone(),
        offset: offset as u64,
    };
    let query = match job.kind {
        QueryKind::Hover { .. } => Query::Hover { position },
        QueryKind::Definition { .. } => Query::Definition { position },
        QueryKind::Completion { .. } => Query::Completion { position },
        _ => unreachable!(),
    };
    let result = config
        .client
        .analyze(
            &analysis.captured.inputs,
            Some((&analysis.response.input, &analysis.captured.inputs)),
            analysis.ticket.generation,
            vec![query],
            &job.cancellation,
        )
        .await
        .map_err(|error| {
            if error.failure().is_some_and(|failure| {
                failure.code == resin_protocol::ErrorCode::ManagedSnapshotUnavailable
            }) {
                RemoteError::ManagedSnapshotUnavailable
            } else {
                RemoteError::Reply {
                    code: if error.is_cancelled() {
                        ErrorCode::RequestCanceled
                    } else {
                        ErrorCode::InternalError
                    },
                    message: error.to_string(),
                }
            }
        })?;
    let paths = config
        .mirrors
        .materialize(
            result.managed_sources.clone(),
            analysis.captured.inputs.managed_snapshot.clone(),
            execution,
            &job.cancellation,
        )
        .await
        .map_err(|error| (ErrorCode::InternalError, error.to_string()))?;
    let managed = result.managed_sources;
    let result = result.results.into_iter().next().ok_or_else(|| {
        (
            ErrorCode::InternalError,
            "server omitted query result".into(),
        )
    })?;
    execution
        .run(&job.cancellation, move |_| {
            let mut renderer = Renderer::new(&analysis);
            for source in managed {
                renderer.texts.insert(source.name, Text::new(&source.text));
            }
            for (name, path) in paths {
                if let Ok(url) = url::Url::from_file_path(path)
                    && let Ok(uri) = url.as_str().parse()
                {
                    renderer.uris.insert(name, uri);
                }
            }
            renderer.query(result)
        })
        .await
        .map_err(|error| (ErrorCode::RequestCanceled, error.to_string()))?
        .map_err(Into::into)
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
) -> super::Result<PreparedDiagnostics> {
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
                for diagnostic in &root.response.diagnostics {
                    let Some(location) = diagnostic
                        .span
                        .as_ref()
                        .and_then(|span| renderer.location(span))
                    else {
                        continue;
                    };
                    let related = diagnostic
                        .related
                        .iter()
                        .filter_map(|note| {
                            Some(lsp_types::DiagnosticRelatedInformation {
                                location: renderer.location(&note.span)?,
                                message: note.message.clone(),
                            })
                        })
                        .collect::<Vec<_>>();
                    let diagnostic = lsp_types::Diagnostic {
                        range: location.range,
                        severity: Some(match diagnostic.severity {
                            resin_protocol::Severity::Error => lsp_types::DiagnosticSeverity::ERROR,
                            resin_protocol::Severity::Warning => {
                                lsp_types::DiagnosticSeverity::WARNING
                            }
                            resin_protocol::Severity::Information => {
                                lsp_types::DiagnosticSeverity::INFORMATION
                            }
                        }),
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

struct Renderer {
    uris: BTreeMap<String, lsp_types::Uri>,
    texts: BTreeMap<String, Text>,
}
impl Renderer {
    fn new(analysis: &RootAnalysis) -> Self {
        Self {
            uris: analysis.uris.clone(),
            texts: analysis
                .texts
                .iter()
                .map(|(name, text)| (name.clone(), Text::new(text)))
                .collect(),
        }
    }
    fn range(&self, span: &resin_protocol::Span) -> Option<lsp_types::Range> {
        Some(self.texts.get(&span.source)?.range(Span {
            start: usize::try_from(span.start).ok()?,
            end: usize::try_from(span.end).ok()?,
        }))
    }
    fn location(&self, span: &resin_protocol::Span) -> Option<lsp_types::Location> {
        Some(lsp_types::Location {
            uri: self.uris.get(&span.source)?.clone(),
            range: self.range(span)?,
        })
    }
    fn query(&self, result: resin_protocol::QueryResult) -> QueryResult {
        match result {
            resin_protocol::QueryResult::Hover { markdown, span } => {
                serde_json::to_value(markdown.map(|value| lsp_types::Hover {
                    contents: lsp_types::HoverContents::Markup(lsp_types::MarkupContent {
                        kind: lsp_types::MarkupKind::Markdown,
                        value,
                    }),
                    range: span.as_ref().and_then(|span| self.range(span)),
                }))
                .map_err(internal)
            }
            resin_protocol::QueryResult::Definition { span } => {
                serde_json::to_value(span.as_ref().and_then(|span| self.location(span)))
                    .map_err(internal)
            }
            resin_protocol::QueryResult::Completion { items } => {
                let items = items
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, item)| {
                        Some(lsp_types::CompletionItem {
                            label: item.label,
                            sort_text: Some(format!("{index:010}")),
                            detail: item.detail,
                            kind: Some(match item.kind {
                                resin_protocol::CompletionKind::Function => {
                                    lsp_types::CompletionItemKind::FUNCTION
                                }
                                resin_protocol::CompletionKind::Variable => {
                                    lsp_types::CompletionItemKind::VARIABLE
                                }
                                resin_protocol::CompletionKind::Field => {
                                    lsp_types::CompletionItemKind::FIELD
                                }
                                resin_protocol::CompletionKind::Type => {
                                    lsp_types::CompletionItemKind::CLASS
                                }
                                resin_protocol::CompletionKind::Module => {
                                    lsp_types::CompletionItemKind::MODULE
                                }
                                resin_protocol::CompletionKind::Keyword => {
                                    lsp_types::CompletionItemKind::KEYWORD
                                }
                            }),
                            text_edit: Some(lsp_types::CompletionTextEdit::Edit(
                                lsp_types::TextEdit {
                                    range: self.range(&item.replace)?,
                                    new_text: item.insert_text,
                                },
                            )),
                            ..Default::default()
                        })
                    })
                    .collect();
                serde_json::to_value(lsp_types::CompletionList {
                    is_incomplete: false,
                    items,
                })
                .map_err(internal)
            }
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
