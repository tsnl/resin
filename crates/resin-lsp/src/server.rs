use crate::{
    text::Text,
    worker::{self, AnalysisUpdate, Change, Update},
};
use crossbeam_channel::{Receiver, Sender, select};
use lsp_server::{Connection, ErrorCode, Message, Notification, Request, RequestId, Response};
use lsp_types::{self as lsp, Position, Uri};
use resin_common::source::SourceLocation;
use resin_compiler::{DefinitionKind, normalize_path};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Options {
    stdlib_path: Option<PathBuf>,
}

pub(crate) fn run(default_stdlib: PathBuf) -> Result<i32> {
    let (connection, io) = Connection::stdio();
    let (id, params) = connection.initialize_start()?;
    let params: lsp::InitializeParams = serde_json::from_value(params)?;
    let options: Options = params
        .initialization_options
        .clone()
        .filter(|v| !v.is_null())
        .map(serde_json::from_value)
        .transpose()?
        .unwrap_or_default();
    let stdlib = options.stdlib_path.unwrap_or(default_stdlib);
    let capabilities = lsp::ServerCapabilities {
        position_encoding: Some(lsp::PositionEncodingKind::UTF16),
        text_document_sync: Some(
            lsp::TextDocumentSyncOptions {
                open_close: Some(true),
                change: Some(lsp::TextDocumentSyncKind::FULL),
                save: Some(
                    lsp::SaveOptions {
                        include_text: Some(false),
                    }
                    .into(),
                ),
                ..Default::default()
            }
            .into(),
        ),
        hover_provider: Some(lsp::HoverProviderCapability::Simple(true)),
        definition_provider: Some(lsp::OneOf::Left(true)),
        document_formatting_provider: Some(lsp::OneOf::Left(true)),
        completion_provider: Some(lsp::CompletionOptions {
            resolve_provider: Some(false),
            trigger_characters: Some(vec![".".into()]),
            ..Default::default()
        }),
        ..Default::default()
    };
    connection.initialize_finish(id, json!({"capabilities": capabilities, "serverInfo": {"name": "resin-lsp", "version": env!("CARGO_PKG_VERSION")}}))?;
    if params
        .capabilities
        .workspace
        .as_ref()
        .and_then(|w| w.did_change_watched_files.as_ref())
        .and_then(|w| w.dynamic_registration)
        == Some(true)
    {
        connection.sender.send(Message::Request(Request::new(String::from("resin-file-watchers").into(), "client/registerCapability".into(), json!({"registrations": [{"id": "resin-files", "method": "workspace/didChangeWatchedFiles", "registerOptions": {"watchers": [{"globPattern": "**/*.resin", "kind": 7}]}}]}))))?;
    }
    let (updates, receiver) = crossbeam_channel::unbounded();
    let (results, snapshots) = crossbeam_channel::unbounded();
    let revision = Arc::new(AtomicU64::new(0));
    let stopping = Arc::new(AtomicBool::new(false));
    let worker = worker::spawn(
        stdlib,
        receiver,
        results,
        revision.clone(),
        stopping.clone(),
    );
    let mut state = State {
        connection,
        updates,
        revision,
        documents: BTreeMap::new(),
        snapshot: None,
        texts: BTreeMap::new(),
        published: BTreeSet::new(),
        pending: HashMap::new(),
        shutdown: false,
    };
    let outcome = state.events(&snapshots);
    stopping.store(true, Ordering::Release);
    drop(state);
    worker.join().map_err(|_| "compiler worker panicked")?;
    io.join()?;
    outcome
}

struct OpenDocument {
    uri: Uri,
    path: PathBuf,
    version: i32,
    source: String,
}
struct Query {
    revision: u64,
    method: String,
    uri: Uri,
    position: Position,
}

struct State {
    connection: Connection,
    updates: Sender<Update>,
    revision: Arc<AtomicU64>,
    documents: BTreeMap<String, OpenDocument>,
    snapshot: Option<AnalysisUpdate>,
    texts: BTreeMap<PathBuf, Text>,
    published: BTreeSet<String>,
    pending: HashMap<RequestId, Query>,
    shutdown: bool,
}

impl State {
    fn events(&mut self, snapshots: &Receiver<AnalysisUpdate>) -> Result<i32> {
        loop {
            select! {
                recv(self.connection.receiver) -> message => {
                    let Ok(message) = message else { return Ok(if self.shutdown { 0 } else { 1 }); };
                    match message {
                        Message::Request(request) => self.request(request)?,
                        Message::Notification(notification) => {
                            if notification.method == "exit" { return Ok(if self.shutdown { 0 } else { 1 }); }
                            if !self.shutdown { self.notification(notification)?; }
                        }
                        Message::Response(response) => {
                            if let Some(error) = response.error { eprintln!("resin-lsp: client request failed: {}", error.message); }
                        }
                    }
                }
                recv(snapshots) -> snapshot => {
                    let snapshot = snapshot.map_err(|_| "compiler worker stopped")?;
                    if !self.shutdown && snapshot.revision == self.revision.load(Ordering::Acquire) {
                        self.texts.clear();
                        for entry in snapshot.entries.values() {
                            for (path, text) in entry.sources() { self.texts.entry(path.to_path_buf()).or_insert_with(|| Text::new(text)); }
                        }
                        self.snapshot = Some(snapshot);
                        self.publish()?;
                        for (id, query) in std::mem::take(&mut self.pending) { self.answer(id, query)?; }
                    }
                }
            }
        }
    }

    fn send(&self, response: Response) -> Result<()> {
        self.connection.sender.send(Message::Response(response))?;
        Ok(())
    }
    fn error(&self, id: RequestId, code: i32, message: impl Into<String>) -> Result<()> {
        self.send(Response::new_err(id, code, message.into()))
    }

    fn request(&mut self, request: Request) -> Result<()> {
        if self.shutdown {
            return self.error(
                request.id,
                ErrorCode::InvalidRequest as i32,
                "server is shutting down",
            );
        }
        if request.method == "shutdown" {
            self.shutdown = true;
            for (id, _) in self.pending.drain().collect::<Vec<_>>() {
                self.error(
                    id,
                    ErrorCode::RequestCanceled as i32,
                    "server is shutting down",
                )?;
            }
            return self.send(Response::new_ok(request.id, Value::Null));
        }
        if request.method == "textDocument/formatting" {
            let params: lsp::DocumentFormattingParams = match serde_json::from_value(request.params)
            {
                Ok(params) => params,
                Err(error) => {
                    return self.error(
                        request.id,
                        ErrorCode::InvalidParams as i32,
                        error.to_string(),
                    );
                }
            };
            let Some(document) = self.documents.get(params.text_document.uri.as_str()) else {
                return self.send(Response::new_ok(request.id, Value::Null));
            };
            // Use the latest accepted buffer even while semantic analysis is busy.
            // Resin has one canonical style, independent of editor indent settings.
            let result = resin_cst::format_source(&document.source)
                .map(|formatted| formatting_edits(&document.source, &formatted));
            return self.send(Response::new_ok(request.id, result));
        }
        if !matches!(
            request.method.as_str(),
            "textDocument/hover" | "textDocument/definition" | "textDocument/completion"
        ) {
            return self.error(
                request.id,
                ErrorCode::MethodNotFound as i32,
                format!("unsupported method: {}", request.method),
            );
        }
        let params: lsp::TextDocumentPositionParams = match serde_json::from_value(request.params) {
            Ok(params) => params,
            Err(error) => {
                return self.error(
                    request.id,
                    ErrorCode::InvalidParams as i32,
                    error.to_string(),
                );
            }
        };
        let query = Query {
            revision: self.revision.load(Ordering::Acquire),
            method: request.method,
            uri: params.text_document.uri,
            position: params.position,
        };
        if !self.documents.contains_key(query.uri.as_str()) {
            return self.send(Response::new_ok(request.id, Value::Null));
        }
        if self
            .snapshot
            .as_ref()
            .is_some_and(|s| s.revision == query.revision)
        {
            self.answer(request.id, query)
        } else {
            self.pending.insert(request.id, query);
            Ok(())
        }
    }

    fn answer(&self, id: RequestId, query: Query) -> Result<()> {
        if query.revision != self.revision.load(Ordering::Acquire) {
            return self.error(
                id,
                ErrorCode::ContentModified as i32,
                "document or dependencies changed",
            );
        }
        let Some(document) = self.documents.get(query.uri.as_str()) else {
            return self.send(Response::new_ok(id, Value::Null));
        };
        let Some(analysis) = self
            .snapshot
            .as_ref()
            .and_then(|s| s.entries.get(&document.path))
        else {
            return self.send(Response::new_ok(id, Value::Null));
        };
        let Some(text) = self.texts.get(&document.path) else {
            return self.send(Response::new_ok(id, Value::Null));
        };
        let Some(offset) = text.offset(query.position) else {
            return self.error(
                id,
                ErrorCode::InvalidParams as i32,
                "position is outside the document or splits a UTF-16 character",
            );
        };
        let result = match query.method.as_str() {
            "textDocument/hover" => {
                serde_json::to_value(analysis.hover(&document.path, offset).map(|hover| {
                    lsp::Hover {
                        contents: lsp::HoverContents::Markup(lsp::MarkupContent {
                            kind: lsp::MarkupKind::Markdown,
                            value: format!("```resin\n{}\n```", hover.text),
                        }),
                        range: Some(text.range(hover.span)),
                    }
                }))?
            }
            "textDocument/definition" => serde_json::to_value(
                analysis
                    .definition(&document.path, offset)
                    .and_then(|location| self.location(&location)),
            )?,
            "textDocument/completion" => {
                let items = analysis
                    .completions(&document.path, offset)
                    .into_iter()
                    .enumerate()
                    .map(|(index, item)| lsp::CompletionItem {
                        label: item.name.clone(),
                        // Preserve analysis ordering when clients sort completion items.
                        sort_text: Some(format!("{index:010}")),
                        detail: Some(item.detail),
                        kind: Some(match item.kind {
                            DefinitionKind::Function => lsp::CompletionItemKind::FUNCTION,
                            DefinitionKind::Type => lsp::CompletionItemKind::CLASS,
                            DefinitionKind::Keyword => lsp::CompletionItemKind::KEYWORD,
                            DefinitionKind::Field => lsp::CompletionItemKind::FIELD,
                            DefinitionKind::Variable | DefinitionKind::Parameter => {
                                lsp::CompletionItemKind::VARIABLE
                            }
                        }),
                        text_edit: Some(lsp::CompletionTextEdit::Edit(lsp::TextEdit {
                            range: text.range(item.replace),
                            new_text: item.name,
                        })),
                        ..Default::default()
                    })
                    .collect::<Vec<_>>();
                serde_json::to_value(lsp::CompletionList {
                    is_incomplete: false,
                    items,
                })?
            }
            _ => unreachable!(),
        };
        self.send(Response::new_ok(id, result))
    }

    fn update(&mut self, change: Change) -> Result<()> {
        let revision = self.revision.fetch_add(1, Ordering::AcqRel) + 1;
        for (id, _) in self.pending.drain().collect::<Vec<_>>() {
            self.error(
                id,
                ErrorCode::ContentModified as i32,
                "document or dependencies changed",
            )?;
        }
        self.updates.send(Update {
            revision,
            roots: self.documents.values().map(|d| d.path.clone()).collect(),
            change,
        })?;
        Ok(())
    }

    fn notification(&mut self, notification: Notification) -> Result<()> {
        // Invalid notifications cannot receive JSON-RPC error responses.
        let result = self.apply_notification(&notification.method, notification.params);
        if let Err(error) = result {
            eprintln!("resin-lsp: {}: {error}", notification.method);
        }
        Ok(())
    }

    fn apply_notification(&mut self, method: &str, params: Value) -> Result<()> {
        match method {
            "textDocument/didOpen" => {
                let params: lsp::DidOpenTextDocumentParams = serde_json::from_value(params)?;
                let doc = params.text_document;
                let path = uri_path(&doc.uri)?;
                if self.documents.contains_key(doc.uri.as_str()) {
                    return Err("document is already open".into());
                }
                if self.documents.values().any(|open| open.path == path) {
                    return Err("the same file is already open under another URI".into());
                }
                self.documents.insert(
                    doc.uri.as_str().into(),
                    OpenDocument {
                        uri: doc.uri,
                        path: path.clone(),
                        version: doc.version,
                        source: doc.text.clone(),
                    },
                );
                self.update(Change::Set(path, doc.text))?;
            }
            "textDocument/didChange" => {
                let params: lsp::DidChangeTextDocumentParams = serde_json::from_value(params)?;
                let Some(document) = self.documents.get_mut(params.text_document.uri.as_str())
                else {
                    return Err("document is not open".into());
                };
                if params.text_document.version <= document.version {
                    return Err("ignoring an out-of-order document version".into());
                }
                if params.content_changes.iter().any(|c| c.range.is_some()) {
                    return Err("server negotiated full-document synchronization".into());
                }
                let Some(change) = params.content_changes.into_iter().last() else {
                    return Ok(());
                };
                document.version = params.text_document.version;
                document.source = change.text.clone();
                let path = document.path.clone();
                self.update(Change::Set(path, change.text))?;
            }
            "textDocument/didSave" => {
                let params: lsp::DidSaveTextDocumentParams = serde_json::from_value(params)?;
                self.update(Change::Disk(vec![uri_path(&params.text_document.uri)?]))?;
            }
            "textDocument/didClose" => {
                let params: lsp::DidCloseTextDocumentParams = serde_json::from_value(params)?;
                if let Some(document) = self.documents.remove(params.text_document.uri.as_str()) {
                    self.update(Change::Close(document.path))?;
                }
            }
            "workspace/didChangeWatchedFiles" => {
                let params: lsp::DidChangeWatchedFilesParams = serde_json::from_value(params)?;
                let paths = params
                    .changes
                    .iter()
                    .map(|event| uri_path(&event.uri))
                    .collect::<Result<Vec<_>>>()?;
                self.update(Change::Disk(paths))?;
            }
            "$/cancelRequest" => {
                if let Some(id) = params
                    .get("id")
                    .and_then(|id| serde_json::from_value::<RequestId>(id.clone()).ok())
                    && self.pending.remove(&id).is_some()
                {
                    self.error(id, ErrorCode::RequestCanceled as i32, "request cancelled")?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn uri(&self, path: &Path) -> Option<Uri> {
        self.documents
            .values()
            .find(|d| d.path == path)
            .map(|d| d.uri.clone())
            .or_else(|| url::Url::from_file_path(path).ok()?.as_str().parse().ok())
    }
    fn location(&self, location: &SourceLocation) -> Option<lsp::Location> {
        Some(lsp::Location {
            uri: self.uri(&location.path)?,
            range: self
                .texts
                .get(&location.path)
                .map(|text| text.range(location.span))
                .unwrap_or_default(),
        })
    }

    fn publish(&mut self) -> Result<()> {
        let mut diagnostics = BTreeMap::<String, Vec<lsp::Diagnostic>>::new();
        if let Some(snapshot) = &self.snapshot {
            for analysis in snapshot.entries.values() {
                for diagnostic in analysis.diagnostics() {
                    let Some(location) = self.location(&diagnostic.location) else {
                        continue;
                    };
                    let related = diagnostic
                        .related
                        .iter()
                        .filter_map(|note| {
                            Some(lsp::DiagnosticRelatedInformation {
                                location: self.location(&note.location)?,
                                message: note.message.clone(),
                            })
                        })
                        .collect::<Vec<_>>();
                    let diagnostic = lsp::Diagnostic {
                        range: location.range,
                        severity: Some(lsp::DiagnosticSeverity::ERROR),
                        source: Some("resin".into()),
                        message: diagnostic.message.clone(),
                        related_information: (!related.is_empty()).then_some(related),
                        ..Default::default()
                    };
                    let list = diagnostics.entry(location.uri.as_str().into()).or_default();
                    if !list.contains(&diagnostic) {
                        list.push(diagnostic);
                    }
                }
            }
        }
        let current = diagnostics
            .keys()
            .chain(self.documents.keys())
            .cloned()
            .collect::<BTreeSet<_>>();
        for uri in current.union(&self.published) {
            let params = lsp::PublishDiagnosticsParams {
                uri: uri.parse()?,
                diagnostics: diagnostics.remove(uri).unwrap_or_default(),
                version: self.documents.get(uri).map(|d| d.version),
            };
            self.connection
                .sender
                .send(Message::Notification(Notification::new(
                    "textDocument/publishDiagnostics".into(),
                    params,
                )))?;
        }
        self.published = current;
        Ok(())
    }
}

/// Trim unchanged text around a replacement, preserving UTF-8 and CRLF boundaries.
fn formatting_edits(source: &str, formatted: &str) -> Vec<lsp::TextEdit> {
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
    vec![lsp::TextEdit {
        range: Text::new(source).range(resin_common::source::Span { start, end }),
        new_text: formatted[start..new_end].into(),
    }]
}

fn uri_path(uri: &Uri) -> Result<PathBuf> {
    let path = url::Url::parse(uri.as_str())?
        .to_file_path()
        .map_err(|_| "only local file:// documents are supported")?;
    Ok(normalize_path(&path)?)
}
