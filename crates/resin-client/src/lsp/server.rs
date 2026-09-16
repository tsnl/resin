//! Accept editor state and bounded requests; compiler work belongs to the coordinator.
use super::worker::{
    self, Channels, DocumentVersion, EditorSnapshot, OpenDocument, QueryKind, RequestJob,
    WorkerConfig,
};
use crossbeam_channel::select;
use lsp_server::{Connection, ErrorCode, Message, Notification, Request, RequestId, Response};
use lsp_types::Uri;
use resin_executor::Cancellation;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::PathBuf,
    sync::Arc,
};

type Result<T> = super::Result<T>;

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
struct Options {
    include_roots: Vec<PathBuf>,
}

pub(crate) fn run(project: PathBuf, include_roots: Vec<PathBuf>) -> Result<i32> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let client = runtime.block_on(crate::interp::connect())?;
    let mirrors = Arc::new(super::mirrors::Mirrors::new()?);
    let (connection, io) = Connection::stdio();
    let outcome = serve(
        connection,
        project,
        include_roots,
        client,
        mirrors,
        &runtime,
    );
    io.join()?;
    outcome
}

fn serve(
    connection: Connection,
    project: PathBuf,
    mut include_roots: Vec<PathBuf>,
    client: crate::Client,
    mirrors: Arc<super::mirrors::Mirrors>,
    runtime: &tokio::runtime::Runtime,
) -> Result<i32> {
    let (id, params) = connection.initialize_start()?;
    let params: lsp_types::InitializeParams = serde_json::from_value(params)?;
    let options: Options = params
        .initialization_options
        .clone()
        .filter(|v| !v.is_null())
        .map(serde_json::from_value)
        .transpose()?
        .unwrap_or_default();
    include_roots.extend(
        options
            .include_roots
            .into_iter()
            .map(|path| project.join(path)),
    );
    connection.initialize_finish(
        id,
        json!({"capabilities": capabilities(), "serverInfo": {
            "name": "resin-lsp", "version": env!("CARGO_PKG_VERSION")
        }}),
    )?;
    let watching = params
        .capabilities
        .workspace
        .as_ref()
        .and_then(|w| w.did_change_watched_files.as_ref());
    if watching.is_some_and(|watching| watching.dynamic_registration == Some(true)) {
        let roots = include_roots.clone();
        let relative =
            watching.is_some_and(|watching| watching.relative_pattern_support == Some(true));
        let watchers =
            runtime.block_on(client.execution.run(&Cancellation::new(), move |_| {
                file_watchers(&roots, relative)
            }))??;
        connection.sender.send(Message::Request(Request::new(
            "resin-file-watchers".to_owned().into(),
            "client/registerCapability".into(),
            json!({"registrations": [{"id": "resin-files",
            "method": "workspace/didChangeWatchedFiles", "registerOptions": {
            "watchers": watchers}}]}),
        )))?;
    }
    let config = WorkerConfig {
        client,
        include_roots,
        mirrors,
    };
    let editor = Arc::new(EditorSnapshot {
        revision: 0,
        registrations: 0,
        disk_revision: 0,
        documents: BTreeMap::new(),
    });
    let (channels, worker) = worker::spawn(config, editor.clone());
    let mut state = State {
        connection,
        channels,
        editor,
        next_epoch: 0,
        next_request: 0,
        published: BTreeSet::new(),
        pending: HashMap::new(),
        shutdown: false,
    };
    let outcome = state.events();
    state.channels.shutdown.cancel();
    drop(state);
    worker.join().map_err(|_| "compiler coordinator panicked")?;
    outcome
}

fn file_watchers(include_roots: &[PathBuf], relative_patterns: bool) -> Result<Vec<Value>> {
    // Header bundles include arbitrary filenames and transitive inputs. Watching
    // only .resin or .h would miss extensionless files and newly shadowing headers.
    let mut watchers = vec![json!({"globPattern": "**/*", "kind": 7})];
    for root in include_roots {
        let root = resin_source::normalize_path(root)?;
        let uri = url::Url::from_directory_path(root)
            .map_err(|_| "header include roots must be absolute directories")?;
        let pattern = if relative_patterns {
            json!({"baseUri": uri.as_str(), "pattern": "**/*"})
        } else {
            let path = uri
                .to_file_path()
                .map_err(|_| "invalid header include root")?;
            json!(recursive_glob(&path))
        };
        let watcher = json!({"globPattern": pattern, "kind": 7});
        if !watchers.contains(&watcher) {
            watchers.push(watcher);
        }
    }
    Ok(watchers)
}

fn recursive_glob(path: &std::path::Path) -> String {
    let mut pattern = String::new();
    for character in path.to_string_lossy().chars() {
        match character {
            '\\' if cfg!(windows) => pattern.push('/'),
            '*' | '?' | '[' | ']' | '{' | '}' => {
                pattern.push('[');
                pattern.push(character);
                pattern.push(']');
            }
            _ => pattern.push(character),
        }
    }
    if !pattern.ends_with('/') {
        pattern.push('/');
    }
    pattern.push_str("**/*");
    pattern
}

fn capabilities() -> lsp_types::ServerCapabilities {
    lsp_types::ServerCapabilities {
        position_encoding: Some(lsp_types::PositionEncodingKind::UTF16),
        text_document_sync: Some(
            lsp_types::TextDocumentSyncOptions {
                open_close: Some(true),
                change: Some(lsp_types::TextDocumentSyncKind::FULL),
                save: Some(
                    lsp_types::SaveOptions {
                        include_text: Some(false),
                    }
                    .into(),
                ),
                ..Default::default()
            }
            .into(),
        ),
        hover_provider: Some(lsp_types::HoverProviderCapability::Simple(true)),
        definition_provider: Some(lsp_types::OneOf::Left(true)),
        document_formatting_provider: Some(lsp_types::OneOf::Left(true)),
        completion_provider: Some(lsp_types::CompletionOptions {
            resolve_provider: Some(false),
            trigger_characters: Some(vec![".".into()]),
            ..Default::default()
        }),
        execute_command_provider: Some(lsp_types::ExecuteCommandOptions {
            commands: vec!["resin.build".into()],
            ..Default::default()
        }),
        ..Default::default()
    }
}

struct State {
    connection: Connection,
    channels: Channels,
    editor: Arc<EditorSnapshot>,
    next_epoch: u64,
    next_request: u64,
    published: BTreeSet<String>,
    pending: HashMap<RequestId, Pending>,
    shutdown: bool,
}

struct Pending {
    serial: u64,
    cancellation: Cancellation,
}

impl State {
    fn events(&mut self) -> Result<i32> {
        let never = crossbeam_channel::never();
        loop {
            // Shutdown drains the coordinator before the client necessarily sends exit.
            let ready = if self.shutdown {
                &never
            } else {
                &self.channels.ready
            };
            select! {
                recv(self.connection.receiver) -> message => match message {
                    Ok(Message::Request(request)) => self.request(request)?,
                    Ok(Message::Notification(notification)) => {
                        if notification.method == "exit" { return Ok(if self.shutdown { 0 } else { 1 }); }
                        if !self.shutdown { self.notification(notification); }
                    }
                    Ok(Message::Response(response)) => {
                        if let Some(error) = response.error { eprintln!("resin-lsp: client request failed: {}", error.message); }
                    }
                    Err(_) => return Ok(if self.shutdown { 0 } else { 1 }),
                },
                recv(ready) -> ready => {
                    if ready.is_err() { return Ok(if self.shutdown { 0 } else { 1 }); }
                    self.receive_results()?;
                }
            }
        }
    }

    fn send(&self, response: Response) -> Result<()> {
        self.connection.sender.send(Message::Response(response))?;
        Ok(())
    }

    fn error(&self, id: RequestId, code: ErrorCode, message: impl Into<String>) -> Result<()> {
        self.send(Response::new_err(id, code as i32, message.into()))
    }

    fn request(&mut self, request: Request) -> Result<()> {
        if self.shutdown {
            return self.error(
                request.id,
                ErrorCode::InvalidRequest,
                "server is shutting down",
            );
        }
        if request.method == "shutdown" {
            return self.begin_shutdown(request.id);
        }
        let kind = match parse_query(&request.method, request.params) {
            Ok(kind) => kind,
            Err((code, message)) => return self.error(request.id, code, message),
        };
        if query_uri(&kind).is_some_and(|uri| !self.editor.documents.contains_key(uri.as_str())) {
            return self.send(Response::new_ok(request.id, Value::Null));
        }
        if self.pending.contains_key(&request.id) {
            return self.error(
                request.id,
                ErrorCode::InvalidRequest,
                "request ID is already pending",
            );
        }
        let Ok(admission) = self.channels.admission.clone().try_acquire_owned() else {
            return self.error(
                request.id,
                ErrorCode::ServerCancelled,
                "server request capacity reached; retry after pending requests finish",
            );
        };
        let id = request.id;
        let serial = self.next_request;
        self.next_request += 1;
        let cancellation = Cancellation::new();
        let job = RequestJob {
            serial,
            id: id.clone(),
            kind,
            editor: self.editor.clone(),
            cancellation: cancellation.clone(),
            admission,
        };
        if self.channels.requests.try_send(job).is_err() {
            return self.error(
                id,
                ErrorCode::ServerCancelled,
                "server request queue is full or shutting down",
            );
        }
        self.pending.insert(
            id,
            Pending {
                serial,
                cancellation,
            },
        );
        Ok(())
    }

    fn begin_shutdown(&mut self, id: RequestId) -> Result<()> {
        self.shutdown = true;
        self.channels.shutdown.cancel();
        for (pending, request) in self.pending.drain().collect::<Vec<_>>() {
            request.cancellation.cancel();
            self.error(
                pending,
                ErrorCode::RequestCanceled,
                "server is shutting down",
            )?;
        }
        self.send(Response::new_ok(id, Value::Null))
    }

    fn receive_results(&mut self) -> Result<()> {
        while let Ok(reply) = self.channels.replies.try_recv() {
            if self
                .pending
                .get(&reply.response.id)
                .is_none_or(|pending| pending.serial != reply.serial)
            {
                continue;
            }
            self.pending.remove(&reply.response.id);
            if reply.freshness.matches(&self.editor) {
                self.send(reply.response)?;
            } else {
                self.error(
                    reply.response.id,
                    ErrorCode::ContentModified,
                    "document or dependencies changed",
                )?;
            }
            drop(reply.admission);
        }
        if self.shutdown {
            return Ok(());
        }
        let publication = self
            .channels
            .publications
            .lock()
            .expect("diagnostic mailbox")
            .take();
        let Some(publication) = publication else {
            return Ok(());
        };
        if publication.revision != self.editor.revision {
            return Ok(());
        }
        for notification in publication.notifications {
            self.connection
                .sender
                .send(Message::Notification(notification))?;
        }
        // Track actual sends: coalescing may skip an earlier prepared clear.
        for uri in self.published.difference(&publication.uris) {
            self.connection
                .sender
                .send(Message::Notification(Notification::new(
                    "textDocument/publishDiagnostics".into(),
                    json!({ "uri": uri, "diagnostics": [] }),
                )))?;
        }
        self.published = publication.uris;
        Ok(())
    }

    fn notification(&mut self, notification: Notification) {
        if let Err(error) = self.apply_notification(&notification.method, notification.params) {
            eprintln!("resin-lsp: {}: {error}", notification.method);
        }
    }

    fn next_editor(&self) -> EditorSnapshot {
        EditorSnapshot {
            revision: self.editor.revision + 1,
            registrations: self.editor.registrations,
            disk_revision: self.editor.disk_revision,
            documents: self.editor.documents.clone(),
        }
    }

    fn accept(&mut self, next: EditorSnapshot) {
        self.editor = Arc::new(next);
        self.channels.editor.send_replace(self.editor.clone());
    }

    fn apply_notification(&mut self, method: &str, params: Value) -> Result<()> {
        match method {
            "textDocument/didOpen" => {
                let params: lsp_types::DidOpenTextDocumentParams = serde_json::from_value(params)?;
                let document = params.text_document;
                let path = uri_path(&document.uri)?;
                if self.editor.documents.contains_key(document.uri.as_str()) {
                    return Err("document is already open".into());
                }
                if self.editor.documents.values().any(|open| open.path == path) {
                    return Err("the same file is already open under another URI".into());
                }
                self.next_epoch += 1;
                let mut next = self.next_editor();
                next.registrations += 1;
                next.documents.insert(
                    document.uri.as_str().to_owned(),
                    Arc::new(OpenDocument {
                        uri: document.uri,
                        path,
                        stamp: DocumentVersion {
                            epoch: self.next_epoch,
                            version: document.version,
                        },
                        text: Arc::new(document.text),
                    }),
                );
                self.accept(next);
            }
            "textDocument/didChange" => {
                let params: lsp_types::DidChangeTextDocumentParams =
                    serde_json::from_value(params)?;
                let Some(document) = self.editor.documents.get(params.text_document.uri.as_str())
                else {
                    return Err("document is not open".into());
                };
                if params.text_document.version <= document.stamp.version {
                    return Err("ignoring an out-of-order document version".into());
                }
                if params
                    .content_changes
                    .iter()
                    .any(|change| change.range.is_some())
                {
                    return Err("server negotiated full-document synchronization".into());
                }
                let Some(change) = params.content_changes.into_iter().last() else {
                    return Ok(());
                };
                let changed = Arc::new(OpenDocument {
                    uri: document.uri.clone(),
                    path: document.path.clone(),
                    stamp: DocumentVersion {
                        epoch: document.stamp.epoch,
                        version: params.text_document.version,
                    },
                    text: Arc::new(change.text),
                });
                let mut next = self.next_editor();
                next.documents
                    .insert(changed.uri.as_str().to_owned(), changed);
                self.accept(next);
            }
            "textDocument/didSave" => {
                let params: lsp_types::DidSaveTextDocumentParams = serde_json::from_value(params)?;
                uri_path(&params.text_document.uri)?;
                let mut next = self.next_editor();
                next.disk_revision += 1;
                self.accept(next);
            }
            "textDocument/didClose" => {
                let params: lsp_types::DidCloseTextDocumentParams = serde_json::from_value(params)?;
                if self
                    .editor
                    .documents
                    .contains_key(params.text_document.uri.as_str())
                {
                    let mut next = self.next_editor();
                    next.documents.remove(params.text_document.uri.as_str());
                    next.registrations += 1;
                    self.accept(next);
                }
            }
            "workspace/didChangeWatchedFiles" => {
                let params: lsp_types::DidChangeWatchedFilesParams =
                    serde_json::from_value(params)?;
                for change in params.changes {
                    uri_path(&change.uri)?;
                }
                let mut next = self.next_editor();
                next.disk_revision += 1;
                self.accept(next);
            }
            "$/cancelRequest" => {
                if let Some(id) = params
                    .get("id")
                    .and_then(|id| serde_json::from_value::<RequestId>(id.clone()).ok())
                    && let Some(request) = self.pending.remove(&id)
                {
                    request.cancellation.cancel();
                    self.error(id, ErrorCode::RequestCanceled, "request cancelled")?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn query_uri(kind: &QueryKind) -> Option<&Uri> {
    match kind {
        QueryKind::Hover { uri, .. }
        | QueryKind::Definition { uri, .. }
        | QueryKind::Completion { uri, .. }
        | QueryKind::Format { uri } => Some(uri),
        QueryKind::Build { .. } => None,
    }
}

fn parse_query(method: &str, params: Value) -> std::result::Result<QueryKind, (ErrorCode, String)> {
    let invalid = |error: serde_json::Error| (ErrorCode::InvalidParams, error.to_string());
    match method {
        "textDocument/formatting" => {
            let params: lsp_types::DocumentFormattingParams =
                serde_json::from_value(params).map_err(invalid)?;
            Ok(QueryKind::Format {
                uri: params.text_document.uri,
            })
        }
        "textDocument/hover" | "textDocument/definition" | "textDocument/completion" => {
            let params: lsp_types::TextDocumentPositionParams =
                serde_json::from_value(params).map_err(invalid)?;
            let (uri, position) = (params.text_document.uri, params.position);
            Ok(match method {
                "textDocument/hover" => QueryKind::Hover { uri, position },
                "textDocument/definition" => QueryKind::Definition { uri, position },
                _ => QueryKind::Completion { uri, position },
            })
        }
        "workspace/executeCommand" => {
            let params: lsp_types::ExecuteCommandParams =
                serde_json::from_value(params).map_err(invalid)?;
            if params.command != "resin.build" {
                return Err((ErrorCode::InvalidParams, "unsupported command".into()));
            }
            let [request]: [Value; 1] = params.arguments.try_into().map_err(|_| {
                (
                    ErrorCode::InvalidParams,
                    "resin.build requires one request object".into(),
                )
            })?;
            Ok(QueryKind::Build {
                request: serde_json::from_value(request).map_err(invalid)?,
            })
        }
        _ => Err((
            ErrorCode::MethodNotFound,
            format!("unsupported method: {method}"),
        )),
    }
}

fn uri_path(uri: &Uri) -> Result<PathBuf> {
    url::Url::parse(uri.as_str())?
        .to_file_path()
        .map_err(|_| "only local file:// documents are supported".into())
}

#[cfg(test)]
mod tests {
    use super::worker::{Freshness, PreparedDiagnostics, PreparedReply};
    use super::*;
    use std::sync::Mutex;
    use tokio::sync::{Semaphore, mpsc, watch};

    struct Harness {
        state: State,
        client: Connection,
        requests: mpsc::Receiver<RequestJob>,
        replies: mpsc::Sender<PreparedReply>,
        publications: Arc<Mutex<Option<PreparedDiagnostics>>>,
        editor: watch::Receiver<Arc<EditorSnapshot>>,
        wake: crossbeam_channel::Sender<()>,
    }

    fn harness() -> Harness {
        let (connection, client) = Connection::memory();
        let editor = Arc::new(EditorSnapshot::default());
        let (editor_tx, editor_rx) = watch::channel(editor.clone());
        let (requests, incoming) = mpsc::channel(64);
        let (replies, receiver) = mpsc::channel(64);
        let publications = Arc::new(Mutex::new(None));
        let (wake, ready) = crossbeam_channel::bounded(1);
        let channels = Channels {
            editor: editor_tx,
            requests,
            admission: Arc::new(Semaphore::new(64)),
            replies: receiver,
            publications: publications.clone(),
            ready,
            shutdown: Cancellation::new(),
        };
        Harness {
            state: State {
                connection,
                channels,
                editor,
                next_epoch: 0,
                next_request: 0,
                published: BTreeSet::new(),
                pending: HashMap::new(),
                shutdown: false,
            },
            client,
            requests: incoming,
            replies,
            publications,
            editor: editor_rx,
            wake,
        }
    }

    fn uri(name: &str) -> Uri {
        url::Url::from_file_path(std::env::temp_dir().join(name))
            .unwrap()
            .as_str()
            .parse()
            .unwrap()
    }

    fn open(state: &mut State, uri: &Uri, text: &str) {
        state
            .apply_notification(
                "textDocument/didOpen",
                json!({"textDocument": {
                    "uri": uri, "version": 1, "languageId": "resin", "text": text
                }}),
            )
            .unwrap();
    }

    fn change(state: &mut State, uri: &Uri, version: i32, text: &str) {
        state
            .apply_notification(
                "textDocument/didChange",
                json!({"textDocument": {
            "uri": uri, "version": version
        }, "contentChanges": [{"text": text}]}),
            )
            .unwrap();
    }

    fn format(state: &mut State, id: i32, uri: &Uri) {
        state
            .request(Request::new(
                id.into(),
                "textDocument/formatting".into(),
                json!({
                    "textDocument": {"uri": uri}, "options": {"tabSize": 4, "insertSpaces": true}
                }),
            ))
            .unwrap();
    }

    fn response(client: &Connection) -> Response {
        match client.receiver.try_recv().unwrap() {
            Message::Response(response) => response,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn include_root_watchers_support_external_directories_and_literal_glob_characters() {
        let root = std::env::temp_dir().join("native[2]").join("{headers}?");
        let roots = vec![root.clone(), root.clone()];
        let relative = file_watchers(&roots, true).unwrap();
        assert_eq!(relative.len(), 2);
        assert_eq!(relative[0]["globPattern"], "**/*");
        assert_eq!(relative[1]["globPattern"]["pattern"], "**/*");
        let base =
            url::Url::parse(relative[1]["globPattern"]["baseUri"].as_str().unwrap()).unwrap();
        assert_eq!(
            base,
            url::Url::from_directory_path(resin_source::normalize_path(&root).unwrap()).unwrap()
        );
        let absolute = file_watchers(&roots, false).unwrap();
        let pattern = absolute[1]["globPattern"].as_str().unwrap();
        assert!(
            pattern.ends_with("native[[]2[]]/[{]headers[}][?]/**/*"),
            "{pattern}"
        );
    }

    #[test]
    fn full_edits_coalesce_and_reopened_epochs_survive_skipped_snapshots() {
        let mut harness = harness();
        let file = uri("resin-protocol-coalesce.resin");
        let other = uri("resin-protocol-other.resin");
        open(&mut harness.state, &file, "old text");
        open(&mut harness.state, &other, "still dirty");
        let old_epoch = harness.state.editor.documents[file.as_str()].stamp.epoch;
        let old_text = Arc::downgrade(&harness.state.editor.documents[file.as_str()].text);
        for version in 2..1002 {
            change(&mut harness.state, &file, version, "latest edit");
        }
        assert!(old_text.upgrade().is_none());
        harness
            .state
            .apply_notification(
                "textDocument/didSave",
                json!({"textDocument": {"uri": file}}),
            )
            .unwrap();
        assert_eq!(
            &**harness.state.editor.documents[other.as_str()].text,
            "still dirty"
        );
        harness
            .state
            .apply_notification(
                "textDocument/didClose",
                json!({"textDocument": {"uri": file}}),
            )
            .unwrap();
        open(&mut harness.state, &file, "reopened");
        let latest = harness.editor.borrow_and_update().clone();
        assert!(latest.documents[file.as_str()].stamp.epoch > old_epoch);
        assert_eq!(latest.documents[file.as_str()].stamp.version, 1);
        assert_eq!(&**latest.documents[file.as_str()].text, "reopened");
        assert_eq!(latest.disk_revision, 1);
        assert_eq!(latest.registrations, 4);
        assert!(!harness.editor.has_changed().unwrap());
    }

    #[test]
    fn admission_survives_cancellation_until_the_queued_job_is_released() {
        let mut harness = harness();
        let file = uri("resin-protocol-capacity.resin");
        open(&mut harness.state, &file, "fn main()  {} ");
        for id in 0..64 {
            format(&mut harness.state, id, &file);
        }
        format(&mut harness.state, 64, &file);
        assert_eq!(
            response(&harness.client).error.unwrap().code,
            ErrorCode::ServerCancelled as i32
        );
        harness
            .state
            .apply_notification("$/cancelRequest", json!({"id": 0}))
            .unwrap();
        assert_eq!(
            response(&harness.client).error.unwrap().code,
            ErrorCode::RequestCanceled as i32
        );
        assert!(
            !harness.state.pending[&RequestId::from(1)]
                .cancellation
                .is_cancelled()
        );
        format(&mut harness.state, 65, &file);
        assert_eq!(
            response(&harness.client).error.unwrap().code,
            ErrorCode::ServerCancelled as i32
        );
        let canceled = harness.requests.try_recv().unwrap();
        assert!(canceled.cancellation.is_cancelled());
        drop(canceled);
        format(&mut harness.state, 66, &file);
        assert!(harness.state.pending.contains_key(&RequestId::from(66)));
        harness
            .state
            .request(Request::new(67.into(), "unrecognized".into(), Value::Null))
            .unwrap();
        assert_eq!(
            response(&harness.client).error.unwrap().code,
            ErrorCode::MethodNotFound as i32
        );
    }

    #[test]
    fn a_prepared_reply_is_rechecked_after_later_editor_events() {
        let mut harness = harness();
        let file = uri("resin-protocol-stale.resin");
        open(&mut harness.state, &file, "fn main(){}");
        format(&mut harness.state, 1, &file);
        let job = harness.requests.try_recv().unwrap();
        let freshness = Freshness::document(&job.editor.documents[file.as_str()]);
        change(&mut harness.state, &file, 2, "fn main()  { 42 } ");
        harness
            .replies
            .try_send(PreparedReply {
                serial: job.serial,
                response: Response::new_ok(job.id, json!([])),
                freshness,
                admission: job.admission,
            })
            .ok()
            .unwrap();
        harness.state.receive_results().unwrap();
        assert_eq!(
            response(&harness.client).error.unwrap().code,
            ErrorCode::ContentModified as i32
        );
        assert_eq!(harness.state.channels.admission.available_permits(), 64);
    }

    #[test]
    fn coalesced_diagnostics_still_clear_every_uri_actually_published() {
        let mut harness = harness();
        let first = uri("resin-protocol-first.resin");
        let second = uri("resin-protocol-second.resin");
        let publication = |uri: &Uri| PreparedDiagnostics {
            revision: 0,
            notifications: vec![Notification::new(
                "textDocument/publishDiagnostics".into(),
                json!({"uri": uri, "diagnostics": []}),
            )],
            uris: BTreeSet::from([uri.as_str().to_owned()]),
        };
        *harness.publications.lock().unwrap() = Some(publication(&first));
        harness.state.receive_results().unwrap();
        let _ = harness.client.receiver.try_recv().unwrap();
        *harness.publications.lock().unwrap() = Some(PreparedDiagnostics::default());
        *harness.publications.lock().unwrap() = Some(publication(&second));
        harness.state.receive_results().unwrap();
        let messages = harness.client.receiver.try_iter().collect::<Vec<_>>();
        assert_eq!(messages.len(), 2);
        assert!(messages.iter().any(|message| matches!(message, Message::Notification(notification)
            if notification.params["uri"] == first.as_str() && notification.params["diagnostics"] == json!([]))));
        assert_eq!(
            harness.state.published,
            BTreeSet::from([second.as_str().to_owned()])
        );
    }

    #[test]
    fn reused_wire_id_cannot_receive_the_cancelled_requests_late_reply() {
        let mut harness = harness();
        let file = uri("resin-protocol-reused-id.resin");
        open(&mut harness.state, &file, "fn main()  {} ");
        format(&mut harness.state, 7, &file);
        let old = harness.requests.try_recv().unwrap();
        harness
            .state
            .apply_notification("$/cancelRequest", json!({"id": 7}))
            .unwrap();
        assert_eq!(
            response(&harness.client).error.unwrap().code,
            ErrorCode::RequestCanceled as i32
        );
        format(&mut harness.state, 7, &file);
        let new = harness.requests.try_recv().unwrap();
        assert_ne!(old.serial, new.serial);
        for (job, text) in [(old, "old"), (new, "new")] {
            harness
                .replies
                .try_send(PreparedReply {
                    serial: job.serial,
                    response: Response::new_ok(job.id, text),
                    freshness: Freshness::default(),
                    admission: job.admission,
                })
                .ok()
                .unwrap();
            harness.state.receive_results().unwrap();
        }
        assert_eq!(response(&harness.client).result.unwrap(), "new");
        assert!(harness.client.receiver.try_recv().is_err());
        assert_eq!(harness.state.channels.admission.available_permits(), 64);
    }

    #[test]
    fn shutdown_waits_for_exit_after_the_coordinator_wake_channel_closes() {
        let harness = harness();
        let mut state = harness.state;
        let pending = std::thread::spawn(move || state.events());
        harness
            .client
            .sender
            .send(Message::Request(Request::new(
                1.into(),
                "shutdown".into(),
                Value::Null,
            )))
            .unwrap();
        let reply = harness
            .client
            .receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        assert!(matches!(reply, Message::Response(response) if response.id == RequestId::from(1)));
        drop(harness.wake);
        harness
            .client
            .sender
            .send(Message::Notification(Notification::new(
                "exit".into(),
                Value::Null,
            )))
            .unwrap();
        assert_eq!(pending.join().unwrap().unwrap(), 0);
    }
}
