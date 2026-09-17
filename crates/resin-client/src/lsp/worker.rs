//! Bounded asynchronous scheduling for accepted editor snapshots and independent requests.
use super::analysis::{self, RootAnalysis, RootTicket};
use lsp_server::{ErrorCode, RequestId, Response};
use lsp_types::{Position, Uri};
use resin_executor::{Cancellation, Execution};
use resin_source::{Loader, Source};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    path::PathBuf,
    sync::Arc,
    thread,
};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore, mpsc, watch},
    task::JoinSet,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DocumentVersion {
    pub epoch: u64,
    pub version: i32,
}

pub(super) struct OpenDocument {
    pub uri: Uri,
    pub path: PathBuf,
    pub stamp: DocumentVersion,
    pub text: Arc<String>,
}

#[derive(Default)]
pub(super) struct EditorSnapshot {
    pub revision: u64,
    pub registrations: u64,
    pub disk_revision: u64,
    pub documents: BTreeMap<String, Arc<OpenDocument>>,
}

#[derive(Clone, Default)]
pub(super) struct Freshness {
    pub document: Option<(Uri, DocumentVersion)>,
    pub dependencies: Vec<(Uri, DocumentVersion)>,
    pub registrations: Option<u64>,
    pub disk_revision: Option<u64>,
}

impl Freshness {
    pub(super) fn document(document: &OpenDocument) -> Self {
        Self {
            document: Some((document.uri.clone(), document.stamp)),
            ..Self::default()
        }
    }

    pub(super) fn matches(&self, editor: &EditorSnapshot) -> bool {
        self.document
            .iter()
            .chain(self.dependencies.iter())
            .all(|(uri, stamp)| {
                editor
                    .documents
                    .get(uri.as_str())
                    .is_some_and(|document| document.stamp == *stamp)
            })
            && self
                .registrations
                .is_none_or(|revision| revision == editor.registrations)
            && self
                .disk_revision
                .is_none_or(|revision| revision == editor.disk_revision)
    }
}

pub(super) enum QueryKind {
    Hover { uri: Uri, position: Position },
    Definition { uri: Uri, position: Position },
    Completion { uri: Uri, position: Position },
    Format { uri: Uri },
    Build { request: super::build::BuildRequest },
}

impl QueryKind {
    fn uri(&self) -> Option<&Uri> {
        match self {
            Self::Hover { uri, .. }
            | Self::Definition { uri, .. }
            | Self::Completion { uri, .. }
            | Self::Format { uri } => Some(uri),
            Self::Build { .. } => None,
        }
    }
}

pub(super) struct RequestJob {
    pub serial: u64,
    pub id: RequestId,
    pub kind: QueryKind,
    pub editor: Arc<EditorSnapshot>,
    pub cancellation: Cancellation,
    pub admission: OwnedSemaphorePermit,
}

pub(super) struct PreparedReply {
    pub serial: u64,
    pub response: Response,
    pub freshness: Freshness,
    pub admission: OwnedSemaphorePermit,
}

#[derive(Default)]
pub(super) struct PreparedDiagnostics {
    pub revision: u64,
    pub notifications: Vec<lsp_server::Notification>,
    pub uris: BTreeSet<String>,
}

pub(super) struct Channels {
    pub editor: watch::Sender<Arc<EditorSnapshot>>,
    pub requests: mpsc::Sender<RequestJob>,
    pub admission: Arc<Semaphore>,
    pub replies: mpsc::Receiver<PreparedReply>,
    pub publications: Arc<std::sync::Mutex<Option<PreparedDiagnostics>>>,
    pub ready: crossbeam_channel::Receiver<()>,
    pub shutdown: Cancellation,
}

pub(super) struct WorkerConfig {
    pub client: crate::Client,
    pub include_roots: Vec<PathBuf>,
    pub mirrors: Arc<super::mirrors::Mirrors>,
}

pub(super) fn spawn(
    config: WorkerConfig,
    editor: Arc<EditorSnapshot>,
) -> (Channels, thread::JoinHandle<()>) {
    let (editor_sender, editor_receiver) = watch::channel(editor);
    let (requests, incoming) = mpsc::channel(64);
    let (outgoing, replies) = mpsc::channel(64);
    let publications = Arc::new(std::sync::Mutex::new(None));
    let published = publications.clone();
    let (wake, ready) = crossbeam_channel::bounded(1);
    let shutdown = Cancellation::new();
    let stopping = shutdown.clone();
    let thread = thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(error) => {
                eprintln!("resin-lsp: cannot start compiler runtime: {error}");
                return;
            }
        };
        runtime.block_on(coordinate(
            config,
            editor_receiver,
            incoming,
            outgoing,
            publications,
            wake,
            stopping,
        ));
    });
    (
        Channels {
            editor: editor_sender,
            requests,
            admission: Arc::new(Semaphore::new(64)),
            replies,
            publications: published,
            ready,
            shutdown,
        },
        thread,
    )
}

pub(super) struct RegisteredEditor {
    pub editor: Arc<EditorSnapshot>,
    pub loader: Loader,
    pub documents: BTreeMap<String, Source>,
}

async fn register_editor(
    previous: Arc<RegisteredEditor>,
    next: Arc<EditorSnapshot>,
    execution: Execution,
    cancellation: Cancellation,
) -> super::Result<Arc<RegisteredEditor>> {
    execution
        .run(&cancellation, move |cancellation| {
            let mut loader = previous.loader.supplied_snapshot();
            let retained: BTreeSet<_> = previous.documents.iter().filter_map(|(uri, source)| {
                let old_epoch = previous.editor.documents[uri].stamp.epoch;
                next.documents.get(uri).filter(|document| document.stamp.epoch == old_epoch).map(|_| source.id())
            }).collect();
            for source in previous.documents.values() {
                cancellation.check()?;
                // Several URIs can own one physical registration. Closing one must
                // not remove supplied text still owned by another open epoch.
                if !retained.contains(&source.id())
                    && let Some(path) = loader.path(source).map(std::path::Path::to_owned)
                {
                    loader.remove_source(&path)?;
                }
            }
            loader = loader.supplied_snapshot();
            let mut documents: BTreeMap<String, Source> = BTreeMap::new();
            let mut owners = BTreeMap::new();
            for (uri, document) in &next.documents {
                cancellation.check()?;
                let previous_document = previous
                    .editor
                    .documents
                    .get(uri.as_str())
                    .filter(|old| old.stamp.epoch == document.stamp.epoch);
                let registered =
                    previous_document.and_then(|_| previous.documents.get(uri.as_str()));
                let path = match registered.and_then(|source| previous.loader.path(source)) {
                    Some(path) => path.to_owned(),
                    // A new epoch must normalize even when another alias retains
                    // a supplied registration under this path's former identity.
                    None => resin_source::normalize_path(&document.path)?,
                };
                let source = if previous_document.is_some_and(|old| old.stamp == document.stamp) {
                    registered.expect("registered open document").clone()
                } else {
                    loader.source_from_text(&path, document.text.as_str())?
                };
                if let Some(owner) = owners.get(&source.id()) {
                    if documents[owner] != source {
                        return Err(format!(
                            "conflicting editor text for {}: {owner} and {uri}; synchronize or close one alias",
                            path.display()
                        ).into());
                    }
                } else {
                    owners.insert(source.id(), uri.clone());
                }
                documents.insert(uri.clone(), source);
            }
            // Keep no closed registrations or older cached text after applying the snapshot.
            let loader = loader.supplied_snapshot();
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(Arc::new(RegisteredEditor {
                editor: next,
                loader,
                documents,
            }))
        })
        .await?
}

struct RunningRoot {
    ticket: RootTicket,
    cancellation: Cancellation,
}
struct RootState {
    ticket: RootTicket,
    running: Option<RunningRoot>,
    current: Option<Arc<RootAnalysis>>,
    previous: Option<Arc<RootAnalysis>>,
    failure: Option<String>,
    pending: bool,
}

struct Scheduler {
    config: Arc<WorkerConfig>,
    execution: Execution,
    latest: Arc<EditorSnapshot>,
    registered: Arc<RegisteredEditor>,
    roots: BTreeMap<String, RootState>,
    pending: VecDeque<RequestJob>,
    requests: HashMap<u64, Cancellation>,
    registration_tasks: JoinSet<(u64, super::Result<Arc<RegisteredEditor>>)>,
    failed_registration: Option<u64>,
    root_tasks: JoinSet<(RootTicket, super::Result<RootAnalysis>)>,
    reply_tasks: JoinSet<PreparedReply>,
    diagnostic_tasks: JoinSet<(String, super::Result<PreparedDiagnostics>)>,
    diagnostics_dirty: bool,
}

impl Scheduler {
    fn new(config: WorkerConfig) -> Self {
        let empty = Arc::new(EditorSnapshot::default());
        Self {
            registered: Arc::new(RegisteredEditor {
                editor: empty.clone(),
                loader: Loader::new(PathBuf::new()),
                documents: BTreeMap::new(),
            }),
            config: Arc::new(config),
            execution: Execution::default(),
            latest: empty,
            roots: BTreeMap::new(),
            pending: VecDeque::new(),
            requests: HashMap::new(),
            registration_tasks: JoinSet::new(),
            failed_registration: None,
            root_tasks: JoinSet::new(),
            reply_tasks: JoinSet::new(),
            diagnostic_tasks: JoinSet::new(),
            diagnostics_dirty: true,
        }
    }

    fn accept(&mut self, editor: Arc<EditorSnapshot>) {
        if Arc::ptr_eq(&self.latest, &editor) {
            return;
        }
        self.roots.retain(|uri, state| {
            if editor.documents.contains_key(uri) {
                return true;
            }
            if let Some(running) = &state.running {
                running.cancellation.cancel();
            }
            false
        });
        for (uri, document) in &editor.documents {
            let state = self.roots.entry(uri.clone()).or_insert_with(|| RootState {
                ticket: RootTicket {
                    uri: document.uri.clone(),
                    epoch: document.stamp.epoch,
                    generation: editor.revision,
                },
                running: None,
                current: None,
                previous: None,
                failure: None,
                pending: true,
            });
            if state
                .current
                .as_ref()
                .is_some_and(|result| result.freshness.matches(&editor))
            {
                continue;
            }
            state.ticket = RootTicket {
                uri: document.uri.clone(),
                epoch: document.stamp.epoch,
                generation: editor.revision,
            };
            if let Some(current) = state.current.take() {
                state.previous = Some(current);
            }
            state.failure = None;
            state.pending = true;
            if let Some(running) = &state.running {
                running.cancellation.cancel();
            }
        }
        self.latest = editor;
        self.diagnostics_dirty = true;
    }

    fn schedule(&mut self, shutdown: &Cancellation) {
        self.invalidate_managed_snapshot();
        if self.registered.editor.revision != self.latest.revision
            && self.failed_registration != Some(self.latest.revision)
            && self.registration_tasks.is_empty()
        {
            let (previous, next, execution, cancellation) = (
                self.registered.clone(),
                self.latest.clone(),
                self.execution.clone(),
                shutdown.clone(),
            );
            self.registration_tasks.spawn(async move {
                let revision = next.revision;
                (
                    revision,
                    register_editor(previous, next, execution, cancellation).await,
                )
            });
        }
        if self.registered.editor.revision == self.latest.revision {
            for state in self.roots.values_mut() {
                if self.root_tasks.len() >= 4 {
                    break;
                }
                if !state.pending || state.running.is_some() {
                    continue;
                }
                let ticket = state.ticket.clone();
                let cancellation = Cancellation::new();
                state.running = Some(RunningRoot {
                    ticket: ticket.clone(),
                    cancellation: cancellation.clone(),
                });
                state.pending = false;
                let (registered, config, previous, execution) = (
                    self.registered.clone(),
                    self.config.clone(),
                    state.previous.take(),
                    self.execution.clone(),
                );
                self.root_tasks.spawn(async move {
                    let result = analysis::analyze_root(
                        ticket.clone(),
                        registered,
                        config,
                        previous,
                        execution,
                        cancellation,
                    )
                    .await;
                    (ticket, result)
                });
            }
        }
        self.schedule_requests();
        if self.diagnostics_dirty && self.diagnostic_tasks.is_empty() {
            let snapshot = self.config.client.capabilities().managed_snapshot.clone();
            let (revision, execution, cancellation) = (
                self.latest.revision,
                self.execution.clone(),
                shutdown.clone(),
            );
            let roots = self
                .roots
                .values()
                .filter_map(|state| state.current.clone())
                .filter(|root| root.captured.inputs.managed_snapshot == snapshot)
                .collect();
            self.diagnostic_tasks.spawn(async move {
                (
                    snapshot,
                    super::query::prepare_diagnostics(revision, roots, &execution, &cancellation)
                        .await,
                )
            });
            self.diagnostics_dirty = false;
        }
    }

    fn invalidate_managed_snapshot(&mut self) {
        let snapshot = self.config.client.capabilities().managed_snapshot.clone();
        for state in self.roots.values_mut() {
            if state
                .current
                .as_ref()
                .is_some_and(|current| current.captured.inputs.managed_snapshot != snapshot)
            {
                state.previous = state.current.take();
                state.failure = None;
                state.pending = true;
                self.diagnostics_dirty = true;
            }
        }
    }

    fn schedule_requests(&mut self) {
        for _ in 0..self.pending.len() {
            let job = self.pending.pop_front().expect("counted pending request");
            if job.cancellation.is_cancelled() {
                self.reply_tasks.spawn(async move {
                    error_reply(job, ErrorCode::RequestCanceled, "request cancelled")
                });
                continue;
            }
            match &job.kind {
                QueryKind::Build { .. } => {
                    let (config, execution) = (self.config.clone(), self.execution.clone());
                    self.reply_tasks.spawn(build_reply(job, config, execution));
                }
                QueryKind::Format { uri } => {
                    let Some(document) = job.editor.documents.get(uri.as_str()).cloned() else {
                        self.reply_tasks.spawn(async move { empty_reply(job) });
                        continue;
                    };
                    let execution = self.execution.clone();
                    self.reply_tasks.spawn(async move {
                        super::query::prepare_format(job, document, &execution).await
                    });
                }
                _ => {
                    let uri = job.kind.uri().expect("semantic query URI");
                    let Some(document) = job.editor.documents.get(uri.as_str()) else {
                        self.reply_tasks.spawn(async move { empty_reply(job) });
                        continue;
                    };
                    let expected = Freshness {
                        registrations: Some(job.editor.registrations),
                        disk_revision: Some(job.editor.disk_revision),
                        ..Freshness::document(document)
                    };
                    if !expected.matches(&self.latest) {
                        self.reply_tasks.spawn(async move {
                            error_reply(job, ErrorCode::ContentModified, "document changed")
                        });
                        continue;
                    }
                    let state = self.roots.get(uri.as_str()).expect("accepted root");
                    if let Some(result) = &state.current {
                        if !result.freshness.matches(&job.editor) {
                            self.reply_tasks.spawn(async move {
                                error_reply(job, ErrorCode::ContentModified, "dependencies changed")
                            });
                            continue;
                        }
                        let (result, registered, config, execution) = (
                            result.clone(),
                            self.registered.clone(),
                            self.config.clone(),
                            self.execution.clone(),
                        );
                        self.reply_tasks.spawn(async move {
                            super::query::prepare_query(job, result, registered, config, &execution)
                                .await
                        });
                    } else if let Some(error) = &state.failure {
                        let error = error.clone();
                        self.reply_tasks.spawn(async move {
                            error_reply(job, ErrorCode::InternalError, error)
                        });
                    } else {
                        self.pending.push_back(job);
                    }
                }
            }
        }
    }

    fn completed_root(&mut self, ticket: RootTicket, result: super::Result<RootAnalysis>) {
        let Some(state) = self.roots.get_mut(ticket.uri.as_str()) else {
            return;
        };
        if state
            .running
            .as_ref()
            .is_some_and(|running| running.ticket == ticket)
        {
            state.running = None;
        }
        if state.ticket != ticket || !analysis::root_is_requested(&ticket, &self.latest) {
            return;
        }
        match result {
            Ok(result)
                if result.freshness.matches(&self.latest)
                    && result.captured.inputs.managed_snapshot
                        == self.config.client.capabilities().managed_snapshot =>
            {
                state.current = Some(Arc::new(result))
            }
            Ok(_) => state.pending = true,
            Err(error) => state.failure = Some(error.to_string()),
        }
        self.diagnostics_dirty = true;
    }

    async fn stop(mut self) {
        for state in self.roots.values() {
            if let Some(running) = &state.running {
                running.cancellation.cancel();
            }
        }
        for cancellation in self.requests.values() {
            cancellation.cancel();
        }
        self.pending.clear();
        while self.registration_tasks.join_next().await.is_some() {}
        while self.root_tasks.join_next().await.is_some() {}
        while self.reply_tasks.join_next().await.is_some() {}
        while self.diagnostic_tasks.join_next().await.is_some() {}
        self.execution.wait_idle().await;
    }
}

async fn coordinate(
    config: WorkerConfig,
    mut editor: watch::Receiver<Arc<EditorSnapshot>>,
    mut incoming: mpsc::Receiver<RequestJob>,
    outgoing: mpsc::Sender<PreparedReply>,
    publications: Arc<std::sync::Mutex<Option<PreparedDiagnostics>>>,
    wake: crossbeam_channel::Sender<()>,
    shutdown: Cancellation,
) {
    let mut scheduler = Scheduler::new(config);
    loop {
        scheduler.accept(editor.borrow_and_update().clone());
        scheduler.schedule(&shutdown);
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            update = editor.changed() => if update.is_err() { break; },
            registered = scheduler.registration_tasks.join_next(), if !scheduler.registration_tasks.is_empty() => {
                match registered.expect("registration task") {
                    Ok((_, Ok(registered))) => scheduler.registered = registered,
                    Ok((revision, Err(error))) => {
                        eprintln!("resin-lsp: cannot register editor snapshot: {error}");
                        // Do not retry a failed normalization in a busy loop; await another accepted event.
                        scheduler.failed_registration = Some(revision);
                        if scheduler.latest.revision == revision {
                            for state in scheduler.roots.values_mut() { state.pending = false; state.failure = Some(error.to_string()); }
                        }
                    }
                    Err(error) => { eprintln!("resin-lsp: registration task failed: {error}"); break; }
                }
            }
            completed = scheduler.root_tasks.join_next(), if !scheduler.root_tasks.is_empty() => {
                match completed.expect("root task") {
                    Ok((ticket, result)) => scheduler.completed_root(ticket, result),
                    Err(error) => { eprintln!("resin-lsp: root task failed: {error}"); break; }
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(10)), if !scheduler.pending.is_empty() => {},
            job = incoming.recv() => match job {
                Some(job) => { scheduler.requests.insert(job.serial, job.cancellation.clone()); scheduler.pending.push_back(job); }
                None => break,
            },
            completed = scheduler.reply_tasks.join_next(), if !scheduler.reply_tasks.is_empty() => {
                match completed.expect("reply task") {
                    Ok(reply) => {
                        scheduler.requests.remove(&reply.serial);
                        if outgoing.send(reply).await.is_err() { break; }
                        let _ = wake.try_send(());
                    }
                    Err(error) => { eprintln!("resin-lsp: request task failed: {error}"); break; }
                }
            }
            completed = scheduler.diagnostic_tasks.join_next(), if !scheduler.diagnostic_tasks.is_empty() => {
                match completed.expect("diagnostic task") {
                    Ok((snapshot, Ok(publication))) if publication.revision == scheduler.latest.revision
                        && snapshot == scheduler.config.client.capabilities().managed_snapshot => {
                        let previous = publications.lock().expect("diagnostic mailbox").replace(publication);
                        drop(previous);
                        let _ = wake.try_send(());
                    }
                    Ok((_, Ok(_))) => scheduler.diagnostics_dirty = true,
                    Ok((_, Err(error))) => eprintln!("resin-lsp: diagnostic rendering failed: {error}"),
                    Err(error) => { eprintln!("resin-lsp: diagnostic task failed: {error}"); break; }
                }
            }
        }
    }
    shutdown.cancel();
    incoming.close();
    scheduler.stop().await;
}

fn error_reply(job: RequestJob, code: ErrorCode, message: impl Into<String>) -> PreparedReply {
    PreparedReply {
        serial: job.serial,
        response: Response::new_err(job.id, code as i32, message.into()),
        freshness: Freshness::default(),
        admission: job.admission,
    }
}

fn empty_reply(job: RequestJob) -> PreparedReply {
    PreparedReply {
        serial: job.serial,
        response: Response::new_ok(job.id, serde_json::Value::Null),
        freshness: Freshness::default(),
        admission: job.admission,
    }
}

async fn build_reply(
    job: RequestJob,
    config: Arc<WorkerConfig>,
    execution: Execution,
) -> PreparedReply {
    let RequestJob {
        serial,
        id,
        kind,
        cancellation,
        admission,
        ..
    } = job;
    let QueryKind::Build { request } = kind else {
        unreachable!("build job")
    };
    let result = super::build::run(request, &config, &execution, &cancellation).await;
    let response = match result {
        Ok(value) => Response::new_ok(id, value),
        Err(_) if cancellation.is_cancelled() => Response::new_err(
            id,
            ErrorCode::RequestCanceled as i32,
            "build cancelled".into(),
        ),
        Err(error) => Response::new_err(id, ErrorCode::InternalError as i32, error.to_string()),
    };
    PreparedReply {
        serial,
        response,
        freshness: Freshness::default(),
        admission,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, path::Path};
    use tempfile::TempDir;

    fn document(path: &Path, epoch: u64, version: i32, text: &str) -> Arc<OpenDocument> {
        let uri = url::Url::from_file_path(path)
            .unwrap()
            .as_str()
            .parse()
            .unwrap();
        Arc::new(OpenDocument {
            uri,
            path: path.to_owned(),
            stamp: DocumentVersion { epoch, version },
            text: Arc::new(text.into()),
        })
    }

    fn editor(
        revision: u64,
        registrations: u64,
        documents: Vec<Arc<OpenDocument>>,
    ) -> Arc<EditorSnapshot> {
        Arc::new(EditorSnapshot {
            revision,
            registrations,
            disk_revision: 0,
            documents: documents
                .into_iter()
                .map(|document| (document.uri.as_str().to_owned(), document))
                .collect(),
        })
    }

    fn empty(library: &Path) -> Arc<RegisteredEditor> {
        Arc::new(RegisteredEditor {
            editor: Arc::new(EditorSnapshot::default()),
            loader: Loader::new(library.into()),
            documents: BTreeMap::new(),
        })
    }

    #[tokio::test]
    async fn closing_registrations_releases_accepted_text_after_outstanding_snapshots_drop() {
        let temp = TempDir::new().unwrap();
        let document = document(&temp.path().join("main.resin"), 1, 1, "fn main()  {}");
        let weak_text = Arc::downgrade(&document.text);
        let registered = register_editor(
            empty(temp.path()),
            editor(1, 1, vec![document]),
            Execution::default(),
            Cancellation::new(),
        )
        .await
        .unwrap();
        let closed = register_editor(
            registered.clone(),
            editor(2, 2, vec![]),
            Execution::default(),
            Cancellation::new(),
        )
        .await
        .unwrap();
        assert!(closed.documents.is_empty());
        assert!(weak_text.upgrade().is_some());
        drop(registered);
        assert!(weak_text.upgrade().is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn closing_either_alias_keeps_the_other_open_physical_registration() {
        for close_alias in [false, true] {
            let temp = TempDir::new().unwrap();
            let library = temp.path().join("lib.resin");
            let alias = temp.path().join("alias.resin");
            let main = temp.path().join("main.resin");
            fs::write(
                &library,
                "export { answer }; fn answer() -> str  { \"disk\" }",
            )
            .unwrap();
            std::os::unix::fs::symlink(&library, &alias).unwrap();
            let supplied = "export { answer }; fn answer() -> i32  { 7 }";
            let canonical = document(&library, 1, 1, supplied);
            let aliased = document(&alias, 2, 1, supplied);
            let entry = document(
                &main,
                3,
                1,
                "import { \"lib.resin\" }; fn main() -> i32  { answer() }",
            );
            let registered = register_editor(
                empty(temp.path()),
                editor(
                    1,
                    3,
                    vec![canonical.clone(), aliased.clone(), entry.clone()],
                ),
                Execution::default(),
                Cancellation::new(),
            )
            .await
            .unwrap();
            let survivor = if close_alias { canonical } else { aliased };
            let original = registered.documents[survivor.uri.as_str()].clone();
            let closed = register_editor(
                registered,
                editor(2, 4, vec![survivor.clone(), entry.clone()]),
                Execution::default(),
                Cancellation::new(),
            )
            .await
            .unwrap();
            assert_eq!(closed.documents[survivor.uri.as_str()], original);
            assert_eq!(closed.loader.path(&original).unwrap(), library);
            let mut request = closed.loader.supplied_snapshot();
            assert_eq!(
                request
                    .load_import(&closed.documents[entry.uri.as_str()], "lib.resin")
                    .unwrap()
                    .text(),
                supplied
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn conflicting_open_alias_text_is_rejected_without_replacing_a_registration() {
        let temp = TempDir::new().unwrap();
        let library = temp.path().join("lib.resin");
        let alias = temp.path().join("alias.resin");
        fs::write(&library, "disk").unwrap();
        std::os::unix::fs::symlink(&library, &alias).unwrap();
        let canonical = document(&library, 1, 1, "accepted text");
        let registered = register_editor(
            empty(temp.path()),
            editor(1, 1, vec![canonical.clone()]),
            Execution::default(),
            Cancellation::new(),
        )
        .await
        .unwrap();
        let conflict = document(&alias, 2, 1, "conflicting text");
        let error = register_editor(
            registered.clone(),
            editor(2, 2, vec![canonical.clone(), conflict.clone()]),
            Execution::default(),
            Cancellation::new(),
        )
        .await
        .err()
        .unwrap();
        let message = error.to_string();
        assert!(message.contains("conflicting editor text"), "{message}");
        assert!(
            message.contains(canonical.uri.as_str()) && message.contains(conflict.uri.as_str()),
            "{message}"
        );
        assert_eq!(
            registered.documents[canonical.uri.as_str()].text(),
            "accepted text"
        );
        assert_eq!(
            registered
                .loader
                .path(&registered.documents[canonical.uri.as_str()])
                .unwrap(),
            library
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn reopening_a_retargeted_path_normalizes_even_while_an_alias_keeps_the_old_identity() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("open.resin");
        let alias = temp.path().join("alias.resin");
        let target = temp.path().join("target.resin");
        fs::write(&path, "original disk").unwrap();
        fs::write(&target, "target disk").unwrap();
        std::os::unix::fs::symlink(&path, &alias).unwrap();
        let first = document(&path, 1, 1, "original supplied");
        let survivor = document(&alias, 2, 1, "original supplied");
        let registered = register_editor(
            empty(temp.path()),
            editor(1, 2, vec![first.clone(), survivor.clone()]),
            Execution::default(),
            Cancellation::new(),
        )
        .await
        .unwrap();
        let original = registered.documents[first.uri.as_str()].id();
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(&target, &path).unwrap();
        let reopened = document(&path, 3, 1, "new supplied");
        let next = register_editor(
            registered,
            editor(3, 4, vec![reopened.clone(), survivor.clone()]),
            Execution::default(),
            Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(next.documents[survivor.uri.as_str()].id(), original);
        assert_eq!(
            next.loader
                .path(&next.documents[survivor.uri.as_str()])
                .unwrap(),
            path
        );
        assert_ne!(next.documents[reopened.uri.as_str()].id(), original);
        assert_eq!(
            next.loader
                .path(&next.documents[reopened.uri.as_str()])
                .unwrap(),
            target
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn coalesced_close_and_reopen_registers_a_new_physical_identity() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("open.resin");
        let target = temp.path().join("target.resin");
        fs::write(&path, "original disk").unwrap();
        fs::write(&target, "target disk").unwrap();
        let first = document(&path, 1, 1, "first edit");
        let uri = first.uri.clone();
        let registered = register_editor(
            empty(temp.path()),
            editor(1, 1, vec![first]),
            Execution::default(),
            Cancellation::new(),
        )
        .await
        .unwrap();
        let original = registered.documents[uri.as_str()].id();
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(&target, &path).unwrap();
        let edited = register_editor(
            registered,
            editor(2, 1, vec![document(&path, 1, 2, "later edit")]),
            Execution::default(),
            Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(edited.documents[uri.as_str()].id(), original);
        assert_eq!(
            edited.loader.path(&edited.documents[uri.as_str()]).unwrap(),
            path
        );
        let reopened = register_editor(
            edited,
            editor(4, 3, vec![document(&path, 2, 1, "reopened edit")]),
            Execution::default(),
            Cancellation::new(),
        )
        .await
        .unwrap();
        assert_ne!(reopened.documents[uri.as_str()].id(), original);
        assert_eq!(
            reopened
                .loader
                .path(&reopened.documents[uri.as_str()])
                .unwrap(),
            target
        );
    }
}
