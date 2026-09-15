//! Capture one accepted editor root and submit its immutable graph to the service.
use super::worker::{EditorSnapshot, Freshness, RegisteredEditor, WorkerConfig};
use crate::inputs::{self, CaptureOptions, CapturedInputs};
use lsp_types::Uri;
use resin_executor::{Cancellation, Execution};
use resin_protocol::{AnalyzeResponse, ErrorCode, HeaderInputs, Inputs};
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
    pub captured: CapturedInputs,
    pub response: AnalyzeResponse,
    pub uris: BTreeMap<String, Uri>,
    pub texts: BTreeMap<String, String>,
}

pub(super) async fn analyze_root(
    ticket: RootTicket,
    registered: Arc<RegisteredEditor>,
    config: Arc<WorkerConfig>,
    previous: Option<Arc<RootAnalysis>>,
    execution: Execution,
    cancellation: Cancellation,
) -> super::Result<RootAnalysis> {
    for attempt in 0..2 {
        let captured = capture(&ticket, &registered, &config, &execution, &cancellation).await?;
        let previous = previous
            .as_ref()
            .map(|previous| (&previous.response.input, &previous.captured.inputs));
        let response = match config
            .client
            .analyze(
                &captured.inputs,
                previous,
                ticket.generation,
                Vec::new(),
                &cancellation,
            )
            .await
        {
            Err(error)
                if attempt == 0
                    && error.failure().is_some_and(|failure| {
                        failure.code == ErrorCode::ManagedSnapshotUnavailable
                    }) =>
            {
                continue;
            }
            result => result?,
        };
        let mirrored = config
            .mirrors
            .materialize(
                response.managed_sources.clone(),
                captured.inputs.managed_snapshot.clone(),
                &execution,
                &cancellation,
            )
            .await?;
        let ticket = ticket.clone();
        let registered = registered.clone();
        return execution
            .run(&cancellation, move |_| {
                let (freshness, mut uris) = presentation(&ticket, &captured, &registered);
                for (name, path) in mirrored {
                    if let Some(uri) = file_uri(&path) {
                        uris.insert(name, uri);
                    }
                }
                uris.insert(captured.inputs.entry.clone(), ticket.uri.clone());
                let texts = captured
                    .origins
                    .iter()
                    .map(|(name, local)| (name.clone(), local.source.text().to_owned()))
                    .chain(
                        response
                            .managed_sources
                            .iter()
                            .map(|source| (source.name.clone(), source.text.clone())),
                    )
                    .collect();
                RootAnalysis {
                    ticket,
                    freshness,
                    captured,
                    response,
                    uris,
                    texts,
                }
            })
            .await
            .map_err(Into::into);
    }
    unreachable!("at most one managed snapshot retry")
}

async fn capture(
    ticket: &RootTicket,
    registered: &Arc<RegisteredEditor>,
    config: &WorkerConfig,
    execution: &Execution,
    cancellation: &Cancellation,
) -> super::Result<CapturedInputs> {
    let path = &registered.editor.documents[ticket.uri.as_str()].path;
    if let Some(managed) = config.mirrors.source(path) {
        if registered.editor.documents[ticket.uri.as_str()]
            .text
            .as_str()
            != managed.source.text
        {
            return Err("server-provided library sources are read-only".into());
        }
        if managed.snapshot != config.client.capabilities().managed_snapshot {
            return Err("managed source belongs to an earlier server snapshot; navigate to its current definition again".into());
        }
        return Ok(CapturedInputs {
            inputs: Inputs {
                entry: managed.source.name.clone(),
                sources: Vec::new(),
                imports: Vec::new(),
                headers: HeaderInputs {
                    bundles: Vec::new(),
                    bindings: Vec::new(),
                    include_roots: Vec::new(),
                },
                acquisition_diagnostics: Vec::new(),
                managed_snapshot: managed.snapshot,
            },
            origins: BTreeMap::from([(
                managed.source.name,
                crate::inputs::LocalSource {
                    source: registered.documents[ticket.uri.as_str()].clone(),
                    path: path.clone(),
                },
            )]),
        });
    }
    let (source, mut loader) = execution
        .run(cancellation, {
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
    let options = CaptureOptions {
        include_roots: config.include_roots.clone(),
        capabilities: config.client.capabilities(),
    };
    inputs::capture(source, &mut loader, &options, execution, cancellation).await
}

fn presentation(
    ticket: &RootTicket,
    captured: &CapturedInputs,
    registered: &RegisteredEditor,
) -> (Freshness, BTreeMap<String, Uri>) {
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
    for (logical, local) in &captured.origins {
        if let Some(uri) = file_uri(&local.path) {
            uris.insert(logical.clone(), uri);
        }
        for (uri, supplied) in &registered.documents {
            if supplied.id() == local.source.id() {
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
    uris.insert(captured.inputs.entry.clone(), ticket.uri.clone());
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
