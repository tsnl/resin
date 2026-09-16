//! Negotiated routes, structured errors, delta fallback, and cancellation races.
use crate::{BuiltArtifact, Client, Error};
use resin_executor::{Cancellation, Execution};
use resin_protocol::{
    AnalyzeRequest, AnalyzeResponse, BuildContract, BuildRequest, Capabilities, ErrorCode, Failure,
    InputHandle, InputSelection, Inputs, Query,
};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    future::Future,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

const MAX_JSON_BYTES: usize = 64 * 1024 * 1024;
const CANCELLATION_TIMEOUT: Duration = Duration::from_secs(5);

pub(super) async fn connect(url: &str) -> Result<Client, Error> {
    let base = base_url(url)?;
    let http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .build()
        .map_err(Error::new)?;
    let execution = Execution::default();
    let capabilities = capabilities(&http, &base, &execution).await?;
    Ok(Client {
        http,
        base,
        capabilities: Arc::new(Mutex::new(capabilities)),
        execution,
    })
}

fn base_url(text: &str) -> Result<reqwest::Url, Error> {
    if text.trim() != text || text.is_empty() {
        return Err(Error::new("RESIN_SERVER must be an absolute HTTP(S) URL"));
    }
    let mut url = reqwest::Url::parse(text)
        .map_err(|_| Error::new("RESIN_SERVER must be an absolute HTTP(S) URL"))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(Error::new(
            "RESIN_SERVER must be an absolute HTTP(S) base URL without credentials, query, or fragment",
        ));
    }
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    Ok(url)
}

fn endpoint(base: &reqwest::Url, route: &str) -> reqwest::Url {
    base.join(route).expect("validated relative service route")
}

pub(super) async fn capabilities(
    http: &reqwest::Client,
    base: &reqwest::Url,
    execution: &Execution,
) -> Result<Arc<Capabilities>, Error> {
    let response = http
        .get(endpoint(base, "v1/capabilities"))
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(Error::new)?;
    let capabilities: Capabilities = json_response(response, execution).await?;
    if capabilities.protocol != resin_protocol::PROTOCOL_VERSION {
        return Err(Failure {
            code: ErrorCode::IncompatibleProtocol,
            message: format!(
                "unsupported compiler protocol {}; client requires {}",
                capabilities.protocol,
                resin_protocol::PROTOCOL_VERSION
            ),
            diagnostics: Vec::new(),
            managed_sources: Vec::new(),
        }
        .into());
    }
    if capabilities.instance.is_empty()
        || capabilities.managed_snapshot.is_empty()
        || capabilities.targets.is_empty()
        || !capabilities
            .entry_profiles
            .contains(&resin_protocol::EntryProfile::Host)
    {
        return Err(Error::new("server returned incomplete capabilities"));
    }
    Ok(Arc::new(capabilities))
}

pub(super) async fn analyze(
    client: &Client,
    current: &Inputs,
    previous: Option<(&InputHandle, &Inputs)>,
    revision: u64,
    queries: Vec<Query>,
    cancellation: &Cancellation,
) -> Result<AnalyzeResponse, Error> {
    let mut selection = crate::inputs::selection(current, previous);
    for retry in 0..2 {
        let expected_instance = client.capabilities().instance.clone();
        let request = AnalyzeRequest {
            request: uuid::Uuid::new_v4().to_string(),
            revision,
            inputs: selection,
            queries: queries.clone(),
        };
        let token = request.request.clone();
        let result = perform(client, &token, cancellation, async {
            let bytes = serialize(client, request, cancellation).await?;
            let response = client
                .http
                .post(endpoint(&client.base, "v1/analyze"))
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(bytes)
                .send()
                .await
                .map_err(Error::new)?;
            let response: AnalyzeResponse = json_response(response, &client.execution).await?;
            if response.revision != revision || response.results.len() != queries.len() {
                return Err(Error::new("analysis response does not match its request"));
            }
            if response.input.instance != expected_instance {
                return Err(Error::instance_changed());
            }
            validate_analysis(&response, current, &queries)?;
            Ok(response)
        })
        .await;
        match result {
            Err(error)
                if retry == 0
                    && (matches!(error.kind, crate::ErrorKind::InstanceChanged)
                        || (previous.is_some()
                            && code(&error) == Some(ErrorCode::InputUnavailable))) =>
            {
                client.refresh_capabilities().await?;
                selection = InputSelection::Full {
                    inputs: current.clone(),
                };
            }
            Err(error) if code(&error) == Some(ErrorCode::ManagedSnapshotUnavailable) => {
                client.refresh_capabilities().await?;
                return Err(error);
            }
            result => return result,
        }
    }
    unreachable!("full-input retry returns its result")
}

pub(super) async fn build(
    client: &Client,
    current: &Inputs,
    previous: Option<(&InputHandle, &Inputs)>,
    revision: u64,
    contract: BuildContract,
    destination: &Path,
    cancellation: &Cancellation,
) -> Result<BuiltArtifact, Error> {
    let capabilities = client.capabilities();
    if !capabilities.targets.contains(&contract.target)
        || !capabilities
            .entry_profiles
            .contains(&contract.entry.profile)
        || contract.entry.profile != resin_protocol::EntryProfile::Host
    {
        return Err(Failure {
            code: ErrorCode::UnsupportedTarget,
            message: "server does not support the requested executable target".into(),
            diagnostics: Vec::new(),
            managed_sources: Vec::new(),
        }
        .into());
    }
    let mut selection = crate::inputs::selection(current, previous);
    for retry in 0..2 {
        let expected_instance = client.capabilities().instance.clone();
        let request = BuildRequest {
            request: uuid::Uuid::new_v4().to_string(),
            revision,
            inputs: selection,
            contract: contract.clone(),
        };
        let token = request.request.clone();
        let result = perform(client, &token, cancellation, async {
            let bytes = serialize(client, request, cancellation).await?;
            let response = client
                .http
                .post(endpoint(&client.base, "v1/build"))
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(bytes)
                .send()
                .await
                .map_err(Error::new)?;
            if !response.status().is_success() {
                return Err(remote_failure(response, &client.execution).await);
            }
            crate::download::receive(
                response,
                destination,
                revision,
                &contract.target,
                &expected_instance,
                &client.execution,
                cancellation,
            )
            .await
        })
        .await;
        match result {
            Err(error)
                if retry == 0
                    && (matches!(error.kind, crate::ErrorKind::InstanceChanged)
                        || (previous.is_some()
                            && code(&error) == Some(ErrorCode::InputUnavailable))) =>
            {
                client.refresh_capabilities().await?;
                selection = InputSelection::Full {
                    inputs: current.clone(),
                };
            }
            Err(error) if code(&error) == Some(ErrorCode::ManagedSnapshotUnavailable) => {
                client.refresh_capabilities().await?;
                return Err(error);
            }
            result => return result,
        }
    }
    unreachable!("full-input retry returns its result")
}

fn code(error: &Error) -> Option<ErrorCode> {
    error.failure().map(|failure| failure.code)
}

fn validate_analysis(
    response: &AnalyzeResponse,
    inputs: &Inputs,
    queries: &[Query],
) -> Result<(), Error> {
    use resin_protocol::QueryResult;
    let mut sources = std::collections::BTreeMap::new();
    for source in &inputs.sources {
        sources.insert(source.name.as_str(), source.text.as_str());
    }
    for source in &response.managed_sources {
        if !source.name.starts_with("$/")
            || sources
                .insert(source.name.as_str(), source.text.as_str())
                .is_some()
        {
            return Err(Error::new(
                "analysis response contains a conflicting or non-managed source",
            ));
        }
    }
    if response.input.id.is_empty() || response.results.len() != queries.len() {
        return Err(Error::new(
            "analysis response has an invalid input handle or result count",
        ));
    }
    for diagnostic in &response.diagnostics {
        for span in diagnostic
            .span
            .iter()
            .chain(diagnostic.related.iter().map(|note| &note.span))
        {
            validate_span(span, &sources)?;
        }
    }
    for (query, result) in queries.iter().zip(&response.results) {
        match (query, result) {
            (Query::Hover { position }, QueryResult::Hover { span, .. }) => {
                if let Some(span) = span {
                    validate_edit_span(span, &position.source, &sources)?;
                }
            }
            (Query::Definition { .. }, QueryResult::Definition { span }) => {
                if let Some(span) = span {
                    validate_span(span, &sources)?;
                }
            }
            (Query::Completion { position }, QueryResult::Completion { items }) => {
                for item in items {
                    validate_edit_span(&item.replace, &position.source, &sources)?;
                }
            }
            _ => return Err(Error::new("analysis result kind does not match its query")),
        }
    }
    Ok(())
}

fn validate_edit_span(
    span: &resin_protocol::Span,
    source: &str,
    sources: &std::collections::BTreeMap<&str, &str>,
) -> Result<(), Error> {
    if span.source != source {
        return Err(Error::new(
            "analysis edit or hover span belongs to another source",
        ));
    }
    validate_span(span, sources)
}

fn validate_span(
    span: &resin_protocol::Span,
    sources: &std::collections::BTreeMap<&str, &str>,
) -> Result<(), Error> {
    let text = sources
        .get(span.source.as_str())
        .ok_or_else(|| Error::new("analysis span refers to an unknown source"))?;
    let start = usize::try_from(span.start).map_err(Error::new)?;
    let end = usize::try_from(span.end).map_err(Error::new)?;
    if start > end
        || end > text.len()
        || !text.is_char_boundary(start)
        || !text.is_char_boundary(end)
    {
        return Err(Error::new(
            "analysis span is outside its source or splits a UTF-8 character",
        ));
    }
    Ok(())
}

async fn serialize<T: Serialize + Send + 'static>(
    client: &Client,
    request: T,
    cancellation: &Cancellation,
) -> Result<Vec<u8>, Error> {
    client
        .execution
        .run(cancellation, move |_| serde_json::to_vec(&request))
        .await
        .map_err(|error| {
            if error == resin_executor::Error::Cancelled {
                Error::cancelled()
            } else {
                Error::new(error)
            }
        })?
        .map_err(Error::new)
}

async fn json_response<T: DeserializeOwned + Send + 'static>(
    response: reqwest::Response,
    execution: &Execution,
) -> Result<T, Error> {
    if !response.status().is_success() {
        return Err(remote_failure(response, execution).await);
    }
    let bytes = bounded_body(response).await?;
    execution
        .run(&Cancellation::new(), move |_| {
            serde_json::from_slice(&bytes)
        })
        .await
        .map_err(Error::new)?
        .map_err(Error::new)
}

async fn remote_failure(response: reqwest::Response, execution: &Execution) -> Error {
    let status = response.status();
    match bounded_body(response).await {
        Ok(bytes) => match execution
            .run(&Cancellation::new(), move |_| {
                serde_json::from_slice::<Failure>(&bytes)
            })
            .await
        {
            Ok(Ok(failure)) => failure.into(),
            _ => Error::new(format!(
                "compiler service returned HTTP {status} without a valid failure message"
            )),
        },
        Err(error) => error,
    }
}

async fn bounded_body(mut response: reqwest::Response) -> Result<Vec<u8>, Error> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_JSON_BYTES as u64)
    {
        return Err(Error::new("service JSON response exceeds the size limit"));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(Error::new)? {
        if bytes.len().saturating_add(chunk.len()) > MAX_JSON_BYTES {
            return Err(Error::new("service JSON response exceeds the size limit"));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// Allow remote cleanup after cancellation, then close a stalled HTTP exchange.
async fn perform<T>(
    client: &Client,
    request: &str,
    cancellation: &Cancellation,
    post: impl Future<Output = Result<T, Error>>,
) -> Result<T, Error> {
    if cancellation.is_cancelled() {
        return Err(Error::cancelled());
    }
    tokio::pin!(post);
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => {},
        result = &mut post => return result,
    }
    let _ = tokio::time::timeout(CANCELLATION_TIMEOUT, cancel_request(client, request, post)).await;
    Err(Error::cancelled())
}

/// A 404 can precede admission. Keep the POST alive while retrying cancellation,
/// including while draining an acknowledged request, within the caller's deadline.
async fn cancel_request<T>(
    client: &Client,
    request: &str,
    post: impl Future<Output = Result<T, Error>>,
) {
    tokio::pin!(post);
    loop {
        let delete = client
            .http
            .delete(endpoint(&client.base, &format!("v1/requests/{request}")))
            .timeout(Duration::from_secs(2))
            .send();
        tokio::pin!(delete);
        tokio::select! {
            biased;
            _ = &mut post => return,
            response = &mut delete => if response.is_ok_and(|response| response.status() == reqwest::StatusCode::NO_CONTENT) {
                let _ = post.await;
                return;
            },
        }
        tokio::select! {
            biased;
            _ = &mut post => return,
            _ = tokio::time::sleep(Duration::from_millis(25)) => {},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::{
        Json, Router,
        extract::State,
        http::StatusCode,
        response::IntoResponse,
        routing::{delete, get, post},
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn advertised() -> Capabilities {
        Capabilities {
            protocol: resin_protocol::PROTOCOL_VERSION,
            instance: "instance".into(),
            managed_snapshot: "managed".into(),
            targets: vec![crate::host_target()],
            entry_profiles: vec![resin_protocol::EntryProfile::Host],
            header_roots: vec![],
        }
    }

    fn captured() -> Inputs {
        Inputs {
            entry: "main.resin".into(),
            sources: vec![],
            imports: vec![],
            headers: Default::default(),
            acquisition_diagnostics: vec![],
            managed_snapshot: "managed".into(),
        }
    }

    fn rejected(code: ErrorCode) -> Failure {
        Failure {
            code,
            message: "test failure".into(),
            diagnostics: vec![],
            managed_sources: vec![],
        }
    }

    async fn mock(router: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        (url, task)
    }

    #[tokio::test]
    async fn negotiation_rejects_an_incompatible_protocol() {
        let (url, task) = mock(Router::new().route(
            "/v1/capabilities",
            get(|| async {
                let mut caps = advertised();
                caps.protocol += 1;
                Json(caps)
            }),
        ))
        .await;
        let error = match Client::connect(&url).await {
            Ok(_) => panic!("incompatible server accepted"),
            Err(error) => error,
        };
        assert_eq!(code(&error), Some(ErrorCode::IncompatibleProtocol));
        task.abort();
    }

    #[tokio::test]
    async fn an_evicted_delta_retries_exactly_once_with_complete_inputs() {
        async fn analyze(
            State(attempts): State<Arc<AtomicUsize>>,
            Json(request): Json<AnalyzeRequest>,
        ) -> axum::response::Response {
            let attempt = attempts.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                assert!(matches!(request.inputs, InputSelection::Delta { .. }));
                return (
                    StatusCode::CONFLICT,
                    Json(rejected(ErrorCode::InputUnavailable)),
                )
                    .into_response();
            }
            assert_eq!(attempt, 1);
            assert_eq!(request.inputs, InputSelection::Full { inputs: captured() });
            Json(AnalyzeResponse {
                revision: request.revision,
                input: InputHandle {
                    instance: "instance".into(),
                    id: "new".into(),
                },
                diagnostics: vec![],
                results: vec![],
                managed_sources: vec![],
            })
            .into_response()
        }
        let attempts = Arc::new(AtomicUsize::new(0));
        let router = Router::new()
            .route("/v1/capabilities", get(|| async { Json(advertised()) }))
            .route("/v1/analyze", post(analyze))
            .with_state(attempts.clone());
        let (url, task) = mock(router).await;
        let client = Client::connect(&url).await.unwrap();
        let inputs = captured();
        let old = InputHandle {
            instance: "old-instance".into(),
            id: "old".into(),
        };
        let response = client
            .analyze(
                &inputs,
                Some((&old, &inputs)),
                7,
                vec![],
                &Cancellation::new(),
            )
            .await
            .unwrap();
        assert_eq!(response.revision, 7);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        task.abort();
    }

    #[test]
    fn analysis_validation_rejects_wrong_query_kinds_and_foreign_or_invalid_edit_ranges() {
        use resin_protocol::{
            CompletionItem, CompletionKind, QueryPosition, QueryResult, SourceFile, Span,
        };
        let mut inputs = captured();
        inputs.sources = vec![
            SourceFile {
                name: "main.resin".into(),
                text: "λx".into(),
            },
            SourceFile {
                name: "other.resin".into(),
                text: "text".into(),
            },
        ];
        let query = Query::Completion {
            position: QueryPosition {
                source: "main.resin".into(),
                offset: 2,
            },
        };
        let mut response = AnalyzeResponse {
            revision: 1,
            input: InputHandle {
                instance: "instance".into(),
                id: "inputs".into(),
            },
            diagnostics: vec![],
            results: vec![QueryResult::Definition { span: None }],
            managed_sources: vec![],
        };
        assert!(validate_analysis(&response, &inputs, std::slice::from_ref(&query)).is_err());
        for (source, start, end) in [
            ("other.resin", 0, 1),
            ("main.resin", 1, 2),
            ("main.resin", 3, 2),
            ("main.resin", 2, 9),
            ("missing.resin", 0, 0),
        ] {
            response.results = vec![QueryResult::Completion {
                items: vec![CompletionItem {
                    label: "name".into(),
                    detail: None,
                    kind: CompletionKind::Variable,
                    insert_text: "name".into(),
                    replace: Span {
                        source: source.into(),
                        start,
                        end,
                    },
                }],
            }];
            assert!(
                validate_analysis(&response, &inputs, std::slice::from_ref(&query)).is_err(),
                "{source}:{start}:{end}"
            );
        }
        response.managed_sources = vec![SourceFile {
            name: "$/library.resin".into(),
            text: "λ".into(),
        }];
        response.results = vec![QueryResult::Definition {
            span: Some(Span {
                source: "$/library.resin".into(),
                start: 0,
                end: 2,
            }),
        }];
        let definition = Query::Definition {
            position: QueryPosition {
                source: "main.resin".into(),
                offset: 2,
            },
        };
        assert!(validate_analysis(&response, &inputs, std::slice::from_ref(&definition)).is_ok());
        response
            .managed_sources
            .push(response.managed_sources[0].clone());
        assert!(validate_analysis(&response, &inputs, &[definition]).is_err());
    }

    #[tokio::test]
    async fn malformed_query_results_are_rejected_at_the_http_boundary() {
        let router = Router::new()
            .route("/v1/capabilities", get(|| async { Json(advertised()) }))
            .route(
                "/v1/analyze",
                post(|Json(request): Json<AnalyzeRequest>| async move {
                    Json(AnalyzeResponse {
                        revision: request.revision,
                        input: InputHandle {
                            instance: "instance".into(),
                            id: "inputs".into(),
                        },
                        diagnostics: vec![],
                        results: vec![resin_protocol::QueryResult::Definition { span: None }],
                        managed_sources: vec![],
                    })
                }),
            );
        let (url, server) = mock(router).await;
        let client = Client::connect(&url).await.unwrap();
        let result = client
            .analyze(
                &captured(),
                None,
                1,
                vec![Query::Hover {
                    position: resin_protocol::QueryPosition {
                        source: "main.resin".into(),
                        offset: 0,
                    },
                }],
                &Cancellation::new(),
            )
            .await;
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("kind does not match")
        );
        server.abort();
    }

    #[derive(Default)]
    struct Restart {
        capabilities: AtomicUsize,
        requests: AtomicUsize,
        posted: tokio::sync::Notify,
        release: tokio::sync::Notify,
    }

    async fn restarted_capabilities(State(restart): State<Arc<Restart>>) -> Json<Capabilities> {
        let mut capabilities = advertised();
        if restart.capabilities.fetch_add(1, Ordering::SeqCst) > 0 {
            capabilities.instance = "restarted".into();
        }
        Json(capabilities)
    }

    #[tokio::test]
    async fn full_analyze_and_build_retry_once_after_a_same_snapshot_restart() {
        async fn analyze(
            State(restart): State<Arc<Restart>>,
            Json(request): Json<AnalyzeRequest>,
        ) -> Json<AnalyzeResponse> {
            restart.requests.fetch_add(1, Ordering::SeqCst);
            assert!(matches!(request.inputs, InputSelection::Full { .. }));
            Json(AnalyzeResponse {
                revision: request.revision,
                input: InputHandle {
                    instance: "restarted".into(),
                    id: "current".into(),
                },
                diagnostics: vec![],
                results: vec![],
                managed_sources: vec![],
            })
        }
        async fn build(
            State(restart): State<Arc<Restart>>,
            Json(request): Json<BuildRequest>,
        ) -> impl IntoResponse {
            use base64::Engine;
            restart.requests.fetch_add(1, Ordering::SeqCst);
            assert!(matches!(request.inputs, InputSelection::Full { .. }));
            let bytes = b"executable bytes".to_vec();
            let metadata = resin_protocol::BuildMetadata {
                revision: request.revision,
                input: InputHandle {
                    instance: "restarted".into(),
                    id: "current".into(),
                },
                artifact: resin_protocol::ArtifactMetadata {
                    name: "program".into(),
                    kind: resin_protocol::ArtifactKind::Executable,
                    target: request.contract.target,
                    length: bytes.len() as u64,
                    blake3: blake3::hash(&bytes).to_hex().to_string(),
                    executable: true,
                },
            };
            let metadata = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(serde_json::to_vec(&metadata).unwrap());
            (
                [
                    ("content-type", "application/octet-stream".to_owned()),
                    (resin_protocol::METADATA_HEADER, metadata),
                ],
                bytes,
            )
        }
        for native in [false, true] {
            let restart = Arc::new(Restart::default());
            let router = Router::new()
                .route("/v1/capabilities", get(restarted_capabilities))
                .route("/v1/analyze", post(analyze))
                .route("/v1/build", post(build))
                .with_state(restart.clone());
            let (url, server) = mock(router).await;
            let client = Client::connect(&url).await.unwrap();
            let cancellation = Cancellation::new();
            if native {
                let destination = tempfile::tempdir().unwrap();
                let contract = BuildContract {
                    target: crate::host_target(),
                    entry: resin_protocol::EntryTarget {
                        export: "main".into(),
                        profile: resin_protocol::EntryProfile::Host,
                    },
                    profile: resin_protocol::BuildProfile::Debug,
                    options: Default::default(),
                };
                let result = client
                    .build(
                        &captured(),
                        None,
                        1,
                        contract,
                        &destination.path().join("program"),
                        &cancellation,
                    )
                    .await
                    .unwrap();
                assert_eq!(result.metadata().input.instance, "restarted");
            } else {
                let result = client
                    .analyze(&captured(), None, 1, vec![], &cancellation)
                    .await
                    .unwrap();
                assert_eq!(result.input.instance, "restarted");
            }
            assert_eq!(restart.requests.load(Ordering::SeqCst), 2);
            assert_eq!(restart.capabilities.load(Ordering::SeqCst), 2);
            server.abort();
        }
    }

    #[tokio::test]
    async fn concurrent_capability_refresh_does_not_change_an_inflight_expected_instance() {
        async fn analyze(
            State(restart): State<Arc<Restart>>,
            Json(request): Json<AnalyzeRequest>,
        ) -> Json<AnalyzeResponse> {
            restart.requests.fetch_add(1, Ordering::SeqCst);
            restart.posted.notify_one();
            restart.release.notified().await;
            Json(AnalyzeResponse {
                revision: request.revision,
                input: InputHandle {
                    instance: "instance".into(),
                    id: "current".into(),
                },
                diagnostics: vec![],
                results: vec![],
                managed_sources: vec![],
            })
        }
        let restart = Arc::new(Restart::default());
        let router = Router::new()
            .route("/v1/capabilities", get(restarted_capabilities))
            .route("/v1/analyze", post(analyze))
            .with_state(restart.clone());
        let (url, server) = mock(router).await;
        let client = Client::connect(&url).await.unwrap();
        let request = tokio::spawn({
            let client = client.clone();
            async move {
                client
                    .analyze(&captured(), None, 1, vec![], &Cancellation::new())
                    .await
            }
        });
        restart.posted.notified().await;
        assert_eq!(
            client.refresh_capabilities().await.unwrap().instance,
            "restarted"
        );
        restart.release.notify_one();
        assert_eq!(request.await.unwrap().unwrap().input.instance, "instance");
        assert_eq!(restart.requests.load(Ordering::SeqCst), 1);
        server.abort();
    }

    #[derive(Default)]
    struct Race {
        posted: tokio::sync::Notify,
        admission: tokio::sync::Notify,
        cancelled: Cancellation,
        admitted: AtomicBool,
        finished: AtomicBool,
        deletes: AtomicUsize,
    }

    #[tokio::test]
    async fn cancellation_retries_404_while_original_post_can_still_be_admitted() {
        async fn analyze(State(race): State<Arc<Race>>) -> impl IntoResponse {
            race.posted.notify_one();
            race.admission.notified().await;
            race.admitted.store(true, Ordering::SeqCst);
            race.cancelled.cancelled().await;
            race.finished.store(true, Ordering::SeqCst);
            (
                StatusCode::from_u16(499).unwrap(),
                Json(rejected(ErrorCode::Cancelled)),
            )
        }
        async fn cancel(State(race): State<Arc<Race>>) -> StatusCode {
            race.deletes.fetch_add(1, Ordering::SeqCst);
            if race.admitted.load(Ordering::SeqCst) {
                race.cancelled.cancel();
                StatusCode::NO_CONTENT
            } else {
                race.admission.notify_one();
                StatusCode::NOT_FOUND
            }
        }
        let race = Arc::new(Race::default());
        let router = Router::new()
            .route("/v1/capabilities", get(|| async { Json(advertised()) }))
            .route("/v1/analyze", post(analyze))
            .route("/v1/requests/{id}", delete(cancel))
            .with_state(race.clone());
        let (url, server) = mock(router).await;
        let client = Client::connect(&url).await.unwrap();
        let cancellation = Cancellation::new();
        let request = tokio::spawn({
            let cancellation = cancellation.clone();
            async move {
                client
                    .analyze(&captured(), None, 1, vec![], &cancellation)
                    .await
            }
        });
        race.posted.notified().await;
        cancellation.cancel();
        let error = tokio::time::timeout(Duration::from_secs(5), request)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert!(error.is_cancelled());
        assert!(race.finished.load(Ordering::SeqCst));
        assert!(race.deletes.load(Ordering::SeqCst) >= 2);
        server.abort();
    }

    struct Stalled {
        posted: tokio::sync::Notify,
        release: tokio::sync::Notify,
        deletes: AtomicUsize,
        cancellation_status: Option<StatusCode>,
    }

    async fn stalled_cancellation(cancellation_status: Option<StatusCode>) {
        async fn analyze(State(stalled): State<Arc<Stalled>>) -> impl IntoResponse {
            stalled.posted.notify_one();
            stalled.release.notified().await;
            (
                StatusCode::from_u16(499).unwrap(),
                Json(rejected(ErrorCode::Cancelled)),
            )
        }
        async fn cancel(State(stalled): State<Arc<Stalled>>) -> StatusCode {
            stalled.deletes.fetch_add(1, Ordering::SeqCst);
            if let Some(status) = stalled.cancellation_status {
                return status;
            }
            std::future::pending().await
        }
        let stalled = Arc::new(Stalled {
            posted: tokio::sync::Notify::new(),
            release: tokio::sync::Notify::new(),
            deletes: AtomicUsize::new(0),
            cancellation_status,
        });
        let router = Router::new()
            .route("/v1/capabilities", get(|| async { Json(advertised()) }))
            .route("/v1/analyze", post(analyze))
            .route("/v1/requests/{id}", delete(cancel))
            .with_state(stalled.clone());
        let (url, server) = mock(router).await;
        let client = Client::connect(&url).await.unwrap();
        let cancellation = Cancellation::new();
        let request = tokio::spawn({
            let cancellation = cancellation.clone();
            async move {
                client
                    .analyze(&captured(), None, 1, vec![], &cancellation)
                    .await
            }
        });
        stalled.posted.notified().await;
        cancellation.cancel();
        let outcome =
            tokio::time::timeout(CANCELLATION_TIMEOUT + Duration::from_secs(2), request).await;
        // Release the mock response only after observing whether cancellation returned.
        stalled.release.notify_one();
        let error = outcome
            .expect("cancellation must finish while the original POST remains stalled")
            .unwrap()
            .unwrap_err();
        assert!(error.is_cancelled());
        let deletes = stalled.deletes.load(Ordering::SeqCst);
        if cancellation_status == Some(StatusCode::NO_CONTENT) {
            assert_eq!(deletes, 1, "acknowledged cancellation only drains the POST");
        } else {
            assert!(
                deletes >= 2,
                "unacknowledged cancellation retries until the deadline"
            );
        }
        server.abort();
    }

    #[tokio::test]
    async fn cancellation_deadline_bounds_stalled_posts_and_cancel_requests() {
        tokio::join!(
            stalled_cancellation(Some(StatusCode::NO_CONTENT)),
            stalled_cancellation(Some(StatusCode::NOT_FOUND)),
            stalled_cancellation(None),
        );
    }

    #[test]
    fn strict_base_urls_preserve_nested_service_paths() {
        for invalid in [
            "",
            "localhost:8080",
            "/tmp/service",
            "file:///tmp/service",
            " http://localhost",
            "http://u:p@localhost",
            "http://localhost/?x=1",
        ] {
            assert!(base_url(invalid).is_err(), "{invalid}");
        }
        assert_eq!(
            endpoint(
                &base_url("https://example.test/compiler").unwrap(),
                "v1/analyze"
            )
            .as_str(),
            "https://example.test/compiler/v1/analyze"
        );
    }
}
