//! HTTP admission, cancellation ownership, and streaming completed artifacts.
use crate::Server;
use axum::{
    Json, Router,
    body::Body,
    extract::{Path, Request, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use futures::StreamExt;
use resin_executor::Cancellation;
use resin_protocol::{AnalyzeRequest, BuildRequest, ErrorCode, Failure};
use serde::{Serialize, de::DeserializeOwned};
use std::sync::Arc;
use tokio::sync::OwnedSemaphorePermit;
use tokio_util::io::ReaderStream;

const MAX_REQUEST_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn failure(code: ErrorCode, message: impl Into<String>) -> Failure {
    Failure {
        code,
        message: message.into(),
        diagnostics: Vec::new(),
        managed_sources: Vec::new(),
    }
}

pub(crate) async fn serve(
    server: Arc<Server>,
    listener: tokio::net::TcpListener,
    cancellation: Cancellation,
) -> Result<(), Failure> {
    let _shutdown = Shutdown(server.clone());
    let router = Router::new()
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/analyze", post(analyze))
        .route("/v1/build", post(build))
        .route("/v1/requests/{request}", delete(cancel))
        .method_not_allowed_fallback(|| async {
            error_response(failure(
                ErrorCode::InvalidRequest,
                "unsupported HTTP method",
            ))
        })
        .fallback(|| async {
            error_response(failure(
                ErrorCode::InvalidRequest,
                "unknown compiler service endpoint",
            ))
        })
        .with_state(server.clone());
    let stopping = server.clone();
    let result = axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            tokio::select! {
                _ = cancellation.cancelled() => {},
                _ = stopping.stopping.cancelled() => {},
            }
            stop(&stopping);
        })
        .await;
    stop(&server);
    server.execution.wait_idle().await;
    result.map_err(|error| failure(ErrorCode::Internal, error.to_string()))
}

fn stop(server: &Server) {
    server.stopping.cancel();
    for token in server.requests.lock().expect("active request map").values() {
        token.cancel();
    }
}

struct Shutdown(Arc<Server>);

impl Drop for Shutdown {
    fn drop(&mut self) {
        stop(&self.0);
    }
}

async fn capabilities(State(server): State<Arc<Server>>) -> Json<resin_protocol::Capabilities> {
    Json(server.capabilities())
}

async fn analyze(State(server): State<Arc<Server>>, request: Request) -> Response {
    let (request, permit) = match decode::<AnalyzeRequest>(&server, request).await {
        Ok(decoded) => decoded,
        Err(error) => return error_response(error),
    };
    let active = match ActiveRequest::new(server.clone(), request.request.clone(), permit) {
        Ok(active) => active,
        Err(error) => return error_response(error),
    };
    let result = async {
        let response = crate::analyze::run(&server, request, &active.cancellation).await;
        check_cancellation(&active.cancellation)?;
        let bytes = serialize(&server, response?, &active.cancellation).await?;
        Ok::<_, Failure>(([(header::CONTENT_TYPE, "application/json")], bytes).into_response())
    }
    .await;
    let response = match result {
        Ok(response) => response,
        Err(error) => render_error(&server, error).await,
    };
    retain_request(response, active)
}

async fn build(State(server): State<Arc<Server>>, request: Request) -> Response {
    let (request, permit) = match decode::<BuildRequest>(&server, request).await {
        Ok(decoded) => decoded,
        Err(error) => return error_response(error),
    };
    let active = match ActiveRequest::new(server.clone(), request.request.clone(), permit) {
        Ok(active) => active,
        Err(error) => return error_response(error),
    };
    let result = async {
        if !server
            .capabilities()
            .targets
            .contains(&request.contract.target)
            || request.contract.entry.profile != resin_protocol::EntryProfile::Host
        {
            return Err(failure(
                ErrorCode::UnsupportedTarget,
                "requested output is not supported by this server",
            ));
        }
        let artifact = crate::build::run(&server, request, &active.cancellation).await;
        check_cancellation(&active.cancellation)?;
        artifact_response(&server, artifact?, &active.cancellation).await
    }
    .await;
    let response = match result {
        Ok(response) => response,
        Err(error) => render_error(&server, error).await,
    };
    retain_request(response, active)
}

async fn artifact_response(
    server: &Server,
    artifact: crate::OwnedArtifact,
    cancellation: &Cancellation,
) -> Result<Response, Failure> {
    let length = artifact.metadata.artifact.length;
    let encoded = URL_SAFE_NO_PAD.encode(serialize(server, artifact.metadata, cancellation).await?);
    if encoded.len() > resin_protocol::MAX_METADATA_BYTES {
        return Err(failure(
            ErrorCode::Internal,
            "artifact metadata exceeds the wire limit",
        ));
    }
    let file = tokio::fs::File::open(artifact.executable.path())
        .await
        .map_err(|error| failure(ErrorCode::Internal, error.to_string()))?;
    let stream = futures::stream::unfold(
        (
            ReaderStream::new(file),
            cancellation.clone(),
            artifact.executable,
        ),
        |(mut reader, cancellation, executable)| async move {
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => None,
                bytes = reader.next() => bytes.map(|bytes| (bytes, (reader, cancellation, executable))),
            }
        },
    );
    let mut response = Body::from_stream(stream).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/octet-stream"),
    );
    response.headers_mut().insert(
        header::CONTENT_LENGTH,
        header::HeaderValue::from_str(&length.to_string()).expect("numeric length"),
    );
    response.headers_mut().insert(
        resin_protocol::METADATA_HEADER,
        header::HeaderValue::from_str(&encoded).expect("base64 metadata"),
    );
    Ok(response)
}

// Keep admission through serialization and body consumption, including compiler errors.
fn retain_request(response: Response, active: ActiveRequest) -> Response {
    let (parts, body) = response.into_parts();
    let stream = futures::stream::unfold(
        (body.into_data_stream(), active),
        |(mut body, active)| async move { body.next().await.map(|bytes| (bytes, (body, active))) },
    );
    Response::from_parts(parts, Body::from_stream(stream))
}

async fn cancel(State(server): State<Arc<Server>>, Path(request): Path<String>) -> Response {
    let requests = server.requests.lock().expect("active request map");
    match requests.get(&request) {
        Some(cancellation) => {
            cancellation.cancel();
            StatusCode::NO_CONTENT.into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

fn check_cancellation(cancellation: &Cancellation) -> Result<(), Failure> {
    cancellation
        .check()
        .map_err(|_| failure(ErrorCode::Cancelled, "request cancelled"))
}

async fn decode<T: DeserializeOwned + Send + 'static>(
    server: &Server,
    request: Request,
) -> Result<(T, OwnedSemaphorePermit), Failure> {
    if server.stopping.is_cancelled() {
        return Err(failure(ErrorCode::Cancelled, "server is shutting down"));
    }
    let permit = server
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| failure(ErrorCode::Busy, "server request capacity is full"))?;
    let bytes = tokio::select! {
        biased;
        _ = server.stopping.cancelled() => return Err(failure(ErrorCode::Cancelled, "server is shutting down")),
        bytes = axum::body::to_bytes(request.into_body(), MAX_REQUEST_BYTES) => bytes.map_err(|error| failure(ErrorCode::InvalidRequest, format!("invalid request body: {error}")))?,
    };
    let value = server
        .execution
        .run(&server.stopping, move |_| {
            serde_json::from_slice::<T>(&bytes)
        })
        .await
        .map_err(|error| failure(ErrorCode::Cancelled, error.to_string()))?
        .map_err(|error| failure(ErrorCode::InvalidRequest, error.to_string()))?;
    Ok((value, permit))
}

async fn serialize<T: Serialize + Send + 'static>(
    server: &Server,
    value: T,
    cancellation: &Cancellation,
) -> Result<Vec<u8>, Failure> {
    server
        .execution
        .run(cancellation, move |_| serde_json::to_vec(&value))
        .await
        .map_err(|error| failure(ErrorCode::Cancelled, error.to_string()))?
        .map_err(|error| failure(ErrorCode::Internal, error.to_string()))
}

async fn render_error(server: &Server, error: Failure) -> Response {
    if error.diagnostics.is_empty() && error.managed_sources.is_empty() {
        return error_response(error);
    }
    let status = status(error.code);
    match serialize(server, error, &server.stopping).await {
        Ok(bytes) => (status, [(header::CONTENT_TYPE, "application/json")], bytes).into_response(),
        Err(error) => error_response(error),
    }
}

fn error_response(error: Failure) -> Response {
    (status(error.code), Json(error)).into_response()
}

fn status(code: ErrorCode) -> StatusCode {
    match code {
        ErrorCode::InvalidRequest => StatusCode::BAD_REQUEST,
        ErrorCode::IncompatibleProtocol
        | ErrorCode::InputUnavailable
        | ErrorCode::ManagedSnapshotUnavailable
        | ErrorCode::UnsupportedTarget => StatusCode::CONFLICT,
        ErrorCode::CompilationFailed => StatusCode::UNPROCESSABLE_ENTITY,
        ErrorCode::Cancelled => StatusCode::from_u16(499).expect("cancelled status"),
        ErrorCode::Busy => StatusCode::TOO_MANY_REQUESTS,
        ErrorCode::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

struct ActiveRequest {
    server: Arc<Server>,
    request: String,
    cancellation: Cancellation,
    _permit: OwnedSemaphorePermit,
}

impl ActiveRequest {
    fn new(
        server: Arc<Server>,
        request: String,
        permit: OwnedSemaphorePermit,
    ) -> Result<Self, Failure> {
        if request.is_empty()
            || request.len() > 128
            || !request
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"-_".contains(&byte))
        {
            return Err(failure(
                ErrorCode::InvalidRequest,
                "request IDs must be nonempty portable opaque tokens",
            ));
        }
        let cancellation = Cancellation::new();
        {
            let mut requests = server.requests.lock().expect("active request map");
            if server.stopping.is_cancelled() {
                return Err(failure(ErrorCode::Cancelled, "server is shutting down"));
            }
            if requests.contains_key(&request) {
                return Err(failure(
                    ErrorCode::InvalidRequest,
                    "request ID is already active",
                ));
            }
            requests.insert(request.clone(), cancellation.clone());
        }
        Ok(Self {
            server,
            request,
            cancellation,
            _permit: permit,
        })
    }
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.server
            .requests
            .lock()
            .expect("active request map")
            .remove(&self.request);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;
    use tokio::io::AsyncWriteExt;

    async fn server(directory: &tempfile::TempDir) -> Arc<Server> {
        let library = directory.path().join("library");
        std::fs::create_dir(&library).unwrap();
        let mut environment = resin_toolchain::Environment::capture().unwrap();
        environment.directory = directory.path().to_owned();
        environment.temporary = directory.path().to_owned();
        Arc::new(
            Server::new(crate::Config {
                library_root: library,
                runtime_include: resin_runtime::INCLUDE_DIR.into(),
                dependencies: vec![],
                storage: directory.path().join("storage"),
                temporary: directory.path().to_owned(),
                tools: environment.toolchain(None, None),
                target: crate::host_target(),
                host_backend: crate::HostBackend::C,
                capacities: crate::Capacities::default(),
            })
            .await
            .unwrap(),
        )
    }

    #[tokio::test]
    async fn connection_loss_drops_admitted_request_ownership() {
        async fn blocked(State(server): State<Arc<Server>>) -> StatusCode {
            let permit = server.admission.clone().acquire_owned().await.unwrap();
            let active = ActiveRequest::new(server, "blocked".into(), permit).unwrap();
            active.cancellation.cancelled().await;
            StatusCode::NO_CONTENT
        }
        let directory = tempfile::tempdir().unwrap();
        let server = server(&directory).await;
        let router = Router::new()
            .route("/blocked", post(blocked))
            .with_state(server.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let running = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let mut connection = tokio::net::TcpStream::connect(address).await.unwrap();
        connection
            .write_all(b"POST /blocked HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
            .await
            .unwrap();
        let cancellation = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(token) = server.requests.lock().unwrap().get("blocked").cloned() {
                    break token;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(connection);
        tokio::time::timeout(Duration::from_secs(5), cancellation.cancelled())
            .await
            .unwrap();
        assert!(server.requests.lock().unwrap().is_empty());
        assert_eq!(
            server.admission.available_permits(),
            server.config.capacities.requests
        );
        running.abort();
    }

    #[tokio::test]
    async fn shutdown_cancels_requests_and_incomplete_bodies_before_draining() {
        let directory = tempfile::tempdir().unwrap();
        let server = server(&directory).await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let cancellation = Cancellation::new();
        let running = tokio::spawn(server.clone().serve(listener, cancellation.clone()));
        let mut connection = tokio::net::TcpStream::connect(address).await.unwrap();
        connection
            .write_all(
                b"POST /v1/analyze HTTP/1.1\r\nHost: localhost\r\nContent-Length: 99999\r\n\r\n{",
            )
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while server.admission.available_permits() == server.config.capacities.requests {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        cancellation.cancel();
        tokio::time::timeout(Duration::from_secs(5), running)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(server.requests.lock().unwrap().is_empty());
        assert_eq!(
            server.admission.available_permits(),
            server.config.capacities.requests
        );
    }

    #[tokio::test]
    async fn malformed_requests_and_capacity_failures_are_structured() {
        let directory = tempfile::tempdir().unwrap();
        let server = server(&directory).await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let cancellation = Cancellation::new();
        let running = tokio::spawn(server.clone().serve(listener, cancellation.clone()));
        let client = reqwest::Client::new();
        for request in [
            client.post(format!("{url}/v1/analyze")).body("{bad"),
            client.put(format!("{url}/v1/analyze")),
            client.get(format!("{url}/missing")),
        ] {
            let response = request.send().await.unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
            assert_eq!(
                response.json::<Failure>().await.unwrap().code,
                ErrorCode::InvalidRequest
            );
        }
        let permit = server
            .admission
            .clone()
            .acquire_many_owned(server.config.capacities.requests as u32)
            .await
            .unwrap();
        let response = client
            .post(format!("{url}/v1/analyze"))
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response.json::<Failure>().await.unwrap().code,
            ErrorCode::Busy
        );
        drop(permit);
        assert_eq!(server.counters(), crate::Counters::default());
        cancellation.cancel();
        running.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn compiler_error_responses_retain_admission_until_the_body_is_released() {
        let directory = tempfile::tempdir().unwrap();
        let server = server(&directory).await;
        let request = BuildRequest {
            request: "failed-build".into(),
            revision: 1,
            inputs: resin_protocol::InputSelection::Full {
                inputs: resin_protocol::Inputs {
                    entry: "main.resin".into(),
                    sources: vec![resin_protocol::SourceFile {
                        name: "main.resin".into(),
                        text: "export { main }; def main() = { missing };".into(),
                    }],
                    imports: vec![],
                    headers: Default::default(),
                    acquisition_diagnostics: vec![],
                    managed_snapshot: server.managed.snapshot().into(),
                },
            },
            contract: resin_protocol::BuildContract {
                target: crate::host_target(),
                entry: resin_protocol::EntryTarget {
                    export: "main".into(),
                    profile: resin_protocol::EntryProfile::Host,
                },
                profile: resin_protocol::BuildProfile::Debug,
                options: Default::default(),
            },
        };
        let response = build(
            State(server.clone()),
            Request::new(Body::from(serde_json::to_vec(&request).unwrap())),
        )
        .await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            server.admission.available_permits(),
            server.config.capacities.requests - 1
        );
        assert!(server.requests.lock().unwrap().contains_key("failed-build"));
        drop(response);
        assert_eq!(
            server.admission.available_permits(),
            server.config.capacities.requests
        );
        assert!(server.requests.lock().unwrap().is_empty());
    }

    #[test]
    fn failure_codes_have_explicit_http_statuses() {
        assert_eq!(status(ErrorCode::InvalidRequest), StatusCode::BAD_REQUEST);
        assert_eq!(status(ErrorCode::InputUnavailable), StatusCode::CONFLICT);
        assert_eq!(
            status(ErrorCode::CompilationFailed),
            StatusCode::UNPROCESSABLE_ENTITY
        );
        assert_eq!(status(ErrorCode::Cancelled).as_u16(), 499);
        assert_eq!(status(ErrorCode::Busy), StatusCode::TOO_MANY_REQUESTS);
    }
}
