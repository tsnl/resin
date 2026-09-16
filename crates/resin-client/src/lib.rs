//! Resin's local syntax, editor, and command-line client for an explicitly selected service.
use resin_executor::{Cancellation, Execution};
use resin_protocol::{
    AnalyzeResponse, BuildContract, BuildMetadata, Capabilities, Failure, InputHandle, Inputs,
    Query,
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

mod cli;
mod connection;
mod download;
mod headers;
mod inputs;
mod interp;
mod lsp;

pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Run the merged command-line client. Build and editor operations require RESIN_SERVER.
pub fn main() -> ! {
    cli::main()
}

/// A negotiated HTTP connection. Clones share capabilities and bounded local work.
#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base: reqwest::Url,
    capabilities: Arc<Mutex<Arc<Capabilities>>>,
    execution: Execution,
}

impl Client {
    /// Require an absolute HTTP(S) URL and negotiate before acquiring source inputs.
    pub async fn connect(url: &str) -> std::result::Result<Self, Error> {
        connection::connect(url).await
    }

    pub fn capabilities(&self) -> Arc<Capabilities> {
        self.capabilities
            .lock()
            .expect("client capabilities")
            .clone()
    }

    pub async fn refresh_capabilities(&self) -> std::result::Result<Arc<Capabilities>, Error> {
        let capabilities =
            connection::capabilities(&self.http, &self.base, &self.execution).await?;
        *self.capabilities.lock().expect("client capabilities") = capabilities.clone();
        Ok(capabilities)
    }

    /// Send a complete capture or a source delta, retrying an unavailable base once with Full.
    pub async fn analyze(
        &self,
        current: &Inputs,
        previous: Option<(&InputHandle, &Inputs)>,
        revision: u64,
        queries: Vec<Query>,
        cancellation: &Cancellation,
    ) -> std::result::Result<AnalyzeResponse, Error> {
        connection::analyze(self, current, previous, revision, queries, cancellation).await
    }

    /// Download and verify one executable, then atomically replace the requested local file.
    pub async fn build(
        &self,
        current: &Inputs,
        previous: Option<(&InputHandle, &Inputs)>,
        revision: u64,
        contract: BuildContract,
        destination: &Path,
        cancellation: &Cancellation,
    ) -> std::result::Result<BuiltArtifact, Error> {
        connection::build(
            self,
            current,
            previous,
            revision,
            contract,
            destination,
            cancellation,
        )
        .await
    }
}

/// A verified download at a caller-selected path. The caller owns its local file lifetime.
#[derive(Debug)]
pub struct BuiltArtifact {
    path: PathBuf,
    metadata: BuildMetadata,
}

impl BuiltArtifact {
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn metadata(&self) -> &BuildMetadata {
        &self.metadata
    }
}

/// Transport, integrity, negotiation, or structured service failure.
#[derive(Debug)]
pub struct Error {
    message: String,
    kind: ErrorKind,
}

#[derive(Debug)]
enum ErrorKind {
    Local,
    Remote { failure: Failure },
    InstanceChanged,
}

impl Error {
    pub fn failure(&self) -> Option<&Failure> {
        match &self.kind {
            ErrorKind::Remote { failure } => Some(failure),
            ErrorKind::Local | ErrorKind::InstanceChanged => None,
        }
    }
    pub fn is_cancelled(&self) -> bool {
        self.failure()
            .is_some_and(|failure| failure.code == resin_protocol::ErrorCode::Cancelled)
    }
    fn new(message: impl std::fmt::Display) -> Self {
        Self {
            message: message.to_string(),
            kind: ErrorKind::Local,
        }
    }
    fn instance_changed() -> Self {
        Self {
            message: "response belongs to another server instance".into(),
            kind: ErrorKind::InstanceChanged,
        }
    }
    fn cancelled() -> Self {
        Self::from(Failure {
            code: resin_protocol::ErrorCode::Cancelled,
            message: "request cancelled".into(),
            diagnostics: Vec::new(),
            managed_sources: Vec::new(),
        })
    }
}

impl From<Failure> for Error {
    fn from(failure: Failure) -> Self {
        Self {
            message: failure.message.clone(),
            kind: ErrorKind::Remote { failure },
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(output)
    }
}
impl std::error::Error for Error {}

/// Local execution compatibility; this does not select or discover a service.
pub fn host_target() -> resin_protocol::Target {
    resin_protocol::Target {
        os: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        abi: if cfg!(target_env = "msvc") {
            "msvc"
        } else if cfg!(target_env = "musl") {
            "musl"
        } else if cfg!(target_os = "macos") {
            "darwin"
        } else {
            "gnu"
        }
        .into(),
        runtime: format!("resin-runtime-{}", env!("CARGO_PKG_VERSION")),
    }
}
