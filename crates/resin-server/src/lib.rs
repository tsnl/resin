//! HTTP compiler application over immutable inputs and shared compiler cache heads.
use resin_executor::{Cancellation, Execution};
use resin_protocol::{BuildMetadata, Capabilities, EntryProfile, Failure, Target};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tokio::sync::Semaphore;

mod analyze;
mod build;
mod caches;
mod headers;
mod http;
mod inputs;
mod managed;
mod publication;

/// One exact dependency revision, resolved and frozen before accepting requests.
#[derive(Clone, Debug)]
pub struct Dependency {
    pub name: String,
    pub repository: String,
    pub commit: String,
    pub source_subdirectory: PathBuf,
    pub header_subdirectories: Vec<PathBuf>,
}

/// Entry-count bounds. Active requests retain selected values beyond head eviction.
#[derive(Clone, Debug)]
pub struct Capacities {
    pub inputs: usize,
    pub sources: usize,
    pub syntax: usize,
    pub ast: usize,
    pub hir: usize,
    pub verified: usize,
    pub generated: usize,
    pub requests: usize,
}

impl Default for Capacities {
    fn default() -> Self {
        Self {
            inputs: 64,
            sources: 4096,
            syntax: 4096,
            ast: 4096,
            hir: 64,
            verified: 64,
            generated: 32,
            requests: 64,
        }
    }
}

/// Complete server-owned filesystem and native build settings. User paths are never resolved here.
pub struct Config {
    pub library_root: PathBuf,
    pub runtime_include: PathBuf,
    pub dependencies: Vec<Dependency>,
    pub storage: PathBuf,
    pub temporary: PathBuf,
    pub tools: resin_toolchain::Toolchain,
    pub target: Target,
    pub host_backend: HostBackend,
    pub capacities: Capacities,
}

/// Host emission selected for this server instance. Cranelift is a scalar-only prototype.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum HostBackend {
    #[default]
    C,
    Cranelift,
}

/// Observation counters for embedding applications and tests, not an HTTP administration API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub source_builds: u64,
    pub syntax_builds: u64,
    pub ast_builds: u64,
    pub hir_builds: u64,
    pub verified_builds: u64,
    pub generated_builds: u64,
    pub native_object_builds: u64,
}

/// Independent server state. Its immutable cache heads are shared by analysis and builds.
pub struct Server {
    pub(crate) config: Config,
    pub(crate) managed: managed::Managed,
    pub(crate) caches: caches::Caches,
    pub(crate) execution: Execution,
    pub(crate) instance: String,
    pub(crate) requests: Mutex<BTreeMap<String, Cancellation>>,
    pub(crate) admission: Arc<Semaphore>,
    pub(crate) stopping: Cancellation,
}

impl Server {
    /// Freeze managed inputs and initialize empty cache heads before serving.
    pub async fn new(config: Config) -> Result<Self, Failure> {
        if config.target != host_target() {
            return Err(http::failure(
                resin_protocol::ErrorCode::UnsupportedTarget,
                "server target must match its installed host toolchain",
            ));
        }
        if config.capacities.requests == 0 {
            return Err(http::failure(
                resin_protocol::ErrorCode::InvalidRequest,
                "request capacity must be positive",
            ));
        }
        let execution = Execution::default();
        let stopping = Cancellation::new();
        let managed = managed::load(&config, &execution, &stopping).await?;
        let caches = caches::Caches::new(&config.capacities);
        let admission = Arc::new(Semaphore::new(config.capacities.requests));
        Ok(Self {
            config,
            managed,
            caches,
            execution,
            instance: token(),
            requests: Mutex::new(BTreeMap::new()),
            admission,
            stopping,
        })
    }

    pub fn capabilities(&self) -> Capabilities {
        Capabilities {
            protocol: resin_protocol::PROTOCOL_VERSION,
            instance: self.instance.clone(),
            managed_snapshot: self.managed.snapshot().to_owned(),
            targets: vec![self.config.target.clone()],
            entry_profiles: vec![EntryProfile::Host],
            header_roots: self.managed.header_roots(),
        }
    }

    pub fn counters(&self) -> Counters {
        self.caches.counters()
    }

    /// Serve until cancelled, then cancel admitted work and drain its native/CPU ownership.
    pub async fn serve(
        self: Arc<Self>,
        listener: tokio::net::TcpListener,
        cancellation: Cancellation,
    ) -> Result<(), Failure> {
        http::serve(self, listener, cancellation).await
    }
}

/// This binary's supported native contract. Cross-compilation is not advertised.
pub fn host_target() -> Target {
    Target {
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

pub(crate) struct OwnedArtifact {
    pub metadata: BuildMetadata,
    pub executable: resin_toolchain::Executable,
}

pub(crate) fn token() -> String {
    uuid::Uuid::new_v4().to_string()
}
