//! Explicit service process and its native byte-embedding helper.
mod embed;
use clap::Parser;
use resin_executor::Cancellation;
use std::{ffi::OsString, path::PathBuf, sync::Arc};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Parser)]
#[command(name = "resin-server", version)]
struct Args {
    /// HTTP listen address. Expose shared deployments behind an authenticated proxy.
    #[arg(long, default_value = "127.0.0.1:7412")]
    listen: std::net::SocketAddr,
    /// Server-provided Resin libraries, snapshotted at startup.
    #[arg(long)]
    library_root: Option<PathBuf>,
    /// Server-owned dependency storage; contains no client workspace discovery files.
    #[arg(long)]
    storage: Option<PathBuf>,
    /// Parent of owned native build and download staging directories.
    #[arg(long)]
    temporary: Option<PathBuf>,
    #[arg(long)]
    cc: Option<OsString>,
    #[arg(long)]
    spirv_opt: Option<OsString>,
    /// Host code generator; Cranelift currently supports scalar-only, headerless programs.
    #[arg(long, value_enum, default_value = "c")]
    host_backend: resin_server::HostBackend,
    /// Entry capacity applied to each cache (requested overflow is retained with a warning).
    #[arg(long)]
    cache_capacity: Option<usize>,
    #[arg(long, default_value_t = 64)]
    requests: usize,
    /// JSON array of pinned Git dependencies, administered by the service operator.
    #[arg(long)]
    dependencies: Option<PathBuf>,
    /// Native build helper: embed arbitrary bytes into a C header.
    #[arg(long, requires_all = ["output", "symbol"])]
    embed: Option<PathBuf>,
    #[arg(short = 'o', long, requires = "embed")]
    output: Option<PathBuf>,
    #[arg(long, requires = "embed")]
    symbol: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Dependency {
    name: String,
    repository: String,
    commit: String,
    source_subdirectory: PathBuf,
    #[serde(default)]
    header_subdirectories: Vec<PathBuf>,
}

fn main() {
    let _ = log::set_logger(&Warnings);
    log::set_max_level(log::LevelFilter::Warn);
    if let Err(error) = run(Args::parse()) {
        eprintln!("resin-server: {error}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    if let Some(input) = args.embed {
        embed::run(&input, &args.output.unwrap(), &args.symbol.unwrap())?;
        return Ok(());
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(serve(args))
}

async fn serve(args: Args) -> Result<()> {
    let environment = resin_toolchain::Environment::capture()?;
    let temporary = args
        .temporary
        .unwrap_or_else(|| environment.temporary.clone());
    let storage = args
        .storage
        .unwrap_or_else(|| temporary.join("resin-server-dependencies"));
    let mut capacities = resin_server::Capacities::default();
    if let Some(capacity) = args.cache_capacity {
        capacities.inputs = capacity;
        capacities.sources = capacity;
        capacities.syntax = capacity;
        capacities.ast = capacity;
        capacities.hir = capacity;
        capacities.verified = capacity;
        capacities.generated = capacity;
    }
    capacities.requests = args.requests;
    let dependencies: Vec<Dependency> = match args.dependencies {
        Some(path) => serde_json::from_slice(&tokio::fs::read(path).await?)?,
        None => Vec::new(),
    };
    let config = resin_server::Config {
        runtime_include: environment.path("RESIN_RUNTIME_INCLUDE", resin_runtime::INCLUDE_DIR),
        library_root: args.library_root.unwrap_or_else(|| {
            environment.path("RESIN_LIBRARY_ROOT", resin_source::library_root())
        }),
        dependencies: dependencies
            .into_iter()
            .map(|dependency| resin_server::Dependency {
                name: dependency.name,
                repository: dependency.repository,
                commit: dependency.commit,
                source_subdirectory: dependency.source_subdirectory,
                header_subdirectories: dependency.header_subdirectories,
            })
            .collect(),
        storage,
        temporary,
        tools: environment.toolchain(args.cc.as_deref(), args.spirv_opt.as_deref()),
        target: resin_server::host_target(),
        host_backend: args.host_backend,
        capacities,
    };
    let server = Arc::new(
        resin_server::Server::new(config)
            .await
            .map_err(|error| error.message)?,
    );
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    eprintln!(
        "resin-server listening on http://{}",
        listener.local_addr()?
    );
    let cancellation = Cancellation::new();
    let stopping = cancellation.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        stopping.cancel();
    });
    server
        .serve(listener, cancellation)
        .await
        .map_err(|error| error.message.into())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        if let Ok(mut terminate) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}

struct Warnings;
impl log::Log for Warnings {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}: {}", record.level(), record.args());
        }
    }
    fn flush(&self) {}
}
