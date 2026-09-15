//! Explicit loopback service fixture; native tools and managed libraries belong to this process.
#![allow(dead_code)]
use resin_executor::Cancellation;
use std::{path::PathBuf, process::Command, sync::Arc, thread};

pub struct Service {
    pub server: Arc<resin_server::Server>,
    pub url: String,
    pub directory: tempfile::TempDir,
    cancellation: Cancellation,
    worker: Option<thread::JoinHandle<()>>,
}

impl Service {
    pub fn new() -> Self {
        Self::configured(|_, _| {})
    }

    pub fn configured(
        configure: impl FnOnce(&mut resin_server::Config, &mut resin_toolchain::Environment),
    ) -> Self {
        Self::configured_at("127.0.0.1:0".parse().unwrap(), configure)
    }

    pub fn restart(
        &mut self,
        configure: impl FnOnce(&mut resin_server::Config, &mut resin_toolchain::Environment),
    ) {
        let address = self.url.strip_prefix("http://").unwrap().parse().unwrap();
        self.cancellation.cancel();
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
        *self = Self::configured_at(address, configure);
    }

    fn configured_at(
        address: std::net::SocketAddr,
        configure: impl FnOnce(&mut resin_server::Config, &mut resin_toolchain::Environment),
    ) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let mut environment = resin_toolchain::Environment::capture().unwrap();
        environment.directory = directory.path().into();
        environment.temporary = directory.path().into();
        environment.executable = PathBuf::from(env!("CARGO_BIN_EXE_resin"));
        let mut config = resin_server::Config {
            runtime_include: environment.path("RESIN_RUNTIME_INCLUDE", resin_runtime::INCLUDE_DIR),
            library_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resin"),
            dependencies: Vec::new(),
            storage: directory.path().join("managed"),
            temporary: directory.path().into(),
            tools: environment.toolchain(None, None),
            target: resin_server::host_target(),
            capacities: resin_server::Capacities::default(),
        };
        configure(&mut config, &mut environment);
        config.tools = environment.toolchain(None, None);
        config.runtime_include =
            environment.path("RESIN_RUNTIME_INCLUDE", resin_runtime::INCLUDE_DIR);
        let cancellation = Cancellation::new();
        let stopping = cancellation.clone();
        let (ready, receiver) = std::sync::mpsc::sync_channel(1);
        let worker = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let server = Arc::new(resin_server::Server::new(config).await.unwrap());
                let listener = tokio::net::TcpListener::bind(address).await.unwrap();
                let url = format!("http://{}", listener.local_addr().unwrap());
                ready.send((server.clone(), url)).unwrap();
                server.serve(listener, stopping).await.unwrap();
            });
        });
        let (server, url) = receiver.recv().unwrap();
        Self {
            server,
            url,
            directory,
            cancellation,
            worker: Some(worker),
        }
    }

    pub fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_resin"));
        command.env("RESIN_SERVER", &self.url);
        command
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.cancellation.cancel();
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}
