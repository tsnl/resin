//! Real HTTP/native ownership: disconnects reap work and cache eviction cannot truncate downloads.
use base64::Engine;
use resin_executor::Cancellation;
use resin_protocol::*;
use resin_server::{Capacities, Config, Server};
use std::{path::Path, sync::Arc, time::Duration};
use tempfile::TempDir;
use tokio::net::TcpListener;

struct Service {
    _directory: TempDir,
    server: Arc<Server>,
    address: std::net::SocketAddr,
    stop: Cancellation,
    running: Option<tokio::task::JoinHandle<Result<(), Failure>>>,
}
impl Service {
    async fn start(directory: TempDir, compiler: Option<&Path>) -> Self {
        let root = directory.path();
        let library = root.join("library");
        std::fs::create_dir(&library).unwrap();
        let mut environment = resin_toolchain::Environment::capture().unwrap();
        environment.directory = root.into();
        environment.temporary = root.into();
        let server = Arc::new(
            Server::new(Config {
                library_root: library,
                runtime_include: resin_runtime::INCLUDE_DIR.into(),
                dependencies: vec![],
                storage: root.join("storage"),
                temporary: root.into(),
                tools: environment.toolchain(compiler.map(Path::as_os_str), None),
                target: resin_server::host_target(),
                capacities: Capacities {
                    generated: 1,
                    ..Default::default()
                },
            })
            .await
            .unwrap(),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let stop = Cancellation::new();
        let running = tokio::spawn(server.clone().serve(listener, stop.clone()));
        Self {
            _directory: directory,
            server,
            address,
            stop,
            running: Some(running),
        }
    }
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.address)
    }
    fn inputs(&self, name: &str, source: &str) -> Inputs {
        Inputs {
            entry: name.into(),
            sources: vec![SourceFile {
                name: name.into(),
                text: source.into(),
            }],
            imports: vec![],
            headers: HeaderInputs::default(),
            acquisition_diagnostics: vec![],
            managed_snapshot: self.server.capabilities().managed_snapshot,
        }
    }
    fn build(&self, inputs: Inputs) -> BuildRequest {
        BuildRequest {
            request: uuid::Uuid::new_v4().to_string(),
            revision: 1,
            inputs: InputSelection::Full { inputs },
            contract: BuildContract {
                target: resin_server::host_target(),
                entry: EntryTarget {
                    export: "main".into(),
                    profile: EntryProfile::Host,
                },
                profile: BuildProfile::Debug,
                options: BuildOptions::default(),
            },
        }
    }
    async fn finish(mut self) {
        self.stop.cancel();
        tokio::time::timeout(Duration::from_secs(10), self.running.take().unwrap())
            .await
            .expect("service shutdown hung")
            .unwrap()
            .unwrap();
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

fn bundle(path: &str, text: &str) -> HeaderBundle {
    let mut hash = blake3::Hasher::new();
    hash.update(b"resin-header-bundle-v1\0");
    for bytes in [path.as_bytes(), text.as_bytes()] {
        hash.update(&(bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    HeaderBundle {
        id: hash.finalize().to_hex().to_string(),
        files: vec![BundleFile {
            path: path.into(),
            contents_base64: base64::engine::general_purpose::STANDARD.encode(text),
        }],
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unread_artifact_stream_survives_eviction_of_its_generation() {
    let service = Service::start(TempDir::new().unwrap(), None).await;
    let client = reqwest::Client::new();
    let mut inputs = service.inputs("large.resin", "export { main }; extern { \"large.h\": { fn retained_value() -> i32; } }; fn main() -> i32  { retained_value() }");
    // A tiny header creates a file larger than socket/HTTP buffering. The client
    // deliberately leaves its body unread while a different generation is built.
    let large = bundle(
        "large.h",
        "static volatile unsigned char retained_bytes[32 * 1024 * 1024] = {7};\nstatic inline int retained_value(void) { return retained_bytes[0]; }\n",
    );
    inputs.headers.bindings.push(HeaderBinding {
        source: "large.resin".into(),
        spelling: "large.h".into(),
        target: HeaderTarget::Uploaded {
            bundle: large.id.clone(),
            path: "large.h".into(),
        },
    });
    inputs.headers.bundles.push(large);
    let request = service.build(inputs);
    let response = tokio::time::timeout(
        Duration::from_secs(30),
        client.post(service.url("/v1/build")).json(&request).send(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(response.status().is_success(), "{:?}", response.status());
    let metadata: BuildMetadata = serde_json::from_slice(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(response.headers()[METADATA_HEADER].as_bytes())
            .unwrap(),
    )
    .unwrap();
    assert!(metadata.artifact.length > 32 * 1024 * 1024);
    // The duplicate request token proves the first response still owns admission.
    let duplicate = AnalyzeRequest {
        request: request.request.clone(),
        revision: 1,
        inputs: InputSelection::Full {
            inputs: service.inputs("other.resin", "fn helper()  {}"),
        },
        queries: vec![],
    };
    let duplicate = client
        .post(service.url("/v1/analyze"))
        .json(&duplicate)
        .send()
        .await
        .unwrap();
    assert_eq!(duplicate.status(), reqwest::StatusCode::BAD_REQUEST);
    assert!(
        duplicate
            .json::<Failure>()
            .await
            .unwrap()
            .message
            .contains("already active")
    );

    let other =
        service.build(service.inputs("other.resin", "export { main }; fn main() -> i32  { 3 }"));
    let other = tokio::time::timeout(
        Duration::from_secs(30),
        client.post(service.url("/v1/build")).json(&other).send(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(other.status().is_success(), "{:?}", other.status());
    let _other = other.bytes().await.unwrap();
    assert_eq!(service.server.counters().generated_builds, 2);
    let bytes = tokio::time::timeout(Duration::from_secs(15), response.bytes())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(bytes.len() as u64, metadata.artifact.length);
    assert_eq!(
        blake3::hash(&bytes).to_hex().as_str(),
        metadata.artifact.blake3
    );
    service.finish().await;
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disconnect_during_native_compilation_reaps_the_real_compiler_process() {
    use std::{os::unix::fs::PermissionsExt, path::PathBuf};
    use tokio::{io::AsyncWriteExt, net::TcpStream};
    let directory = TempDir::new().unwrap();
    let compiler = directory.path().join("blocked-compiler");
    let marker = directory.path().join("compiler.pid");
    let real = std::env::var_os("CC").unwrap_or_else(|| "cc".into());
    fn quote(value: &std::ffi::OsStr) -> String {
        format!("'{}'", value.to_string_lossy().replace('\'', "'\\''"))
    }
    let script = format!(
        "#!/bin/sh\nfor argument in \"$@\"; do\n case \"$argument\" in\n *.i) printf '%s' \"$$\" > {}; exec sleep 60 ;;\n esac\ndone\nexec {} \"$@\"\n",
        quote(marker.as_os_str()),
        quote(&real)
    );
    std::fs::write(&compiler, script).unwrap();
    std::fs::set_permissions(&compiler, std::fs::Permissions::from_mode(0o755)).unwrap();
    let service = Service::start(directory, Some(&compiler)).await;
    let request = service.build(service.inputs(
        "cancelled.resin",
        "export { main }; fn main() -> i32  { 7 }",
    ));
    let bytes = serde_json::to_vec(&request).unwrap();
    let mut connection = TcpStream::connect(service.address).await.unwrap();
    connection.write_all(format!("POST /v1/build HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", bytes.len()).as_bytes()).await.unwrap();
    connection.write_all(&bytes).await.unwrap();
    let pid = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Ok(pid) = tokio::fs::read_to_string(&marker).await
                && !pid.is_empty()
            {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("native compiler did not reach its gated .i compilation");
    drop(connection);
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let alive = tokio::process::Command::new("sh")
                .args(["-c", "kill -0 \"$1\" 2>/dev/null", "probe", &pid])
                .status()
                .await
                .unwrap()
                .success();
            if !alive {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("compiler process survived client disconnect");
    // The server remains available; no explicit DELETE or service shutdown caused reaping.
    let response = reqwest::Client::new()
        .get(service.url("/v1/capabilities"))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let mut programs = vec![];
    fn walk(path: &Path, programs: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                walk(&entry.path(), programs);
            } else if entry.file_name() == "program" {
                programs.push(entry.path());
            }
        }
    }
    walk(service._directory.path(), &mut programs);
    assert!(
        programs.is_empty(),
        "cancelled native work published an artifact: {programs:?}"
    );
    service.finish().await;
}

#[tokio::test]
async fn specialization_diagnostics_survive_cache_publication_and_http_transport() {
    let service = Service::start(TempDir::new().unwrap(), None).await;
    let source = r#"export { main };
        struct Cell { value: i32 }
        fn get(cell: Ref<Cell>) -> Ref<i32> { cell.value }
        fn address<T>(cell: Ref<T>) -> Ptr<i32> { &cell:get() }
        fn main() { let cell = Cell { value = 1 }; address(cell); }
    "#;
    let client = reqwest::Client::new();
    for _ in 0..2 {
        let request = service.build(service.inputs("main.resin", source));
        let response = client
            .post(service.url("/v1/build"))
            .json(&request)
            .send()
            .await
            .unwrap();
        assert!(!response.status().is_success());
        let failure = response.json::<Failure>().await.unwrap();
        assert_eq!(failure.code, ErrorCode::CompilationFailed, "{failure:?}");
        let diagnostic = &failure.diagnostics[0];
        assert_eq!(diagnostic.code.as_deref(), Some("invalid-specialization"));
        assert!(
            diagnostic
                .message
                .contains("cannot take the address of a Ref")
        );
        let span = diagnostic.span.as_ref().unwrap();
        assert_eq!(span.source, "main.resin");
        assert!(source[span.start as usize..span.end as usize].contains("&cell:get()"));
        assert!(
            diagnostic
                .related
                .iter()
                .any(|note| note.message.contains("while specializing address"))
        );
    }
    service.finish().await;
}
