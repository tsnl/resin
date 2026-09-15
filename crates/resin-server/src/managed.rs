//! Freeze only configured server libraries and pinned packages before serving clients.
use crate::{Config, http::failure};
use resin_executor::{Cancellation, Execution};
use resin_protocol::{ErrorCode, Failure, ManagedHeaderRoot};
use resin_source::Source;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone)]
pub(crate) struct Managed {
    snapshot: String,
    pub sources: BTreeMap<String, Source>,
    pub headers: BTreeMap<String, BTreeMap<String, Arc<[u8]>>>,
    pub header_order: Vec<String>,
}
impl Managed {
    pub fn snapshot(&self) -> &str {
        &self.snapshot
    }
    pub fn header_roots(&self) -> Vec<ManagedHeaderRoot> {
        self.header_order
            .iter()
            .map(|id| ManagedHeaderRoot {
                id: id.clone(),
                headers: self.headers[id].keys().cloned().collect(),
            })
            .chain(std::iter::once(ManagedHeaderRoot {
                id: "system".into(),
                headers: vec![],
            }))
            .collect()
    }
}

pub(crate) async fn load(
    config: &Config,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<Managed, Failure> {
    tokio::fs::create_dir_all(&config.storage)
        .await
        .map_err(internal)?;
    tokio::fs::create_dir_all(&config.temporary)
        .await
        .map_err(internal)?;
    let mut sources = BTreeMap::new();
    let mut headers = BTreeMap::new();
    let mut header_order = vec!["runtime".to_owned()];
    add_sources(
        &mut sources,
        "$/",
        tree(&config.library_root, execution, cancellation).await?,
    )?;
    headers.insert(
        "runtime".into(),
        tree(&config.runtime_include, execution, cancellation).await?,
    );
    let mut names = BTreeSet::new();
    for dependency in &config.dependencies {
        cancellation.check().map_err(internal)?;
        if dependency.name.is_empty()
            || !dependency
                .name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
            || !names.insert(dependency.name.clone())
            || ![40, 64].contains(&dependency.commit.len())
            || !dependency.commit.bytes().all(|c| c.is_ascii_hexdigit())
        {
            return Err(failure(
                ErrorCode::InvalidRequest,
                "dependency names must be unique and revisions must be full Git commit IDs",
            ));
        }
        let checkout = tempfile::Builder::new()
            .prefix("resin-dependency-")
            .tempdir_in(&config.storage)
            .map_err(internal)?;
        git(checkout.path(), &["init", "--quiet"], cancellation).await?;
        git(
            checkout.path(),
            &[
                "fetch",
                "--quiet",
                "--depth=1",
                "--",
                &dependency.repository,
                &dependency.commit,
            ],
            cancellation,
        )
        .await?;
        git(
            checkout.path(),
            &["checkout", "--quiet", "--detach", "FETCH_HEAD"],
            cancellation,
        )
        .await?;
        let actual = git(checkout.path(), &["rev-parse", "HEAD"], cancellation).await?;
        if actual.trim() != dependency.commit.to_ascii_lowercase() {
            return Err(failure(
                ErrorCode::InvalidRequest,
                "fetched dependency commit does not match its configured pin",
            ));
        }
        let source = subdirectory(
            checkout.path(),
            &dependency.source_subdirectory,
            execution,
            cancellation,
        )
        .await?;
        add_sources(
            &mut sources,
            &format!("$/deps/{}/", dependency.name),
            tree(&source, execution, cancellation).await?,
        )?;
        for (index, subdir) in dependency.header_subdirectories.iter().enumerate() {
            let directory = subdirectory(checkout.path(), subdir, execution, cancellation).await?;
            let root = format!("package/{}/{index}", dependency.name);
            headers.insert(
                root.clone(),
                tree(&directory, execution, cancellation).await?,
            );
            header_order.push(root);
        }
    }
    let mut digest = blake3::Hasher::new();
    digest.update(b"resin-managed-snapshot-v1\0");
    for root in &header_order {
        hash(&mut digest, root.as_bytes());
    }
    for (name, source) in &sources {
        hash(&mut digest, name.as_bytes());
        hash(&mut digest, source.text().as_bytes());
    }
    for (root, files) in &headers {
        hash(&mut digest, root.as_bytes());
        for (name, bytes) in files {
            hash(&mut digest, name.as_bytes());
            hash(&mut digest, bytes);
        }
    }
    Ok(Managed {
        snapshot: digest.finalize().to_hex().to_string(),
        sources,
        headers,
        header_order,
    })
}

fn add_sources(
    sources: &mut BTreeMap<String, Source>,
    prefix: &str,
    files: BTreeMap<String, Arc<[u8]>>,
) -> Result<(), Failure> {
    for (name, bytes) in files
        .into_iter()
        .filter(|(name, _)| name.ends_with(".resin"))
    {
        let name = format!("{prefix}{name}");
        let text = std::str::from_utf8(&bytes).map_err(internal)?;
        if sources
            .insert(name.clone(), Source::new(name.as_str(), text))
            .is_some()
        {
            return Err(failure(
                ErrorCode::InvalidRequest,
                "managed source namespaces overlap",
            ));
        }
    }
    Ok(())
}

async fn subdirectory(
    root: &Path,
    path: &Path,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<PathBuf, Failure> {
    if path.components().any(|part| {
        !matches!(
            part,
            std::path::Component::Normal(_) | std::path::Component::CurDir
        )
    }) {
        return Err(failure(
            ErrorCode::InvalidRequest,
            "dependency subdirectories must be relative and contained",
        ));
    }
    let root = root.to_owned();
    let path = path.to_owned();
    execution
        .run(cancellation, move |_| {
            let root = std::fs::canonicalize(root).map_err(internal)?;
            let path = std::fs::canonicalize(root.join(path)).map_err(internal)?;
            if !path.starts_with(&root) {
                return Err(failure(
                    ErrorCode::InvalidRequest,
                    "dependency subdirectory escapes its pinned checkout",
                ));
            }
            Ok(path)
        })
        .await
        .map_err(internal)?
}

async fn git(
    directory: &Path,
    arguments: &[&str],
    cancellation: &Cancellation,
) -> Result<String, Failure> {
    use std::process::Stdio;
    let mut command = tokio::process::Command::new("git");
    command
        .args(arguments)
        .current_dir(directory)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn().map_err(internal)?;
    let stdout = child.stdout.take().expect("piped Git stdout");
    let stderr = child.stderr.take().expect("piped Git stderr");
    let out = tokio::spawn(read_output(stdout));
    let err = tokio::spawn(read_output(stderr));
    let status = tokio::select! {
        result = child.wait() => result.map_err(internal)?,
        _ = cancellation.cancelled() => {
            let _ = child.kill().await; let _ = child.wait().await;
            let _ = out.await; let _ = err.await;
            return Err(failure(ErrorCode::Cancelled, "dependency acquisition cancelled"));
        }
    };
    let stdout = out.await.map_err(internal)?.map_err(internal)?;
    let stderr = err.await.map_err(internal)?.map_err(internal)?;
    if !status.success() {
        return Err(internal(String::from_utf8_lossy(&stderr)));
    }
    String::from_utf8(stdout).map_err(internal)
}
async fn read_output(mut output: impl tokio::io::AsyncRead + Unpin) -> std::io::Result<Vec<u8>> {
    use tokio::io::AsyncReadExt;
    let mut bytes = Vec::new();
    output.read_to_end(&mut bytes).await?;
    Ok(bytes)
}

pub(crate) async fn tree(
    root: &Path,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<BTreeMap<String, Arc<[u8]>>, Failure> {
    let root = root.to_owned();
    execution
        .run(cancellation, move |cancellation| {
            fn visit(
                root: &Path,
                directory: &Path,
                relative: &str,
                ancestors: &mut BTreeSet<PathBuf>,
                files: &mut BTreeMap<String, Arc<[u8]>>,
                cancellation: &Cancellation,
            ) -> Result<(), Failure> {
                cancellation.check().map_err(internal)?;
                let canonical = std::fs::canonicalize(directory).map_err(internal)?;
                if !canonical.starts_with(root) || !ancestors.insert(canonical.clone()) {
                    return Err(failure(
                        ErrorCode::InvalidRequest,
                        "managed directory contains an escaping symlink or cycle",
                    ));
                }
                for entry in std::fs::read_dir(directory).map_err(internal)? {
                    cancellation.check().map_err(internal)?;
                    let entry = entry.map_err(internal)?;
                    let file_name = entry.file_name().into_string().map_err(|_| {
                        failure(
                            ErrorCode::InvalidRequest,
                            "managed file names must be UTF-8",
                        )
                    })?;
                    if file_name == ".git" {
                        continue;
                    }
                    let name = if relative.is_empty() {
                        file_name
                    } else {
                        format!("{relative}/{file_name}")
                    };
                    let target = std::fs::canonicalize(entry.path()).map_err(internal)?;
                    if !target.starts_with(root) {
                        return Err(failure(
                            ErrorCode::InvalidRequest,
                            "managed symlink escapes its configured root",
                        ));
                    }
                    let metadata = std::fs::metadata(&target).map_err(internal)?;
                    if metadata.is_dir() {
                        visit(root, &entry.path(), &name, ancestors, files, cancellation)?;
                    } else if metadata.is_file() {
                        files.insert(name, std::fs::read(&target).map_err(internal)?.into());
                    } else {
                        return Err(failure(
                            ErrorCode::InvalidRequest,
                            "managed snapshots require regular files",
                        ));
                    }
                }
                ancestors.remove(&canonical);
                Ok(())
            }
            let root = std::fs::canonicalize(root).map_err(internal)?;
            let mut files = BTreeMap::new();
            visit(
                &root,
                &root,
                "",
                &mut BTreeSet::new(),
                &mut files,
                cancellation,
            )?;
            Ok(files)
        })
        .await
        .map_err(internal)?
}
fn hash(hash: &mut blake3::Hasher, bytes: &[u8]) {
    hash.update(&(bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}
fn internal(error: impl std::fmt::Display) -> Failure {
    failure(ErrorCode::Internal, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Capacities, Dependency, Server};
    use resin_protocol::{AnalyzeRequest, HeaderInputs, InputSelection, Inputs};

    fn config(root: &Path, dependency: Dependency) -> Config {
        let library = root.join("library");
        std::fs::create_dir_all(&library).unwrap();
        let mut environment = resin_toolchain::Environment::capture().unwrap();
        environment.directory = root.into();
        environment.temporary = root.into();
        Config {
            library_root: library,
            runtime_include: resin_runtime::INCLUDE_DIR.into(),
            dependencies: vec![dependency],
            storage: root.join("storage"),
            temporary: root.into(),
            tools: environment.toolchain(None, None),
            target: crate::host_target(),
            host_backend: crate::HostBackend::C,
            capacities: Capacities::default(),
        }
    }
    fn git(root: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().into()
    }

    #[tokio::test]
    async fn pinned_git_sources_and_headers_are_frozen_and_managed_entries_need_no_uploads() {
        let directory = tempfile::TempDir::new().unwrap();
        let repository = directory.path().join("repository");
        std::fs::create_dir_all(repository.join("src")).unwrap();
        std::fs::create_dir(repository.join("include")).unwrap();
        let text = "export { answer }; extern { \"native.h\": { def native_answer() -> int; } }; def answer() -> int = { native_answer() };\n";
        std::fs::write(repository.join("src/value.resin"), text).unwrap();
        std::fs::write(
            repository.join("include/native.h"),
            "static inline int native_answer(void) { return 19; }\n",
        )
        .unwrap();
        git(&repository, &["init", "--quiet"]);
        git(&repository, &["add", "."]);
        git(
            &repository,
            &[
                "-c",
                "user.name=Resin tests",
                "-c",
                "user.email=tests@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "pinned fixture",
            ],
        );
        let commit = git(&repository, &["rev-parse", "HEAD"]);
        let dependency = Dependency {
            name: "sample".into(),
            repository: repository.to_str().unwrap().into(),
            commit,
            source_subdirectory: "src".into(),
            header_subdirectories: vec!["include".into()],
        };
        let server = Server::new(config(directory.path(), dependency.clone()))
            .await
            .unwrap();
        std::fs::write(
            repository.join("src/value.resin"),
            "bad source after server startup",
        )
        .unwrap();
        std::fs::write(
            repository.join("include/native.h"),
            "#error changed checkout\n",
        )
        .unwrap();
        assert_eq!(
            server.managed.sources["$/deps/sample/value.resin"].text(),
            text
        );
        let inputs = Inputs {
            entry: "$/deps/sample/value.resin".into(),
            sources: vec![],
            imports: vec![],
            headers: HeaderInputs::default(),
            acquisition_diagnostics: vec![],
            managed_snapshot: server.managed.snapshot().into(),
        };
        let response = crate::analyze::run(
            &server,
            AnalyzeRequest {
                request: crate::token(),
                revision: 1,
                inputs: InputSelection::Full { inputs },
                queries: vec![],
            },
            &Cancellation::new(),
        )
        .await
        .unwrap();
        assert!(
            response.diagnostics.is_empty(),
            "{:?}",
            response.diagnostics
        );
        assert!(
            std::str::from_utf8(&server.managed.headers["package/sample/0"]["native.h"])
                .unwrap()
                .contains("return 19")
        );
        let invalid = Dependency {
            commit: "main".into(),
            ..dependency
        };
        assert_eq!(
            Server::new(config(directory.path(), invalid))
                .await
                .err()
                .unwrap()
                .code,
            ErrorCode::InvalidRequest
        );
    }
}

#[cfg(all(test, unix))]
mod containment_tests {
    use super::*;
    #[tokio::test]
    async fn package_subdirectory_symlinks_cannot_select_an_unrelated_server_tree() {
        let checkout = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        std::os::unix::fs::symlink(outside.path(), checkout.path().join("src")).unwrap();
        let error = subdirectory(
            checkout.path(),
            Path::new("src"),
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidRequest);
    }
}
