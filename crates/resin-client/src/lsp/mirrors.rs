//! Immutable local files make server-owned definitions navigable with ordinary file URIs.
use resin_executor::{Cancellation, Execution};
use resin_protocol::SourceFile;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub(super) struct Mirrors {
    directory: tempfile::TempDir,
    files: Mutex<BTreeMap<PathBuf, ManagedSource>>,
}

#[derive(Clone)]
pub(super) struct ManagedSource {
    pub source: SourceFile,
    pub snapshot: String,
}

impl Mirrors {
    pub(super) fn new() -> super::Result<Self> {
        Ok(Self {
            directory: tempfile::tempdir()?,
            files: Mutex::new(BTreeMap::new()),
        })
    }

    pub(super) fn source(&self, path: &Path) -> Option<ManagedSource> {
        self.files
            .lock()
            .expect("managed mirrors")
            .get(path)
            .cloned()
    }

    pub(super) async fn materialize(
        self: &Arc<Self>,
        sources: Vec<SourceFile>,
        snapshot: String,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> super::Result<BTreeMap<String, PathBuf>> {
        let mirrors = self.clone();
        execution
            .run(cancellation, move |cancellation| {
                let mut paths = BTreeMap::new();
                for source in sources {
                    cancellation.check()?;
                    if !source.name.starts_with("$/") {
                        return Err("server returned a non-managed mirror name".into());
                    }
                    let mut hash = blake3::Hasher::new();
                    for value in [&snapshot, &source.name, &source.text] {
                        hash.update(&(value.len() as u64).to_le_bytes());
                        hash.update(value.as_bytes());
                    }
                    let path = mirrors
                        .directory
                        .path()
                        .join(format!("{}.resin", hash.finalize().to_hex()));
                    let mut files = mirrors.files.lock().expect("managed mirrors");
                    if !files.contains_key(&path) {
                        std::fs::write(&path, &source.text)?;
                        let mut permissions = std::fs::metadata(&path)?.permissions();
                        permissions.set_readonly(true);
                        std::fs::set_permissions(&path, permissions)?;
                        files.insert(
                            path.clone(),
                            ManagedSource {
                                source: source.clone(),
                                snapshot: snapshot.clone(),
                            },
                        );
                    }
                    paths.insert(source.name, path);
                }
                Ok(paths)
            })
            .await?
    }
}
