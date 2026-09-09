use crossbeam_channel::{Receiver, Sender};
use resin_compiler::{Compilation, Compiler};
use resin_source::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
};
#[cfg(test)]
use tempfile::TempDir;

pub enum Change {
    Set(PathBuf, String),
    Close(PathBuf),
    Refresh,
}

pub struct Update {
    pub revision: u64,
    pub roots: BTreeSet<PathBuf>,
    pub change: Change,
}

pub struct AnalysisUpdate {
    pub revision: u64,
    pub entries: BTreeMap<PathBuf, Arc<Compilation>>,
    pub paths: BTreeMap<SourceId, PathBuf>,
}

pub fn spawn(
    stdlib: PathBuf,
    updates: Receiver<Update>,
    results: Sender<AnalysisUpdate>,
    current: Arc<AtomicU64>,
    stopping: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut compiler = Compiler::new();
        let mut loader = resin_source::Loader::new(stdlib);
        let mut documents = BTreeMap::new();
        while let Ok(first) = updates.recv() {
            if stopping.load(Ordering::Relaxed) {
                break;
            }
            let batch = std::iter::once(first).chain(updates.try_iter());
            let (revision, roots) = apply_updates(&mut loader, &mut documents, batch);
            let result = compile_roots(
                &mut compiler,
                &mut loader,
                &documents,
                roots,
                revision,
                &current,
                &stopping,
            );
            if !superseded(revision, &current, &stopping) && results.send(result).is_err() {
                break;
            }
        }
    })
}

fn apply_updates(
    loader: &mut resin_source::Loader,
    documents: &mut BTreeMap<PathBuf, Source>,
    updates: impl Iterator<Item = Update>,
) -> (u64, BTreeSet<PathBuf>) {
    let mut revision = 0;
    let mut roots = BTreeSet::new();
    for update in updates {
        revision = update.revision;
        roots = update.roots;
        if let Err(error) = apply_change(loader, documents, update.change) {
            eprintln!("resin-lsp: {error}");
        }
    }
    (revision, roots)
}

fn apply_change(
    loader: &mut resin_source::Loader,
    documents: &mut BTreeMap<PathBuf, Source>,
    change: Change,
) -> std::io::Result<()> {
    match change {
        Change::Set(path, text) => {
            let source = loader.source_from_text(&path, text)?;
            documents.insert(path, source);
        }
        Change::Close(path) => {
            let registered = documents
                .get(&path)
                .and_then(|source| loader.path(source))
                .unwrap_or(&path)
                .to_path_buf();
            loader.remove_source(&registered)?;
            documents.remove(&path);
        }
        Change::Refresh => {}
    }
    Ok(())
}

fn superseded(revision: u64, current: &AtomicU64, stopping: &AtomicBool) -> bool {
    stopping.load(Ordering::Relaxed) || current.load(Ordering::Acquire) != revision
}

fn compile_roots(
    compiler: &mut Compiler,
    loader: &mut resin_source::Loader,
    documents: &BTreeMap<PathBuf, Source>,
    roots: BTreeSet<PathBuf>,
    revision: u64,
    current: &AtomicU64,
    stopping: &AtomicBool,
) -> AnalysisUpdate {
    let mut result = AnalysisUpdate {
        revision,
        entries: BTreeMap::new(),
        paths: BTreeMap::new(),
    };
    for path in roots {
        if superseded(revision, current, stopping) {
            break;
        }
        let source = documents
            .get(&path)
            .cloned()
            .map(Ok)
            .unwrap_or_else(|| loader.load_file(&path));
        match source {
            Ok(source) => retain_entry(&mut result, compiler.compile(source, loader), loader, path),
            Err(error) => eprintln!("resin-lsp: {}: {error}", path.display()),
        }
    }
    result
}

fn retain_entry(
    result: &mut AnalysisUpdate,
    compilation: Arc<Compilation>,
    files: &resin_source::Loader,
    path: PathBuf,
) {
    result
        .paths
        .extend(compilation.sources().filter_map(|source| {
            files
                .path(source)
                .map(|path| (source.id(), path.to_path_buf()))
        }));
    result.entries.insert(path, compilation);
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::symlink};

    #[test]
    fn replacing_an_open_file_with_a_symlink_does_not_leak_its_later_edits() {
        let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
        let path = directory.path().join("open.resin");
        let target = directory.path().join("target.resin");
        fs::write(&path, "original disk").unwrap();
        fs::write(&target, "target disk").unwrap();
        let path = resin_source::normalize_path(&path).unwrap();
        let mut loader = resin_source::Loader::new(directory.path().into());
        let mut documents = BTreeMap::new();
        apply_change(
            &mut loader,
            &mut documents,
            Change::Set(path.clone(), "first edit".into()),
        )
        .unwrap();
        let original = documents[&path].clone();
        fs::remove_file(&path).unwrap();
        symlink(&target, &path).unwrap();
        apply_change(
            &mut loader,
            &mut documents,
            Change::Set(path.clone(), "later edit".into()),
        )
        .unwrap();
        assert_eq!(documents[&path].id(), original.id());
        assert_eq!(documents[&path].text(), "later edit");
        apply_change(&mut loader, &mut documents, Change::Close(path.clone())).unwrap();
        assert!(!documents.contains_key(&path));
        let root = loader
            .source_from_text(&directory.path().join("root.resin"), "")
            .unwrap();
        assert_eq!(
            loader.load_import(&root, "target.resin").unwrap().text(),
            "target disk"
        );
        assert_eq!(
            loader.load_import(&root, "open.resin").unwrap().text(),
            "target disk"
        );
    }
}
