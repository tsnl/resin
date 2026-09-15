use crossbeam_channel::{Receiver, Sender};
use resin_cache::Cache;
use resin_executor::{Cancellation, Execution};
use resin_hir::Hir;
use resin_source::SourceGraph;
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
    pub entries: BTreeMap<PathBuf, Hir>,
    pub paths: BTreeMap<PathBuf, BTreeMap<SourceId, PathBuf>>,
}

pub fn spawn(
    library_root: PathBuf,
    updates: Receiver<Update>,
    results: Sender<AnalysisUpdate>,
    current: Arc<AtomicU64>,
    stopping: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(error) => {
                eprintln!("resin-lsp: cannot start compiler runtime: {error}");
                return;
            }
        };
        let execution = Execution::default();
        let mut loader = resin_source::Loader::new(library_root);
        let mut documents = BTreeMap::new();
        let mut caches = Caches::new();
        while let Ok(first) = updates.recv() {
            if stopping.load(Ordering::Acquire) {
                break;
            }
            let batch: Vec<_> = std::iter::once(first).chain(updates.try_iter()).collect();
            // Supplied editor text is accepted in order, including edits superseded
            // before analysis starts. Hashing and path normalization use a worker.
            let applied = runtime.block_on(execution.run(&Cancellation::new(), move |_| {
                let (revision, roots) =
                    apply_updates(&mut loader, &mut documents, batch.into_iter());
                (loader, documents, revision, roots)
            }));
            let Ok((next_loader, next_documents, revision, roots)) = applied else {
                break;
            };
            loader = next_loader;
            documents = next_documents;
            let cancellation = Cancellation::new();
            let result = runtime.block_on(async {
                let compile = compile_roots(
                    &mut caches,
                    &mut loader,
                    &documents,
                    roots,
                    revision,
                    &execution,
                    &cancellation,
                );
                tokio::pin!(compile);
                tokio::select! {
                    result = &mut compile => result,
                    _ = wait_superseded(revision, &current, &stopping) => {
                        cancellation.cancel();
                        compile.await
                    }
                }
            });
            if superseded(revision, &current, &stopping) {
                continue;
            }
            match result {
                Ok(result) => {
                    if results.send(result).is_err() {
                        break;
                    }
                }
                Err(error) => eprintln!("resin-lsp: {error}"),
            }
        }
        runtime.block_on(execution.wait_idle());
    })
}

// Caches belong to this application worker and share completed phase values across roots.
// Requested files survive capacity overflow; ordinary idle entries can be evicted.
struct Caches {
    syntax: Cache<Source, resin_cst::Document>,
    ast: Cache<Source, resin_ast::ModuleDocument>,
    hir: Cache<SourceGraph, Hir>,
}

impl Caches {
    fn new() -> Self {
        Self {
            syntax: Cache::new(4096),
            ast: Cache::new(4096),
            hir: Cache::new(64),
        }
    }
}

async fn wait_superseded(revision: u64, current: &AtomicU64, stopping: &AtomicBool) {
    while !superseded(revision, current, stopping) {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
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

async fn compile_roots(
    caches: &mut Caches,
    loader: &mut resin_source::Loader,
    documents: &BTreeMap<PathBuf, Source>,
    roots: BTreeSet<PathBuf>,
    revision: u64,
    execution: &Execution,
    cancellation: &Cancellation,
) -> crate::Result<AnalysisUpdate> {
    let mut result = AnalysisUpdate {
        revision,
        entries: BTreeMap::new(),
        paths: BTreeMap::new(),
    };
    let mut syntax = caches.syntax.clone();
    let mut ast = caches.ast.clone();
    let mut selected_syntax = BTreeMap::new();
    let mut programs = BTreeMap::new();
    let mut paths = BTreeMap::new();
    let mut failed = Vec::new();
    for path in roots {
        cancellation.check()?;
        let source = match documents.get(&path).cloned() {
            Some(source) => source,
            None => match loader.load_file_async(&path, execution, cancellation).await {
                Ok(source) => source,
                Err(resin_source::LoadError::Io { error }) => {
                    eprintln!("resin-lsp: {}: {error}", path.display());
                    continue;
                }
                Err(error) => return Err(error.into()),
            },
        };
        let captured =
            crate::inputs::capture(source, loader, &syntax, execution, cancellation).await?;
        result.paths.insert(
            path.clone(),
            captured
                .origins
                .iter()
                .filter_map(|(logical, original)| {
                    loader
                        .path(original)
                        .map(|path| (logical.clone(), path.to_path_buf()))
                })
                .collect(),
        );
        for source in captured.graph.sources() {
            selected_syntax.insert(
                source.clone(),
                captured
                    .syntax
                    .get(source)
                    .expect("captured syntax")
                    .clone(),
            );
        }
        syntax = syntax
            .update(
                selected_syntax.keys().cloned(),
                |source| {
                    let document = selected_syntax[&source].clone();
                    async move { Ok::<_, resin_executor::Error>(document) }
                },
                execution,
                cancellation,
            )
            .await?;
        ast = ast
            .update(
                selected_syntax.keys().cloned(),
                |source| {
                    let syntax = syntax.get(&source).expect("selected syntax").clone();
                    async move {
                        let parsed =
                            resin_ast::build_ast(syntax.clone(), execution, cancellation).await?;
                        Ok::<_, resin_executor::Error>(Arc::new(resin_ast::ModuleDocument {
                            source,
                            syntax,
                            file: Arc::new(parsed.file),
                            errors: parsed.errors,
                        }))
                    }
                },
                execution,
                cancellation,
            )
            .await?;
        let files = captured
            .graph
            .sources()
            .map(|source| {
                (
                    source.clone(),
                    ast.get(source).expect("selected AST").clone(),
                )
            })
            .collect();
        let graph = captured.graph;
        let mut program =
            resin_ast::build_program(graph.clone(), files, execution, cancellation).await?;
        if captured.diagnostics.is_empty() {
            paths.insert(path, graph.clone());
            programs.insert(graph, Arc::new(program));
        } else {
            // Acquisition errors are current request state, outside semantic graph keys.
            crate::inputs::acquisition_diagnostics(&mut program, captured.diagnostics);
            failed.push((path, Arc::new(program)));
        }
    }
    let hir = caches
        .hir
        .update(
            programs.keys().cloned(),
            |graph| {
                let program = programs[&graph].clone();
                async move {
                    Hir::build(program, execution, cancellation)
                        .await
                        .map(Arc::new)
                }
            },
            execution,
            cancellation,
        )
        .await?;
    for (path, graph) in paths {
        retain_entry(
            &mut result,
            hir.get(&graph).expect("selected HIR").as_ref().clone(),
            path,
        );
    }
    for (path, program) in failed {
        retain_entry(
            &mut result,
            Hir::build(program, execution, cancellation).await?,
            path,
        );
    }
    cancellation.check()?;
    *caches = Caches { syntax, ast, hir };
    Ok(result)
}

fn retain_entry(result: &mut AnalysisUpdate, compilation: Hir, path: PathBuf) {
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

#[cfg(test)]
mod async_tests {
    use super::*;
    use std::{fs, num::NonZeroUsize, time::Duration};

    #[tokio::test]
    async fn shared_roots_reuse_phase_values_and_retain_old_results_after_an_import_edit() {
        let directory = TempDir::new().unwrap();
        let library = directory.path().join("library.resin");
        let left = directory.path().join("left.resin");
        let right = directory.path().join("right.resin");
        fs::write(&library, "export { value }; def value() -> int = { 7 };").unwrap();
        for path in [&left, &right] {
            fs::write(
                path,
                "import { \"library.resin\" }; def main() -> int = { value() };",
            )
            .unwrap();
        }
        let roots = BTreeSet::from([left.clone(), right.clone()]);
        let mut loader = resin_source::Loader::new(directory.path().into());
        let mut caches = Caches {
            syntax: Cache::new(1),
            ast: Cache::new(1),
            hir: Cache::new(1),
        };
        let execution = Execution::new(NonZeroUsize::new(2).unwrap());
        let cancellation = Cancellation::new();
        let documents = BTreeMap::new();
        let first = compile_roots(
            &mut caches,
            &mut loader,
            &documents,
            roots.clone(),
            1,
            &execution,
            &cancellation,
        )
        .await
        .unwrap();
        assert!(
            first
                .entries
                .values()
                .all(|hir| hir.diagnostics().is_empty())
        );
        assert_eq!(
            caches.syntax.len(),
            3,
            "all requested files survive capacity overflow"
        );
        assert_eq!(caches.ast.len(), 3);
        assert_eq!(caches.hir.len(), 2);
        let library_source = first.entries[&left]
            .sources()
            .find(|source| first.paths[&left].get(&source.id()) == Some(&library))
            .unwrap()
            .clone();
        assert!(Arc::ptr_eq(
            &first.entries[&left].inputs().documents[&library_source],
            &first.entries[&right].inputs().documents[&library_source]
        ));
        let reused = compile_roots(
            &mut caches,
            &mut loader,
            &documents,
            roots.clone(),
            2,
            &execution,
            &cancellation,
        )
        .await
        .unwrap();
        assert!(Arc::ptr_eq(
            first.entries[&left].inputs(),
            reused.entries[&left].inputs()
        ));
        fs::write(
            &library,
            "export { value }; def value() -> int = { missing };",
        )
        .unwrap();
        let changed = compile_roots(
            &mut caches,
            &mut loader,
            &documents,
            roots,
            3,
            &execution,
            &cancellation,
        )
        .await
        .unwrap();
        assert!(
            changed
                .entries
                .values()
                .all(|hir| !hir.diagnostics().is_empty())
        );
        assert!(
            first
                .entries
                .values()
                .all(|hir| hir.diagnostics().is_empty())
        );
        assert_eq!(
            library_source.text(),
            "export { value }; def value() -> int = { 7 };"
        );
    }

    #[tokio::test]
    async fn cancelled_analysis_does_not_replace_completed_cache_snapshots() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("main.resin");
        fs::write(&path, "def main() -> int = { 1 };").unwrap();
        let mut loader = resin_source::Loader::new(directory.path().into());
        let mut caches = Caches::new();
        let execution = Execution::new(NonZeroUsize::new(1).unwrap());
        let roots = BTreeSet::from([path]);
        compile_roots(
            &mut caches,
            &mut loader,
            &BTreeMap::new(),
            roots.clone(),
            1,
            &execution,
            &Cancellation::new(),
        )
        .await
        .unwrap();
        let (key, before) = caches
            .hir
            .iter()
            .next()
            .map(|(key, value)| (key.clone(), value.clone()))
            .unwrap();
        let held = execution.acquire(&Cancellation::new()).await.unwrap();
        let cancellation = Cancellation::new();
        let cancel = async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            cancellation.cancel();
        };
        let documents = BTreeMap::new();
        let compile = compile_roots(
            &mut caches,
            &mut loader,
            &documents,
            roots,
            2,
            &execution,
            &cancellation,
        );
        let (result, ()) = tokio::join!(compile, cancel);
        assert!(result.is_err());
        assert!(Arc::ptr_eq(caches.hir.get(&key).unwrap(), &before));
        drop(held);
        execution.wait_idle().await;
    }

    #[test]
    fn worker_coalesces_edits_and_shuts_down_without_losing_accepted_text() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("main.resin");
        let (updates, receiver) = crossbeam_channel::unbounded();
        let (results, snapshots) = crossbeam_channel::unbounded();
        let current = Arc::new(AtomicU64::new(2));
        let stopping = Arc::new(AtomicBool::new(false));
        for (revision, value) in [(1, 1), (2, 2)] {
            updates
                .send(Update {
                    revision,
                    roots: BTreeSet::from([path.clone()]),
                    change: Change::Set(
                        path.clone(),
                        format!("def main() -> int = {{ {value} }};"),
                    ),
                })
                .unwrap();
        }
        let worker = spawn(
            directory.path().into(),
            receiver,
            results,
            current,
            stopping.clone(),
        );
        let snapshot = snapshots.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(snapshot.revision, 2);
        assert_eq!(
            snapshot.entries[&path].source().text(),
            "def main() -> int = { 2 };"
        );
        stopping.store(true, Ordering::Release);
        drop(updates);
        worker.join().unwrap();
    }
}

#[cfg(test)]
mod checkout_tests {
    use super::*;

    #[tokio::test]
    async fn shared_graphs_keep_each_roots_definition_paths() {
        let directory = TempDir::new().unwrap();
        let mut roots = BTreeSet::new();
        for name in ["left", "right"] {
            let root = directory.path().join(name);
            tokio::fs::create_dir(&root).await.unwrap();
            tokio::fs::write(
                root.join("main.resin"),
                "import { \"library.resin\" }; def main() -> int = { value() };",
            )
            .await
            .unwrap();
            tokio::fs::write(
                root.join("library.resin"),
                "export { value }; def value() -> int = { 42 };",
            )
            .await
            .unwrap();
            roots.insert(root.join("main.resin"));
        }
        let mut loader = resin_source::Loader::new(directory.path().join("builtin"));
        let mut caches = Caches::new();
        let output = compile_roots(
            &mut caches,
            &mut loader,
            &BTreeMap::new(),
            roots.clone(),
            1,
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap();
        let paths: Vec<_> = roots.into_iter().collect();
        let first = &output.entries[&paths[0]];
        let second = &output.entries[&paths[1]];
        assert!(first.same(second));
        assert_eq!(caches.hir.len(), 1);
        assert_eq!(caches.ast.len(), 2);
        let offset = first.source().text().find("value()").unwrap();
        let location = first.definition(first.source(), offset).unwrap();
        for root in paths {
            assert_eq!(
                output.paths[&root][&location.source.id()],
                root.parent().unwrap().join("library.resin")
            );
        }
    }
}
