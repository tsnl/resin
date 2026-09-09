use crossbeam_channel::{Receiver, Sender};
use resin::{analysis::Analysis, compiler::Session};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
};

pub enum Change {
    Set(PathBuf, String),
    Close(PathBuf),
    Disk(Vec<PathBuf>),
}

pub struct Update {
    pub revision: u64,
    pub roots: BTreeSet<PathBuf>,
    pub change: Change,
}

pub struct Snapshot {
    pub revision: u64,
    pub entries: BTreeMap<PathBuf, Arc<Analysis>>,
}

pub fn spawn(
    stdlib: PathBuf,
    updates: Receiver<Update>,
    results: Sender<Snapshot>,
    current: Arc<AtomicU64>,
    stopping: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut compiler = Session::new(stdlib);
        while let Ok(first) = updates.recv() {
            if stopping.load(Ordering::Relaxed) {
                break;
            }
            let mut revision = first.revision;
            let mut roots = BTreeSet::new();
            // Apply every edit, but check only the latest pending state.
            for update in std::iter::once(first).chain(updates.try_iter()) {
                revision = update.revision;
                roots = update.roots;
                let changed = match update.change {
                    Change::Set(path, text) => compiler.set_overlay(&path, text),
                    Change::Close(path) => compiler.remove_overlay(&path),
                    Change::Disk(paths) => paths
                        .iter()
                        .try_for_each(|path| compiler.file_changed(path)),
                };
                if let Err(error) = changed {
                    eprintln!("resin-lsp: {error}");
                }
            }
            let mut entries = BTreeMap::new();
            for path in &roots {
                if stopping.load(Ordering::Relaxed) || current.load(Ordering::Acquire) != revision {
                    break;
                }
                match compiler.analyze(path) {
                    Ok(snapshot) => {
                        entries.insert(path.clone(), snapshot);
                    }
                    Err(error) => eprintln!("resin-lsp: {}: {error}", path.display()),
                }
            }
            compiler.retain_entries(&roots);
            if stopping.load(Ordering::Relaxed) {
                break;
            }
            if current.load(Ordering::Acquire) == revision
                && results.send(Snapshot { revision, entries }).is_err()
            {
                break;
            }
        }
    })
}
