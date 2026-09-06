//! A long-lived compiler session shared by command-line tools and editor adapters.
//!
//! Hosts supply buffer/disk changes; the session owns parsed trees, dependencies,
//! and checked snapshots. Queries without intervening changes reuse the same
//! snapshot. Existing snapshots remain valid when the session advances.

use crate::{
    analysis::{Analysis, Sources, normalize_path, syntax::Document},
    ast,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

pub struct Session {
    sources: Sources,
    stdlib: PathBuf,
    parsed: BTreeMap<PathBuf, Arc<Document>>,
    checked: BTreeMap<PathBuf, Arc<Analysis>>,
    dependents: BTreeMap<PathBuf, BTreeSet<PathBuf>>,
    revision: u64,
}

impl Default for Session {
    fn default() -> Self {
        Self::new(ast::stdlib_path())
    }
}

impl Session {
    pub fn new(stdlib: PathBuf) -> Self {
        Self {
            sources: Sources::default(),
            stdlib,
            parsed: BTreeMap::new(),
            checked: BTreeMap::new(),
            dependents: BTreeMap::new(),
            revision: 0,
        }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn set_overlay(&mut self, path: &Path, text: String) -> io::Result<()> {
        let path = normalize_path(path)?;
        if self.sources.overlays.get(&path) == Some(&text) {
            return Ok(());
        }
        self.sources.overlays.insert(path.clone(), text);
        self.invalidate(&path);
        Ok(())
    }

    pub fn remove_overlay(&mut self, path: &Path) -> io::Result<()> {
        let path = normalize_path(path)?;
        if self.sources.overlays.remove(&path).is_some() {
            self.invalidate(&path);
        }
        Ok(())
    }

    /// Notify the compiler after an external modification, creation, or deletion.
    /// Open buffers continue to own their contents until their overlay is removed.
    pub fn file_changed(&mut self, path: &Path) -> io::Result<()> {
        let path = normalize_path(path)?;
        if !self.sources.overlays.contains_key(&path) {
            self.invalidate(&path);
        }
        Ok(())
    }

    pub fn analyze(&mut self, entry: &Path) -> io::Result<Arc<Analysis>> {
        let entry = normalize_path(entry)?;
        if let Some(snapshot) = self.checked.get(&entry) {
            return Ok(snapshot.clone());
        }
        let snapshot = Arc::new(Analysis::build(
            &entry,
            &self.sources,
            &self.stdlib,
            &mut self.parsed,
        ));
        for dependency in &snapshot.dependencies {
            self.dependents
                .entry(dependency.clone())
                .or_default()
                .insert(entry.clone());
        }
        self.checked.insert(entry, snapshot.clone());
        Ok(snapshot)
    }

    pub fn set_stdlib(&mut self, stdlib: PathBuf) {
        if self.stdlib == stdlib {
            return;
        }
        self.stdlib = stdlib;
        self.checked.clear();
        self.dependents.clear();
        self.revision += 1;
    }

    /// Release entries a host no longer needs while retaining shared dependencies.
    pub fn retain_entries(&mut self, entries: &BTreeSet<PathBuf>) {
        let removed = self
            .checked
            .keys()
            .filter(|p| !entries.contains(*p))
            .cloned()
            .collect::<Vec<_>>();
        for entry in removed {
            self.forget(&entry);
        }
        self.parsed.retain(|path, _| {
            self.dependents.contains_key(path) || self.sources.overlays.contains_key(path)
        });
    }

    fn invalidate(&mut self, path: &Path) {
        self.revision += 1;
        let mut entries = self.dependents.get(path).cloned().unwrap_or_default();
        entries.insert(path.to_path_buf());
        for entry in entries {
            self.forget(&entry);
        }
    }

    fn forget(&mut self, entry: &Path) {
        self.checked.remove(entry);
        self.dependents.retain(|_, entries| {
            entries.remove(entry);
            !entries.is_empty()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_snapshots_are_shared_and_only_dependents_are_invalidated() {
        let root = Path::new("/tmp/resin-compiler-session-test");
        let mut compiler = Session::new(root.join("std"));
        let entry = root.join("main.resin");
        let library = root.join("lib.resin");
        let independent = root.join("other.resin");
        compiler
            .set_overlay(
                &entry,
                "import { \"lib.resin\" }; def main () -> int = { value() };".into(),
            )
            .unwrap();
        compiler
            .set_overlay(
                &library,
                "export { value }; def value () -> int = { 1 };".into(),
            )
            .unwrap();
        compiler
            .set_overlay(&independent, "def other () -> int = { 2 };".into())
            .unwrap();
        let old = compiler.analyze(&entry).unwrap();
        let other = compiler.analyze(&independent).unwrap();
        assert!(old.module().is_ok());
        assert!(Arc::ptr_eq(&old, &compiler.analyze(&entry).unwrap()));
        compiler
            .set_overlay(
                &library,
                "export { renamed }; def renamed () -> int = { 1 };".into(),
            )
            .unwrap();
        let new = compiler.analyze(&entry).unwrap();
        assert!(!Arc::ptr_eq(&old, &new));
        assert!(new.module().is_err());
        assert!(old.module().is_ok(), "previous snapshots must remain valid");
        assert!(Arc::ptr_eq(
            &other,
            &compiler.analyze(&independent).unwrap()
        ));
    }

    #[test]
    fn creating_a_missing_dependency_recovers_the_cached_error() {
        let root = Path::new("/tmp/resin-compiler-missing-test");
        let mut compiler = Session::default();
        compiler
            .set_overlay(&root.join("main.resin"), "import { \"new.resin\" };".into())
            .unwrap();
        assert!(
            compiler
                .analyze(&root.join("main.resin"))
                .unwrap()
                .module()
                .is_err()
        );
        compiler
            .set_overlay(&root.join("new.resin"), "export {};".into())
            .unwrap();
        assert!(
            compiler
                .analyze(&root.join("main.resin"))
                .unwrap()
                .module()
                .is_ok()
        );
    }
}
