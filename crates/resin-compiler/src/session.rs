use crate::syntax::Document;
use crate::{Compilation, Sources, normalize_path, stdlib_path};
use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

pub(super) struct State {
    sources: Sources,
    stdlib: PathBuf,
    parsed: BTreeMap<PathBuf, Arc<Document>>,
    checked: BTreeMap<PathBuf, Arc<Compilation>>,
    dependents: BTreeMap<PathBuf, BTreeSet<PathBuf>>,
    revision: u64,
}

impl Default for State {
    fn default() -> Self {
        Self::new(stdlib_path())
    }
}

impl State {
    pub(super) fn new(stdlib: PathBuf) -> Self {
        Self {
            sources: Sources::default(),
            stdlib,
            parsed: BTreeMap::new(),
            checked: BTreeMap::new(),
            dependents: BTreeMap::new(),
            revision: 0,
        }
    }

    pub(super) fn revision(&self) -> u64 {
        self.revision
    }

    pub(super) fn set_overlay(&mut self, path: &Path, text: String) -> io::Result<()> {
        let spelling = std::path::absolute(path)?;
        let canonical = normalize_path(path)?;
        if self.sources.overlays.get(&canonical) == Some(&text)
            && !self.retargeted(&spelling, &canonical)
        {
            return Ok(());
        }
        self.sources.overlays.insert(canonical.clone(), text);
        self.invalidate_paths(&[spelling, canonical]);
        Ok(())
    }

    pub(super) fn remove_overlay(&mut self, path: &Path) -> io::Result<()> {
        let spelling = std::path::absolute(path)?;
        let canonical = normalize_path(path)?;
        if self.sources.overlays.remove(&canonical).is_some()
            || self.retargeted(&spelling, &canonical)
        {
            self.invalidate_paths(&[spelling, canonical]);
        }
        Ok(())
    }

    /// Notify the compiler after an external modification, creation, or deletion.
    /// Open buffers continue to own their contents until their overlay is removed.
    pub(super) fn file_changed(&mut self, path: &Path) -> io::Result<()> {
        let spelling = std::path::absolute(path)?;
        let canonical = normalize_path(path)?;
        if !self.sources.overlays.contains_key(&canonical) || self.retargeted(&spelling, &canonical)
        {
            // A retargeted symlink has a new canonical destination. Its old import
            // spelling still identifies the entries that need to be rebuilt.
            self.invalidate_paths(&[spelling, canonical]);
        }
        Ok(())
    }

    pub(super) fn analyze(&mut self, entry: &Path) -> io::Result<Arc<Compilation>> {
        let entry = normalize_path(entry)?;
        if let Some(compilation) = self.checked.get(&entry) {
            return Ok(compilation.clone());
        }
        let compilation = Arc::new(Compilation {
            data: crate::compilation::Data::build(
                &entry,
                &self.sources,
                &self.stdlib,
                &mut self.parsed,
            ),
        });
        for dependency in compilation.dependencies() {
            self.dependents
                .entry(dependency.clone())
                .or_default()
                .insert(entry.clone());
        }
        self.checked.insert(entry, compilation.clone());
        Ok(compilation)
    }

    pub(super) fn set_stdlib(&mut self, stdlib: PathBuf) {
        if self.stdlib == stdlib {
            return;
        }
        self.stdlib = stdlib;
        self.checked.clear();
        self.dependents.clear();
        self.revision += 1;
    }

    /// Release entries a host no longer needs while retaining shared dependencies.
    pub(super) fn retain_entries(&mut self, entries: &BTreeSet<PathBuf>) {
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

    /// Compare against the resolution retained by each cached entry, not today's filesystem.
    fn retargeted(&self, spelling: &Path, canonical: &Path) -> bool {
        self.dependents
            .get(spelling)
            .into_iter()
            .flatten()
            .any(|entry| {
                self.checked[entry]
                    .data
                    .resolutions
                    .get(spelling)
                    .is_some_and(|old| old != canonical)
            })
    }

    fn invalidate_paths(&mut self, paths: &[PathBuf]) {
        self.revision += 1;
        let mut entries = BTreeSet::new();
        for path in paths {
            entries.insert(path.clone());
            entries.extend(self.dependents.get(path).into_iter().flatten().cloned());
        }
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
    use crate::Session;

    #[test]
    fn unchanged_compilations_are_shared_and_only_dependents_are_invalidated() {
        let root = std::env::temp_dir().join("resin-compiler-session-test");
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
        assert!(old.module().is_ok(), "{:?}", old.diagnostics());
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
        assert!(
            old.module().is_ok(),
            "previous compilations must remain valid"
        );
        assert!(Arc::ptr_eq(
            &other,
            &compiler.analyze(&independent).unwrap()
        ));
    }

    #[test]
    fn creating_a_missing_dependency_recovers_the_cached_error() {
        let root = std::env::temp_dir().join("resin-compiler-missing-test");
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
        let compilation = compiler.analyze(&root.join("main.resin")).unwrap();
        assert!(
            compilation.module().is_ok(),
            "{:?}",
            compilation.diagnostics()
        );
    }
}

#[cfg(test)]
mod verification_tests {
    use crate::Session;
    use crate::{build::emit_c, shaders::emit_glsl};
    use resin_common::prelude::*;

    #[test]
    fn compilations_reuse_verification_for_multiple_backends_and_invalidate_on_edit() {
        let path = std::env::temp_dir().join("resin-verification-cache.resin");
        let source = "export { main, a, b }; def main() -> int = { 0 }; @compute_shader def a(invocation: ulong, p: Ptr<uint>) = { var i = uint(invocation); p.* := i; }; @compute_shader def b(invocation: ulong, p: Ptr<uint>) = { var i = uint(invocation); p.* := i + 1_ui; };";
        let mut session = Session::default();
        session.set_overlay(&path, source.into()).unwrap();
        let compilation = session.analyze(&path).unwrap();
        let checked = compilation.data.verified().unwrap();
        let mut shaders = Vec::new();
        for name in ["a", "b"] {
            let id = checked.module().entries[name];
            shaders.push(emit_glsl(checked, id, Stage::Compute).unwrap());
        }
        let host = emit_c(checked, "main", &[]).unwrap();
        assert!(std::ptr::eq(
            checked.analysis(),
            compilation.data.verified().unwrap().analysis()
        ));
        assert_eq!(
            host,
            resin_codegen::emit_c(checked.module(), "main").unwrap()
        );
        for (name, expected) in ["a", "b"].into_iter().zip(shaders) {
            assert_eq!(
                expected,
                resin_codegen::emit_glsl(checked.module(), name, Stage::Compute).unwrap()
            );
        }
        session
            .set_overlay(&path, source.replace("{ 0 }", "{ 1 }"))
            .unwrap();
        let changed = session.analyze(&path).unwrap();
        let new = changed.data.verified().unwrap();
        assert!(!std::ptr::eq(checked.module(), new.module()));
        assert_ne!(host, emit_c(new, "main", &[]).unwrap());
        assert_eq!(
            host,
            emit_c(compilation.data.verified().unwrap(), "main", &[]).unwrap()
        );
    }
}
