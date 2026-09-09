use resin_source::Loader;
use resin_types::prelude::*;
use std::fs;
use tempfile::TempDir;

#[test]
#[cfg(unix)]
fn unchanged_roots_follow_retargeted_import_symlinks_without_notifications() {
    use std::{os::unix::fs::symlink, sync::Arc};
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    fs::write(
        directory.path().join("first.resin"),
        "export { answer }; def answer() -> int = { 41 };",
    )
    .unwrap();
    fs::write(
        directory.path().join("second.resin"),
        "export { answer }; def answer() -> long = { 42 };",
    )
    .unwrap();
    let alias = directory.path().join("alias.resin");
    symlink("first.resin", &alias).unwrap();
    let mut loader = Loader::new(directory.path().into());
    let source = loader
        .source_from_text(
            &directory.path().join("root.resin"),
            "import { \"alias.resin\" }; def value() -> _ = { answer() };",
        )
        .unwrap();
    let mut compiler = resin_compiler::Compiler::new();
    let first = compiler.compile(source.clone(), &mut loader);
    assert!(first.diagnostics().is_empty(), "{:?}", first.diagnostics());
    fs::remove_file(&alias).unwrap();
    symlink("second.resin", &alias).unwrap();
    let second = compiler.compile(source, &mut loader);
    assert!(
        second.diagnostics().is_empty(),
        "{:?}",
        second.diagnostics()
    );
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(first.entry(), second.entry());
    for (compilation, expected) in [(first, Ty::Int32), (second, Ty::Int64)] {
        let module = compilation.module().unwrap();
        let function = module
            .functions
            .iter()
            .find(|function| function.name.as_deref() == Some("value"))
            .unwrap();
        assert_eq!(function.result, expected);
    }
}

#[test]
fn unchanged_roots_retry_missing_transitive_imports_without_notifications() {
    use std::sync::Arc;
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    fs::write(
        directory.path().join("middle.resin"),
        "export { answer }; import { \"leaf.resin\" }; def answer() -> int = { leaf() };",
    )
    .unwrap();
    let mut loader = Loader::new(directory.path().into());
    let source = loader
        .source_from_text(
            &directory.path().join("root.resin"),
            "import { \"middle.resin\" }; def value() -> int = { answer() };",
        )
        .unwrap();
    let mut compiler = resin_compiler::Compiler::new();
    let missing = compiler.compile(source.clone(), &mut loader);
    assert!(missing.module().is_err());
    assert!(
        missing
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message.contains("leaf.resin"))
    );
    let leaf = directory.path().join("leaf.resin");
    fs::write(&leaf, "export { leaf }; def leaf() -> int = { 42 };").unwrap();
    let repaired = compiler.compile(source.clone(), &mut loader);
    assert!(
        repaired.diagnostics().is_empty(),
        "{:?}",
        repaired.diagnostics()
    );
    assert!(repaired.module().is_ok());
    assert!(!Arc::ptr_eq(&missing, &repaired));
    assert!(missing.module().is_err());
    assert_eq!(repaired.sources().count(), 3);
    fs::remove_file(leaf).unwrap();
    assert!(compiler.compile(source, &mut loader).module().is_err());
    assert!(repaired.module().is_ok());
}

#[test]
fn custom_library_root_edits_recompile_an_unchanged_entry() {
    use std::sync::Arc;
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let library_root = directory.path().join("custom-library");
    fs::create_dir(&library_root).unwrap();
    let library = library_root.join("math.resin");
    fs::write(&library, "export { answer }; def answer() -> int = { 41 };").unwrap();
    let mut loader = Loader::new(library_root);
    let source = loader
        .source_from_text(
            &directory.path().join("main.resin"),
            "import { \"$/math.resin\" }; def value() -> _ = { answer() };",
        )
        .unwrap();
    let mut compiler = resin_compiler::Compiler::new();
    let before = compiler.compile(source.clone(), &mut loader);
    assert!(
        before.diagnostics().is_empty(),
        "{:?}",
        before.diagnostics()
    );
    fs::write(
        &library,
        "export { answer }; def answer() -> long = { 42 };",
    )
    .unwrap();
    let after = compiler.compile(source, &mut loader);
    assert!(after.diagnostics().is_empty(), "{:?}", after.diagnostics());
    assert_eq!(before.entry(), after.entry());
    assert!(!Arc::ptr_eq(&before, &after));
    for (compilation, expected) in [(before, Ty::Int32), (after, Ty::Int64)] {
        let module = compilation.module().unwrap();
        let function = module
            .functions
            .iter()
            .find(|function| function.name.as_deref() == Some("value"))
            .unwrap();
        assert_eq!(function.result, expected);
    }
}
