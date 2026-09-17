#[allow(dead_code)]
mod support;

use resin_hir::Type;
use resin_source::Loader;
use std::fs;
use tempfile::TempDir;

#[test]
#[cfg(unix)]
fn unchanged_roots_follow_retargeted_import_symlinks_without_notifications() {
    use std::os::unix::fs::symlink;
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    fs::write(
        directory.path().join("first.resin"),
        "export { answer }; fn answer() -> i32  { 41 }",
    )
    .unwrap();
    fs::write(
        directory.path().join("second.resin"),
        "export { answer }; fn answer() -> i64  { 42 }",
    )
    .unwrap();
    let alias = directory.path().join("alias.resin");
    symlink("first.resin", &alias).unwrap();
    let mut loader = Loader::new(directory.path().into());
    let source = loader
        .source_from_text(
            &directory.path().join("root.resin"),
            "import { \"alias.resin\" }; fn value() -> _  { answer() }",
        )
        .unwrap();
    let first = support::frontend::analyze(source.clone(), &mut loader, None);
    assert!(first.diagnostics().is_empty(), "{:?}", first.diagnostics());
    fs::remove_file(&alias).unwrap();
    symlink("second.resin", &alias).unwrap();
    let second = support::frontend::analyze(source, &mut loader, Some(&first));
    assert!(
        second.diagnostics().is_empty(),
        "{:?}",
        second.diagnostics()
    );
    assert!(!first.same(&second));
    assert_eq!(first.source(), second.source());
    for (compilation, expected) in [(first, Type::Int32), (second, Type::Int64)] {
        let module = compilation.hir().unwrap();
        let function = module
            .functions
            .iter()
            .find(|function| function.name.as_ref() == "value")
            .unwrap();
        assert_eq!(function.signature.result.ty, expected);
    }
}

#[test]
fn unchanged_roots_retry_missing_transitive_imports_without_notifications() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    fs::write(
        directory.path().join("middle.resin"),
        "export { answer }; import { \"leaf.resin\" }; fn answer() -> i32  { leaf() }",
    )
    .unwrap();
    let mut loader = Loader::new(directory.path().into());
    let source = loader
        .source_from_text(
            &directory.path().join("root.resin"),
            "import { \"middle.resin\" }; fn value() -> i32  { answer() }",
        )
        .unwrap();
    let missing = support::frontend::analyze(source.clone(), &mut loader, None);
    assert!(missing.hir().is_err());
    assert!(
        missing
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message.contains("leaf.resin"))
    );
    let leaf = directory.path().join("leaf.resin");
    fs::write(&leaf, "export { leaf }; fn leaf() -> i32  { 42 }").unwrap();
    let repaired = support::frontend::analyze(source.clone(), &mut loader, Some(&missing));
    assert!(
        repaired.diagnostics().is_empty(),
        "{:?}",
        repaired.diagnostics()
    );
    assert!(repaired.hir().is_ok());
    assert!(!missing.same(&repaired));
    assert!(missing.hir().is_err());
    assert_eq!(repaired.sources().count(), 3);
    fs::remove_file(leaf).unwrap();
    assert!(
        support::frontend::analyze(source, &mut loader, Some(&repaired))
            .hir()
            .is_err()
    );
    assert!(repaired.hir().is_ok());
}

#[test]
fn custom_library_root_edits_recompile_an_unchanged_entry() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let library_root = directory.path().join("custom-library");
    fs::create_dir(&library_root).unwrap();
    let library = library_root.join("math.resin");
    fs::write(&library, "export { answer }; fn answer() -> i32  { 41 }").unwrap();
    let mut loader = Loader::new(library_root);
    let source = loader
        .source_from_text(
            &directory.path().join("main.resin"),
            "import { \"$/math.resin\" }; fn value() -> _  { answer() }",
        )
        .unwrap();
    let before = support::frontend::analyze(source.clone(), &mut loader, None);
    assert!(
        before.diagnostics().is_empty(),
        "{:?}",
        before.diagnostics()
    );
    fs::write(&library, "export { answer }; fn answer() -> i64  { 42 }").unwrap();
    let after = support::frontend::analyze(source, &mut loader, Some(&before));
    assert!(after.diagnostics().is_empty(), "{:?}", after.diagnostics());
    assert_eq!(before.source(), after.source());
    assert!(!before.same(&after));
    for (compilation, expected) in [(before, Type::Int32), (after, Type::Int64)] {
        let module = compilation.hir().unwrap();
        let function = module
            .functions
            .iter()
            .find(|function| function.name.as_ref() == "value")
            .unwrap();
        assert_eq!(function.signature.result.ty, expected);
    }
}
