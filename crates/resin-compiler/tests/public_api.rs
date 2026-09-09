//! Immutable compiler behavior with an importer-scoped, entirely in-memory loader.
use resin_compiler::{Compilation, Compiler};
use resin_source::Loader;
use resin_source::prelude::*;
use std::{collections::BTreeSet, sync::Arc};

fn sources(compilation: &Compilation) -> BTreeSet<Source> {
    compilation.sources().cloned().collect()
}

fn valid(compilation: &Compilation) {
    assert!(
        compilation.module().is_ok(),
        "{:?}",
        compilation.diagnostics()
    );
}

#[test]
fn unchanged_graph_reuses_the_completed_result_after_resolving_imports() {
    let entry = Source::new(
        "entry",
        "import { \"dependency\" }; def main() -> int = { value() };",
    );
    let dependency = Source::new(
        "dependency",
        "export { value }; def value() -> int = { 1 };",
    );
    let mut loader = Loader::new(resin_source::stdlib_path());
    loader
        .set_import(&entry, "dependency", dependency.clone())
        .unwrap();
    let mut compiler = Compiler::new();
    let first = compiler.compile(entry.clone(), &mut loader);
    valid(&first);
    let second = compiler.compile(entry.clone(), &mut loader);
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(second.entry(), &entry);
    assert_eq!(sources(&second), [entry, dependency].into());
}

#[test]
fn changing_a_transitive_source_invalidates_an_unchanged_entry() {
    let entry = Source::new(
        "entry",
        "import { \"middle\" }; def main() -> int = { value() };",
    );
    let middle = Source::new(
        "middle",
        "export { value }; import { \"leaf\" }; def value() -> int = { leaf() };",
    );
    let leaf = Source::new("leaf", "export { leaf }; def leaf() -> int = { 1 };");
    let mut loader = Loader::new(resin_source::stdlib_path());
    loader.set_import(&entry, "middle", middle.clone()).unwrap();
    loader.set_import(&middle, "leaf", leaf.clone()).unwrap();
    let mut compiler = Compiler::default();
    let before = compiler.compile(entry.clone(), &mut loader);
    valid(&before);
    let changed = leaf.with_text("export { leaf }; def leaf() -> bool = { 1 == 1 };");
    loader.set_import(&middle, "leaf", changed.clone()).unwrap();
    let after = compiler.compile(entry.clone(), &mut loader);
    assert!(!Arc::ptr_eq(&before, &after));
    assert!(after.module().is_err());
    assert_eq!(after.entry(), &entry);
    assert!(sources(&before).contains(&leaf));
    assert!(sources(&after).contains(&changed));
    assert!(!sources(&after).contains(&leaf));
    valid(&before);
}

#[test]
fn a_missing_transitive_import_recovers_without_notifications() {
    let entry = Source::new("entry", "import { \"middle\" };");
    let middle = Source::new("middle", "import { \"leaf\" };");
    let leaf = Source::new("leaf", "def leaf() = {};");
    let mut loader = Loader::new(resin_source::stdlib_path());
    loader.set_import(&entry, "middle", middle.clone()).unwrap();
    let mut compiler = Compiler::default();
    let before = compiler.compile(entry.clone(), &mut loader);
    assert!(before.module().is_err());
    assert!(before.diagnostics().iter().any(|diagnostic| {
        diagnostic.location.source == middle
            && diagnostic.location.span.end > diagnostic.location.span.start
    }));
    loader.set_import(&middle, "leaf", leaf.clone()).unwrap();
    let after = compiler.compile(entry, &mut loader);
    valid(&after);
    assert!(!Arc::ptr_eq(&before, &after));
    assert!(
        before.module().is_err(),
        "retained failure remains immutable"
    );
    assert!(sources(&after).contains(&leaf));
}

#[test]
fn changed_import_edges_invalidate_cache_even_with_the_same_source_set() {
    let entry = Source::new(
        "entry",
        "import { \"first\", \"second\" }; def main() -> int = { first() };",
    );
    let first = Source::new(
        "first",
        "export { first }; import { \"value\" }; def first() -> int = { value() };",
    );
    let second = Source::new(
        "second",
        "export { second }; import { \"value\" }; def second() -> bool = { value() };",
    );
    let integer = Source::new("integer", "export { value }; def value() -> int = { 1 };");
    let boolean = Source::new(
        "boolean",
        "export { value }; def value() -> bool = { 1 == 1 };",
    );
    let mut loader = Loader::new(resin_source::stdlib_path());
    loader.set_import(&entry, "first", first.clone()).unwrap();
    loader.set_import(&entry, "second", second.clone()).unwrap();
    loader.set_import(&first, "value", integer.clone()).unwrap();
    loader
        .set_import(&second, "value", boolean.clone())
        .unwrap();
    let mut compiler = Compiler::default();
    let before = compiler.compile(entry.clone(), &mut loader);
    valid(&before);
    loader.set_import(&first, "value", boolean.clone()).unwrap();
    loader
        .set_import(&second, "value", integer.clone())
        .unwrap();
    let after = compiler.compile(entry, &mut loader);
    assert_eq!(sources(&before), sources(&after));
    assert!(!Arc::ptr_eq(&before, &after));
    assert!(after.module().is_err());
    valid(&before);
}

#[test]
fn inconsistent_versions_of_one_logical_source_are_diagnosed() {
    let entry = Source::new("entry", "import { \"first\", \"second\" };");
    let first = Source::new("module", "def value() -> int = { 1 };");
    let second = first.with_text("def value() -> int = { 2 };");
    assert_eq!(first.id(), second.id());
    assert_ne!(first, second);
    let mut loader = Loader::new(resin_source::stdlib_path());
    loader.set_import(&entry, "first", first.clone()).unwrap();
    loader.set_import(&entry, "second", second.clone()).unwrap();
    let result = Compiler::default().compile(entry.clone(), &mut loader);
    assert!(result.module().is_err());
    assert!(
        result
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message.contains("version"))
    );
    assert!(result.recovered_file(&entry).is_some());
}

#[test]
fn import_cycles_are_diagnosed_with_source_ranges() {
    let entry = Source::new("entry", "import { \"next\" };");
    let next = Source::new("next", "import { \"entry\" };");
    let mut loader = Loader::new(resin_source::stdlib_path());
    loader.set_import(&entry, "next", next.clone()).unwrap();
    loader.set_import(&next, "entry", entry.clone()).unwrap();
    let result = Compiler::default().compile(entry.clone(), &mut loader);
    assert!(result.module().is_err());
    assert!(result.diagnostics().iter().any(|diagnostic| {
        diagnostic.message.contains("cyclic")
            && diagnostic.location.span.end > diagnostic.location.span.start
    }));
    assert!(result.recovered_file(&entry).is_some());
    assert!(result.recovered_file(&next).is_some());
}

#[test]
fn identical_diagnostic_names_do_not_merge_distinct_sources() {
    let entry = Source::new("entry", "import { \"first\", \"second\" };");
    let first = Source::new("generated", "def first() -> int = { 1 == 1 };");
    let second = Source::new("generated", "def second() -> bool = { 1 };");
    let mut loader = Loader::new(resin_source::stdlib_path());
    loader.set_import(&entry, "first", first.clone()).unwrap();
    loader.set_import(&entry, "second", second.clone()).unwrap();
    let result = Compiler::default().compile(entry.clone(), &mut loader);
    assert!(result.module().is_err());
    assert_eq!(
        sources(&result),
        [entry, first.clone(), second.clone()].into()
    );
    let origins: BTreeSet<_> = result
        .diagnostics()
        .iter()
        .map(|diagnostic| diagnostic.location.source.clone())
        .collect();
    assert_eq!(origins, [first, second].into());
}

#[test]
fn retained_compilations_keep_their_own_source_versions_and_editor_queries() {
    let before = Source::new(
        "editor",
        "def value() -> int = { 1 }; def main() -> int = { value() };",
    );
    let mut loader = Loader::new(resin_source::stdlib_path());
    let mut compiler = Compiler::default();
    let old = compiler.compile(before.clone(), &mut loader);
    let after =
        before.with_text("def value() -> bool = { 1 == 1 }; def main() -> bool = { value() };");
    let new = compiler.compile(after.clone(), &mut loader);
    valid(&old);
    valid(&new);
    assert_eq!(before.id(), after.id());
    assert!(!Arc::ptr_eq(&old, &new));
    for (result, source, ty) in [(&old, &before, "int"), (&new, &after, "bool")] {
        let offset = source.text().rfind("value").unwrap();
        assert_eq!(result.definition(source, offset).unwrap().source, *source);
        assert!(result.hover(source, offset).unwrap().text.contains(ty));
        assert!(
            result
                .completions(source, offset + 3)
                .iter()
                .any(|item| item.name == "value")
        );
        assert_eq!(sources(result), [source.clone()].into());
    }
    assert!(old.recovered_file(&after).is_none());
    assert!(old.hover(&after, 0).is_none());
    assert!(new.recovered_file(&before).is_none());
}

#[test]
fn later_errors_preserve_completed_earlier_passes_and_recovered_syntax() {
    let source = Source::new(
        "entry",
        "def first() -> int = { var n: int; n }; def second() -> bool = { var b: bool; b };",
    );
    let mut loader = Loader::new(resin_source::stdlib_path());
    let mut compiler = Compiler::default();
    let lowered = compiler.compile(source.clone(), &mut loader);
    assert!(lowered.program().is_ok());
    assert!(lowered.hir().is_ok());
    assert!(lowered.module().is_err());
    assert_eq!(lowered.diagnostics().len(), 2);
    assert!(
        lowered
            .diagnostics()
            .iter()
            .all(|diagnostic| diagnostic.location.source == source)
    );
    let typed_source = source.with_text("def main() -> int = { 1 == 2 };");
    let typed = compiler.compile(typed_source, &mut loader);
    assert!(typed.program().is_ok());
    assert!(typed.hir().is_err());
    let parsed_source = source.with_text("def main( = { 1 == 2 };");
    let parsed = compiler.compile(parsed_source.clone(), &mut loader);
    assert!(parsed.program().is_err());
    assert!(parsed.recovered_file(&parsed_source).is_some());
    assert_eq!(lowered.diagnostics().len(), 2);
}

#[test]
fn conflicting_versions_do_not_displace_the_first_accepted_version() {
    let entry = Source::new("entry", "import { \"first\", \"conflict\", \"original\" };");
    let original = Source::new("module", "def value() -> int = { 1 };");
    let conflict = original.with_text("def value() -> int = { 2 };");
    let mut loader = Loader::new(resin_source::stdlib_path());
    loader
        .set_import(&entry, "first", original.clone())
        .unwrap();
    loader
        .set_import(&entry, "conflict", conflict.clone())
        .unwrap();
    loader
        .set_import(&entry, "original", original.clone())
        .unwrap();
    let result = Compiler::default().compile(entry.clone(), &mut loader);
    assert!(result.module().is_err());
    assert_eq!(result.diagnostics().len(), 1, "{:?}", result.diagnostics());
    let diagnostic = &result.diagnostics()[0];
    assert!(diagnostic.message.contains("different versions"));
    assert_eq!(diagnostic.location.source, entry);
    let span = diagnostic.location.span;
    assert_eq!(&entry.text()[span.start..span.end], "\"conflict\"");
    let offset = entry.text().find("original").unwrap();
    assert_eq!(result.definition(&entry, offset).unwrap().source, original);
}
