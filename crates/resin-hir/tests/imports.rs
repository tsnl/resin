//! Immutable frontend behavior with an importer-scoped, entirely in-memory loader.
mod common;

use common::application::{acquire, request};
use resin_cache::Cache;
use resin_hir::Hir;
use resin_source::GraphError;
use resin_source::Loader;
use resin_source::prelude::*;
use std::collections::BTreeSet;
use std::sync::Arc;

fn sources(compilation: &Hir) -> BTreeSet<Source> {
    compilation.sources().cloned().collect()
}

fn valid(compilation: &Hir) {
    assert!(compilation.hir().is_ok(), "{:?}", compilation.diagnostics());
}

#[tokio::test]
async fn unchanged_graph_reuses_the_completed_result_after_resolving_imports() {
    let entry = Source::new(
        "entry",
        "import { \"dependency\" }; fn main() -> i32  { value() }",
    );
    let dependency = Source::new("dependency", "export { value }; fn value() -> i32  { 1 }");
    let mut loader = Loader::new(resin_source::library_root());
    let cache = Cache::new(16);
    loader
        .set_import(&entry, "dependency", dependency.clone())
        .unwrap();
    let (cache, first) = request(&cache, entry.clone(), &mut loader).await.unwrap();
    valid(&first);
    // A new caller reconstructs equal source values without sharing their allocations.
    let reconstructed = Source::new(entry.name(), entry.text());
    loader
        .set_import(
            &reconstructed,
            "dependency",
            Source::new(dependency.name(), dependency.text()),
        )
        .unwrap();
    let (_, second) = request(&cache, reconstructed, &mut loader).await.unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(second.source(), &entry);
    assert_eq!(sources(&second), [entry, dependency].into());
}

#[tokio::test]
async fn changing_a_transitive_source_invalidates_an_unchanged_entry() {
    let entry = Source::new(
        "entry",
        "import { \"middle\" }; fn main() -> i32  { value() }",
    );
    let middle = Source::new(
        "middle",
        "export { value }; import { \"leaf\" }; fn value() -> i32  { leaf() }",
    );
    let leaf = Source::new("leaf", "export { leaf }; fn leaf() -> i32  { 1 }");
    let mut loader = Loader::new(resin_source::library_root());
    let cache = Cache::new(16);
    loader.set_import(&entry, "middle", middle.clone()).unwrap();
    loader.set_import(&middle, "leaf", leaf.clone()).unwrap();
    let (cache, before) = request(&cache, entry.clone(), &mut loader).await.unwrap();
    valid(&before);
    let changed = leaf.with_text("export { leaf }; fn leaf() -> bool  { 1 == 1 }");
    loader.set_import(&middle, "leaf", changed.clone()).unwrap();
    let (_, after) = request(&cache, entry.clone(), &mut loader).await.unwrap();
    assert!(!Arc::ptr_eq(&before, &after));
    assert!(after.hir().is_err());
    assert_eq!(after.source(), &entry);
    assert!(sources(&before).contains(&leaf));
    assert!(sources(&after).contains(&changed));
    assert!(!sources(&after).contains(&leaf));
    valid(&before);
}

#[tokio::test]
async fn a_missing_transitive_import_recovers_without_notifications() {
    let entry = Source::new("entry", "import { \"middle\" };");
    let middle = Source::new("middle", "import { \"leaf\" };");
    let leaf = Source::new("leaf", "fn leaf()  {}");
    let mut loader = Loader::new(resin_source::library_root());
    let cache = Cache::new(16);
    loader.set_import(&entry, "middle", middle.clone()).unwrap();
    let (cache, before) = request(&cache, entry.clone(), &mut loader).await.unwrap();
    assert!(before.hir().is_err());
    assert!(before.diagnostics().iter().any(|diagnostic| {
        diagnostic.location.source == middle
            && diagnostic.location.span.end > diagnostic.location.span.start
    }));
    loader.set_import(&middle, "leaf", leaf.clone()).unwrap();
    let (_, after) = request(&cache, entry, &mut loader).await.unwrap();
    valid(&after);
    assert!(!Arc::ptr_eq(&before, &after));
    assert!(before.hir().is_err(), "retained failure remains immutable");
    assert!(sources(&after).contains(&leaf));
}

#[tokio::test]
async fn changed_import_edges_invalidate_cache_even_with_the_same_source_set() {
    let entry = Source::new(
        "entry",
        "import { \"first\", \"second\" }; fn main() -> i32  { first() }",
    );
    let first = Source::new(
        "first",
        "export { first }; import { \"value\" }; fn first() -> i32  { value() }",
    );
    let second = Source::new(
        "second",
        "export { second }; import { \"value\" }; fn second() -> bool  { value() }",
    );
    let integer = Source::new("integer", "export { value }; fn value() -> i32  { 1 }");
    let boolean = Source::new(
        "boolean",
        "export { value }; fn value() -> bool  { 1 == 1 }",
    );
    let mut loader = Loader::new(resin_source::library_root());
    let cache = Cache::new(16);
    loader.set_import(&entry, "first", first.clone()).unwrap();
    loader.set_import(&entry, "second", second.clone()).unwrap();
    loader.set_import(&first, "value", integer.clone()).unwrap();
    loader
        .set_import(&second, "value", boolean.clone())
        .unwrap();
    let (cache, before) = request(&cache, entry.clone(), &mut loader).await.unwrap();
    valid(&before);
    loader.set_import(&first, "value", boolean.clone()).unwrap();
    loader
        .set_import(&second, "value", integer.clone())
        .unwrap();
    let (_, after) = request(&cache, entry, &mut loader).await.unwrap();
    assert_eq!(sources(&before), sources(&after));
    assert!(!Arc::ptr_eq(&before, &after));
    assert!(after.hir().is_err());
    valid(&before);
}

#[tokio::test]
async fn inconsistent_versions_of_one_logical_source_are_diagnosed() {
    let entry = Source::new("entry", "import { \"first\", \"second\" };");
    let first = Source::new("module", "fn value() -> i32  { 1 }");
    let second = first.with_text("fn value() -> i32  { 2 }");
    assert_eq!(first.id(), second.id());
    assert_ne!(first, second);
    let mut loader = Loader::new(resin_source::library_root());
    let cache = Cache::<resin_source::SourceGraph, Hir>::new(16);
    loader.set_import(&entry, "first", first.clone()).unwrap();
    loader.set_import(&entry, "second", second.clone()).unwrap();
    assert!(matches!(acquire(entry, &mut loader).await.err().unwrap(),
        GraphError::ConflictingSource { source } if source == first.id()));
    assert!(cache.is_empty());
}

#[tokio::test]
async fn import_cycles_are_diagnosed_with_source_ranges() {
    let entry = Source::new("entry", "import { \"next\" };");
    let next = Source::new("next", "import { \"entry\" };");
    let mut loader = Loader::new(resin_source::library_root());
    let cache = Cache::new(16);
    loader.set_import(&entry, "next", next.clone()).unwrap();
    loader.set_import(&next, "entry", entry.clone()).unwrap();
    let (_, result) = request(&cache, entry.clone(), &mut loader).await.unwrap();
    assert!(result.hir().is_err());
    assert!(result.diagnostics().iter().any(|diagnostic| {
        diagnostic.message.contains("cyclic")
            && diagnostic.location.span.end > diagnostic.location.span.start
    }));
    assert!(result.recovered_file(&entry).is_some());
    assert!(result.recovered_file(&next).is_some());
}

#[tokio::test]
async fn identical_diagnostic_names_do_not_merge_distinct_sources() {
    let entry = Source::new("entry", "import { \"first\", \"second\" };");
    let first = Source::with_identity(
        SourceId::new("first"),
        "generated",
        "fn first() -> i32  { 1 == 1 }",
    );
    let second = Source::with_identity(
        SourceId::new("second"),
        "generated",
        "fn second() -> bool  { 1 }",
    );
    let mut loader = Loader::new(resin_source::library_root());
    let cache = Cache::new(16);
    loader.set_import(&entry, "first", first.clone()).unwrap();
    loader.set_import(&entry, "second", second.clone()).unwrap();
    let (_, result) = request(&cache, entry.clone(), &mut loader).await.unwrap();
    assert!(result.hir().is_err());
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

#[tokio::test]
async fn retained_compilations_keep_their_own_source_versions_and_editor_queries() {
    let before = Source::new(
        "editor",
        "fn value() -> i32  { 1 } fn main() -> i32  { value() }",
    );
    let mut loader = Loader::new(resin_source::library_root());
    let cache = Cache::new(16);
    let (cache, old) = request(&cache, before.clone(), &mut loader).await.unwrap();
    let after = before.with_text("fn value() -> bool  { 1 == 1 } fn main() -> bool  { value() }");
    let (_, new) = request(&cache, after.clone(), &mut loader).await.unwrap();
    valid(&old);
    valid(&new);
    assert_eq!(before.id(), after.id());
    assert!(!Arc::ptr_eq(&old, &new));
    for (result, source, ty) in [(&old, &before, "i32"), (&new, &after, "bool")] {
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

#[tokio::test]
async fn later_errors_preserve_completed_earlier_passes_and_recovered_syntax() {
    let source = Source::new(
        "entry",
        "export { first, second }; fn first() -> bool { (1 == 1) + (1 == 1) } fn second() -> i32 { let mut r = (1,); r + r; 0 }",
    );
    let mut loader = Loader::new(resin_source::library_root());
    let cache = Cache::new(16);
    let (cache, lowered) = request(&cache, source.clone(), &mut loader).await.unwrap();
    assert!(lowered.program().is_ok());
    assert!(lowered.hir().is_ok());
    let typed_source = source.with_text("fn main() -> i32  { 1 == 2 }");
    let (cache, typed) = request(&cache, typed_source, &mut loader).await.unwrap();
    assert!(typed.program().is_ok());
    assert!(typed.hir().is_err());
    let parsed_source = source.with_text("fn main( = { 1 == 2 };");
    let (_, parsed) = request(&cache, parsed_source.clone(), &mut loader)
        .await
        .unwrap();
    assert!(parsed.program().is_err());
    assert!(parsed.recovered_file(&parsed_source).is_some());
}

#[tokio::test]
async fn imported_initialization_errors_keep_the_dependency_source_version() {
    let entry = Source::new(
        "entry",
        "import { \"dependency\" }; fn main() -> i32  { value() }",
    );
    let dependency = Source::new(
        "dependency",
        "export { value }; fn value() -> i32  { let mut n: i32; n }",
    );
    let read = dependency.text().rfind("n }").unwrap();
    let mut loader = Loader::new(resin_source::library_root());
    let cache = Cache::new(16);
    loader
        .set_import(&entry, "dependency", dependency.clone())
        .unwrap();
    let (cache, failed) = request(&cache, entry.clone(), &mut loader).await.unwrap();
    assert!(failed.hir().is_err());
    assert!(failed.hir().is_err());
    assert_eq!(failed.diagnostics().len(), 1);
    let diagnostic = &failed.diagnostics()[0];
    assert_eq!(diagnostic.location.source, dependency);
    assert_eq!(
        diagnostic.location.span,
        Span {
            start: read,
            end: read + 1
        }
    );
    assert!(diagnostic.message.contains("UninitializedValue"));
    let call = entry.text().rfind("value").unwrap();
    assert_eq!(failed.definition(&entry, call).unwrap().source, dependency);

    let repaired = dependency.with_text("export { value }; fn value() -> i32  { 7 }");
    loader.set_import(&entry, "dependency", repaired).unwrap();
    let (_, repaired) = request(&cache, entry, &mut loader).await.unwrap();
    valid(&repaired);
    drop(loader);
    let diagnostic = &failed.diagnostics()[0];
    assert_eq!(diagnostic.location.source, dependency);
    assert_eq!(
        &diagnostic.location.source.text()
            [diagnostic.location.span.start..diagnostic.location.span.end],
        "n"
    );
}

#[tokio::test]
async fn conflicting_versions_cannot_publish_a_partial_cache_generation() {
    let entry = Source::new("entry", "import { \"first\", \"conflict\", \"original\" };");
    let original = Source::new("module", "fn value() -> i32  { 1 }");
    let conflict = original.with_text("fn value() -> i32  { 2 }");
    let mut loader = Loader::new(resin_source::library_root());
    let cache = Cache::new(16);
    loader
        .set_import(&entry, "first", original.clone())
        .unwrap();
    loader
        .set_import(&entry, "conflict", conflict.clone())
        .unwrap();
    loader
        .set_import(&entry, "original", original.clone())
        .unwrap();
    assert!(
        matches!(request(&cache, entry.clone(), &mut loader).await.err().unwrap(),
        GraphError::ConflictingSource { source } if source == original.id())
    );
    assert!(
        cache.is_empty(),
        "a rejected input cannot publish a cache generation"
    );
    loader
        .set_import(&entry, "conflict", original.clone())
        .unwrap();
    let (_, result) = request(&cache, entry.clone(), &mut loader).await.unwrap();
    valid(&result);
    let offset = entry.text().find("original").unwrap();
    assert_eq!(result.definition(&entry, offset).unwrap().source, original);
}
