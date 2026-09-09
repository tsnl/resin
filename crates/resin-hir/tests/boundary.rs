use resin_ast::{Program, SourceModule};
use resin_source::prelude::*;
use resin_types::prelude::*;

use resin_cst::Document;
use std::{collections::BTreeMap, sync::Arc};

fn module(name: &str, text: &str) -> SourceModule {
    source_module(Source::new(name, text))
}

fn source_module(source: Source) -> SourceModule {
    let file = resin_ast::generate(&Document::reparse(source.text().into(), None)).unwrap();
    SourceModule {
        source,
        file,
        imports: vec![],
    }
}

#[test]
fn malformed_dependency_order_returns_diagnostics() {
    for dependency in [0, 1, usize::MAX] {
        let mut entry = module("entry.resin", "def main() = {};");
        entry.imports.push((Span { start: 0, end: 1 }, dependency));
        let program = Program {
            modules: vec![entry],
        };
        let analysis = resin_hir::analyze_program(&program);
        assert!(analysis.module.is_none());
        assert_eq!(analysis.diagnostics.len(), 1);
        assert!(resin_hir::generate_program(&program).is_err());
    }
}

#[test]
fn checking_a_standalone_file_rejects_unresolved_imports() {
    let entry = module(
        "entry.resin",
        r#"import { "library.resin" }; def main() = {};"#,
    );
    assert!(resin_hir::generate(&entry.file).is_err());
}

#[test]
fn resolved_program_infers_through_an_import_and_exposes_only_root_exports() {
    let library = module(
        "library.resin",
        "export { narrow }; def narrow(value: int) -> int = { value };",
    );
    let mut entry = module(
        "entry.resin",
        "export { main }; def main() -> _ = { narrow(42) };",
    );
    entry.imports.push((Span { start: 0, end: 0 }, 0));
    let module = resin_hir::generate_program(&Program {
        modules: vec![library, entry],
    })
    .unwrap();
    let main = &module.functions[module.entries["main"].index()];
    assert_eq!(main.signature.result.ty, Ty::Int32);
    assert_eq!(module.entries.len(), 1);
    assert!(main.body.is_some());
}

fn syntax(sources: &[Source]) -> BTreeMap<Source, Arc<Document>> {
    sources
        .iter()
        .map(|source| {
            let document = Document::reparse(source.text().into(), None);
            (source.clone(), Arc::new(document))
        })
        .collect()
}

#[test]
fn analysis_keeps_editor_queries_after_an_unrelated_type_error() {
    let source =
        "// é🌲\ndef broken() -> int = { missing() }; def healthy(value: int) -> int = { value };";
    let entry = module("entry.resin", source);
    let input = entry.source.clone();
    let syntax = syntax(std::slice::from_ref(&input));
    let program = Program {
        modules: vec![entry],
    };
    let analysis = resin_hir::analyze_program(&program);
    assert!(analysis.module.is_none());
    assert!(!analysis.diagnostics.is_empty());
    let offset = source.rfind("value").unwrap();
    let definition = analysis
        .semantics
        .definition(&syntax, &input, offset)
        .unwrap();
    assert_eq!(definition.span.start, source.find("value").unwrap());
    assert!(
        analysis
            .semantics
            .hover(&syntax, &input, offset)
            .unwrap()
            .text
            .contains("int")
    );
    for invalid in [4, 6, source.len() + 1, usize::MAX] {
        assert!(
            analysis
                .semantics
                .definition(&syntax, &input, invalid)
                .is_none()
        );
        assert!(analysis.semantics.hover(&syntax, &input, invalid).is_none());
        assert!(
            analysis
                .semantics
                .completions(&syntax, &input, invalid)
                .is_empty()
        );
    }
}

#[test]
fn sources_with_equal_names_have_distinct_editor_facts() {
    let integer = Source::new("memory", "def local(value: int) -> int = { value };");
    let boolean = Source::new("memory", "def local(value: bool) -> bool = { value };");
    let syntax = syntax(&[integer.clone(), boolean.clone()]);
    let program = Program {
        modules: vec![
            source_module(integer.clone()),
            source_module(boolean.clone()),
        ],
    };
    let analysis = resin_hir::analyze_program(&program);
    assert!(analysis.module.is_some(), "{:?}", analysis.diagnostics);
    for (source, expected) in [(integer, "value: int"), (boolean, "value: bool")] {
        let offset = source.text().rfind("value").unwrap();
        let definition = analysis
            .semantics
            .definition(&syntax, &source, offset)
            .unwrap();
        assert_eq!(definition.source, source);
        assert_eq!(definition.span.start, source.text().find("value").unwrap());
        assert_eq!(
            analysis
                .semantics
                .hover(&syntax, &source, offset)
                .unwrap()
                .text,
            expected
        );
    }
}

#[test]
fn revised_source_cannot_borrow_editor_facts_from_its_previous_version() {
    let original = Source::new("memory", "def local(value: int) -> int = { value };");
    let revised = original.with_text("def local(value: bool) -> bool = { value };");
    let syntax = syntax(&[original.clone(), revised.clone()]);
    let analysis = resin_hir::analyze_program(&Program {
        modules: vec![source_module(original.clone())],
    });
    let offset = revised.text().rfind("value").unwrap();
    assert_eq!(original.id(), revised.id());
    assert!(
        analysis
            .semantics
            .definition(&syntax, &revised, offset)
            .is_none()
    );
    assert!(
        analysis
            .semantics
            .hover(&syntax, &revised, offset)
            .is_none()
    );
    let offset = original.text().rfind("value").unwrap();
    assert_eq!(
        analysis
            .semantics
            .hover(&syntax, &original, offset)
            .unwrap()
            .text,
        "value: int"
    );
}
