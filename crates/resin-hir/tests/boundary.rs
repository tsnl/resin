use resin_ast::{Program, SourceModule};
use resin_common::prelude::*;

use resin_cst::Document;
use std::path::Path;

fn module(path: &str, source: &str) -> SourceModule {
    SourceModule {
        path: path.into(),
        source: source.into(),
        imports: vec![],
        file: resin_ast::generate(&Document::reparse(source.into(), None)).unwrap(),
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

struct Syntax(Document);
impl resin_hir::Documents for Syntax {
    fn get(&self, path: &Path) -> Option<&Document> {
        (path == Path::new("entry.resin")).then_some(&self.0)
    }
}

#[test]
fn analysis_keeps_editor_queries_after_an_unrelated_type_error() {
    let source =
        "// é🌲\ndef broken() -> int = { missing() }; def healthy(value: int) -> int = { value };";
    let syntax = Syntax(Document::reparse(source.into(), None));
    let program = Program {
        modules: vec![module("entry.resin", source)],
    };
    let analysis = resin_hir::analyze_program(&program);
    assert!(analysis.module.is_none());
    assert!(!analysis.diagnostics.is_empty());
    let path = Path::new("entry.resin");
    let offset = source.rfind("value").unwrap();
    let definition = analysis
        .semantics
        .definition(&syntax, path, offset)
        .unwrap();
    assert_eq!(definition.span.start, source.find("value").unwrap());
    assert!(
        analysis
            .semantics
            .hover(&syntax, path, offset)
            .unwrap()
            .text
            .contains("int")
    );
    for invalid in [4, 6, source.len() + 1, usize::MAX] {
        assert!(
            analysis
                .semantics
                .definition(&syntax, path, invalid)
                .is_none()
        );
        assert!(analysis.semantics.hover(&syntax, path, invalid).is_none());
        assert!(
            analysis
                .semantics
                .completions(&syntax, path, invalid)
                .is_empty()
        );
    }
}
