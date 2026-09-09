use resin_ast::{SourceModule, StmtKind, TermKind};
use resin_cst::Document;
use resin_source::prelude::*;

#[test]
fn recovery_keeps_original_byte_spans_and_incomplete_function_bodies() {
    let source = "// é🌲\ndef main() = { print(\"hello\")";
    let document = Document::reparse(source.into(), None);
    assert!(resin_ast::generate(&document).is_err());
    let recovered = resin_ast::recover(&document);
    assert!(!recovered.errors.is_empty());
    let function = &recovered.file.stmts[0];
    let StmtKind::Function { body, .. } = &function.val else {
        panic!("function lost")
    };
    let TermKind::Block { tail, .. } = &body.val else {
        panic!("block lost")
    };
    for span in [function.span, body.span, tail.span] {
        assert!(span.start <= span.end && span.end <= source.len());
        assert!(source.is_char_boundary(span.start) && source.is_char_boundary(span.end));
    }
    assert!(matches!(tail.val, TermKind::Call { .. }));
}

#[test]
fn recovering_valid_syntax_matches_strict_generation() {
    let source =
        "struct Point { x: int }; impl Point { def get(self: Point) -> int = { self.x }; }";
    let document = Document::reparse(source.into(), None);
    let recovered = resin_ast::recover(&document);
    let strict = resin_ast::generate(&document).unwrap();
    assert!(recovered.errors.is_empty());
    assert_eq!(
        resin_ast::format_source(&strict),
        resin_ast::format_source(&recovered.file)
    );
    assert_eq!(strict.declarations().count(), 2);
}

#[test]
fn source_locations_tolerate_editor_offsets_inside_utf8() {
    let source = "// é🌲\ndef main() = {};";
    let module = SourceModule {
        source: Source::new("memory.resin", source),
        file: resin_ast::generate(&Document::reparse(source.into(), None)).unwrap(),
        imports: vec![],
    };
    assert_eq!(
        module.location(Span { start: 4, end: 4 }),
        "memory.resin:1:4"
    );
    assert_eq!(
        module.location(Span { start: 10, end: 10 }),
        "memory.resin:2:1"
    );
    assert!(
        module
            .location(Span {
                start: usize::MAX,
                end: usize::MAX
            })
            .starts_with("memory.resin:2:")
    );
}
