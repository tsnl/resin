mod common;

use resin_ast::{SourceModule, StmtKind, TermKind};

use resin_source::prelude::*;

#[test]
fn recovery_keeps_original_byte_spans_and_incomplete_function_bodies() {
    let source = "// é🌲\nfn main() = { print(\"hello\")";
    let document = common::syntax(source);
    assert!(!common::ast(document.clone()).errors.is_empty());
    let recovered = common::ast(document.clone());
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
    let source = "struct Point { x: int,  }\nfn get(self: Point) -> int  { self.x }\n ";
    let document = common::syntax(source);
    let recovered = common::ast(document.clone());
    let strict = common::ast(document.clone()).file;
    assert!(recovered.errors.is_empty());
    assert_eq!(
        resin_ast::format_source(&strict),
        resin_ast::format_source(&recovered.file)
    );
    assert_eq!(strict.declarations().count(), 2);
}

#[test]
fn source_locations_tolerate_editor_offsets_inside_utf8() {
    let source = "// é🌲\nfn main()  {}";
    let module = SourceModule {
        source: Source::new("memory.resin", source),
        file: std::sync::Arc::new(common::parse(source).file),
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

#[test]
fn generic_wrapper_types_lower_with_their_element_annotations() {
    let source = "fn first(values: GpuSpan<uint>) -> GpuPtr<_>  { values:at(0_ul) }";
    let file = common::parse(source).file;
    let StmtKind::Function { params, result, .. } = &file.stmts[0].val else {
        panic!("expected function");
    };
    let resin_ast::TypeKind::App { head, args } = &params[0].1.val else {
        panic!("expected GPU span type former");
    };
    assert_eq!(head.val.as_ref(), "GpuSpan");
    assert!(
        matches!(&args[0].val, resin_ast::TypeKind::Atom { name } if name.val.as_ref() == "uint")
    );
    let resin_ast::TypeKind::App { head, args } = &result.val else {
        panic!("expected GPU pointer type former");
    };
    assert_eq!(head.val.as_ref(), "GpuPtr");
    assert!(matches!(args[0].val, resin_ast::TypeKind::Infer));
}

#[test]
fn generic_pipeline_annotations_keep_both_arguments() {
    let source = "fn pipeline(value: GpuComputePipeline<Root, ArcPtr<Owner>>) -> GpuGraphicsPipeline<None, _>  { value }";
    let file = common::parse(source).file;
    let StmtKind::Function { params, result, .. } = &file.stmts[0].val else {
        panic!("expected function");
    };
    let resin_ast::TypeKind::App { head, args } = &params[0].1.val else {
        panic!("expected compute pipeline annotation");
    };
    assert_eq!(head.val.as_ref(), "GpuComputePipeline");
    assert!(
        matches!(&args[0].val, resin_ast::TypeKind::Atom { name } if name.val.as_ref() == "Root")
    );
    assert!(
        matches!(&args[1].val, resin_ast::TypeKind::App { head, .. } if head.val.as_ref() == "ArcPtr")
    );
    let resin_ast::TypeKind::App { head, args } = &result.val else {
        panic!("expected graphics pipeline annotation");
    };
    assert_eq!(head.val.as_ref(), "GpuGraphicsPipeline");
    assert!(
        matches!(&args[0].val, resin_ast::TypeKind::Atom { name } if name.val.as_ref() == "None")
    );
    assert!(matches!(args[1].val, resin_ast::TypeKind::Infer));
}

#[test]
fn recovery_retains_struct_fields_and_free_function_declarations() {
    let source = "struct Item { value: int,  }\nfn read(self: Item) -> int  { self. }\n fn later() -> int  { 42 }";
    let document = common::syntax(source);
    let file = common::ast(document.clone()).file;
    let StmtKind::Struct { body, methods, .. } = &file.stmts[0].val else {
        panic!("expected a struct");
    };
    let resin_ast::TypeKind::Record { fields } = &body.val else {
        panic!()
    };
    assert_eq!(fields[0].0.val.as_ref(), "value");
    assert!(methods.is_empty());
    assert!(
        matches!(&file.stmts[1].val, StmtKind::Function { name, .. } if name.val.as_ref() == "read")
    );
    assert!(
        matches!(&file.stmts[2].val, StmtKind::Function { name, .. } if name.val.as_ref() == "later")
    );
    assert_eq!(file.declarations().count(), 3);
}

#[test]
fn fields_must_precede_struct_methods() {
    let source = "struct Item { def read() -> int = { 42 }, value: int, }";
    let document = common::syntax(source);
    assert!(!common::ast(document.clone()).errors.is_empty());
}
