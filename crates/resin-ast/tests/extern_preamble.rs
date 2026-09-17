mod common;

use resin_ast::{StmtKind, TypeKind};

#[test]
fn header_groups_retain_dependencies_and_flatten_module_declarations() {
    let source = r#"export { first };
        extern {
            "empty.h": {},
            "native.h": {
                fn first(value: i32) -> i32;
                fn second();
            },
        };
        import { "types.resin" };
        extern type Native;
        fn main()  {}
    "#;
    let parsed = common::parse(source);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let file = parsed.file;
    assert_eq!(
        file.foreign_headers
            .iter()
            .map(|h| h.val.as_ref())
            .collect::<Vec<_>>(),
        ["empty.h", "native.h"]
    );
    for header in &file.foreign_headers {
        assert_eq!(
            &source[header.span.start..header.span.end],
            format!("\"{}\"", header.val)
        );
    }
    assert_eq!(file.imports[0].val.as_ref(), "types.resin");
    assert_eq!(file.declarations().count(), 4);
    for (statement, expected) in file.stmts[..2].iter().zip(["first", "second"]) {
        let StmtKind::ForeignFunction { header, name, .. } = &statement.val else {
            panic!("foreign declaration lost");
        };
        assert_eq!(header.as_ref(), "native.h");
        assert_eq!(name.val.as_ref(), expected);
        assert_eq!(&source[name.span.start..name.span.end], expected);
        assert!(source[statement.span.start..statement.span.end].starts_with("fn "));
    }
    assert!(
        matches!(&file.stmts[1].val, StmtKind::ForeignFunction { result, .. } if matches!(result.val, TypeKind::Unit))
    );
    assert!(matches!(file.stmts[2].val, StmtKind::ForeignType { .. }));
    let printed = resin_ast::format_source(&file);
    assert!(printed.contains("\"empty.h\""), "{printed}");
    assert!(printed.contains("\"native.h\""), "{printed}");
}

#[test]
fn foreign_declarations_survive_unrelated_body_errors() {
    let source =
        r#"extern { "native.h": { fn native() -> i32; } }; fn broken()  { let mut x = ; }"#;
    let parsed = common::parse(source);
    assert!(!parsed.errors.is_empty());
    assert_eq!(parsed.file.foreign_headers[0].val.as_ref(), "native.h");
    assert!(
        matches!(&parsed.file.stmts[0].val, StmtKind::ForeignFunction { name, .. } if name.val.as_ref() == "native")
    );
}

#[test]
fn header_strings_use_the_same_decoding_as_imports_and_literals() {
    let source = r#"extern { "dir\\header.h": {} }; import { "dir\\types.resin" };"#;
    let parsed = common::parse(source);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    assert_eq!(parsed.file.foreign_headers[0].val.as_ref(), "dir\\header.h");
    assert_eq!(parsed.file.imports[0].val.as_ref(), "dir\\types.resin");
}
