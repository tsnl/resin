mod common;
use common::parse;

#[tokio::test]
async fn preamble_queries_decode_paths_and_preserve_empty_header_groups() {
    let source = r#"// import { "comment.resin" };
export { call };
extern {
    "local/é\"quoted.h": { fn call(); },
    "empty.h": {},
};
import { "../types.resin", "back\\slash.resin" };
fn body()  { "import { \"body.resin\" };" }
"#;
    let document = parse(source, None).await;
    let preamble = document.preamble();
    assert!(preamble.diagnostics.is_empty(), "{preamble:?}");
    assert_eq!(preamble.headers.len(), 2);
    assert_eq!(&*preamble.headers[0].val, "local/é\"quoted.h");
    assert_eq!(&*preamble.headers[1].val, "empty.h");
    assert_eq!(preamble.imports.len(), 2);
    assert_eq!(&*preamble.imports[0].val, "../types.resin");
    assert_eq!(&*preamble.imports[1].val, "back\\slash.resin");
    for entry in preamble.headers.iter().chain(&preamble.imports) {
        assert_eq!(
            resin_cst::decode_string(&source[entry.span.start..entry.span.end]).as_deref(),
            Some(&*entry.val)
        );
    }
}

#[tokio::test]
async fn unrelated_body_errors_do_not_invalidate_dependency_declarations() {
    let prefix = "extern { \"local.h\": {} }; import { \"types.resin\" };";
    for body in [
        "fn main()  { let mut x = ; }",
        "fn main() = { \"unterminated",
        "fn main(",
        "struct Broken { value:",
        "const broken = ;",
        "const unfinished =",
        "extern type Handle; fn main()  { missing( }",
    ] {
        let document = parse(format!("{prefix}{body}"), None).await;
        assert!(document.tree().root_node().has_error(), "{body}");
        let preamble = document.preamble();
        assert!(preamble.diagnostics.is_empty(), "{body}: {preamble:?}");
        assert_eq!(preamble.headers.len(), 1, "{body}");
        assert_eq!(preamble.imports.len(), 1, "{body}");
    }
}

#[tokio::test]
async fn incomplete_preambles_retain_complete_paths_and_report_incompleteness() {
    for source in [
        "import { \"one.resin\", \"two.resin\"",
        "import { \"one.resin\", \"unfinished",
        "extern { \"local.h\": { fn call();",
        "extern { \"local.h\":",
    ] {
        let document = parse(source, None).await;
        let preamble = document.preamble();
        assert!(!preamble.diagnostics.is_empty(), "{source}: {preamble:?}");
        let dependencies = if source.starts_with("import") {
            &preamble.imports
        } else {
            &preamble.headers
        };
        assert!(!dependencies.is_empty(), "{source}: {preamble:?}");
        for dependency in dependencies {
            assert!(dependency.span.end <= source.len());
        }
        assert_eq!(document.source(), source);
    }
}

#[tokio::test]
async fn malformed_preambles_never_claim_to_be_complete() {
    for source in [
        "import",
        "extern {",
        "export { broken\nimport { \"one.resin\" };",
        "import {}; extern { \"late.h\": {} };",
        "extern { \"bad\\q.h\": {} };",
        "extern \"legacy.h\" fn legacy();",
        "fn first()  {} import { \"late.resin\" };",
        "extern type Native; extern { \"late.h\": {} };",
        "const count = 1; import { \"late.resin\" };",
    ] {
        let preamble = parse(source, None).await.preamble();
        assert!(!preamble.diagnostics.is_empty(), "{source}: {preamble:?}");
    }
}

#[tokio::test]
async fn misplaced_clauses_report_errors_without_collecting_body_dependencies() {
    let source = r#"import { "valid.resin" };
        fn broken()  { import { "body.resin" }; }
        import { "late.resin" };
    "#;
    let preamble = parse(source, None).await.preamble();
    assert!(!preamble.diagnostics.is_empty());
    assert_eq!(preamble.imports.len(), 1);
    assert_eq!(&*preamble.imports[0].val, "valid.resin");
    let body_only = source
        .strip_suffix("import { \"late.resin\" };\n    ")
        .unwrap();
    assert!(
        parse(body_only, None)
            .await
            .preamble()
            .diagnostics
            .is_empty()
    );
}

#[tokio::test]
async fn opaque_foreign_types_are_body_declarations() {
    let preamble = parse("extern type Handle;", None).await.preamble();
    assert!(preamble.headers.is_empty());
    assert!(preamble.diagnostics.is_empty());
}

#[test]
fn grouped_foreign_functions_have_stable_multiline_formatting() {
    let source = "extern{\"local.h\":{fn call(value:int)->int;},\"empty.h\":{},};";
    let formatted = resin_cst::format_source(source).unwrap();
    assert_eq!(
        formatted,
        "extern {\n\t\"local.h\": {\n\t\tfn call(value: int) -> int;\n\t},\n\t\"empty.h\": {},\n};\n"
    );
    assert_eq!(resin_cst::format_source(&formatted).unwrap(), formatted);
}

#[test]
fn string_decoding_matches_grammar_escapes() {
    assert_eq!(
        resin_cst::decode_string(r#""é\n\r\t\0\"\\""#),
        Some("é\n\r\t\0\"\\".into())
    );
    for text in [
        "",
        "\"",
        "plain",
        "\"unfinished",
        r#""bad\q""#,
        "\"line\nbreak\"",
        r#""bad"quote""#,
    ] {
        assert!(resin_cst::decode_string(text).is_none(), "{text}");
    }
}
