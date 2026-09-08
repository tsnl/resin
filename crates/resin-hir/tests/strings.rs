use resin_source::prelude::*;
use resin_types::prelude::*;

fn generate(text: &str) -> Result<resin_hir::Module, SourceError> {
    let source = Source::new("strings", text);
    let document = resin_cst::Document::reparse(source.text().into(), None);
    let file = resin_ast::generate(&document).unwrap();
    resin_hir::generate_program(&resin_ast::Program {
        modules: vec![resin_ast::SourceModule {
            source,
            file,
            imports: vec![],
        }],
    })
}

fn result(module: &resin_hir::Module, name: &str) -> Ty {
    module
        .functions
        .iter()
        .find(|function| function.name.as_ref() == name)
        .unwrap()
        .signature
        .result
        .ty
        .clone()
}

#[test]
fn literals_infer_str_across_function_calls_and_annotations() {
    let module = generate(
        r#"
        def text() -> _ = { "hello" };
        def identity(value: str) -> str = { value };
        def answer() -> _ = { identity(text()) };
        "#,
    )
    .unwrap();
    for name in ["text", "identity", "answer"] {
        assert_eq!(result(&module, name), Ty::Str);
    }
}

#[test]
fn string_views_support_fields_and_indexing() {
    let module = generate(
        r#"
        def data(value: str) -> _ = { value.data };
        def length(value: str) -> _ = { value.length };
        def byte(value: str) -> _ = { value.at(0_ul) };
        def legacy(value: str) -> _ = { value(0_ul) };
        "#,
    )
    .unwrap();
    assert_eq!(result(&module, "length"), Ty::UInt64);
    for name in ["data", "byte", "legacy"] {
        assert_eq!(
            result(&module, name),
            Ty::Pointer {
                pointee: Box::new(Ty::UInt8)
            }
        );
    }
}

#[test]
fn string_views_convert_explicitly_and_copy_into_owned_strings() {
    let module = generate(
        r#"
        def bytes() -> _ = { Span<_>("hello") };
        def forwarded_bytes() -> _ = { Span<_>(text()) };
        def text() -> _ = { "hello" };
        def recursive(again: bool) -> _ = {
            if (again) { Span<_>(recursive(0 == 1)); "hello" } else { "world" }
        };
        def literal_copy() -> String = { String.from_str("hello") };
        def bytes_copy(value: Span<ubyte>) -> String = { String.from_bytes(value) };
        "#,
    )
    .unwrap();
    assert_eq!(result(&module, "bytes"), Ty::byte_span());
    assert_eq!(result(&module, "forwarded_bytes"), Ty::byte_span());
}

#[test]
fn string_views_cannot_be_confused_with_bytes_or_owned_strings() {
    for source in [
        r#"def value() -> Span<ubyte> = { "text" };"#,
        r#"def value() -> String = { "text" };"#,
        r#"def value(bytes: Span<ubyte>) -> str = { str(bytes) };"#,
        r#"def value(data: Ptr<ubyte>) -> str = { str { data = data, length = 1_ul } };"#,
        r#"def value(bytes: Span<ubyte>) -> String = { String.from_str(bytes) };"#,
        r#"def value() -> String = { String.from_bytes("text") };"#,
        r#"def value() -> _ = { Span<uint>("text") };"#,
        r#"def value() -> _ = { "text".unknown };"#,
    ] {
        assert!(generate(source).is_err(), "accepted {source}");
    }
}
