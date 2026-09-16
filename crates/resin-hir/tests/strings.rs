use resin_hir::Type;
use resin_source::prelude::*;

mod common;

fn generate(text: &str) -> Result<resin_hir::Module, SourceError> {
    common::hir_module(text)
}

fn result(module: &resin_hir::Module, name: &str) -> Type {
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
        assert_eq!(result(&module, name), Type::Str);
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
    assert_eq!(result(&module, "length"), Type::UInt64);
    assert_eq!(
        result(&module, "data"),
        Type::Pointer {
            pointee: Box::new(Type::UInt8)
        }
    );
    for name in ["byte", "legacy"] {
        assert_eq!(result(&module, name), Type::UInt8);
    }
}

#[test]
fn owned_strings_are_ordinary_source_declarations() {
    let module = generate(
        r#"
        intrinsic "string_from_bytes" def copy(data: Ptr<ubyte>, length: ulong) -> StrongOwner;
        struct String { owner: StrongOwner,
            def from_str(text: str) -> String = { String { owner = copy(text.data, text.length) } };
        };
        def text() -> str = { "hello" };
        def owned() -> String = { String.from_str(text()) };
    "#,
    )
    .unwrap();
    assert_eq!(result(&module, "text"), Type::Str);
    assert!(matches!(result(&module, "owned"), Type::Defined { .. }));
    // The spelling carries no compiler representation or privileged methods.
    generate("struct String { count: uint }; def make() -> String = { String { count = 7_ui } };")
        .unwrap();
}

#[test]
fn string_literals_cannot_be_forged_from_arbitrary_storage() {
    for source in [
        r#"struct Bytes (Ptr<ubyte>, ulong); def bad() -> Bytes = { "text" };"#,
        r#"struct String { owner: StrongOwner }; def bad() -> String = { "text" };"#,
        r#"struct Bytes (Ptr<ubyte>, ulong); def bad(bytes: Bytes) -> str = { str(bytes) };"#,
        r#"def bad(data: Ptr<ubyte>) -> str = { str { data = data, length = 1_ul } };"#,
        r#"def bad() -> _ = { "text".unknown };"#,
    ] {
        assert!(generate(source).is_err(), "accepted {source}");
    }
}
