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
        r#"fn text() -> _  { "hello" }
        fn identity(value: str) -> str  { value }
        fn answer() -> _  { identity(text()) }
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
        r#"fn data(value: str) -> _  { value.data }
        fn length(value: str) -> _  { value.length }
        fn byte(value: str) -> _  { value:at(u64(0)) }
        fn legacy(value: str) -> _  { value(u64(0)) }
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
        r#"intrinsic "string_from_bytes" fn copy(data: Ptr<u8>, length: u64) -> StrongOwner;
        struct String { owner: StrongOwner,
            
        }
fn string_from_str(text: str) -> String  { String { owner = copy(text.data, text.length) } }

        fn text() -> str  { "hello" }
        fn owned() -> String  { string_from_str(text()) }
    "#,
    )
    .unwrap();
    assert_eq!(result(&module, "text"), Type::Str);
    assert!(matches!(result(&module, "owned"), Type::Defined { .. }));
    // The spelling carries no compiler representation or privileged methods.
    generate("struct String { count: u32, } fn make() -> String  { String { count = u32(7) } }")
        .unwrap();
}

#[test]
fn string_literals_cannot_be_forged_from_arbitrary_storage() {
    for source in [
        r#"struct Bytes (Ptr<u8>, u64); fn bad() -> Bytes  { "text" }"#,
        r#"struct String { owner: StrongOwner, } fn bad() -> String  { "text" }"#,
        r#"struct Bytes (Ptr<u8>, u64); fn bad(bytes: Bytes) -> str  { str(bytes) }"#,
        r#"fn bad(data: Ptr<u8>) -> str  { str { data = data, length = u64(1) } }"#,
        r#"fn bad() -> _  { "text".unknown }"#,
    ] {
        assert!(generate(source).is_err(), "accepted {source}");
    }
}
