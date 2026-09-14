use resin_hir::{TermKind, Type};

fn generate(source: &str) -> Result<resin_hir::Module, resin_hir::GenerateError> {
    let document = resin_cst::Document::reparse(source.into(), None);
    resin_hir::generate(&resin_ast::generate(&document).unwrap())
}

#[test]
fn intrinsic_declarations_keep_generic_signatures_and_ordinary_calls() {
    let module = generate(
        r#"
        intrinsic "pointer_index" def at<T>(data: Ptr<T>, length: ulong, index: ulong) -> Ptr<T>;
        def index<T>(data: Ptr<T>, length: ulong) -> Ptr<T> = { at(data, length, 0) };
    "#,
    )
    .unwrap();
    let primitive = &module.functions[0];
    let TermKind::Intrinsic { args, .. } = &primitive.body.as_ref().unwrap().kind else {
        panic!("primitive operation")
    };
    assert_eq!(args.values.len(), 3);
    assert!(matches!(args.params[0], Type::Pointer { .. }));
    assert_eq!(primitive.signature.type_params.len(), 1);
    assert_eq!(primitive.signature.params.len(), 3);
}

#[test]
fn intrinsic_contracts_reject_unknown_operations_and_forged_signatures() {
    for source in [
        r#"intrinsic "missing" def wrong() -> ulong;"#,
        r#"intrinsic "pointer_index" def wrong<T>(data: Ptr<T>, length: ulong, index: int) -> Ptr<T>;"#,
        r#"intrinsic "pointer_index" def wrong<T>(data: Ptr<T>, length: ulong, index: ulong) -> T;"#,
        r#"intrinsic "pointer_index" def wrong<T, U>(data: Ptr<T>, length: ulong, index: ulong) -> Ptr<U>;"#,
    ] {
        assert!(generate(source).is_err(), "{source}");
    }
}
