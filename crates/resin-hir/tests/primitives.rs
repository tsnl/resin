use resin_hir::{TermKind, Type};

mod common;
use common::hir_module;

fn generate(source: &str) -> Result<resin_hir::Module, resin_source::SourceError> {
    hir_module(source)
}

#[test]
fn intrinsic_declarations_keep_generic_signatures_and_ordinary_calls() {
    let module = generate(
        r#"intrinsic "pointer_index" fn at<T>(data: Ptr<T>, length: ulong, index: ulong) -> Ptr<T>;
        fn index<T>(data: Ptr<T>, length: ulong) -> Ptr<T>  { at(data, length, 0) }
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
        r#"intrinsic "missing" fn wrong() -> ulong;"#,
        r#"intrinsic "pointer_index" fn wrong<T>(data: Ptr<T>, length: ulong, index: int) -> Ptr<T>;"#,
        r#"intrinsic "pointer_index" fn wrong<T>(data: Ptr<T>, length: ulong, index: ulong) -> T;"#,
        r#"intrinsic "pointer_index" fn wrong<T, U>(data: Ptr<T>, length: ulong, index: ulong) -> Ptr<U>;"#,
    ] {
        assert!(generate(source).is_err(), "{source}");
    }
}

#[test]
fn pointer_and_array_methods_keep_generic_nominal_payloads_symbolic() {
    let source = r#"struct Cell<T> { value: T, }
        fn replace<T>(storage: Ptr<Cell<T>>, value: T) -> Cell<T>  {
            storage:replace(Cell<T> { value = value })
        }
        fn first<T>(value: T) -> T  {
            let mut cells = [Cell<T> { value = value }];
            cells:at(0).value
        }
    "#;
    generate(source).unwrap();
}

#[test]
fn gpu_argument_operations_accept_explicit_values_in_both_call_spellings() {
    generate(
        r#"fn record(arguments: Ptr<GpuArguments>, commands: Ptr<ubyte>) -> int  {
            arguments.*:dispatch_native(commands, 1, 1, 1);
            arguments.*:draw_native(commands, 3);
            dispatch_native(arguments.*, commands, 1, 1, 1);
            draw_native(arguments.*, commands, 3)
        }
    "#,
    )
    .unwrap();
}
