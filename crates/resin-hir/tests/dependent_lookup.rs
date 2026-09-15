use resin_hir::{Term, TermKind, Type};

fn compile(source: &str) -> Result<resin_hir::Module, resin_hir::GenerateError> {
    let document = resin_cst::Document::build(source.into(), None);
    resin_hir::build_hir(&resin_ast::build_ast(&document).unwrap())
}

fn tail(function: &resin_hir::Function) -> &Term {
    let TermKind::Block { tail, .. } = &function.body.as_ref().unwrap().kind else {
        panic!("function block")
    };
    tail
}

#[test]
fn dependent_methods_retain_the_receiver_and_determining_result() {
    let module = compile("def read<T>(value: T) -> _ = { value.read() };").unwrap();
    let read = &module.functions[0];
    let Type::FunctionResult { function } = &read.signature.result.ty else {
        panic!("method result relation")
    };
    let Type::Method { lookup } = function.as_ref() else {
        panic!("method type relation")
    };
    assert_eq!(lookup.name.as_ref(), "read");
    assert_eq!(
        lookup.receiver,
        Type::Parameter {
            parameter: read.signature.type_params[0].id
        }
    );
    assert!(!lookup.associated);
    assert!(lookup.type_args.is_empty());
    assert!(matches!(
        tail(read).kind,
        TermKind::DependentMethodCall { .. }
    ));
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn dependent_field_and_method_results_compose_without_concrete_declarations() {
    let module = compile(
        "def field_method<T>(value: T) -> _ = { value.item.read() }; \
         def method_field<T>(value: T) -> _ = { value.read().item }; \
         def method_method<T>(value: T) -> _ = { value.read().next() };",
    )
    .unwrap();
    assert!(matches!(
        module.functions[0].signature.result.ty,
        Type::FunctionResult { .. }
    ));
    assert!(matches!(
        module.functions[1].signature.result.ty,
        Type::Member { .. }
    ));
    assert!(matches!(
        module.functions[2].signature.result.ty,
        Type::FunctionResult { .. }
    ));
}

#[test]
fn dependent_associated_references_retain_explicit_method_arguments() {
    let module = compile("def select<T, U>() -> _ = { T.make::<U> };").unwrap();
    let select = &module.functions[0];
    let Type::Method { lookup } = &select.signature.result.ty else {
        panic!("associated method reference")
    };
    assert!(lookup.associated);
    assert_eq!(lookup.name.as_ref(), "make");
    assert_eq!(
        lookup.type_args,
        [Type::Parameter {
            parameter: select.signature.type_params[1].id
        }]
    );
    assert!(matches!(
        tail(select).kind,
        TermKind::DependentMethod { .. }
    ));
}

#[test]
fn dependent_callable_fields_remain_ordinary_calls() {
    let module = compile("def call<T>(value: T) -> _ = { (value.callback)(41) };").unwrap();
    let TermKind::Call { func, .. } = &tail(&module.functions[0]).kind else {
        panic!("function-valued field call")
    };
    assert!(matches!(func.kind, TermKind::Field { .. }));
    assert!(matches!(
        module.functions[0].signature.result.ty,
        Type::FunctionResult { .. }
    ));
}

#[test]
fn weak_receiver_and_method_variables_are_not_template_parameters() {
    for source in [
        "def make<T>() -> T = { 0 }; def main() -> int = { make().read() };",
        "def read<T>(value: T) -> _ = { value.read::<_>() };",
    ] {
        let error = compile(source).unwrap_err();
        assert!(error.to_string().contains("annotat"), "{source}: {error}");
    }
}
