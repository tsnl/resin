use resin_hir::Type;

fn compile(source: &str) -> Result<resin_hir::Module, resin_hir::GenerateError> {
    let document = resin_cst::Document::reparse(source.into(), None);
    resin_hir::generate(&resin_ast::generate(&document).unwrap())
}

#[test]
fn nominal_definitions_keep_one_body_with_rigid_field_binders() {
    let module = compile("struct Pair<T> { first: T, second: T }; def small(value: Pair<int>) -> Pair<int> = { value }; def large(value: Pair<ulong>) -> Pair<ulong> = { value };").unwrap();
    let pairs = module
        .types
        .iter()
        .filter(|definition| definition.name.as_ref() == "Pair")
        .collect::<Vec<_>>();
    assert_eq!(pairs.len(), 1);
    let pair = pairs[0];
    assert_eq!(pair.type_params.len(), 1);
    let Type::Record { fields } = &pair.body else {
        panic!("record")
    };
    for field in fields {
        assert_eq!(
            field.ty,
            Type::Parameter {
                parameter: pair.type_params[0].id
            }
        );
    }
    let Type::Defined {
        definition: first,
        arguments: first_args,
    } = &module.functions[0].signature.result.ty
    else {
        panic!("nominal")
    };
    let Type::Defined {
        definition: second,
        arguments: second_args,
    } = &module.functions[1].signature.result.ty
    else {
        panic!("nominal")
    };
    assert_eq!(first, second);
    assert_eq!(first_args, &[Type::Int32]);
    assert_eq!(second_args, &[Type::UInt64]);
}

#[test]
fn nominal_arguments_deduce_function_types_without_equating_binders() {
    let module = compile("struct Cell<T> { value: T }; def read<T>(cell: Cell<T>) -> T = { cell.value }; def main() -> int = { read(Cell<int> { value = 42 }) };").unwrap();
    let cell = module
        .types
        .iter()
        .find(|definition| definition.name.as_ref() == "Cell")
        .unwrap();
    let read = &module.functions[0];
    assert_ne!(cell.type_params[0].id, read.signature.type_params[0].id);
    assert_eq!(
        read.signature.result.ty,
        Type::Parameter {
            parameter: read.signature.type_params[0].id
        }
    );
    assert_eq!(module.functions[1].signature.result.ty, Type::Int32);
}

#[test]
fn nominal_names_and_arguments_remain_distinct_during_inference() {
    for source in [
        "struct Cell<T> { value: T }; def take(cell: Cell<int>) = {}; def main() = { take(Cell<ulong> { value = 42 }); };",
        "struct Left<T> { value: T }; struct Right<T> { value: T }; def take(value: Left<int>) = {}; def main() = { take(Right<int> { value = 42 }); };",
        "struct Cell<T> { value: T }; def main() = { var cell = Cell<int> { value = 42 }; cell := Cell<ulong> { value = 42 }; };",
    ] {
        assert!(compile(source).is_err(), "{source}");
    }
}

#[test]
fn generic_struct_declaration_and_application_errors_are_source_errors() {
    for source in [
        "struct Pair<T, T> { value: T };",
        "struct Pair<T> { value: _ };",
        "struct Pair<T> { value: Missing };",
        "struct Pair<T> { value: T }; def take(value: Pair) = {};",
        "struct Pair<T> { value: T }; def take(value: Pair<int, int>) = {};",
        "struct Pair<T> { value: T }; def main() = { Pair<int> { missing = 42 }; };",
        "struct Pair<T> { first: T, second: T }; def main() = { Pair<int> { first = 42 }; };",
        "struct Pair<T> { value: T }; def main() = { Pair<int> { value = 1 == 1 }; };",
    ] {
        assert!(compile(source).is_err(), "{source}");
    }
}

#[test]
fn nested_generic_aliases_expand_to_the_original_nominal_declaration() {
    let module = compile("struct Cell<T> { value: T }; type Renamed<U> = Cell<U>; type Nested<V> = Renamed<Ptr<V>>; def take(value: Nested<int>) -> Cell<Ptr<int>> = { value };").unwrap();
    let function = &module.functions[0];
    assert_eq!(
        function.signature.params[0].annotation.ty,
        function.signature.result.ty
    );
    let Type::Defined { arguments, .. } = &function.signature.result.ty else {
        panic!("nominal")
    };
    assert_eq!(
        arguments,
        &[Type::Pointer {
            pointee: Box::new(Type::Int32)
        }]
    );
}

#[test]
fn local_nominal_results_carry_captures_as_well_as_explicit_arguments() {
    let module = compile("def pair<T>(value: T) -> _ = { struct Local<U> { outer: T, inner: U }; Local<int> { outer = value, inner = 35 } }; def small() -> _ = { pair(7_i) }; def large() -> _ = { pair(4294967296_ul) };").unwrap();
    let pair = &module.functions[0];
    let local = module
        .types
        .iter()
        .find(|definition| definition.name.as_ref() == "Local")
        .unwrap();
    assert_eq!(local.type_params.len(), 2);
    let Type::Defined { arguments, .. } = &pair.signature.result.ty else {
        panic!("local nominal")
    };
    assert_eq!(arguments.len(), 2);
    assert!(arguments.contains(&Type::Parameter {
        parameter: pair.signature.type_params[0].id
    }));
    assert!(arguments.contains(&Type::Int32));
    let Type::Defined {
        definition: small,
        arguments: small_args,
    } = &module.functions[1].signature.result.ty
    else {
        panic!("small nominal")
    };
    let Type::Defined {
        definition: large,
        arguments: large_args,
    } = &module.functions[2].signature.result.ty
    else {
        panic!("large nominal")
    };
    assert_eq!(small, large);
    assert_ne!(small_args, large_args);
    assert!(large_args.contains(&Type::UInt64));
}

#[test]
fn generic_foreign_values_fail_without_panicking() {
    for source in [
        "struct Cell<T> { value: T }; extern \"native.h\" def native(value: Cell<int>);",
        "struct Cell<T> { value: T }; extern \"native.h\" def native() -> Cell<int>;",
    ] {
        let error = compile(source).unwrap_err();
        assert!(error.to_string().contains("Foreign"), "{error}");
    }
}

#[test]
fn shader_interface_expansion_bounds_duplicated_generic_fields() {
    let mut ty = "float32".to_owned();
    for _ in 0..16 {
        ty = format!("Pair<{ty}>");
    }
    let source = format!(
        "struct Pair<T> {{ left: T, right: T }}; @fragment_shader def fragment(color: {ty}) -> {ty} = {{ color }};"
    );
    let error = compile(&source).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("shader interface expansion exceeds the HIR size limit"),
        "{error}"
    );
}

#[test]
fn shader_interface_expansion_counts_record_depth() {
    let mut source = "struct Layer0 { value: float32 };".to_owned();
    for layer in 1..=128 {
        source.push_str(&format!(
            "struct Layer{layer} {{ value: Layer{} }};",
            layer - 1
        ));
    }
    source.push_str("@fragment_shader def fragment(color: Layer128) -> Layer128 = { color };");
    let error = compile(&source).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("shader interface expansion exceeds the HIR depth limit"),
        "{error}"
    );
}
