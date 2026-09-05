use super::*;
use crate::ir::RecordField;

#[test]
fn context_creates_distinct_nominal_identities_without_a_module() {
    let mut context = TyperContext::default();
    assert!(context.definitions().is_empty());
    let first = context.create_type("Meters", Ty::Int32).unwrap();
    let second = context.create_type("Meters", Ty::Int32).unwrap();
    assert_ne!(first, second);
    assert_eq!(first.index(), 0);
    assert_eq!(second.index(), 1);
    assert_eq!(context.definitions().len(), 2);
    assert_eq!(context.definition(first).unwrap().body(), Some(&Ty::Int32));
    assert!(
        context
            .same(
                &Ty::Defined { definition: first },
                &Ty::Defined { definition: second },
            )
            .is_err()
    );
}

#[test]
fn unfinished_definitions_are_not_unit_and_cannot_be_exported() {
    let mut context = TyperContext::new();
    let definition = context.reserve_type("Pending");
    let ty = Ty::Defined { definition };
    let incomplete = TypeErrorKind::IncompleteTypeDefinition { definition };
    assert!(context.definition(definition).unwrap().body().is_none());
    assert_eq!(context.body(&ty).unwrap_err().kind, incomplete);
    assert_eq!(
        context.ascribe(&Ty::Unit, &ty).unwrap_err().kind,
        incomplete
    );
    assert_eq!(context.as_bool(&ty).unwrap_err().kind, incomplete);
    assert_eq!(context.as_record(&ty).unwrap_err().kind, incomplete);
    assert_eq!(
        context.clone().into_definitions().unwrap_err().kind,
        incomplete
    );

    context.define_type(definition, Ty::Unit).unwrap();
    assert_eq!(context.body(&ty).unwrap(), Ty::Unit);
    assert_eq!(
        context.into_definitions().unwrap()[0].body(),
        Some(&Ty::Unit)
    );
}

#[test]
fn completed_definitions_cannot_be_redefined_even_after_export_and_import() {
    let mut context = TyperContext::new();
    let definition = context.create_type("Value", Ty::Int32).unwrap();
    for body in [Ty::Int32, Ty::Float64] {
        assert_eq!(
            context.define_type(definition, body).unwrap_err().kind,
            TypeErrorKind::TypeAlreadyDefined { definition }
        );
    }
    let mut context = TyperContext::from_definitions(context.into_definitions().unwrap());
    assert_eq!(
        context.define_type(definition, Ty::Unit).unwrap_err().kind,
        TypeErrorKind::TypeAlreadyDefined { definition }
    );
    assert_eq!(
        context.definition(definition).unwrap().body(),
        Some(&Ty::Int32)
    );
}

#[test]
fn inline_dependencies_must_be_complete_but_pointer_dependencies_can_be_pending() {
    let mut context = TyperContext::new();
    let first = context.reserve_type("First");
    let second = context.reserve_type("Second");
    let second_ty = Ty::Defined { definition: second };
    assert_eq!(
        context
            .define_type(first, second_ty.clone())
            .unwrap_err()
            .kind,
        TypeErrorKind::IncompleteTypeDefinition { definition: second }
    );
    assert!(context.definition(first).unwrap().body().is_none());
    context
        .define_type(
            second,
            Ty::Pointer {
                pointee: Box::new(Ty::Defined { definition: first }),
            },
        )
        .unwrap();
    context.define_type(first, second_ty).unwrap();
    let module = crate::ir::Module {
        types: context.into_definitions().unwrap(),
        ..Default::default()
    };
    crate::ir::verify(&module).unwrap();
}

#[test]
fn invalid_references_do_not_initialize_a_reserved_body() {
    let mut context = TyperContext::new();
    let definition = context.reserve_type("Value");
    let missing = TypeId::from_index(99);
    assert_eq!(
        context
            .define_type(
                definition,
                Ty::Pointer {
                    pointee: Box::new(Ty::Defined {
                        definition: missing
                    }),
                }
            )
            .unwrap_err()
            .kind,
        TypeErrorKind::InvalidTypeDefinition {
            definition: missing
        }
    );
    assert!(context.definition(definition).unwrap().body().is_none());
    context.define_type(definition, Ty::Int32).unwrap();
}

#[test]
fn recursive_definitions_can_be_reserved_and_completed_in_the_context() {
    let mut context = TyperContext::new();
    let definition = context.reserve_type("Node");
    let node = Ty::Defined { definition };
    let pointer = Ty::Pointer {
        pointee: Box::new(node.clone()),
    };
    let body = Ty::Record {
        fields: vec![
            RecordField {
                name: "value".into(),
                ty: Ty::Int32,
            },
            RecordField {
                name: "next".into(),
                ty: pointer.clone(),
            },
        ],
    };
    context.define_type(definition, body.clone()).unwrap();
    assert_eq!(context.body(&node).unwrap(), body);
    assert_eq!(context.type_field(&node, "next").unwrap().ty, pointer);

    let span_id = context.reserve_type("Span");
    let span = Ty::Defined {
        definition: span_id,
    };
    context
        .define_type(
            span_id,
            Ty::Span {
                element: Box::new(span),
            },
        )
        .unwrap();

    let function_id = context.reserve_type("Function");
    let function = Ty::Defined {
        definition: function_id,
    };
    context
        .define_type(
            function_id,
            Ty::Function {
                param: Box::new(function.clone()),
                result: Box::new(function),
            },
        )
        .unwrap();
    assert_eq!(context.into_definitions().unwrap().len(), 3);
}

#[test]
fn invalid_recursive_layouts_leave_the_reservation_retryable() {
    let mut context = TyperContext::new();
    let definition = context.reserve_type("Value");
    let value = Ty::Defined { definition };
    for body in [
        value.clone(),
        Ty::Array {
            element: Box::new(value.clone()),
            length: 1,
        },
        Ty::Record {
            fields: vec![RecordField {
                name: "self".into(),
                ty: value.clone(),
            }],
        },
    ] {
        assert_eq!(
            context.define_type(definition, body).unwrap_err().kind,
            TypeErrorKind::RecursiveTypeWithoutIndirection { definition }
        );
        assert!(context.definition(definition).unwrap().body().is_none());
    }

    context.define_type(definition, Ty::Int32).unwrap();
    assert_eq!(context.body(&value).unwrap(), Ty::Int32);
}

#[test]
fn bad_references_do_not_leave_a_partially_created_definition() {
    let mut context = TyperContext::new();
    let missing = TypeId::from_index(99);
    let invalid = Ty::Defined {
        definition: missing,
    };
    for body in [
        invalid.clone(),
        Ty::Pointer {
            pointee: Box::new(invalid.clone()),
        },
        Ty::Span {
            element: Box::new(invalid.clone()),
        },
        Ty::Function {
            param: Box::new(Ty::Unit),
            result: Box::new(invalid),
        },
    ] {
        assert_eq!(
            context.create_type("Invalid", body).unwrap_err().kind,
            TypeErrorKind::InvalidTypeDefinition {
                definition: missing
            }
        );
        assert!(context.definitions().is_empty());
    }
    assert_eq!(
        context.define_type(missing, Ty::Int32).unwrap_err().kind,
        TypeErrorKind::InvalidTypeDefinition {
            definition: missing
        }
    );
    assert_eq!(context.create_type("Valid", Ty::Int32).unwrap().index(), 0);
}

#[test]
fn definition_tables_move_between_checking_passes_without_changing_ids() {
    let mut context = TyperContext::new();
    let definition = context.create_type("Meters", Ty::Int32).unwrap();
    let table = context.into_definitions().unwrap();
    let allocation = table.as_ptr();
    let mut context = TyperContext::from_definitions(table);
    assert_eq!(context.definitions().as_ptr(), allocation);
    let next = context.create_type("Seconds", Ty::Float64).unwrap();
    assert_eq!(next.index(), 1);
    let meters = Ty::Defined { definition };
    assert_eq!(
        context.type_ascription(&meters, &Ty::Int32).unwrap(),
        meters
    );
    assert_eq!(
        context.definition(definition).unwrap().name.as_ref(),
        "Meters"
    );
}

#[test]
fn numeric_literals_keep_their_default_types() {
    let typer = TyperContext::new();
    for literal in ["0", "123", "-123", "0x1e", "0X10", "-0x1e", "-0XFE"] {
        assert_eq!(typer.type_num(literal), Ty::Int32, "{literal}");
    }
    for literal in ["1.5", "-1.5", "1e3", "-1E3", "1e-3"] {
        assert_eq!(typer.type_num(literal), Ty::Float64, "{literal}");
    }
}

#[test]
fn type_terms_inhabit_the_type_universe() {
    assert_eq!(TyperContext::new().type_type(&Ty::Int32), Ty::Type);
}

#[test]
fn empty_arrays_need_an_injected_element_type() {
    let typer = TyperContext::new();

    assert_eq!(
        typer.type_array(&[]).unwrap_err().kind,
        TypeErrorKind::EmptyArrayNeedsElementType
    );
    assert_eq!(
        typer.type_array_of(&Ty::UInt8, &[]).unwrap(),
        Ty::Array {
            element: Box::new(Ty::UInt8),
            length: 0,
        }
    );
}

#[test]
fn nominal_records_expose_fields_without_losing_identity() {
    let mut typer = TyperContext::new();
    let body = Ty::Record {
        fields: vec![RecordField {
            name: "value".into(),
            ty: Ty::Int32,
        }],
    };
    let definition = typer.create_type("Node", body.clone()).unwrap();
    let node = Ty::Defined { definition };

    assert_eq!(
        typer.type_field(&node, "value").unwrap(),
        FieldAccess {
            ty: Ty::Int32,
            index: 0,
            steps: vec![Conv::Unwrap { definition }],
        }
    );
    assert!(matches!(
        typer.type_assign(&node, &body),
        Err(TypeError {
            kind: TypeErrorKind::TypeMismatch { .. }
        })
    ));
}

#[test]
fn pointer_dereference_is_explicit() {
    let typer = TyperContext::new();
    let pointer = Ty::Pointer {
        pointee: Box::new(Ty::Int64),
    };

    assert_eq!(typer.type_deref(&pointer).unwrap(), Ty::Int64);
    assert!(matches!(
        typer.type_field(&pointer, "value"),
        Err(TypeError {
            kind: TypeErrorKind::ExpectedRecord { .. }
        })
    ));
}

#[test]
fn field_access_autoderefs_pointers() {
    let record = Ty::Record {
        fields: vec![RecordField {
            name: "x".into(),
            ty: Ty::Int32,
        }],
    };
    let pointer = Ty::Pointer {
        pointee: Box::new(record),
    };
    let access = TyperContext::new().type_field(&pointer, "x").unwrap();
    assert_eq!(access.ty, Ty::Int32);
    assert_eq!(access.steps, vec![Conv::Deref]);
}

#[test]
fn field_access_stops_at_recursive_pointers() {
    let mut typer = TyperContext::new();
    let definition = typer.reserve_type("Loop");
    let recursive = Ty::Defined { definition };
    typer
        .define_type(
            definition,
            Ty::Pointer {
                pointee: Box::new(recursive.clone()),
            },
        )
        .unwrap();
    assert_eq!(
        typer.type_field(&recursive, "value").unwrap_err().kind,
        TypeErrorKind::ExpectedRecord { found: recursive }
    );
}

#[test]
fn child_types_compose_into_a_function_call() {
    let typer = TyperContext::new();
    let function = typer.type_function(&Ty::Int32, &Ty::Float64);

    assert_eq!(typer.type_call(&function, &Ty::Int32).unwrap(), Ty::Float64);
}

#[test]
fn convert_does_not_unwrap_function_arguments() {
    let mut typer = TyperContext::new();
    let meters = Ty::Defined {
        definition: typer.create_type("Meters", Ty::Int32).unwrap(),
    };
    let callee = Ty::Function {
        param: Box::new(Ty::Int32),
        result: Box::new(Ty::Int32),
    };
    assert!(matches!(
        typer.type_call(&callee, &meters),
        Err(TypeError {
            kind: TypeErrorKind::TypeMismatch { .. }
        })
    ));
}

#[test]
fn ascription_moves_one_nominal_layer() {
    let mut typer = TyperContext::new();
    let meters = Ty::Defined {
        definition: typer.create_type("Meters", Ty::Int32).unwrap(),
    };
    let distance = Ty::Defined {
        definition: typer.create_type("Distance", meters.clone()).unwrap(),
    };

    assert_eq!(typer.type_ascription(&meters, &Ty::Int32).unwrap(), meters);
    assert_eq!(
        typer.type_ascription(&Ty::Int32, &meters).unwrap(),
        Ty::Int32
    );
    assert_eq!(typer.type_ascription(&distance, &meters).unwrap(), distance);
    assert_eq!(typer.type_ascription(&meters, &distance).unwrap(), meters);
    assert!(matches!(
        typer.type_ascription(&distance, &Ty::Int32),
        Err(TypeError {
            kind: TypeErrorKind::TypeMismatch { .. }
        })
    ));
    assert!(matches!(
        typer.type_ascription(&Ty::Int32, &distance),
        Err(TypeError {
            kind: TypeErrorKind::TypeMismatch { .. }
        })
    ));
}

#[test]
fn builtin_signatures_follow_the_operator() {
    let typer = TyperContext::new();
    for (names, args, result) in [
        (&["+", "-", "~"][..], vec![Ty::Int32], Ty::Int32),
        (
            &["+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"][..],
            vec![Ty::Int32, Ty::Int32],
            Ty::Int32,
        ),
        (
            &["==", "!=", "<", "<=", ">", ">="][..],
            vec![Ty::Int32, Ty::Int32],
            Ty::Bool,
        ),
        (&["!"][..], vec![Ty::Bool], Ty::Bool),
        (&["&&", "||"][..], vec![Ty::Bool, Ty::Bool], Ty::Bool),
    ] {
        for name in names {
            assert_eq!(
                typer.type_builtin_call(name, &args).unwrap(),
                BuiltinCall {
                    params: args.clone(),
                    result: result.clone(),
                },
                "{name}({args:?})"
            );
        }
    }
}

#[test]
fn builtin_arithmetic_has_no_type_trait_constraint() {
    let typer = TyperContext::new();
    let unusual_operand = Ty::Record {
        fields: vec![RecordField {
            name: "value".into(),
            ty: Ty::Int32,
        }],
    };

    let call = typer
        .type_builtin_call("+", &[unusual_operand.clone(), unusual_operand.clone()])
        .unwrap();

    assert_eq!(
        call.params,
        vec![unusual_operand.clone(), unusual_operand.clone()]
    );
    assert_eq!(call.result, unusual_operand);
}

#[test]
fn builtin_errors_distinguish_arity_names_and_types() {
    let typer = TyperContext::new();
    for (name, args) in [
        ("+", vec![]),
        ("*", vec![Ty::Bool]),
        ("!", vec![Ty::Int32, Ty::Int32]),
        ("&&", vec![Ty::Int32; 3]),
    ] {
        assert_eq!(
            typer.type_builtin_call(name, &args).unwrap_err().kind,
            TypeErrorKind::InvalidBuiltinArgumentCount {
                name: name.into(),
                found: args.len(),
            }
        );
    }
    assert_eq!(
        typer.type_builtin_call("unknown", &[]).unwrap_err().kind,
        TypeErrorKind::UnknownBuiltin {
            name: "unknown".into(),
        }
    );
    assert_eq!(
        typer
            .type_builtin_call("+", &[Ty::Int32, Ty::Float64])
            .unwrap_err()
            .kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Int32,
            found: Ty::Float64,
        }
    );
    for (name, args) in [
        ("!", vec![Ty::Int32]),
        ("&&", vec![Ty::Bool, Ty::Int32]),
        ("||", vec![Ty::Int32, Ty::Bool]),
    ] {
        assert_eq!(
            typer.type_builtin_call(name, &args).unwrap_err().kind,
            TypeErrorKind::ExpectedBoolean { found: Ty::Int32 }
        );
    }
}
