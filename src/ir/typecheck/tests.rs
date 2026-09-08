use super::*;
use crate::ir::RecordField;

fn record(ty: Ty) -> Ty {
    Ty::Record {
        fields: vec![RecordField {
            name: "value".into(),
            ty,
        }],
    }
}

#[test]
fn nominal_records_keep_distinct_identities_across_contexts() {
    let mut context = TyperContext::new();
    let first = context.create_type("Meters", record(Ty::Int32)).unwrap();
    let table = context.into_definitions().unwrap();
    let allocation = table.as_ptr();
    let mut context = TyperContext::from_definitions(table);
    assert_eq!(context.definitions().as_ptr(), allocation);
    let second = context.create_type("Meters", record(Ty::Int32)).unwrap();
    assert_ne!(first, second);
    assert!(
        context
            .same(
                &Ty::Defined { definition: first },
                &Ty::Defined { definition: second }
            )
            .is_err()
    );
    assert_eq!(
        context.definition(first).unwrap().body(),
        Some(&record(Ty::Int32))
    );
    assert_eq!(
        context
            .define_type(first, record(Ty::Float64))
            .unwrap_err()
            .kind,
        TypeErrorKind::TypeAlreadyDefined { definition: first }
    );
}

#[test]
fn unfinished_records_cannot_be_used_or_exported() {
    let mut context = TyperContext::new();
    let definition = context.reserve_type("Pending");
    let ty = Ty::Defined { definition };
    let incomplete = TypeErrorKind::IncompleteTypeDefinition { definition };
    assert_eq!(context.body(&ty).unwrap_err().kind, incomplete);
    assert_eq!(
        context.ascribe(&record(Ty::Int32), &ty).unwrap_err().kind,
        incomplete
    );
    assert_eq!(context.as_record(&ty).unwrap_err().kind, incomplete);
    assert_eq!(
        context.clone().into_definitions().unwrap_err().kind,
        incomplete
    );
    context
        .define_type(definition, Ty::Record { fields: vec![] })
        .unwrap();
    context.into_definitions().unwrap();
}

#[test]
fn nominal_nonrecords_are_rejected_at_creation_and_verification() {
    for body in [
        Ty::Unit,
        Ty::Bool,
        Ty::Int32,
        Ty::Pointer {
            pointee: Box::new(Ty::Int32),
        },
        Ty::Function {
            param: Box::new(Ty::Unit),
            result: Box::new(Ty::Unit),
        },
        Ty::Span {
            element: Box::new(Ty::Int32),
        },
        Ty::Array {
            element: Box::new(Ty::Int32),
            length: 1,
        },
    ] {
        let mut context = TyperContext::new();
        let definition = TypeId::from_index(0);
        assert_eq!(
            context
                .create_type("Invalid", body.clone())
                .unwrap_err()
                .kind,
            TypeErrorKind::NominalTypeMustBeRecord { definition }
        );
        assert!(context.definitions().is_empty());
        let module = crate::ir::Module {
            types: vec![TypeDef::new("Invalid", body)].into(),
            ..Default::default()
        };
        assert_eq!(
            crate::ir::verify(&module).unwrap_err().kind,
            crate::ir::VerifyErrorKind::NominalTypeMustBeRecord { definition }
        );
    }
}

#[test]
fn inline_dependencies_require_completion_but_indirect_fields_can_be_pending() {
    let mut context = TyperContext::new();
    let first = context.reserve_type("First");
    let second = context.reserve_type("Second");
    let first_ty = Ty::Defined { definition: first };
    let second_ty = Ty::Defined { definition: second };
    assert_eq!(
        context
            .define_type(first, record(second_ty.clone()))
            .unwrap_err()
            .kind,
        TypeErrorKind::IncompleteTypeDefinition { definition: second }
    );
    context
        .define_type(
            second,
            record(Ty::Pointer {
                pointee: Box::new(first_ty),
            }),
        )
        .unwrap();
    context.define_type(first, record(second_ty)).unwrap();
    crate::ir::verify(&crate::ir::Module {
        types: context.into_definitions().unwrap(),
        ..Default::default()
    })
    .unwrap();
}

#[test]
fn invalid_references_and_layouts_leave_the_reservation_retryable() {
    let mut context = TyperContext::new();
    let definition = context.reserve_type("Node");
    let named = Ty::Defined { definition };
    let missing = TypeId::from_index(99);
    let invalid = Ty::Defined {
        definition: missing,
    };
    for field in [
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
            context
                .define_type(definition, record(field))
                .unwrap_err()
                .kind,
            TypeErrorKind::InvalidTypeDefinition {
                definition: missing
            }
        );
    }
    for field in [
        named.clone(),
        Ty::Array {
            element: Box::new(named.clone()),
            length: 1,
        },
    ] {
        assert_eq!(
            context
                .define_type(definition, record(field))
                .unwrap_err()
                .kind,
            TypeErrorKind::RecursiveTypeWithoutIndirection { definition }
        );
    }
    assert!(context.definition(definition).unwrap().body().is_none());
    context
        .define_type(
            definition,
            record(Ty::Pointer {
                pointee: Box::new(named),
            }),
        )
        .unwrap();
    assert!(context.into_definitions().is_ok());
}

#[test]
fn recursive_span_and_function_fields_have_finite_layouts() {
    for indirect in [0, 1] {
        let mut context = TyperContext::new();
        let definition = context.reserve_type("Node");
        let named = Ty::Defined { definition };
        let field = if indirect == 0 {
            Ty::Span {
                element: Box::new(named),
            }
        } else {
            Ty::Function {
                param: Box::new(named.clone()),
                result: Box::new(named),
            }
        };
        context.define_type(definition, record(field)).unwrap();
        crate::ir::verify(&crate::ir::Module {
            types: context.into_definitions().unwrap(),
            ..Default::default()
        })
        .unwrap();
    }
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
fn field_access_preserves_nominal_identity_and_autoderefs_pointers() {
    let mut typer = TyperContext::new();
    let body = record(Ty::Int32);
    let definition = typer.create_type("Node", body.clone()).unwrap();
    let node = Ty::Defined { definition };
    let pointer = Ty::Pointer {
        pointee: Box::new(body.clone()),
    };
    for (base, steps) in [
        (node.clone(), vec![Conv::Unwrap { definition }]),
        (pointer, vec![Conv::Deref]),
    ] {
        assert_eq!(
            typer.type_field(&base, "value").unwrap(),
            FieldAccess {
                ty: Ty::Int32,
                index: 0,
                steps
            }
        );
    }
    assert!(matches!(
        typer.type_assign(&node, &body),
        Err(TypeError {
            kind: TypeErrorKind::TypeMismatch { .. }
        })
    ));
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
fn convert_does_not_unwrap_function_arguments() {
    let mut typer = TyperContext::new();
    let meters = Ty::Defined {
        definition: typer.create_type("Meters", record(Ty::Int32)).unwrap(),
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
fn ascription_wraps_records_but_does_not_flatten_nested_fields() {
    let mut typer = TyperContext::new();
    let body = record(Ty::Int32);
    let meters = Ty::Defined {
        definition: typer.create_type("Meters", body.clone()).unwrap(),
    };
    let distance_body = record(meters.clone());
    let distance = Ty::Defined {
        definition: typer
            .create_type("Distance", distance_body.clone())
            .unwrap(),
    };
    assert_eq!(
        typer.type_ascription(&distance, &distance_body).unwrap(),
        distance
    );
    assert!(typer.type_ascription(&distance, &meters).is_err());
    assert!(typer.type_ascription(&meters, &Ty::Int32).is_err());
    assert!(typer.as_bool(&meters).is_err());
    assert!(typer.type_deref(&meters).is_err());
    assert!(typer.type_call(&meters, &Ty::Unit).is_err());
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

#[test]
fn explicit_conversions_preserve_ascription_and_pointer_boundaries() {
    let mut context = TyperContext::new();
    let first = context.create_type("First", record(Ty::Int32)).unwrap();
    let second = context.create_type("Second", record(Ty::Int32)).unwrap();
    let nominal = Ty::Defined { definition: first };
    let other = Ty::Defined { definition: second };
    let union = Ty::union([first, second]);
    let pointer = |ty| Ty::Pointer {
        pointee: Box::new(ty),
    };
    let span = Ty::Span {
        element: Box::new(Ty::Int32),
    };
    let span_record = span.span_record().unwrap();
    for (from, to, expected) in [
        (Ty::Int32, Ty::Int32, ExplicitConversion::Ascribe(vec![])),
        (
            pointer(Ty::Int32),
            pointer(Ty::Int32),
            ExplicitConversion::Ascribe(vec![]),
        ),
        (
            record(Ty::Int32),
            nominal.clone(),
            ExplicitConversion::Ascribe(vec![Conv::Wrap { definition: first }]),
        ),
        (
            nominal.clone(),
            record(Ty::Int32),
            ExplicitConversion::Ascribe(vec![Conv::Unwrap { definition: first }]),
        ),
        (
            span.clone(),
            span_record.clone(),
            ExplicitConversion::Ascribe(vec![Conv::SpanRecord]),
        ),
        (
            span_record,
            span,
            ExplicitConversion::Ascribe(vec![Conv::MakeSpan]),
        ),
        (Ty::Int32, Ty::Float64, ExplicitConversion::NumericCast),
        (nominal.clone(), union.clone(), ExplicitConversion::Widen),
        (
            pointer(nominal.clone()),
            pointer(union.clone()),
            ExplicitConversion::PointerCast,
        ),
        (
            pointer(Ty::Int32),
            Ty::UInt64,
            ExplicitConversion::PointerCast,
        ),
        (
            Ty::UInt64,
            pointer(Ty::Int32),
            ExplicitConversion::PointerCast,
        ),
    ] {
        assert_eq!(context.explicit_conversion(&from, &to).unwrap(), expected);
        if !matches!(expected, ExplicitConversion::Ascribe(_)) {
            assert!(context.ascribe(&from, &to).is_err());
        }
    }
    assert!(!pointer(nominal.clone()).widens_to(&pointer(union.clone())));
    for (from, to) in [
        (nominal, other),
        (union, Ty::Defined { definition: first }),
        (pointer(Ty::Int32), Ty::Int64),
        (Ty::Int32, pointer(Ty::Int32)),
        (Ty::Unit, Ty::Record { fields: vec![] }),
    ] {
        assert!(context.explicit_conversion(&from, &to).is_err());
    }
}
