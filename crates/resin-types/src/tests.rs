use super::*;
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
