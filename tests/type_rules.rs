use resin_common::types::{RecordField, Ty, TypeDef, TypeErrorKind, TypeId, TyperContext};
fn record(ty: Ty) -> Ty {
    Ty::Record {
        fields: vec![RecordField {
            name: "value".into(),
            ty,
        }],
    }
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
        let module = resin_lir::Module {
            types: vec![TypeDef::new("Invalid", body)].into(),
            ..Default::default()
        };
        assert_eq!(
            resin_lir_verifier::verify(&module).unwrap_err().kind,
            resin_lir_verifier::VerifyErrorKind::NominalTypeMustBeRecord { definition }
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
    resin_lir_verifier::verify(&resin_lir::Module {
        types: context.into_definitions().unwrap(),
        ..Default::default()
    })
    .unwrap();
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
        resin_lir_verifier::verify(&resin_lir::Module {
            types: context.into_definitions().unwrap(),
            ..Default::default()
        })
        .unwrap();
    }
}
