use resin_types::{RecordField, Ty, TypeDef, TypeId, TyperContext, shader};

#[test]
fn arithmetic_schemes_and_concrete_profiles_have_distinct_admission_rules() {
    let typer = TyperContext::new();
    let record = Ty::Record { fields: vec![] };
    assert!(
        typer
            .type_builtin_call("+", &[record.clone(), record.clone()])
            .is_ok()
    );
    assert!(
        typer
            .builtin_instance("+", &[record.clone(), record])
            .is_err()
    );
    assert!(
        typer
            .builtin_instance("/", &[Ty::UInt32, Ty::UInt32])
            .is_ok()
    );
    assert!(
        shader::builtin_instance(&typer, "/", &[Ty::UInt32, Ty::UInt32])
            .unwrap_err()
            .contains("unsupported shader builtin")
    );
    assert!(shader::builtin_instance(&typer, "/", &[Ty::Float32, Ty::Float32]).is_ok());
    assert!(shader::builtin_instance(&typer, "+", &[Ty::UInt8, Ty::UInt8]).is_ok());
    assert!(
        shader::builtin_instance(&typer, "+", &[Ty::Float64, Ty::Float64])
            .unwrap_err()
            .contains("does not support type")
    );
    assert!(shader::builtin_instance(&typer, "+", &[]).is_err());
}

#[test]
fn managed_fields_are_opaque_and_do_not_require_shader_payload_types() {
    let definitions = vec![TypeDef::new(
        "Root",
        Ty::Record {
            fields: vec![
                RecordField {
                    name: "owner".into(),
                    ty: Ty::ArcPtr {
                        pointee: Box::new(Ty::Float64),
                    },
                },
                RecordField {
                    name: "value".into(),
                    ty: Ty::UInt32,
                },
            ],
        },
    )];
    assert!(
        shader::value_type(
            &definitions,
            &Ty::Pointer {
                pointee: Box::new(Ty::Defined {
                    definition: TypeId::from_index(0)
                })
            }
        )
        .is_ok()
    );
    assert!(
        shader::value_type(
            &[],
            &Ty::Pointer {
                pointee: Box::new(Ty::Bool)
            }
        )
        .unwrap_err()
        .contains("no shared host/device layout")
    );
    assert!(
        shader::value_type(
            &[],
            &Ty::Array {
                element: Box::new(Ty::UInt8),
                length: 0
            }
        )
        .unwrap_err()
        .contains("must not be empty")
    );
}
