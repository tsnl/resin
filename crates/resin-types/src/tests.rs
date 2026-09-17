use super::*;

#[test]
fn constant_integer_operations_check_every_width() {
    for (ty, max, min) in [
        (Ty::Int8, "127", "-128"),
        (Ty::UInt8, "255", "0"),
        (Ty::Int16, "32767", "-32768"),
        (Ty::UInt16, "65535", "0"),
        (Ty::Int32, "2147483647", "-2147483648"),
        (Ty::UInt32, "4294967295", "0"),
        (Ty::Int64, "9223372036854775807", "-9223372036854775808"),
        (Ty::UInt64, "18446744073709551615", "0"),
    ] {
        let max = literal::parse(max, &ty).unwrap();
        let min = literal::parse(min, &ty).unwrap();
        let one = literal::parse("1", &ty).unwrap();
        let zero = literal::parse("0", &ty).unwrap();
        assert!(
            constant_operation("+", &[max.clone(), one.clone()]).is_err(),
            "{ty:?}"
        );
        assert!(
            constant_operation("-", &[min, one.clone()]).is_err(),
            "{ty:?}"
        );
        assert!(
            constant_operation("/", &[one, zero.clone()]).is_err(),
            "{ty:?}"
        );
        assert_eq!(constant_operation("^", &[max.clone(), max]).unwrap(), zero);
    }
}

#[test]
fn constant_conversions_preserve_integer_float_rounding_and_check_ranges() {
    let value = Value::UInt64 {
        value: (1_u64 << 63) + (1_u64 << 39) + 1,
    };
    assert_eq!(
        convert_constant(&value, &Ty::Float32).unwrap(),
        Value::Float32 {
            value: ((1_u64 << 63) + (1_u64 << 39) + 1) as f32
        }
    );
    assert!(
        convert_constant(
            &Value::Float64 {
                value: 2_f64.powi(64)
            },
            &Ty::UInt64
        )
        .is_err()
    );
    assert!(convert_constant(&Value::Int32 { value: -1 }, &Ty::UInt8).is_err());
    assert_eq!(
        convert_constant(&Value::Float64 { value: -1.9 }, &Ty::Int8).unwrap(),
        Value::Int8 { value: -1 }
    );
}

#[test]
fn value_layout_preserves_placeholders_and_checks_invalid_types() {
    for ty in [Ty::Unit, Ty::None, Ty::Bool, Ty::Record { fields: vec![] }] {
        let layout = layout::value(&[], &ty).unwrap();
        assert_eq!((layout.size, layout.align), (1, 1));
    }
    assert_eq!(
        layout::value(
            &[],
            &Ty::Array {
                element: Box::new(Ty::UInt32),
                length: 0
            }
        )
        .unwrap()
        .size,
        4
    );
    assert!(
        layout::value(
            &[],
            &Ty::Array {
                element: Box::new(Ty::UInt64),
                length: usize::MAX
            }
        )
        .is_err()
    );
    assert!(
        layout::value(
            &[],
            &Ty::Foreign {
                name: "Opaque".into()
            }
        )
        .is_err()
    );
    let recursive = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    assert!(layout::value(&[TypeDef::new("Loop", recursive.clone())], &recursive).is_err());
    assert!(
        layout::layout(&[], &Ty::Float64).is_err(),
        "shared GPU layout stays restricted"
    );
}
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
        Ty::pointer_length(invalid.clone()),
        Ty::Function {
            params: vec![Ty::Unit],
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
    let span_record = Ty::pointer_length(Ty::Int32);
    let span_id = context.create_type("View", span_record.clone()).unwrap();
    let span = Ty::Defined {
        definition: span_id,
    };
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
            ExplicitConversion::Ascribe(vec![Conv::Unwrap {
                definition: span_id,
            }]),
        ),
        (
            span_record,
            span,
            ExplicitConversion::Ascribe(vec![Conv::Wrap {
                definition: span_id,
            }]),
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

#[test]
fn string_views_expose_bytes_without_accepting_arbitrary_storage() {
    let context = TyperContext::new();
    let pointer = Ty::Pointer {
        pointee: Box::new(Ty::UInt8),
    };
    assert_eq!(context.type_field(&Ty::Str, "data").unwrap().ty, pointer);
    assert_eq!(
        context.type_field(&Ty::Str, "length").unwrap().ty,
        Ty::UInt64
    );
    assert_eq!(
        context
            .ascribe(&Ty::Str, &Ty::Str.view_record().unwrap())
            .unwrap(),
        vec![Conv::ViewRecord]
    );
    assert!(context.ascribe(&Ty::Str, &Ty::byte_span()).is_err());
    assert!(!Ty::Str.widens_to(&Ty::byte_span()));
    assert!(!Ty::byte_span().widens_to(&Ty::Str));
    assert!(context.ascribe(&Ty::byte_span(), &Ty::Str).is_err());
    assert!(
        context
            .ascribe(&Ty::Str.view_record().unwrap(), &Ty::Str)
            .is_err()
    );
}

#[test]
fn opaque_gpu_values_preserve_ownership_without_exposing_pointer_operations() {
    let context = TyperContext::new();
    for gpu in [Ty::GpuView, Ty::GpuArguments, Ty::GpuPipelineContract] {
        assert!(gpu.needs_drop(&[]));
        assert!(record(gpu.clone()).needs_drop(&[]));
        assert!(!gpu.foreign_value());
        assert!(layout::layout(&[], &gpu).is_err());
        assert!(gpu.deref_target().is_none());
        assert!(gpu.view_record().is_none());
        assert!(context.type_field(&gpu, "data").is_err());
        for other in [
            Ty::UInt64,
            Ty::StrongOwner,
            Ty::Pointer {
                pointee: Box::new(Ty::UInt32),
            },
        ] {
            assert!(!gpu.pointer_cast(&other));
            assert!(!other.pointer_cast(&gpu));
            assert!(context.explicit_conversion(&gpu, &other).is_err());
            assert!(context.explicit_conversion(&other, &gpu).is_err());
        }
    }
}

#[test]
fn gpu_element_storage_excludes_references_and_custom_destruction() {
    let mut context = TyperContext::new();
    let id = context.create_type("Element", record(Ty::UInt32)).unwrap();
    let array = Ty::Array {
        element: Box::new(Ty::Defined { definition: id }),
        length: 8,
    };
    assert!(array.gpu_element(context.definitions()));
    context.define_drop(id, FunctionId::from_index(0));
    assert!(!array.gpu_element(context.definitions()));
    for ty in [
        Ty::Pointer {
            pointee: Box::new(Ty::UInt32),
        },
        Ty::pointer_length(Ty::UInt32),
        Ty::GpuView,
        Ty::GpuPipelineContract,
        Ty::GpuArguments,
        Ty::StrongOwner,
        Ty::WeakOwner,
        Ty::Array {
            element: Box::new(Ty::UInt32),
            length: 0,
        },
        Ty::Bool,
        Ty::Str,
    ] {
        assert!(!ty.gpu_element(&[]));
        assert!(!record(ty).gpu_element(&[]));
    }
}

#[test]
fn gpu_primitive_types_intern_and_render_independently_of_source_wrappers() {
    let mut table = TypeTable::default();
    for (ty, name) in [
        (Ty::GpuView, "GpuView"),
        (Ty::GpuArguments, "GpuArguments"),
        (Ty::GpuPipelineContract, "GpuPipelineContract"),
    ] {
        table.intern(&ty);
        assert!(table.id(&ty).is_some());
        assert_eq!(format_type(&ty, &table), name);
    }
}

#[test]
fn gpu_projection_requires_explicit_nominal_metadata_and_preserves_element_types() {
    let pointer = Ty::Pointer {
        pointee: Box::new(Ty::UInt32),
    };
    let view = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    let mut table = [TypeDef::new("DeviceReference", record(Ty::GpuView))];
    assert!(crate::gpu_projection_plan(&table, &view, &pointer).is_err());
    let TypeDef::Nominal { gpu_projection, .. } = &mut table[0] else {
        unreachable!()
    };
    *gpu_projection = Some(crate::GpuProjection {
        kind: crate::GpuProjectionKind::Pointer,
        target: pointer.clone(),
    });
    let plan = crate::gpu_projection_plan(&table, &view, &pointer).unwrap();
    assert_eq!(plan.source, view);
    assert_eq!(plan.target, pointer);
    assert_eq!(
        plan.operation,
        crate::GpuProjectionOperation::Pointer {
            element: Ty::UInt32
        }
    );
    let retagged = Ty::Pointer {
        pointee: Box::new(Ty::Float32),
    };
    assert!(crate::gpu_projection_plan(&table, &view, &retagged).is_err());
    assert!(crate::gpu_projection_plan(&table, &pointer, &pointer).is_err());
    assert!(crate::gpu_projection_plan(&table, &Ty::GpuArguments, &pointer).is_err());
    let TypeDef::Nominal { drop, .. } = &mut table[0] else {
        unreachable!()
    };
    *drop = Some(FunctionId::from_index(0));
    assert!(crate::gpu_projection_plan(&table, &view, &pointer).is_err());
}

#[test]
fn source_pipeline_contract_preserves_stage_root_and_shared_owner() {
    let root = record(Ty::UInt32);
    let owner = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    let pipeline = Ty::Defined {
        definition: TypeId::from_index(1),
    };
    let mut table = [
        TypeDef::new("DeviceOwner", record(Ty::StrongOwner)),
        TypeDef::new("Program", record(Ty::GpuPipelineContract)),
    ];
    assert!(crate::gpu_pipeline_contract(&table, &pipeline).is_err());
    for kind in [
        crate::GpuPipelineKind::Compute,
        crate::GpuPipelineKind::Graphics,
    ] {
        let expected = crate::GpuPipeline {
            kind,
            root: root.clone(),
            owner: owner.clone(),
        };
        let TypeDef::Nominal { gpu_pipeline, .. } = &mut table[1] else {
            unreachable!()
        };
        *gpu_pipeline = Some(expected.clone());
        assert_eq!(
            crate::gpu_pipeline_contract(&table, &pipeline).unwrap(),
            &expected
        );
        assert!(pipeline.needs_drop(&table));
        assert!(!pipeline.gpu_element(&table));
    }
    let TypeDef::Nominal { body, .. } = &mut table[0] else {
        unreachable!()
    };
    *body = Some(record(Ty::UInt64));
    assert!(crate::gpu_pipeline_contract(&table, &pipeline).is_err());
}

fn shader_parameter(input: Ty, root: Ty) -> Vec<Ty> {
    vec![
        input,
        Ty::Pointer {
            pointee: Box::new(root),
        },
    ]
}

fn shader_graphics_types(context: &mut TyperContext) -> (Ty, Ty, Ty) {
    let vector = |names: &[&str]| Ty::Record {
        fields: names
            .iter()
            .map(|name| RecordField {
                name: (*name).into(),
                ty: Ty::Float32,
            })
            .collect(),
    };
    let color = Ty::Defined {
        definition: context
            .create_type("Color", vector(&["r", "g", "b", "a"]))
            .unwrap(),
    };
    let other = Ty::Defined {
        definition: context
            .create_type("OtherColor", vector(&["r", "g", "b", "a"]))
            .unwrap(),
    };
    let vertex = Ty::Record {
        fields: vec![
            RecordField {
                name: "position".into(),
                ty: vector(&["x", "y", "z", "w"]),
            },
            RecordField {
                name: "color".into(),
                ty: color.clone(),
            },
        ],
    };
    (color, other, vertex)
}

#[test]
fn compute_pipeline_root_checks_stage_signature_before_host_projection() {
    let context = TyperContext::new();
    let parameter = shader_parameter(Ty::UInt64, Ty::UInt32);
    assert_eq!(
        shader::pipeline_root(&context, &[(parameter.as_slice(), &Ty::Unit, "compute")]).unwrap(),
        Ty::UInt32
    );
    for stages in [
        vec![],
        vec![(parameter.as_slice(), &Ty::Unit, "vertex")],
        vec![
            (parameter.as_slice(), &Ty::Unit, "compute"),
            (parameter.as_slice(), &Ty::Unit, "compute"),
        ],
        vec![([Ty::UInt64].as_slice(), &Ty::Unit, "compute")],
        vec![(parameter.as_slice(), &Ty::UInt32, "compute")],
    ] {
        assert!(shader::pipeline_root(&context, &stages).is_err());
    }
    let parameter = shader_parameter(Ty::UInt64, Ty::Bool);
    let root =
        shader::pipeline_root(&context, &[(parameter.as_slice(), &Ty::Unit, "compute")]).unwrap();
    assert_eq!(root, Ty::Bool);
    assert!(crate::gpu_projection_plan(&[], &Ty::Bool, &root).is_err());
}

#[test]
fn graphics_pipeline_roots_and_varyings_keep_nominal_type_identity() {
    let mut context = TyperContext::new();
    let (color, other_color, vertex) = shader_graphics_types(&mut context);
    let root = Ty::Defined {
        definition: context.create_type("Root", record(Ty::UInt32)).unwrap(),
    };
    let other_root = Ty::Defined {
        definition: context
            .create_type("OtherRoot", record(Ty::UInt32))
            .unwrap(),
    };
    let rooted_vertex = shader_parameter(Ty::Int32, root.clone());
    let rooted_fragment = shader_parameter(color.clone(), root.clone());
    let pipeline = |vertex_parameter: &[Ty], fragment_parameter: &[Ty]| {
        shader::pipeline_root(
            &context,
            &[
                (vertex_parameter, &vertex, "vertex"),
                (fragment_parameter, &color, "fragment"),
            ],
        )
    };
    assert_eq!(
        pipeline(&[Ty::Int32], std::slice::from_ref(&color)).unwrap(),
        Ty::None
    );
    for (vertex_parameter, fragment_parameter) in [
        (rooted_vertex.as_slice(), std::slice::from_ref(&color)),
        ([Ty::Int32].as_slice(), rooted_fragment.as_slice()),
        (rooted_vertex.as_slice(), rooted_fragment.as_slice()),
    ] {
        assert_eq!(
            pipeline(vertex_parameter, fragment_parameter).unwrap(),
            root
        );
    }
    let mismatched_root = shader_parameter(color.clone(), other_root);
    assert!(
        pipeline(&rooted_vertex, &mismatched_root)
            .unwrap_err()
            .contains("same root type")
    );
    assert!(
        pipeline(&[Ty::Int32], &[other_color])
            .unwrap_err()
            .contains("same type")
    );
    assert!(
        shader::pipeline_root(
            &context,
            &[
                (std::slice::from_ref(&color), &color, "fragment"),
                ([Ty::Int32].as_slice(), &vertex, "vertex")
            ]
        )
        .is_err()
    );
}

#[test]
fn references_preserve_access_without_pointer_conversions_or_shared_layout() {
    let reference = Ty::Reference {
        mutable: false,
        referent: Box::new(Ty::Bool),
    };
    let pointer = Ty::Pointer {
        pointee: Box::new(Ty::Bool),
    };
    let context = TyperContext::new();
    for other in [pointer, Ty::UInt64] {
        assert!(context.explicit_conversion(&reference, &other).is_err());
        assert!(context.explicit_conversion(&other, &reference).is_err());
        assert!(!reference.widens_to(&other));
    }
    assert!(crate::shader::value_type(&[], &reference).is_ok());
    assert!(layout::layout(&[], &reference).is_err());
    assert!(!reference.gpu_element(&[]));
    assert_eq!(crate::format_type(&reference, &[]), "Ref<bool>");
}
