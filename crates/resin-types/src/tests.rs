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
    let span_record = span.view_record().unwrap();
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
            ExplicitConversion::Ascribe(vec![Conv::ViewRecord]),
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
        context.ascribe(&Ty::Str, &Ty::byte_span()).unwrap(),
        vec![Conv::StrSpan]
    );
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
fn gpu_views_preserve_ownership_and_reject_raw_pointer_conversions() {
    let context = TyperContext::new();
    let gpu = Ty::GpuPointer {
        pointee: Box::new(Ty::UInt32),
    };
    let span = Ty::GpuSpan {
        element: Box::new(Ty::UInt32),
    };
    assert_eq!(gpu.deref_target(), Some(&Ty::UInt32));
    assert_eq!(context.type_field(&span, "data").unwrap().ty, gpu);
    assert_eq!(context.type_field(&span, "length").unwrap().ty, Ty::UInt64);
    assert!(context.type_field(&gpu, "host").is_err());
    for ty in [&gpu, &span] {
        assert!(ty.needs_drop(&[]));
        assert!(record(ty.clone()).needs_drop(&[]));
        assert!(!ty.foreign_value());
        assert!(layout::layout(&[], ty).is_err());
    }
    for other in [
        Ty::UInt64,
        Ty::Pointer {
            pointee: Box::new(Ty::UInt32),
        },
        Ty::GpuPointer {
            pointee: Box::new(Ty::UInt64),
        },
    ] {
        assert!(!gpu.pointer_cast(&other));
        assert!(!other.pointer_cast(&gpu));
        assert!(context.explicit_conversion(&gpu, &other).is_err());
        assert!(context.explicit_conversion(&other, &gpu).is_err());
    }
    let wider = Ty::GpuPointer {
        pointee: Box::new(Ty::union_of([Ty::UInt32, Ty::UInt64])),
    };
    assert!(!gpu.widens_to(&wider));
    assert!(
        context
            .ascribe(&span.view_record().unwrap(), &span)
            .is_err()
    );
}

#[test]
fn gpu_element_storage_excludes_references_and_custom_destruction() {
    let mut context = TyperContext::new();
    let id = context.create_type("Element", record(Ty::UInt32)).unwrap();
    let element = Ty::Defined { definition: id };
    let array = Ty::Array {
        element: Box::new(element.clone()),
        length: 8,
    };
    assert!(array.gpu_element(context.definitions()));
    context.define_drop(id, FunctionId::from_index(0));
    assert!(!array.gpu_element(context.definitions()));
    for ty in [
        Ty::Pointer {
            pointee: Box::new(Ty::UInt32),
        },
        Ty::Span {
            element: Box::new(Ty::UInt32),
        },
        Ty::GpuPointer {
            pointee: Box::new(Ty::UInt32),
        },
        Ty::GpuSpan {
            element: Box::new(Ty::UInt32),
        },
        Ty::Arc {
            pointee: Box::new(Ty::UInt32),
        },
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
fn gpu_types_intern_components_and_render_source_spellings() {
    let span = Ty::GpuSpan {
        element: Box::new(Ty::UInt32),
    };
    let pointer = Ty::GpuPointer {
        pointee: Box::new(Ty::UInt32),
    };
    let mut table = TypeTable::default();
    table.intern(&span);
    for ty in [&span, &pointer, &Ty::UInt32, &Ty::UInt64] {
        assert!(table.id(ty).is_some());
    }
    assert_eq!(format_type(&span, &table), "GpuSpan<uint>");
    assert_eq!(format_type(&pointer, &table), "GpuPtr<uint>");
}

#[test]
fn gpu_arguments_are_opaque_managed_values_and_projection_is_type_directed() {
    let arguments = Ty::GpuArguments;
    let context = TyperContext::new();
    assert!(arguments.needs_drop(&[]));
    assert!(!arguments.foreign_value());
    assert!(arguments.view_record().is_none());
    assert!(arguments.deref_target().is_none());
    assert!(layout::layout(&[], &arguments).is_err());
    assert!(
        context
            .explicit_conversion(&Ty::UInt64, &arguments)
            .is_err()
    );
    let root = record(Ty::Span {
        element: Box::new(Ty::Int32),
    });
    let table = [TypeDef::new("Root", root)];
    let root = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    assert_eq!(
        root.gpu_projection(&table),
        Some(record(Ty::GpuSpan {
            element: Box::new(Ty::Int32)
        }))
    );
    let graph = Ty::Pointer {
        pointee: Box::new(root),
    };
    assert_eq!(graph.gpu_projection(&table), None);
    assert_eq!(Ty::GpuArguments.gpu_projection(&[]), None);
}

#[test]
fn pipeline_types_preserve_root_identity_and_opaque_shared_ownership() {
    let context = TyperContext::new();
    let root = record(Ty::Span {
        element: Box::new(Ty::UInt32),
    });
    let owner = Ty::Arc {
        pointee: Box::new(Ty::Int32),
    };
    let compute = Ty::GpuComputePipeline {
        root: Box::new(root.clone()),
        owner: Box::new(owner.clone()),
    };
    let graphics = Ty::GpuGraphicsPipeline {
        root: Box::new(root.clone()),
        owner: Box::new(owner.clone()),
    };
    assert_ne!(compute, graphics);
    for pipeline in [&compute, &graphics] {
        assert_eq!(pipeline.gpu_pipeline(), Some((&root, &owner)));
        assert_eq!(
            pipeline.gpu_pipeline_argument(&[]),
            root.gpu_projection(&[])
        );
        assert!(pipeline.needs_drop(&[]));
        assert!(pipeline.deref_target().is_none());
        assert!(pipeline.view_record().is_none());
        assert!(!pipeline.foreign_value());
        assert!(layout::layout(&[], pipeline).is_err());
        assert!(!pipeline.gpu_element(&[]));
        assert!(pipeline.gpu_projection(&[]).is_none());
        for other in [&owner, &Ty::UInt64] {
            assert!(context.explicit_conversion(pipeline, other).is_err());
            assert!(context.explicit_conversion(other, pipeline).is_err());
        }
    }
    assert!(context.explicit_conversion(&compute, &graphics).is_err());
    let mut table = TypeTable::default();
    table.intern(&compute);
    table.intern(&graphics);
    for ty in [&root, &owner, &Ty::Int32, &Ty::UInt32] {
        assert!(table.id(ty).is_some());
    }
    assert_eq!(
        format_type(&compute, &table),
        "GpuComputePipeline<{ value: Span<uint> }, Arc<int>>"
    );
    assert_eq!(
        format_type(&graphics, &table),
        "GpuGraphicsPipeline<{ value: Span<uint> }, Arc<int>>"
    );
}

#[test]
fn pipeline_argument_contract_rejects_invalid_owners_roots_and_rootless_compute() {
    let owner = Box::new(Ty::Arc {
        pointee: Box::new(Ty::Int32),
    });
    let graphics = Ty::GpuGraphicsPipeline {
        root: Box::new(Ty::None),
        owner: owner.clone(),
    };
    assert_eq!(graphics.gpu_pipeline_argument(&[]), Some(Ty::None));
    for pipeline in [
        Ty::GpuComputePipeline {
            root: Box::new(Ty::None),
            owner: owner.clone(),
        },
        Ty::GpuComputePipeline {
            root: Box::new(Ty::UInt32),
            owner: Box::new(Ty::Int32),
        },
        Ty::GpuGraphicsPipeline {
            root: Box::new(Ty::GpuArguments),
            owner,
        },
    ] {
        assert_eq!(pipeline.gpu_pipeline_argument(&[]), None);
    }
}

fn shader_parameter(input: Ty, root: Ty) -> Ty {
    Ty::Record {
        fields: vec![
            RecordField {
                name: "_0".into(),
                ty: input,
            },
            RecordField {
                name: "_1".into(),
                ty: Ty::Pointer {
                    pointee: Box::new(root),
                },
            },
        ],
    }
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
fn compute_pipeline_root_checks_stage_signature_and_projection_support() {
    let context = TyperContext::new();
    let parameter = shader_parameter(Ty::UInt64, Ty::UInt32);
    assert_eq!(
        shader::pipeline_root(&context, &[(&parameter, &Ty::Unit, "compute")]).unwrap(),
        Ty::UInt32
    );
    for stages in [
        vec![],
        vec![(&parameter, &Ty::Unit, "vertex")],
        vec![
            (&parameter, &Ty::Unit, "compute"),
            (&parameter, &Ty::Unit, "compute"),
        ],
        vec![(&Ty::UInt64, &Ty::Unit, "compute")],
        vec![(&parameter, &Ty::UInt32, "compute")],
    ] {
        assert!(shader::pipeline_root(&context, &stages).is_err());
    }
    let parameter = shader_parameter(Ty::UInt64, Ty::Bool);
    assert!(
        shader::pipeline_root(&context, &[(&parameter, &Ty::Unit, "compute")])
            .unwrap_err()
            .contains("projection")
    );
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
    let pipeline = |vertex_parameter: &Ty, fragment_parameter: &Ty| {
        shader::pipeline_root(
            &context,
            &[
                (vertex_parameter, &vertex, "vertex"),
                (fragment_parameter, &color, "fragment"),
            ],
        )
    };
    assert_eq!(pipeline(&Ty::Int32, &color).unwrap(), Ty::None);
    for (vertex_parameter, fragment_parameter) in [
        (&rooted_vertex, &color),
        (&Ty::Int32, &rooted_fragment),
        (&rooted_vertex, &rooted_fragment),
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
        pipeline(&Ty::Int32, &other_color)
            .unwrap_err()
            .contains("same type")
    );
    assert!(
        shader::pipeline_root(
            &context,
            &[
                (&color, &color, "fragment"),
                (&Ty::Int32, &vertex, "vertex")
            ]
        )
        .is_err()
    );
}
