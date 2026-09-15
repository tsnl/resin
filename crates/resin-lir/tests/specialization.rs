//! Schemes survive in HIR; requests and concrete function identities belong to LIR.
use resin_hir::{
    Annotation, Constant, Function, Module, Parameter, Signature, Statement, Term, TermKind, Type,
    TypeParameter, TypeParameterId,
};
use resin_lir::{ErrorKind, Instr, LoweringOptions};
use resin_source::prelude::*;
use resin_types::{FunctionId, Ty, TypeId};
use std::num::NonZeroUsize;

const SPAN: Span = Span { start: 0, end: 1 };
const T: TypeParameterId = TypeParameterId::from_index(0);

fn annotation(ty: Type) -> Annotation {
    Annotation { ty, span: SPAN }
}
fn term(ty: Type, kind: TermKind) -> Term {
    Term {
        span: SPAN,
        ty,
        kind,
    }
}
fn unit() -> Term {
    term(
        Type::Unit,
        TermKind::Constant {
            value: Constant::Unit,
        },
    )
}
fn function(name: &str, body: Term) -> Function {
    Function {
        location: Some(SourceLocation {
            source: Source::new(format!("test://{name}"), name),
            span: SPAN,
        }),
        name: name.into(),
        signature: Signature {
            type_params: vec![],
            params: vec![],
            result: annotation(body.ty.clone()),
        },
        foreign_header: None,
        body: Some(body),
    }
}
fn template(name: &str, body: Term) -> Function {
    let mut function = function(name, body);
    function.signature.type_params.push(TypeParameter {
        id: T,
        name: Ident {
            val: "T".into(),
            span: SPAN,
        },
    });
    function
}
fn reference(function: usize, argument: Type) -> Term {
    term(
        Type::Function {
            params: vec![],
            result: Box::new(Type::Unit),
        },
        TermKind::Function {
            function: FunctionId::from_index(function),
            type_args: vec![argument],
        },
    )
}
fn block(terms: impl IntoIterator<Item = Term>) -> Term {
    term(
        Type::Unit,
        TermKind::Block {
            stmts: terms
                .into_iter()
                .map(|term| Statement::Expr { term })
                .collect(),
            tail: Box::new(unit()),
        },
    )
}
fn program(arguments: Vec<Type>) -> Module {
    Module {
        functions: vec![
            template("mark", unit()),
            function(
                "main",
                block(arguments.into_iter().map(|ty| reference(0, ty))),
            ),
        ],
        entries: [("main".into(), FunctionId::from_index(1))].into(),
        ..Default::default()
    }
}
fn options(limit: usize) -> LoweringOptions {
    LoweringOptions {
        max_monomorphs_per_function: NonZeroUsize::new(limit).unwrap(),
    }
}

#[test]
fn instances_are_memoized_and_the_exact_allowance_is_admitted() {
    let hir = program(vec![Type::Int32, Type::Bool, Type::Int32, Type::Bool]);
    let lir = resin_lir::build_lir_all_with_options(&hir, &options(2)).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(hir.functions.len(), 2);
    assert_eq!(lir.functions.len(), 3);
    assert_eq!(lir.entries["main"].index(), 0);
    let calls: Vec<_> = lir.functions[0]
        .blocks
        .iter()
        .flat_map(|b| &b.instrs)
        .filter_map(|i| match i {
            Instr::Function { function } => Some(function.index()),
            _ => None,
        })
        .collect();
    assert_eq!(calls, [1, 2, 1, 2]);
    let error = resin_lir::build_lir_all_with_options(&hir, &options(1))
        .unwrap_err()
        .remove(0);
    assert_eq!(
        error.kind,
        ErrorKind::MonomorphLimit {
            function: "mark".into(),
            limit: 1,
            arguments: vec!["bool".into()],
            profile: resin_lir::Profile::Host,
        }
    );
    assert_eq!(error.source.unwrap().name(), "test://main");
    assert_eq!(error.applications[0].function.as_ref(), "main");
}

#[test]
fn substitution_normalizes_unions_before_memoization() {
    let hir = program(vec![
        Type::Union {
            variants: vec![Type::Int32, Type::Int32],
        },
        Type::Int32,
    ]);
    let lir = resin_lir::build_lir_all_with_options(&hir, &options(1)).unwrap();
    assert_eq!(lir.functions.len(), 2);
}

#[test]
fn growing_recursive_requests_fail_with_a_bounded_application_trace() {
    let mut hir = program(vec![Type::Int32]);
    hir.functions[0].body = Some(block([reference(
        0,
        Type::Pointer {
            pointee: Box::new(Type::Parameter { parameter: T }),
        },
    )]));
    let errors = resin_lir::build_lir_all_with_options(&hir, &options(40)).unwrap_err();
    assert_eq!(errors.len(), 1);
    let error = &errors[0];
    assert!(matches!(
        error.kind,
        ErrorKind::MonomorphLimit { limit: 40, .. }
    ));
    assert_eq!(error.source.as_ref().unwrap().name(), "test://mark");
    assert_eq!(error.applications.len(), 32);
    assert!(
        error
            .applications
            .iter()
            .all(|note| note.function.as_ref() == "mark")
    );
}

#[test]
fn recursive_requests_reuse_pending_identities() {
    let mut hir = program(vec![Type::Int32]);
    hir.functions[0].body = Some(block([reference(0, Type::Parameter { parameter: T })]));
    let lir = resin_lir::build_lir_all_with_options(&hir, &options(1)).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.functions.len(), 2);
    assert!(
        lir.functions[1]
            .blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .any(|i| *i
                == Instr::Function {
                    function: FunctionId::from_index(1)
                })
    );
}

#[test]
fn identity_signatures_and_bodies_are_concrete_without_changing_hir() {
    let ty = Type::Parameter { parameter: T };
    let name = Ident {
        val: "value".into(),
        span: SPAN,
    };
    let mut identity = template(
        "identity",
        term(
            ty.clone(),
            TermKind::Local {
                binding: 0,
                name: name.clone(),
            },
        ),
    );
    identity.signature.params = vec![Parameter {
        binding: Some(0),
        name,
        annotation: annotation(ty),
    }];
    let reference = term(
        Type::Function {
            params: vec![Type::Int32],
            result: Box::new(Type::Int32),
        },
        TermKind::Function {
            function: FunctionId::from_index(0),
            type_args: vec![Type::Int32],
        },
    );
    let hir = Module {
        functions: vec![identity, function("main", block([reference]))],
        ..Default::default()
    };
    let lir = resin_lir::build_lir_all(&hir).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.functions[1].locals[0].ty, Ty::Int32);
    assert_eq!(lir.functions[1].result, Ty::Int32);
    assert_eq!(
        hir.functions[0].signature.result.ty,
        Type::Parameter { parameter: T }
    );
}

#[test]
fn unused_families_do_not_constrain_supported_concrete_operations() {
    let ty = Type::Parameter { parameter: T };
    let name = Ident {
        val: "value".into(),
        span: SPAN,
    };
    let value = term(
        ty.clone(),
        TermKind::Local {
            binding: 0,
            name: name.clone(),
        },
    );
    let mut add = template(
        "add",
        term(
            ty.clone(),
            TermKind::Builtin {
                name: "+".into(),
                args: vec![value.clone(), value],
            },
        ),
    );
    add.signature.params.push(Parameter {
        binding: Some(0),
        name,
        annotation: annotation(ty),
    });
    let mut hir = Module {
        functions: vec![add, function("main", unit())],
        ..Default::default()
    };
    assert_eq!(resin_lir::build_lir_all(&hir).unwrap().functions.len(), 1);
    hir.functions[1].body = Some(block([term(
        Type::Function {
            params: vec![Type::Bool],
            result: Box::new(Type::Bool),
        },
        TermKind::Function {
            function: FunctionId::from_index(0),
            type_args: vec![Type::Bool],
        },
    )]));
    let error = resin_lir::build_lir_all(&hir).unwrap_err();
    assert!(matches!(error.kind, ErrorKind::Type { .. }));
    assert_eq!(error.applications[0].function.as_ref(), "add");
}

#[test]
fn implicit_drop_references_use_concrete_function_identities() {
    let mut hir = program(vec![]);
    hir.types.push(resin_hir::TypeDefinition {
        gpu_projection: None,
        gpu_pipeline: None,
        type_params: vec![],
        name: "Owner".into(),
        body: Type::Record { fields: vec![] },
        methods: Default::default(),
        drop: Some(FunctionId::from_index(2)),
    });
    let mut drop = function("drop", unit());
    drop.signature.params.push(Parameter {
        binding: Some(0),
        name: Ident {
            val: "self".into(),
            span: SPAN,
        },
        annotation: annotation(Type::Pointer {
            pointee: Box::new(Type::Defined {
                arguments: vec![],
                definition: TypeId::from_index(0),
            }),
        }),
    });
    hir.functions.push(drop);
    let lir = resin_lir::build_lir_all(&hir).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.types[0].drop_hook(), Some(FunctionId::from_index(1)));
}

#[test]
fn type_expansion_has_a_separate_guard_from_the_function_allowance() {
    let mut hir = program(vec![Type::Int32]);
    hir.functions[0].body = Some(block([reference(
        0,
        Type::Pointer {
            pointee: Box::new(Type::Parameter { parameter: T }),
        },
    )]));
    let error = resin_lir::build_lir_all(&hir).unwrap_err();
    assert!(matches!(error.kind, ErrorKind::TypeExpansionLimit { .. }));
}

#[test]
fn exponentially_growing_arguments_hit_the_type_size_guard() {
    let mut hir = program(vec![Type::Int32]);
    let field = |name: &str| resin_hir::RecordField {
        name: name.into(),
        ty: Type::Parameter { parameter: T },
    };
    hir.functions[0].body = Some(block([reference(
        0,
        Type::Record {
            fields: vec![field("left"), field("right")],
        },
    )]));
    let error = resin_lir::build_lir_all(&hir).unwrap_err();
    assert!(matches!(error.kind, ErrorKind::TypeSizeLimit { .. }));
}

#[test]
fn failed_requests_are_memoized_without_publishing_an_incomplete_module() {
    let mut hir = program(vec![Type::Int32, Type::Int32]);
    hir.functions[0].body = Some(term(
        Type::Parameter {
            parameter: TypeParameterId::from_index(99),
        },
        TermKind::Constant {
            value: Constant::Unit,
        },
    ));
    let errors = resin_lir::build_lir_all_with_options(&hir, &options(1)).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::InvalidHir { .. }));
}

fn entry(name: &str, function: usize, profile: resin_lir::Profile) -> resin_lir::Entry {
    resin_lir::Entry {
        name: name.into(),
        function: FunctionId::from_index(function),
        arguments: vec![],
        profile,
    }
}

#[test]
fn requested_roots_exclude_unused_functions_types_and_drop_hooks() {
    let mut hir = program(vec![Type::Int32]);
    hir.functions.push(function(
        "unused",
        term(
            Type::Bool,
            TermKind::Builtin {
                name: "+".into(),
                args: vec![
                    term(
                        Type::Bool,
                        TermKind::Constant {
                            value: Constant::Bool { value: true }
                        }
                    );
                    2
                ],
            },
        ),
    ));
    hir.types.push(resin_hir::TypeDefinition {
        gpu_projection: None,
        gpu_pipeline: None,
        type_params: vec![],
        name: "Unused".into(),
        body: Type::Defined {
            arguments: vec![],
            definition: TypeId::from_index(99),
        },
        methods: Default::default(),
        drop: Some(FunctionId::from_index(99)),
    });
    let lir = resin_lir::build_lir(
        &hir,
        &[entry("main", 1, resin_lir::Profile::Host)],
        &options(1),
    )
    .unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.functions.len(), 2);
    assert!(lir.types.is_empty());
    assert!(
        resin_lir::build_lir_all(&hir).is_err(),
        "whole-module construction still requests all declarations"
    );
}

#[test]
fn explicit_root_arguments_normalize_and_preserve_recursive_nominal_identity() {
    let mut hir = program(vec![]);
    hir.functions[0].body = Some(block([term(
        Type::UInt64,
        TermKind::Layout {
            of: Type::Parameter { parameter: T },
            size: true,
        },
    )]));
    hir.types = (0..3)
        .map(|index| resin_hir::TypeDefinition {
            gpu_projection: None,
            gpu_pipeline: None,
            type_params: vec![],
            name: format!("Type{index}").into(),
            body: Type::Record { fields: vec![] },
            methods: Default::default(),
            drop: None,
        })
        .collect();
    hir.types[2].body = Type::Record {
        fields: vec![resin_hir::RecordField {
            name: "next".into(),
            ty: Type::Pointer {
                pointee: Box::new(Type::Defined {
                    arguments: vec![],
                    definition: TypeId::from_index(2),
                }),
            },
        }],
    };
    let mut root = entry("mark", 0, resin_lir::Profile::Host);
    let nominal = Type::Defined {
        arguments: vec![],
        definition: TypeId::from_index(2),
    };
    root.arguments = vec![Type::Union {
        variants: vec![nominal.clone(), nominal.clone()],
    }];
    let mut duplicate = root.clone();
    duplicate.arguments = vec![nominal];
    let lir = resin_lir::build_lir(&hir, &[root, duplicate], &options(1)).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.functions.len(), 1);
    assert_eq!(lir.types.len(), 1);
    assert_eq!(lir.types[0].name().unwrap().as_ref(), "Type2");
    assert_eq!(
        lir.types[0].body().unwrap(),
        &Ty::Record {
            fields: vec![resin_types::RecordField {
                name: "next".into(),
                ty: Ty::Pointer {
                    pointee: Box::new(Ty::Defined {
                        definition: TypeId::from_index(0)
                    })
                },
            }]
        }
    );
}

fn add_shader(hir: &mut Module, body: Term) -> usize {
    let index = hir.functions.len();
    let mut shader = function("kernel", body);
    for (binding, ty) in [
        Type::UInt64,
        Type::Pointer {
            pointee: Box::new(Type::UInt32),
        },
    ]
    .into_iter()
    .enumerate()
    {
        shader.signature.params.push(Parameter {
            binding: Some(binding),
            name: Ident {
                val: format!("p{binding}").into(),
                span: SPAN,
            },
            annotation: annotation(ty),
        });
    }
    hir.functions.push(shader);
    hir.shaders.insert(
        FunctionId::from_index(index),
        resin_types::shader::ShaderEntry {
            stage: "compute".into(),
            embedded: false,
        },
    );
    index
}

#[test]
fn shader_artifacts_request_a_separate_profile_and_both_count_toward_the_limit() {
    let mut hir = program(vec![Type::Int32]);
    let shader = add_shader(&mut hir, block([reference(0, Type::Int32)]));
    hir.functions[1].body = Some(block([
        reference(0, Type::Int32),
        term(
            Type::Record {
                fields: vec![
                    resin_hir::RecordField {
                        name: "data".into(),
                        ty: Type::Pointer {
                            pointee: Box::new(Type::UInt8),
                        },
                    },
                    resin_hir::RecordField {
                        name: "length".into(),
                        ty: Type::UInt64,
                    },
                ],
            },
            TermKind::Shader {
                function: FunctionId::from_index(shader),
                stage: "compute".into(),
            },
        ),
    ]));
    let roots = [entry("main", 1, resin_lir::Profile::Host)];
    let lir = resin_lir::build_lir(&hir, &roots, &options(2)).unwrap();
    resin_lir::verify(&lir).unwrap();
    let profiles: Vec<_> = lir
        .functions
        .iter()
        .filter(|f| f.name.as_deref() == Some("mark"))
        .map(|f| f.profile)
        .collect();
    assert_eq!(
        profiles,
        [resin_lir::Profile::Host, resin_lir::Profile::Shader]
    );
    assert_eq!(lir.shaders.len(), 1);
    assert!(lir.shaders.values().all(|shader| shader.embedded));
    let error = resin_lir::build_lir(&hir, &roots, &options(1))
        .unwrap_err()
        .remove(0);
    assert!(matches!(
        error.kind,
        ErrorKind::MonomorphLimit {
            profile: resin_lir::Profile::Shader,
            ..
        }
    ));
    assert_eq!(error.applications[0].profile, resin_lir::Profile::Shader);
}

#[test]
fn requesting_only_a_shader_does_not_create_host_instances_or_exports() {
    let mut hir = program(vec![Type::Bool]);
    let shader = add_shader(&mut hir, block([reference(0, Type::Int32)]));
    let lir = resin_lir::build_lir(
        &hir,
        &[entry("kernel", shader, resin_lir::Profile::Shader)],
        &options(1),
    )
    .unwrap();
    resin_lir::verify(&lir).unwrap();
    assert!(lir.entries.is_empty());
    assert_eq!(lir.functions.len(), 2);
    assert!(
        lir.functions
            .iter()
            .all(|f| f.profile == resin_lir::Profile::Shader)
    );
    assert!(!lir.shaders.values().next().unwrap().embedded);
    let mut invalid = lir.clone();
    invalid.functions[1].profile = resin_lir::Profile::Host;
    assert!(matches!(
        resin_lir::verify(&invalid).unwrap_err().kind,
        resin_lir::VerifyErrorKind::InvalidProfile {
            expected: resin_lir::Profile::Shader,
            found: resin_lir::Profile::Host
        }
    ));
}

#[test]
fn nominal_expansion_is_bounded_across_declaration_boundaries() {
    let mut hir = program(vec![]);
    hir.functions[0].body = Some(block([term(
        Type::UInt64,
        TermKind::Layout {
            of: Type::Parameter { parameter: T },
            size: true,
        },
    )]));
    hir.types = (0..300)
        .map(|index| resin_hir::TypeDefinition {
            gpu_projection: None,
            gpu_pipeline: None,
            type_params: vec![],
            name: format!("Type{index}").into(),
            methods: Default::default(),
            drop: None,
            body: Type::Record {
                fields: vec![resin_hir::RecordField {
                    name: "next".into(),
                    ty: Type::Pointer {
                        pointee: Box::new(Type::Defined {
                            arguments: vec![],
                            definition: TypeId::from_index((index + 1) % 300),
                        }),
                    },
                }],
            },
        })
        .collect();
    let mut root = entry("mark", 0, resin_lir::Profile::Host);
    root.arguments = vec![Type::Defined {
        arguments: vec![],
        definition: TypeId::from_index(0),
    }];
    let error = resin_lir::build_lir(&hir, &[root], &options(1))
        .unwrap_err()
        .remove(0);
    assert!(matches!(error.kind, ErrorKind::TypeExpansionLimit { .. }));
}

const U: TypeParameterId = TypeParameterId::from_index(1);

fn nominal(argument: Type) -> Type {
    Type::Defined {
        definition: TypeId::from_index(0),
        arguments: vec![argument],
    }
}

fn nominal_program(body: Type) -> Module {
    let mut hir = program(vec![]);
    hir.types.push(resin_hir::TypeDefinition {
        gpu_projection: None,
        gpu_pipeline: None,
        type_params: vec![TypeParameter {
            id: U,
            name: Ident::new("U".into(), SPAN),
        }],
        name: "Node".into(),
        body,
        methods: Default::default(),
        drop: None,
    });
    hir
}

fn measure_parameter(hir: &mut Module) {
    hir.functions[0].body = Some(block([term(
        Type::UInt64,
        TermKind::Layout {
            of: Type::Parameter { parameter: T },
            size: true,
        },
    )]));
}

fn nominal_roots() -> Vec<resin_lir::Entry> {
    [Type::Int32, Type::Int64]
        .into_iter()
        .enumerate()
        .map(|(index, argument)| {
            let mut request = entry(&format!("root{index}"), 0, resin_lir::Profile::Host);
            request.arguments = vec![nominal(argument)];
            request
        })
        .collect()
}

#[test]
fn nominal_arguments_have_identity_without_demanding_their_layout() {
    let mut hir = nominal_program(Type::Unit); // A nominal body must be a record when used.
    let requests = nominal_roots();
    let lir = resin_lir::build_lir(&hir, &requests, &options(2)).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.functions.len(), 2);
    assert!(lir.types.is_empty());
    let error = resin_lir::build_lir(&hir, &requests, &options(1))
        .unwrap_err()
        .remove(0);
    assert!(
        matches!(error.kind, ErrorKind::MonomorphLimit { arguments, .. } if arguments == [std::sync::Arc::from("Node<long>")])
    );
    measure_parameter(&mut hir);
    assert!(resin_lir::build_lir(&hir, &requests, &options(2)).is_err());
}

#[test]
fn nominal_instances_substitute_fields_and_close_recursive_edges() {
    let mut hir = nominal_program(Type::Record {
        fields: vec![
            resin_hir::RecordField {
                name: "value".into(),
                ty: Type::Parameter { parameter: U },
            },
            resin_hir::RecordField {
                name: "next".into(),
                ty: Type::Pointer {
                    pointee: Box::new(nominal(Type::Parameter { parameter: U })),
                },
            },
        ],
    });
    measure_parameter(&mut hir);
    let lir = resin_lir::build_lir(&hir, &nominal_roots(), &options(2)).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.types.len(), 2);
    for (index, expected) in [Ty::Int32, Ty::Int64].into_iter().enumerate() {
        let Ty::Record { fields } = lir.types[index].body().unwrap() else {
            panic!("record")
        };
        assert_eq!(fields[0].ty, expected);
        assert_eq!(
            fields[1].ty,
            Ty::Pointer {
                pointee: Box::new(Ty::Defined {
                    definition: TypeId::from_index(index)
                })
            }
        );
    }
    assert_eq!(lir.types[0].name().unwrap().as_ref(), "Node<int>");
    assert_eq!(lir.types[1].name().unwrap().as_ref(), "Node<long>");
}

#[test]
fn nominal_hooks_receive_owner_arguments_before_storage_lowering() {
    let mut hir = nominal_program(Type::Record {
        fields: vec![resin_hir::RecordField {
            name: "value".into(),
            ty: Type::Parameter { parameter: U },
        }],
    });
    measure_parameter(&mut hir);
    let mut drop = function("drop", unit());
    drop.signature.type_params = hir.types[0].type_params.clone();
    drop.signature.params.push(Parameter {
        name: Ident::new("self".into(), SPAN),
        binding: Some(0),
        annotation: annotation(Type::Pointer {
            pointee: Box::new(nominal(Type::Parameter { parameter: U })),
        }),
    });
    hir.types[0].drop = Some(FunctionId::from_index(hir.functions.len()));
    hir.functions.push(drop);
    let lir = resin_lir::build_lir(&hir, &nominal_roots(), &options(2)).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.functions.len(), 4);
    for (index, definition) in lir.types.iter().enumerate() {
        let hook = definition.drop_hook().unwrap();
        assert_eq!(
            lir.functions[hook.index()].locals[0].ty,
            Ty::Pointer {
                pointee: Box::new(Ty::Defined {
                    definition: TypeId::from_index(index)
                })
            }
        );
    }
    assert_ne!(lir.types[0].drop_hook(), lir.types[1].drop_hook());
}

#[test]
fn nominal_recursion_with_growing_arguments_reports_a_type_limit() {
    let mut hir = nominal_program(Type::Record {
        fields: vec![resin_hir::RecordField {
            name: "next".into(),
            ty: Type::Pointer {
                pointee: Box::new(nominal(Type::Pointer {
                    pointee: Box::new(Type::Parameter { parameter: U }),
                })),
            },
        }],
    });
    measure_parameter(&mut hir);
    let error = resin_lir::build_lir(&hir, &nominal_roots()[..1], &options(1))
        .unwrap_err()
        .remove(0);
    assert!(
        matches!(error.kind, ErrorKind::TypeExpansionLimit { .. }),
        "{error:?}"
    );
}

#[test]
fn member_derived_unions_normalize_independently_of_layout_discovery_order() {
    let a = Type::Defined {
        definition: TypeId::from_index(0),
        arguments: vec![],
    };
    let b = Type::Defined {
        definition: TypeId::from_index(1),
        arguments: vec![],
    };
    let holder = Type::Defined {
        definition: TypeId::from_index(2),
        arguments: vec![],
    };
    let choice = Type::Union {
        variants: vec![a, b.clone()],
    };
    let mut hir = program(vec![
        Type::Member {
            base: Box::new(holder),
            name: "choice".into(),
        },
        choice.clone(),
    ]);
    hir.types = ["A", "B", "Holder"]
        .into_iter()
        .map(|name| resin_hir::TypeDefinition {
            gpu_projection: None,
            gpu_pipeline: None,
            type_params: vec![],
            name: name.into(),
            body: Type::Record { fields: vec![] },
            methods: Default::default(),
            drop: None,
        })
        .collect();
    hir.types[2].body = Type::Record {
        fields: vec![
            resin_hir::RecordField {
                name: "first".into(),
                ty: b,
            },
            resin_hir::RecordField {
                name: "choice".into(),
                ty: choice,
            },
        ],
    };
    let lir = resin_lir::build_lir(
        &hir,
        &[entry("main", 1, resin_lir::Profile::Host)],
        &options(1),
    )
    .unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.functions.len(), 2);
}

#[test]
fn shader_dependency_order_is_iterative_and_rejects_cycles_with_bounded_notes() {
    let reference = |index| {
        term(
            Type::Function {
                params: vec![],
                result: Box::new(Type::Unit),
            },
            TermKind::Function {
                function: FunctionId::from_index(index),
                type_args: vec![],
            },
        )
    };
    let count = 2000;
    let mut hir = Module::default();
    for index in 0..count {
        hir.functions.push(function(
            &format!("helper{index}"),
            if index + 1 == count {
                unit()
            } else {
                block([reference(index + 1)])
            },
        ));
    }
    let shader = add_shader(&mut hir, block([reference(0)]));
    let roots = [entry("kernel", shader, resin_lir::Profile::Shader)];
    let lir = resin_lir::build_lir(&hir, &roots, &options(1)).unwrap();
    let checked = resin_lir::VerifiedModule::new(lir).unwrap();
    let root = *checked.view().module().shaders.keys().next().unwrap();
    let order = checked.view().shader_functions(root).unwrap();
    assert_eq!(order.len(), count + 1);
    assert_eq!(order.last(), Some(&root));
    hir.functions[count - 1].body = Some(block([reference(0)]));
    let error = resin_lir::build_lir(&hir, &roots, &options(1))
        .unwrap_err()
        .remove(0);
    assert!(matches!(error.kind, ErrorKind::UnsupportedProfile { .. }));
    assert_eq!(error.applications.len(), 32);
}

fn requested_template(
    function: Function,
    argument: Type,
) -> Result<resin_lir::Module, Vec<resin_lir::Error>> {
    resin_lir::build_lir(
        &Module {
            functions: vec![function],
            ..Default::default()
        },
        &[resin_lir::Entry {
            name: "entry".into(),
            function: FunctionId::from_index(0),
            arguments: vec![argument],
            profile: resin_lir::Profile::Host,
        }],
        &LoweringOptions::default(),
    )
}

#[test]
fn numeric_representation_and_layout_follow_the_selected_argument() {
    let t = Type::Parameter { parameter: T };
    let mut measure = template(
        "measure",
        term(
            Type::UInt64,
            TermKind::Block {
                stmts: vec![Statement::Expr {
                    term: term(t.clone(), TermKind::Numeric { text: "7".into() }),
                }],
                tail: Box::new(term(Type::UInt64, TermKind::Layout { of: t, size: true })),
            },
        ),
    );
    for (argument, value, size) in [
        (Type::UInt8, resin_types::Value::UInt8 { value: 7 }, 1),
        (Type::UInt32, resin_types::Value::UInt32 { value: 7 }, 4),
    ] {
        let lir = requested_template(measure.clone(), argument).unwrap();
        resin_lir::verify(&lir).unwrap();
        let pushes: Vec<_> = lir.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instrs)
            .filter_map(|op| match op {
                Instr::Push { value } => Some(value),
                _ => None,
            })
            .collect();
        assert!(pushes.contains(&&value));
        assert!(pushes.contains(&&resin_types::Value::UInt64 { value: size }));
    }
    measure.body = Some(term(
        Type::UInt64,
        TermKind::Layout {
            of: Type::Parameter { parameter: T },
            size: false,
        },
    ));
    let lir = requested_template(measure, Type::UInt32).unwrap();
    assert!(lir.functions[0].blocks[0].instrs.contains(&Instr::Push {
        value: resin_types::Value::UInt64 { value: 4 }
    }));
}

#[test]
fn numeric_specialization_checks_range_suffix_and_type_without_defaulting() {
    for (text, argument, message) in [
        ("256", Type::UInt8, "out of range"),
        ("7_ui", Type::UInt8, "suffix does not match"),
        ("7", Type::Bool, "cannot use numeric literal"),
        ("1.5", Type::Int32, "invalid integer literal"),
    ] {
        let numeric = template(
            "numeric",
            term(
                Type::Parameter { parameter: T },
                TermKind::Numeric { text: text.into() },
            ),
        );
        let error = requested_template(numeric, argument).unwrap_err().remove(0);
        assert!(
            matches!(&error.kind, ErrorKind::InvalidInstance { message: found } if found.contains(message)),
            "{error}"
        );
        assert_eq!(error.span, SPAN);
        assert_eq!(error.applications.len(), 1);
        assert_eq!(error.applications[0].function.as_ref(), "numeric");
    }
}

#[test]
fn member_types_and_field_indices_are_determined_from_concrete_receivers() {
    let t = Type::Parameter { parameter: T };
    let member = Type::Member {
        base: Box::new(t.clone()),
        name: "value".into(),
    };
    let pointer = Type::Pointer {
        pointee: Box::new(t),
    };
    let name = Ident::new("receiver".into(), SPAN);
    let mut read = template(
        "read",
        term(
            member,
            TermKind::Field {
                base: Box::new(term(
                    pointer.clone(),
                    TermKind::Local {
                        binding: 0,
                        name: name.clone(),
                    },
                )),
                name: "value".into(),
            },
        ),
    );
    read.signature.params.push(Parameter {
        binding: Some(0),
        name,
        annotation: annotation(pointer),
    });
    for (fields, index, result) in [
        (vec![("value", Type::UInt8)], 0, Ty::UInt8),
        (
            vec![("padding", Type::UInt64), ("value", Type::UInt32)],
            1,
            Ty::UInt32,
        ),
    ] {
        let argument = Type::Record {
            fields: fields
                .into_iter()
                .map(|(name, ty)| resin_hir::RecordField {
                    name: name.into(),
                    ty,
                })
                .collect(),
        };
        let lir = requested_template(read.clone(), argument).unwrap();
        resin_lir::verify(&lir).unwrap();
        assert_eq!(lir.functions[0].result, result);
        assert!(
            lir.functions[0]
                .blocks
                .iter()
                .flat_map(|block| &block.instrs)
                .any(|op| matches!(op, Instr::AccessStatic { index: found } if *found == index))
        );
    }
    let error = requested_template(read, Type::UInt32)
        .unwrap_err()
        .remove(0);
    assert!(matches!(error.kind, ErrorKind::Type { .. }));
}

#[test]
fn conversions_select_the_concrete_operation_after_substitution() {
    let name = Ident::new("value".into(), SPAN);
    let mut cast = template(
        "cast",
        term(
            Type::Parameter { parameter: T },
            TermKind::Convert {
                arg: Box::new(term(
                    Type::UInt64,
                    TermKind::Local {
                        binding: 0,
                        name: name.clone(),
                    },
                )),
            },
        ),
    );
    cast.signature.params.push(Parameter {
        binding: Some(0),
        name,
        annotation: annotation(Type::UInt64),
    });
    for (argument, operation) in [
        (Type::UInt8, Instr::NumericCast { ty: Ty::UInt8 }),
        (
            Type::Pointer {
                pointee: Box::new(Type::UInt8),
            },
            Instr::PointerCast {
                ty: Ty::Pointer {
                    pointee: Box::new(Ty::UInt8),
                },
            },
        ),
    ] {
        let lir = requested_template(cast.clone(), argument).unwrap();
        resin_lir::verify(&lir).unwrap();
        assert!(
            lir.functions[0]
                .blocks
                .iter()
                .flat_map(|block| &block.instrs)
                .any(|op| *op == operation)
        );
    }
    let error = requested_template(cast, Type::Record { fields: vec![] })
        .unwrap_err()
        .remove(0);
    assert!(matches!(error.kind, ErrorKind::Type { .. }));
    assert_eq!(error.applications[0].function.as_ref(), "cast");
}

#[test]
fn determining_member_types_normalize_before_instance_memoization() {
    let member = Type::Member {
        base: Box::new(Type::Record {
            fields: vec![resin_hir::RecordField {
                name: "value".into(),
                ty: Type::Int32,
            }],
        }),
        name: "value".into(),
    };
    let hir = program(vec![member, Type::Int32]);
    let lir = resin_lir::build_lir_all_with_options(&hir, &options(1)).unwrap();
    assert_eq!(lir.functions.len(), 2);
}

#[test]
fn generic_conversions_cannot_bypass_custom_destruction() {
    let t = Type::Parameter { parameter: T };
    let name = Ident::new("value".into(), SPAN);
    let mut unwrap = template(
        "unwrap",
        term(
            Type::Record { fields: vec![] },
            TermKind::Convert {
                arg: Box::new(term(
                    t.clone(),
                    TermKind::Local {
                        binding: 0,
                        name: name.clone(),
                    },
                )),
            },
        ),
    );
    unwrap.signature.params.push(Parameter {
        binding: Some(0),
        name: name.clone(),
        annotation: annotation(t),
    });
    let owner = Type::Defined {
        arguments: vec![],
        definition: TypeId::from_index(0),
    };
    let mut drop = function("drop", unit());
    drop.signature.params.push(Parameter {
        binding: Some(0),
        name,
        annotation: annotation(Type::Pointer {
            pointee: Box::new(owner.clone()),
        }),
    });
    let hir = Module {
        functions: vec![unwrap, drop],
        types: vec![resin_hir::TypeDefinition {
            gpu_projection: None,
            gpu_pipeline: None,
            type_params: vec![],
            name: "Owner".into(),
            body: Type::Record { fields: vec![] },
            methods: Default::default(),
            drop: Some(FunctionId::from_index(1)),
        }],
        ..Default::default()
    };
    let mut request = entry("unwrap", 0, resin_lir::Profile::Host);
    request.arguments = vec![owner];
    let error = resin_lir::build_lir(&hir, &[request], &LoweringOptions::default())
        .unwrap_err()
        .remove(0);
    assert!(matches!(
        error.kind,
        ErrorKind::Type {
            kind: resin_types::TypeErrorKind::UnwrapManaged { .. }
        }
    ));
    assert_eq!(error.applications[0].function.as_ref(), "unwrap");
}

fn method_lookup(receiver: Type) -> resin_hir::MethodLookup {
    resin_hir::MethodLookup {
        receiver,
        name: "read".into(),
        type_args: vec![],
        associated: true,
    }
}

fn method_result(lookup: resin_hir::MethodLookup) -> Type {
    Type::FunctionResult {
        function: Box::new(Type::Method {
            lookup: Box::new(lookup),
        }),
    }
}

fn dependent_methods() -> Module {
    let lookup = method_lookup(Type::Parameter { parameter: T });
    Module {
        functions: vec![
            template(
                "invoke",
                term(
                    method_result(lookup.clone()),
                    TermKind::DependentMethodCall {
                        lookup,
                        receiver: None,
                        args: vec![],
                    },
                ),
            ),
            function(
                "Owner.read",
                term(
                    Type::Int32,
                    TermKind::Constant {
                        value: Constant::Int32 { value: 42 },
                    },
                ),
            ),
        ],
        types: vec![resin_hir::TypeDefinition {
            gpu_projection: None,
            gpu_pipeline: None,
            type_params: vec![],
            name: "Owner".into(),
            body: Type::Record { fields: vec![] },
            methods: [("read".into(), FunctionId::from_index(1))].into(),
            drop: None,
        }],
        ..Default::default()
    }
}

fn method_owner() -> Type {
    Type::Defined {
        definition: TypeId::from_index(0),
        arguments: vec![],
    }
}

fn build_lir_method(hir: &Module) -> Result<resin_lir::Module, Vec<resin_lir::Error>> {
    let mut request = entry("invoke", 0, resin_lir::Profile::Host);
    request.arguments = vec![method_owner()];
    resin_lir::build_lir(hir, &[request], &options(1))
}

#[test]
fn dependent_calls_request_selected_methods_and_return_their_results() {
    let lir = build_lir_method(&dependent_methods()).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.functions.len(), 2);
    assert_eq!(lir.functions[0].result, Ty::Int32);
    assert_eq!(lir.functions[1].name.as_deref(), Some("Owner.read"));
}

#[test]
fn method_results_normalize_before_instance_memoization_without_layout_discovery() {
    let mut hir = dependent_methods();
    hir.functions[0] = template("mark", unit());
    hir.types[0].body = method_owner();
    hir.functions.push(function(
        "main",
        block([
            reference(0, method_result(method_lookup(method_owner()))),
            reference(0, Type::Int32),
        ]),
    ));
    let lir = resin_lir::build_lir(
        &hir,
        &[entry("main", 2, resin_lir::Profile::Host)],
        &options(1),
    )
    .unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.functions.len(), 2);
    assert!(lir.types.is_empty());
}

#[test]
fn recursive_method_result_queries_report_a_cycle() {
    let mut hir = dependent_methods();
    hir.functions[1].signature.result = annotation(method_result(method_lookup(method_owner())));
    let error = build_lir_method(&hir).unwrap_err().remove(0);
    assert!(
        matches!(&error.kind, ErrorKind::InvalidInstance { message } if message.contains("cyclic dependent method signature")),
        "{error:?}"
    );
    assert_eq!(error.applications[0].function.as_ref(), "invoke");
}

#[test]
fn growing_method_result_queries_are_bounded_before_exhausting_the_host_stack() {
    let mut hir = dependent_methods();
    let parameter = TypeParameterId::from_index(1);
    hir.functions[1].signature.type_params.push(TypeParameter {
        id: parameter,
        name: Ident::new("U".into(), SPAN),
    });
    let mut next = method_lookup(method_owner());
    next.type_args = vec![Type::Pointer {
        pointee: Box::new(Type::Parameter { parameter }),
    }];
    hir.functions[1].signature.result = annotation(method_result(next));
    let mut initial = method_lookup(Type::Parameter { parameter: T });
    initial.type_args = vec![Type::Int32];
    hir.functions[0].signature.result = annotation(method_result(initial));
    let error = build_lir_method(&hir).unwrap_err().remove(0);
    assert!(
        matches!(error.kind, ErrorKind::TypeExpansionLimit { limit: 32 }),
        "{error:?}"
    );
}

#[test]
fn dependent_lookup_reports_missing_methods_with_the_application_trace() {
    let mut hir = dependent_methods();
    hir.types[0].methods.clear();
    let error = build_lir_method(&hir).unwrap_err().remove(0);
    assert!(
        matches!(&error.kind, ErrorKind::InvalidInstance { message } if message.contains("Owner has no method read")),
        "{error:?}"
    );
    assert_eq!(error.applications[0].function.as_ref(), "invoke");
}

#[test]
fn dependent_method_arguments_are_substituted_without_deduction() {
    let mut hir = dependent_methods();
    hir.functions[1].signature.type_params.push(TypeParameter {
        id: TypeParameterId::from_index(1),
        name: Ident::new("U".into(), SPAN),
    });
    let result = Type::Parameter {
        parameter: TypeParameterId::from_index(1),
    };
    hir.functions[1].signature.result = annotation(result.clone());
    hir.functions[1].body = Some(term(result, TermKind::Numeric { text: "42".into() }));
    let error = build_lir_method(&hir).unwrap_err().remove(0);
    assert!(
        matches!(&error.kind, ErrorKind::InvalidInstance { message } if message.contains("1 explicit type arguments")),
        "{error:?}"
    );
    let mut lookup = method_lookup(Type::Parameter { parameter: T });
    lookup.type_args = vec![Type::Int32];
    hir.functions[0] = template(
        "invoke",
        term(
            method_result(lookup.clone()),
            TermKind::DependentMethodCall {
                lookup,
                receiver: None,
                args: vec![],
            },
        ),
    );
    let lir = build_lir_method(&hir).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.functions.len(), 2);
}

#[test]
fn dependent_calls_preserve_argument_and_result_widening() {
    let mut hir = dependent_methods();
    let optional = Type::Union {
        variants: vec![Type::Int32, Type::None],
    };
    hir.functions[1].signature.params.push(Parameter {
        binding: Some(0),
        name: Ident::new("value".into(), SPAN),
        annotation: annotation(optional.clone()),
    });
    hir.functions[0] = template(
        "invoke",
        term(
            optional,
            TermKind::DependentMethodCall {
                lookup: method_lookup(Type::Parameter { parameter: T }),
                receiver: None,
                args: vec![term(
                    Type::Int32,
                    TermKind::Constant {
                        value: Constant::Int32 { value: 7 },
                    },
                )],
            },
        ),
    );
    let lir = build_lir_method(&hir).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(
        lir.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instrs)
            .filter(|instruction| matches!(instruction, Instr::Widen { .. }))
            .count(),
        2
    );
}

#[test]
fn function_projections_reject_noncallable_instances_with_a_source_error() {
    let error = requested_template(
        template(
            "invoke",
            term(
                Type::FunctionResult {
                    function: Box::new(Type::Parameter { parameter: T }),
                },
                TermKind::Constant {
                    value: Constant::Unit,
                },
            ),
        ),
        Type::Int32,
    )
    .unwrap_err()
    .remove(0);
    assert!(
        matches!(error.kind, ErrorKind::InvalidInstance { .. }),
        "{error:?}"
    );
}
