//! Lower a resolved tree without a parser, source provider, or compiler session.
mod support;
use resin_hir::{Annotation, Constant, Function, Module, Signature, Term, TermKind, Type};
use resin_lir::Instr;
use resin_source::prelude::*;
use resin_types::prelude::*;

fn constant(value: Constant, ty: Type) -> Term {
    Term {
        span: Span { start: 0, end: 0 },
        ty,
        kind: TermKind::Constant { value },
    }
}

fn conditional() -> Module {
    Module {
        functions: vec![Function {
            location: None,
            name: "choose".into(),
            signature: Signature {
                type_params: vec![],
                params: vec![],
                result: Annotation {
                    ty: Type::Int32,
                    span: Span { start: 0, end: 0 },
                },
            },
            foreign_header: None,
            body: Some(Term {
                span: Span { start: 0, end: 0 },
                ty: Type::Int32,
                kind: TermKind::If {
                    cond: Box::new(constant(Constant::Bool { value: true }, Type::Bool)),
                    then: Box::new(constant(Constant::Int32 { value: 1 }, Type::Int32)),
                    els: Box::new(constant(Constant::Int32 { value: 2 }, Type::Int32)),
                },
            }),
        }],
        ..Default::default()
    }
}

#[test]
fn resolved_tree_is_sufficient_to_lower_control_flow() {
    let tree = conditional();
    let module = support::build_lir(&tree, &[], &resin_lir::LoweringOptions::default()).unwrap();
    assert_eq!(
        module,
        support::build_lir(&tree, &[], &resin_lir::LoweringOptions::default()).unwrap()
    );
    drop(tree);
    let function = &module.functions[0];
    assert_eq!(function.parameter_count, 0);
    assert!(function.blocks.len() > 1);
    assert!(
        function
            .blocks
            .iter()
            .any(|block| block.instrs.contains(&Instr::Push {
                value: Value::Int32 { value: 2 },
            }))
    );
    assert!(resin_lir::format_module(&module).contains("choose"));
}

#[test]
fn explicit_header_dependencies_survive_without_foreign_functions() {
    let mut tree = conditional();
    tree.foreign_headers.insert(resin_hir::ForeignHeader {
        source: resin_source::SourceId::new("native.resin"),
        spelling: "empty.h".into(),
    });
    let module = support::build_lir(&tree, &[], &resin_lir::LoweringOptions::default()).unwrap();
    assert_eq!(
        module.foreign_headers.iter().next().unwrap().source,
        tree.foreign_headers.iter().next().unwrap().source
    );
    assert_eq!(
        module
            .foreign_headers
            .iter()
            .next()
            .unwrap()
            .spelling
            .as_ref(),
        "empty.h"
    );
    assert!(
        module
            .functions
            .iter()
            .all(|function| function.foreign.is_none())
    );
    assert!(resin_lir::format_module(&module).contains("\"empty.h\""));
    resin_lir::verify(&module).unwrap();
}

#[test]
fn invalid_standalone_header_dependencies_fail_verification() {
    for header in ["", "bad\nheader.h", "bad>header.h", "bad\"header.h"] {
        let module = resin_lir::Module {
            foreign_headers: [resin_lir::ForeignHeader {
                source: resin_source::SourceId::new("native.resin"),
                spelling: header.into(),
            }]
            .into(),
            ..Default::default()
        };
        let error = resin_lir::verify(&module).unwrap_err();
        assert_eq!(error.location, resin_lir::VerifyLocation::Module);
        assert!(matches!(
            error.kind,
            resin_lir::VerifyErrorKind::InvalidForeignHeader { .. }
        ));
    }
}

#[test]
fn lowering_preserves_source_handles_without_inventing_missing_origins() {
    let source = Source::new("editor://scratch", "choose");
    let other = Source::new("editor://other", "choose");
    let mut tree = conditional();
    tree.functions.push(tree.functions[0].clone());
    tree.functions.push(tree.functions[0].clone());
    tree.functions[0].location = Some(SourceLocation {
        source: source.clone(),
        span: Span { start: 0, end: 6 },
    });
    tree.functions[2].location = Some(SourceLocation {
        source: other.clone(),
        span: Span { start: 0, end: 6 },
    });
    let module = support::build_lir(&tree, &[], &resin_lir::LoweringOptions::default()).unwrap();
    drop(tree);
    assert_eq!(module.origins.functions.len(), 2);
    assert_eq!(
        module.origins.functions[&FunctionId::from_index(0)].source,
        source
    );
    assert_eq!(
        module.origins.functions[&FunctionId::from_index(2)].source,
        other
    );
    assert!(!module.origins.instructions.is_empty());
    for ((function, _, _), origin) in &module.origins.instructions {
        let expected = match function.index() {
            0 => &source,
            2 => &other,
            _ => panic!("source-less function acquired an origin"),
        };
        assert_eq!(&origin.source, expected);
        assert_eq!(origin.source.text(), "choose");
    }
    assert_eq!(
        module
            .origins
            .instructions
            .keys()
            .filter(|(id, _, _)| id.index() == 0)
            .count(),
        module
            .origins
            .instructions
            .keys()
            .filter(|(id, _, _)| id.index() == 2)
            .count(),
    );
}

#[test]
fn failed_functions_keep_their_own_bindings_cleanup_and_error_origins() {
    let source = Source::new("editor://scratch", "owner missing");
    let changed = source.with_text("a later source version");
    let missing_span = Span { start: 6, end: 13 };
    let owner = Ident {
        val: "owner".into(),
        span: Span { start: 0, end: 5 },
    };
    let mut first = parameter_function(&[], false);
    first.location = Some(SourceLocation {
        source: source.clone(),
        span: Span { start: 0, end: 13 },
    });
    first.body = Some(Term {
        span: Span { start: 0, end: 13 },
        ty: Type::Int32,
        kind: TermKind::Block {
            stmts: vec![resin_hir::Statement::Define {
                binding: 0,
                name: owner.clone(),
                init: Term {
                    span: owner.span,
                    ty: Type::StrongOwner,
                    kind: TermKind::Unwrap {
                        value: Box::new(Term {
                            span: owner.span,
                            ty: Type::Union {
                                variants: vec![Type::None, Type::StrongOwner],
                            },
                            kind: TermKind::Intrinsic {
                                op: resin_types::Intrinsic::OwnerAllocate,
                                type_args: vec![Type::Int32],
                                args: resin_hir::Arguments {
                                    values: vec![
                                        constant(Constant::UInt64 { value: 1 }, Type::UInt64),
                                        constant(Constant::Int32 { value: 7 }, Type::Int32),
                                    ],
                                    params: vec![Type::UInt64, Type::Int32],
                                },
                            },
                        }),
                    },
                },
            }],
            tail: Box::new(Term {
                span: missing_span,
                ty: Type::Int32,
                kind: TermKind::Local {
                    mutable: true,
                    binding: 1,
                    name: Ident {
                        val: "missing".into(),
                        span: missing_span,
                    },
                },
            }),
        },
    });
    let detached_span = Span { start: 20, end: 25 };
    let mut last = parameter_function(&[], false);
    last.body = Some(Term {
        span: detached_span,
        ty: Type::Int32,
        kind: TermKind::Local {
            mutable: true,
            binding: 0,
            name: Ident {
                span: detached_span,
                ..owner
            },
        },
    });
    let tree = Module {
        // The middle function has no owner slots for failed cleanup state to reuse.
        functions: vec![first, parameter_function(&[], false), last],
        ..Default::default()
    };
    let errors =
        support::build_lir(&tree, &[], &resin_lir::LoweringOptions::default()).unwrap_err();
    drop(tree);
    drop(source);

    assert_eq!(errors.len(), 2);
    let origin = errors[0].source.as_ref().unwrap();
    assert_eq!(origin.id(), changed.id());
    assert_ne!(origin, &changed);
    assert_eq!(origin.text(), "owner missing");
    assert_eq!(errors[0].span, missing_span);
    assert_eq!(
        errors[0].kind,
        resin_lir::ErrorKind::UnboundValue {
            name: "missing".into(),
        }
    );
    assert!(errors[1].source.is_none());
    assert_eq!(errors[1].span, detached_span);
    assert_eq!(
        errors[1].kind,
        resin_lir::ErrorKind::UnboundValue {
            name: "owner".into(),
        }
    );
}

fn parameter(index: usize, ty: Type, foreign: bool) -> resin_hir::Parameter {
    let span = Span { start: 0, end: 0 };
    resin_hir::Parameter {
        binding: (!foreign).then_some(index),
        name: Ident {
            val: format!("p{index}").into(),
            span,
        },
        annotation: Annotation { ty, span },
    }
}

fn parameter_function(types: &[Type], foreign: bool) -> Function {
    let span = Span { start: 0, end: 0 };
    let params: Vec<_> = types
        .iter()
        .enumerate()
        .map(|(index, ty)| parameter(index, ty.clone(), foreign))
        .collect();
    let body = params.last().filter(|_| !foreign).map_or_else(
        || constant(Constant::Int32 { value: 42 }, Type::Int32),
        |parameter| Term {
            span,
            ty: Type::Int32,
            kind: TermKind::Local {
                mutable: true,
                binding: parameter.binding.unwrap(),
                name: parameter.name.clone(),
            },
        },
    );
    Function {
        location: None,
        name: "callee".into(),
        signature: Signature {
            type_params: vec![],
            params,
            result: Annotation {
                ty: Type::Int32,
                span,
            },
        },
        foreign_header: foreign.then(|| resin_hir::ForeignHeader {
            source: resin_source::SourceId::new("native.resin"),
            spelling: "callee.h".into(),
        }),
        body: (!foreign).then_some(body),
    }
}

#[test]
fn ordinary_and_foreign_parameters_have_separate_locals() {
    for (params, expected) in [
        (vec![], vec![]),
        (vec![Type::Int32], vec![Ty::Int32]),
        (vec![Type::Bool, Type::Int32], vec![Ty::Bool, Ty::Int32]),
    ] {
        for foreign in [false, true] {
            let location = SourceLocation {
                source: Source::new("callee.resin", "callee"),
                span: Span { start: 0, end: 6 },
            };
            let mut declaration = parameter_function(&params, foreign);
            declaration.location = Some(location.clone());
            let tree = Module {
                functions: vec![declaration],
                ..Default::default()
            };
            let checked = resin_lir::VerifiedModule::new(
                support::build_lir(&tree, &[], &resin_lir::LoweringOptions::default()).unwrap(),
            )
            .unwrap();
            let function = &checked.view().module().functions[0];
            assert_eq!(function.foreign.is_some(), foreign);
            assert_eq!(
                checked.view().module().origins.functions[&FunctionId::from_index(0)],
                location,
            );
            assert_eq!(function.parameter_count, expected.len());
            assert_eq!(
                function.locals[..function.parameter_count]
                    .iter()
                    .map(|local| local.ty.clone())
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                function.ty(),
                Some(Ty::Function {
                    params: expected.clone(),
                    result: Box::new(Ty::Int32),
                })
            );
            assert_eq!(function.blocks.is_empty(), foreign);
            if foreign {
                assert_eq!(function.locals.len(), expected.len());
                assert_eq!(function.foreign.as_ref().unwrap().params, expected);
                assert!(checked.view().analysis().functions[0].inputs.is_empty());
                assert!(checked.view().module().origins.instructions.is_empty());
            }
        }
    }
}

#[test]
fn invalid_nominal_type_expressions_report_errors_before_storage_lowering() {
    let invalid_reference = Type::Pointer {
        mutable: true,
        pointee: Box::new(Type::Defined {
            arguments: vec![],
            definition: TypeId::from_index(99),
        }),
    };
    let inline_cycle = Type::Defined {
        arguments: vec![],
        definition: TypeId::from_index(0),
    };
    for body in [
        Type::Int32,
        Type::Record {
            fields: vec![resin_hir::RecordField {
                name: "bad".into(),
                ty: invalid_reference,
            }],
        },
        Type::Record {
            fields: vec![resin_hir::RecordField {
                name: "self".into(),
                ty: inline_cycle,
            }],
        },
    ] {
        let mut tree = conditional();
        tree.types.push(resin_hir::TypeDefinition {
            gpu_projection: None,
            gpu_pipeline: None,
            type_params: vec![],
            name: "Invalid".into(),
            body,
            methods: Default::default(),
            text_view: None,
            drop: None,
        });
        let error = support::build_lir(&tree, &[], &resin_lir::LoweringOptions::default())
            .unwrap_err()
            .remove(0);
        assert!(
            matches!(error.kind, resin_lir::ErrorKind::Type { .. }),
            "{error}"
        );
    }
}
