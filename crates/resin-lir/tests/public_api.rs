//! Lower a resolved tree without a parser, source provider, or compiler session.
use resin_hir::{Annotation, Function, Module, Signature, Term, TermKind};
use resin_lir::Instr;
use resin_source::prelude::*;
use resin_types::prelude::*;

fn constant(value: Value, ty: Ty) -> Term {
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
                params: vec![],
                result: Annotation {
                    ty: Ty::Int32,
                    span: Span { start: 0, end: 0 },
                },
            },
            foreign: None,
            body: Some(Term {
                span: Span { start: 0, end: 0 },
                ty: Ty::Int32,
                kind: TermKind::If {
                    cond: Box::new(constant(Value::Bool { value: true }, Ty::Bool)),
                    then: Box::new(constant(Value::Int32 { value: 1 }, Ty::Int32)),
                    els: Box::new(constant(Value::Int32 { value: 2 }, Ty::Int32)),
                },
            }),
        }],
        ..Default::default()
    }
}

#[test]
fn resolved_tree_is_sufficient_to_lower_control_flow() {
    let tree = conditional();
    let module = resin_lir::generate(&tree).unwrap();
    assert_eq!(module, resin_lir::generate(&tree).unwrap());
    drop(tree);
    let function = &module.functions[0];
    assert_eq!(function.locals[0].ty, Ty::Unit);
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
    let module = resin_lir::generate(&tree).unwrap();
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
        ty: Ty::Int32,
        kind: TermKind::Block {
            stmts: vec![resin_hir::Statement::Define {
                binding: 0,
                name: owner.clone(),
                init: Term {
                    span: owner.span,
                    ty: Ty::Arc {
                        pointee: Box::new(Ty::Int32),
                    },
                    kind: TermKind::ArcNew {
                        value: Box::new(constant(Value::Int32 { value: 7 }, Ty::Int32)),
                    },
                },
            }],
            tail: Box::new(Term {
                span: missing_span,
                ty: Ty::Int32,
                kind: TermKind::Local {
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
        ty: Ty::Int32,
        kind: TermKind::Local {
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
    let errors = resin_lir::analyze(&tree).unwrap_err();
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

fn parameter(index: usize, ty: Ty, foreign: bool) -> resin_hir::Parameter {
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

fn parameter_function(types: &[Ty], foreign: bool) -> Function {
    let span = Span { start: 0, end: 0 };
    let params: Vec<_> = types
        .iter()
        .enumerate()
        .map(|(index, ty)| parameter(index, ty.clone(), foreign))
        .collect();
    let body = params.last().filter(|_| !foreign).map_or_else(
        || constant(Value::Int32 { value: 42 }, Ty::Int32),
        |parameter| Term {
            span,
            ty: Ty::Int32,
            kind: TermKind::Local {
                binding: parameter.binding.unwrap(),
                name: parameter.name.clone(),
            },
        },
    );
    Function {
        location: None,
        name: "callee".into(),
        signature: Signature {
            params,
            result: Annotation {
                ty: Ty::Int32,
                span,
            },
        },
        foreign: foreign.then(|| Foreign {
            header: "callee.h".into(),
            params: types.to_vec(),
        }),
        body: (!foreign).then_some(body),
    }
}

#[test]
fn ordinary_and_foreign_parameters_share_one_unit_single_or_tuple_slot() {
    let tuple = Ty::Record {
        fields: vec![
            RecordField {
                name: "_0".into(),
                ty: Ty::Bool,
            },
            RecordField {
                name: "_1".into(),
                ty: Ty::Int32,
            },
        ],
    };
    for (params, expected) in [
        (vec![], Ty::Unit),
        (vec![Ty::Int32], Ty::Int32),
        (vec![Ty::Bool, Ty::Int32], tuple),
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
            let checked =
                resin_lir::VerifiedModule::new(resin_lir::generate(&tree).unwrap()).unwrap();
            let function = &checked.view().module().functions[0];
            assert_eq!(function.foreign.is_some(), foreign);
            assert_eq!(
                checked.view().module().origins.functions[&FunctionId::from_index(0)],
                location,
            );
            assert_eq!(function.locals[0].ty, expected);
            assert_eq!(
                function.ty(),
                Some(Ty::Function {
                    param: Box::new(expected.clone()),
                    result: Box::new(Ty::Int32),
                })
            );
            assert_eq!(function.blocks.is_empty(), foreign);
            if foreign {
                assert_eq!(function.locals.len(), 1);
                assert_eq!(function.foreign.as_ref().unwrap().params, params);
                assert!(checked.view().analysis().functions[0].inputs.is_empty());
                assert!(checked.view().module().origins.instructions.is_empty());
            }
        }
    }
}
