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
    let mut tree = conditional();
    tree.functions.push(tree.functions[0].clone());
    tree.functions[0].location = Some(SourceLocation {
        source: source.clone(),
        span: Span { start: 0, end: 6 },
    });
    let module = resin_lir::generate(&tree).unwrap();
    drop(tree);
    assert_eq!(module.origins.functions.len(), 1);
    assert!(!module.origins.instructions.is_empty());
    for ((function, _, _), origin) in &module.origins.instructions {
        assert_eq!(*function, FunctionId::from_index(0));
        assert_eq!(origin.source, source);
        assert_eq!(origin.source.text(), "choose");
    }
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
            let tree = Module {
                functions: vec![parameter_function(&params, foreign)],
                ..Default::default()
            };
            let checked =
                resin_lir::VerifiedModule::new(resin_lir::generate(&tree).unwrap()).unwrap();
            let function = &checked.view().module().functions[0];
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
                assert_eq!(function.foreign.as_ref().unwrap().params, params);
                assert!(checked.view().analysis().functions[0].inputs.is_empty());
            }
        }
    }
}
