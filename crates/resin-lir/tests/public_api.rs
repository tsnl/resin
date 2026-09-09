//! Lower a resolved tree without a parser, source provider, or compiler session.
use resin_common::source::Span;
use resin_hir::{Annotation, Function, Module, Signature, Term, TermKind};
use resin_lir::{Instr, Ty, Value};

fn constant(value: Value, ty: Ty) -> Term {
    Term {
        span: Span { start: 0, end: 0 },
        ty,
        kind: TermKind::Constant(value),
    }
}

fn conditional() -> Module {
    Module {
        functions: vec![Function {
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
