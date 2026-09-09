use crate::lower::context::Context;
use crate::lower::infer;
use crate::{GenerateError, GenerateErrorKind};
use resin_source::prelude::*;
use resin_types::prelude::*;

fn infer_relation(
    typer: &mut Context,
    relation: impl FnOnce(infer::Type) -> infer::Constraint,
) -> Result<Ty, GenerateError> {
    let mut inference = infer::Inference::new(typer);
    let (rule, out) = inference.expression();
    inference.constrain(rule, (Span { start: 0, end: 0 }, relation(out.clone())));
    if let Some(error) = inference
        .solve(std::slice::from_ref(&out))
        .into_iter()
        .next()
    {
        return Err(error);
    }
    inference.solver.require(&out, Span { start: 0, end: 0 })
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
fn empty_arrays_need_an_injected_element_type() {
    let compile = |source: &str| {
        crate::generate(
            &resin_ast::generate(&resin_cst::Document::reparse(source.into(), None)).unwrap(),
        )
    };
    assert_eq!(
        compile("def main() = { []; };").unwrap_err().kind,
        GenerateErrorKind::Type {
            kind: TypeErrorKind::EmptyArrayNeedsElementType
        }
    );
    let mut typer = Context::new();
    let empty = Ty::Array {
        element: Box::new(Ty::UInt8),
        length: 0,
    };
    assert_eq!(
        infer_relation(&mut typer, |out| infer::Constraint::Equal(
            empty.clone().into(),
            out
        ))
        .unwrap(),
        empty
    );
}

#[test]
fn field_access_preserves_nominal_identity_and_autoderefs_pointers() {
    let mut typer = Context::new();
    let body = record(Ty::Int32);
    let definition = typer.create_type("Node", body.clone()).unwrap();
    let node = Ty::Defined { definition };
    let pointer = Ty::Pointer {
        pointee: Box::new(body.clone()),
    };
    for (base, steps) in [
        (node.clone(), vec![Conv::Unwrap { definition }]),
        (pointer, vec![Conv::Deref]),
    ] {
        assert_eq!(
            typer.type_field(&base, "value").unwrap(),
            FieldAccess {
                ty: Ty::Int32,
                index: 0,
                steps
            }
        );
    }
    assert!(matches!(
        typer.same(&node, &body),
        Err(TypeError {
            kind: TypeErrorKind::TypeMismatch { .. }
        })
    ));
    let pointer = Ty::Pointer {
        pointee: Box::new(Ty::Int64),
    };
    assert_eq!(
        infer_relation(&mut typer, |out| infer::Constraint::Deref(
            pointer.clone().into(),
            out
        ))
        .unwrap(),
        Ty::Int64
    );
    assert!(matches!(
        typer.type_field(&pointer, "value"),
        Err(TypeError {
            kind: TypeErrorKind::ExpectedRecord { .. }
        })
    ));
}

#[test]
fn convert_does_not_unwrap_function_arguments() {
    let mut typer = Context::new();
    let meters = Ty::Defined {
        definition: typer.create_type("Meters", record(Ty::Int32)).unwrap(),
    };
    let callee = Ty::Function {
        param: Box::new(Ty::Int32),
        result: Box::new(Ty::Int32),
    };
    assert!(matches!(
        infer_relation(&mut typer, |out| infer::Constraint::Call(
            callee.into(),
            meters.into(),
            out
        )),
        Err(GenerateError {
            kind: GenerateErrorKind::Type {
                kind: TypeErrorKind::TypeMismatch { .. }
            },
            ..
        })
    ));
}

#[test]
fn ascription_wraps_records_but_does_not_flatten_nested_fields() {
    let mut typer = Context::new();
    let body = record(Ty::Int32);
    let meters = Ty::Defined {
        definition: typer.create_type("Meters", body.clone()).unwrap(),
    };
    let distance_body = record(meters.clone());
    let distance = Ty::Defined {
        definition: typer
            .create_type("Distance", distance_body.clone())
            .unwrap(),
    };
    assert!(typer.ascribe(&distance_body, &distance).is_ok());
    assert!(typer.ascribe(&meters, &distance).is_err());
    assert!(typer.ascribe(&Ty::Int32, &meters).is_err());
    assert!(typer.as_bool(&meters).is_err());
    assert!(
        infer_relation(&mut typer, |out| infer::Constraint::Deref(
            meters.clone().into(),
            out
        ))
        .is_err()
    );
    assert!(
        infer_relation(&mut typer, |out| infer::Constraint::Call(
            meters.into(),
            Ty::Unit.into(),
            out
        ))
        .is_err()
    );
}

use super::{Solver, Type};
const SPAN: Span = Span { start: 0, end: 1 };

#[test]
fn nested_variables_are_independent() {
    let mut solver = Solver::default();
    let a = solver.fresh();
    let b = solver.fresh();
    let ty = Type::function(Type::pointer(a.clone()), Type::pointer(b.clone()));
    let concrete = Type::function(
        Type::pointer(Ty::Int32.into()),
        Type::pointer(Ty::Bool.into()),
    );
    solver.unify(&ty, &concrete, SPAN).unwrap();
    assert_eq!(solver.resolve(&a), Some(Ty::Int32));
    assert_eq!(solver.resolve(&b), Some(Ty::Bool));
}

#[test]
fn cycles_and_ambiguity_are_errors_not_unit() {
    let mut solver = Solver::default();
    let a = solver.fresh();
    let b = solver.fresh();
    solver.unify(&a, &Type::pointer(b.clone()), SPAN).unwrap();
    assert!(solver.unify(&a, &b, SPAN).is_err());
    assert!(solver.require(&a, SPAN).is_err());
}

#[test]
fn numeric_defaults_wait_for_context() {
    let mut solver = Solver::default();
    let literal = solver.number("1");
    let other = solver.number("1.0");
    let hex = solver.number("-0xdead");
    solver.unify(&literal, &Ty::UInt64.into(), SPAN).unwrap();
    solver.default_numbers(&[literal.clone(), other.clone(), hex.clone()]);
    assert_eq!(solver.resolve(&literal), Some(Ty::UInt64));
    assert_eq!(solver.resolve(&other), Some(Ty::Float64));
    assert_eq!(solver.resolve(&hex), Some(Ty::Int64));
}
