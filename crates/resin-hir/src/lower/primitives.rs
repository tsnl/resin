//! Checked source declarations for compiler operations. Calls use ordinary schemes.
use crate::{Arguments, Function, GenerateError, Term, TermKind, Type};

pub(super) fn define(function: &mut Function, operation: &str) -> Result<(), GenerateError> {
    let span = function.signature.result.span;
    let parameters = function
        .signature
        .type_params
        .iter()
        .map(|parameter| Type::Parameter {
            parameter: parameter.id,
        })
        .collect::<Vec<_>>();
    let (op, params, result) = super::context::primitive_signature(operation, &parameters)
        .ok_or_else(|| {
            GenerateError::inference(span, "unknown intrinsic or incorrect type parameter count")
        })?;
    let actual = function
        .signature
        .params
        .iter()
        .map(|p| p.annotation.ty.clone())
        .collect::<Vec<_>>();
    if actual != params || function.signature.result.ty != result {
        return Err(GenerateError::inference(
            span,
            format!("invalid signature for intrinsic `{operation}`"),
        ));
    }
    body(function, op);
    Ok(())
}

pub(super) fn body(function: &mut Function, op: crate::Intrinsic) {
    let span = function.signature.result.span;
    let result = function.signature.result.ty.clone();
    let params = function
        .signature
        .params
        .iter()
        .map(|p| p.annotation.ty.clone())
        .collect();
    let parameters = function
        .signature
        .type_params
        .iter()
        .map(|p| Type::Parameter { parameter: p.id })
        .collect();
    let values = function
        .signature
        .params
        .iter_mut()
        .enumerate()
        .map(|(index, parameter)| {
            let binding = index;
            parameter.binding = Some(binding);
            let place = Term {
                span: parameter.name.span,
                ty: parameter.annotation.ty.clone(),
                kind: TermKind::Local {
                    binding,
                    name: parameter.name.clone(),
                    mutable: false,
                },
            };
            if place.ty.copies_implicitly() {
                place
            } else {
                Term {
                    span: place.span,
                    ty: place.ty.clone(),
                    kind: TermKind::Move {
                        place: Box::new(place),
                    },
                }
            }
        })
        .collect::<Vec<_>>();
    function.body = Some(Term {
        span,
        ty: result,
        kind: TermKind::Intrinsic {
            op,
            type_args: parameters,
            args: Arguments { values, params },
        },
    });
}
