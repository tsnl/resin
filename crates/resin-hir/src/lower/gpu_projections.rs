//! Register source wrapper contracts without recognizing library names.
use super::{
    context::Context,
    infer::{Solver, Type as Inferred},
};
use crate::{Function, FunctionId, GenerateError, GpuProjection, Type};
use resin_types::{GpuProjectionKind, Intrinsic};

pub(super) fn define(
    context: &mut Context,
    function: &mut Function,
    declaration: FunctionId,
    operation: &str,
) -> Result<bool, GenerateError> {
    let (kind, op) = match operation {
        "gpu_pointer_projection" => (GpuProjectionKind::Pointer, Intrinsic::GpuPointerProjection),
        "gpu_span_projection" => (
            GpuProjectionKind::Sequence,
            Intrinsic::GpuSequenceProjection,
        ),
        _ => return Ok(false),
    };
    let span = function.signature.result.span;
    let invalid = || {
        GenerateError::inference(
            span,
            format!("invalid signature or wrapper storage for intrinsic `{operation}`"),
        )
    };
    let [parameter] = function.signature.type_params.as_slice() else {
        return Err(invalid());
    };
    let [input] = function.signature.params.as_slice() else {
        return Err(invalid());
    };
    let Type::Defined {
        definition,
        arguments,
    } = &input.annotation.ty
    else {
        return Err(invalid());
    };
    let bound = Type::Parameter {
        parameter: parameter.id,
    };
    if arguments != &[bound.clone()] {
        return Err(invalid());
    }
    let source = context
        .nominal_schemes
        .get(definition)
        .ok_or_else(invalid)?;
    if source.type_params.len() != 1 || source.drop.is_some() || source.gpu_projection.is_some() {
        return Err(invalid());
    }
    let output = &function.signature.result.ty;
    let Type::Record { fields } = body(context, &input.annotation.ty).ok_or_else(invalid)? else {
        return Err(invalid());
    };
    match kind {
        GpuProjectionKind::Pointer => {
            if fields.len() != 1
                || fields[0].ty != Type::GpuView
                || *output
                    != (Type::Pointer {
                        pointee: Box::new(bound.clone()),
                    })
            {
                return Err(invalid());
            }
        }
        GpuProjectionKind::Sequence => {
            let Type::Record {
                fields: output_fields,
            } = body(context, output).ok_or_else(invalid)?
            else {
                return Err(invalid());
            };
            if fields.len() != 2
                || output_fields.len() != 2
                || fields[1].ty != Type::UInt64
                || output_fields[1].ty != Type::UInt64
                || output_fields[0].ty
                    != (Type::Pointer {
                        pointee: Box::new(bound.clone()),
                    })
            {
                return Err(invalid());
            }
            let Type::Defined {
                definition: pointer,
                arguments,
            } = &fields[0].ty
            else {
                return Err(invalid());
            };
            let pointer = context
                .nominal_schemes
                .get(pointer)
                .and_then(|scheme| scheme.gpu_projection.as_ref())
                .ok_or_else(invalid)?;
            if arguments != &[bound] || pointer.kind != GpuProjectionKind::Pointer {
                return Err(invalid());
            }
        }
    }
    let target = substitute(
        output,
        &function.signature.type_params,
        source
            .type_params
            .iter()
            .map(|p| Type::Parameter { parameter: p.id })
            .collect(),
    )
    .ok_or_else(invalid)?;
    context
        .nominal_schemes
        .get_mut(definition)
        .unwrap()
        .gpu_projection = Some(GpuProjection {
        declaration,
        kind,
        target,
    });
    super::primitives::body(function, op);
    Ok(true)
}

fn body(context: &Context, ty: &Type) -> Option<Type> {
    let Type::Defined {
        definition,
        arguments,
    } = ty
    else {
        return Some(ty.clone());
    };
    let scheme = context.nominal_schemes.get(definition)?;
    substitute(&scheme.body, &scheme.type_params, arguments.clone())
}

fn substitute(
    ty: &Type,
    parameters: &[crate::TypeParameter],
    arguments: Vec<Type>,
) -> Option<Type> {
    let mut solver = Solver::default();
    let (applied, _) = solver
        .apply(
            Inferred::from_hir(ty),
            parameters,
            Some(arguments.iter().map(Inferred::from_hir).collect()),
            resin_source::Span { start: 0, end: 0 },
        )
        .ok()?;
    solver.complete(&applied)
}
