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
    if matches!(
        operation,
        "gpu_compute_pipeline_type"
            | "gpu_graphics_pipeline_type"
            | "gpu_ray_tracing_pipeline_type"
    ) {
        return pipeline(context, function, declaration, operation);
    }
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
    if arguments != std::slice::from_ref(&bound) {
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
    let pointer_target = match kind {
        GpuProjectionKind::Pointer => output.clone(),
        GpuProjectionKind::Sequence => {
            let Some(Type::Record { fields }) = body(context, output) else {
                return Err(invalid());
            };
            fields.first().ok_or_else(invalid)?.ty.clone()
        }
    };
    let Type::Pointer { pointee, .. } = &pointer_target else {
        return Err(invalid());
    };
    if pointee.as_ref() != &bound {
        return Err(invalid());
    }
    match kind {
        GpuProjectionKind::Pointer => {
            if fields.len() != 1 || fields[0].ty != Type::GpuView {
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
            let scheme = context.nominal_schemes.get(pointer).ok_or_else(invalid)?;
            let projection = scheme.gpu_projection.as_ref().ok_or_else(invalid)?;
            if arguments != std::slice::from_ref(&bound)
                || projection.kind != GpuProjectionKind::Pointer
                || substitute(&projection.target, &scheme.type_params, arguments.clone())
                    != Some(pointer_target.clone())
            {
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

pub(super) fn body(context: &Context, ty: &Type) -> Option<Type> {
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

pub(super) fn substitute(
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

fn pipeline(
    context: &mut Context,
    function: &mut Function,
    declaration: FunctionId,
    operation: &str,
) -> Result<bool, GenerateError> {
    let invalid = || {
        GenerateError::inference(
            function.signature.result.span,
            format!("invalid signature or wrapper storage for intrinsic `{operation}`"),
        )
    };
    let [root, owner] = function.signature.type_params.as_slice() else {
        return Err(invalid());
    };
    let [input] = function.signature.params.as_slice() else {
        return Err(invalid());
    };
    let Type::Defined {
        definition,
        arguments,
    } = &function.signature.result.ty
    else {
        return Err(invalid());
    };
    let source = context
        .nominal_schemes
        .get(definition)
        .ok_or_else(invalid)?;
    let bound = |parameter: &crate::TypeParameter| Type::Parameter {
        parameter: parameter.id,
    };
    if input.annotation.ty != Type::GpuPipelineContract
        || arguments != &[bound(root), bound(owner)]
        || source.type_params.len() != 2
        || source.drop.is_some()
        || source.gpu_pipeline.is_some()
    {
        return Err(invalid());
    }
    let Type::Record { fields } = &source.body else {
        return Err(invalid());
    };
    if fields.len() != 1 || fields[0].ty != Type::GpuPipelineContract {
        return Err(invalid());
    }
    let kind = if operation == "gpu_ray_tracing_pipeline_type" {
        resin_types::GpuPipelineKind::RayTracing
    } else if operation == "gpu_compute_pipeline_type" {
        resin_types::GpuPipelineKind::Compute
    } else {
        resin_types::GpuPipelineKind::Graphics
    };
    if context.nominal_schemes.values().any(|source| {
        source
            .gpu_pipeline
            .as_ref()
            .is_some_and(|pipeline| pipeline.kind == kind)
    }) {
        return Err(invalid());
    }
    let contract = crate::GpuPipeline {
        declaration,
        kind,
        root: bound(&source.type_params[0]),
        owner: bound(&source.type_params[1]),
    };
    context
        .nominal_schemes
        .get_mut(definition)
        .unwrap()
        .gpu_pipeline = Some(contract);
    super::primitives::body(function, Intrinsic::GpuPipelineType);
    Ok(true)
}
