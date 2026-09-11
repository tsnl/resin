//! Bind a declared shader contract to its source owner and project only when recording.
use super::{Slot, types::Types};
use crate::Error;
use resin_lir::Instr;
use resin_types::prelude::*;
use std::fmt::Write;

pub(super) fn instruction(
    types: &Types<'_>,
    name: &str,
    instr: &Instr,
    args: &[Slot],
    result: &Ty,
    out: &mut String,
) -> Result<String, Error> {
    match instr {
        Instr::GpuComputePipeline { factory, shader } => {
            create(types, name, *factory, &[*shader], &args[0], result, out)
        }
        Instr::GpuGraphicsPipeline {
            factory,
            vertex,
            fragment,
        } => create(
            types,
            name,
            *factory,
            &[*vertex, *fragment],
            &args[0],
            result,
            out,
        ),
        Instr::GpuDispatch {
            context,
            allocator,
            record: recorder,
        } => record(
            types,
            name,
            (*context, Some(*allocator), *recorder),
            args,
            result,
            out,
        ),
        Instr::GpuDraw {
            context,
            allocator,
            record: recorder,
        } => record(
            types,
            name,
            (*context, *allocator, *recorder),
            args,
            result,
            out,
        ),
        _ => unreachable!("verified pipeline operation"),
    }
}

fn create(
    types: &Types<'_>,
    name: &str,
    factory: FunctionId,
    shaders: &[FunctionId],
    gpu: &Slot,
    result: &Ty,
    out: &mut String,
) -> Result<String, Error> {
    let function = &types.module.functions[factory.index()];
    let mut args = vec![types.copy(&gpu.ty, &gpu.expr)];
    let span = Ty::Span {
        element: Box::new(Ty::UInt8),
    };
    for &shader in shaders {
        let symbol = crate::shader_symbol(shader);
        args.push(format!(
            "({}){{ (uint8_t *){symbol}, {symbol}_length }}",
            types.name(&span)
        ));
    }
    writeln!(
        out,
        "  {} {name}_factory = {};",
        types.name(&function.result),
        call(types, factory, &args)
    )
    .unwrap();
    // Both success payloads have the owner's representation. The new type is
    // established here, with the declared shader identities verified in LIR.
    writeln!(
        out,
        "  {} {name}_result = {{ .tag = {name}_factory.tag }};",
        types.name(result)
    )
    .unwrap();
    writeln!(out, "  if ({name}_factory.tag == 0u) {{ {name}_result.payload.v0 = {name}_factory.payload.v0; }}").unwrap();
    writeln!(
        out,
        "  else {{ {name}_result.payload.v1 = {name}_factory.payload.v1; }}"
    )
    .unwrap();
    Ok(format!("{name}_result"))
}

fn record(
    types: &Types<'_>,
    name: &str,
    functions: (FunctionId, Option<FunctionId>, FunctionId),
    args: &[Slot],
    result: &Ty,
    out: &mut String,
) -> Result<String, Error> {
    let (context, allocator, recorder) = functions;
    let (root, owner) = args[1]
        .ty
        .gpu_pipeline()
        .expect("verified pipeline operand");
    let draw = matches!(&args[1].ty, Ty::GpuGraphicsPipeline { .. });
    let Some(allocator) = allocator else {
        return Ok(record_call(types, recorder, args, owner, None, draw));
    };
    let gpu_type = &types.module.functions[context.index()].result;
    let gpu = Slot {
        ty: gpu_type.clone(),
        expr: format!("{name}_gpu"),
        live: None,
    };
    writeln!(
        out,
        "  {} {} = {};",
        types.name(gpu_type),
        gpu.expr,
        call(types, context, &[types.copy(owner, &args[1].expr)])
    )
    .unwrap();
    let Ty::Result { error, .. } = result else {
        unreachable!("verified recording result");
    };
    let projection_result = Ty::Result {
        value: Box::new(Ty::GpuArguments),
        error: error.clone(),
    };
    let projected = super::projection::project(
        types,
        &format!("{name}_projected"),
        allocator,
        root,
        &[gpu.clone(), args[2].clone()],
        &projection_result,
        out,
    )?;
    types.drop_value(gpu_type, &gpu.expr, out);
    writeln!(out, "  {} {name}_result = {{0}};", types.name(result)).unwrap();
    writeln!(out, "  if ({projected}.tag == 0u) {{").unwrap();
    writeln!(
        out,
        "    {name}_result = {};",
        record_call(
            types,
            recorder,
            args,
            owner,
            Some(&format!("{projected}.payload.v0")),
            draw
        )
    )
    .unwrap();
    writeln!(
        out,
        "  }} else {{ {name}_result.tag = 1u; {name}_result.payload.v1 = {projected}.payload.v1; }}"
    )
    .unwrap();
    Ok(format!("{name}_result"))
}

fn record_call(
    types: &Types<'_>,
    recorder: FunctionId,
    args: &[Slot],
    owner: &Ty,
    projection: Option<&str>,
    draw: bool,
) -> String {
    let root = if draw {
        let union = Ty::union_of([Ty::GpuArguments, Ty::None]);
        let tag = types.tag(&Case::Type(if projection.is_some() {
            Ty::GpuArguments
        } else {
            Ty::None
        }));
        format!(
            "({}){{ .tag = {tag}u, .payload.v{tag} = {} }}",
            types.name(&union),
            projection.unwrap_or("0")
        )
    } else {
        projection.expect("compute projection").to_owned()
    };
    let mut values = vec![
        types.copy(&args[0].ty, &args[0].expr),
        types.copy(owner, &args[1].expr),
        root,
    ];
    values.extend(args[3..].iter().map(|arg| types.copy(&arg.ty, &arg.expr)));
    call(types, recorder, &values)
}

fn call(types: &Types<'_>, function: FunctionId, args: &[String]) -> String {
    let parameter = &types.module.functions[function.index()].locals[0].ty;
    let arg = if args.len() == 1 {
        args[0].clone()
    } else {
        format!("({}){{ {} }}", types.name(parameter), args.join(", "))
    };
    format!("r_fn{}({arg})", function.index())
}
