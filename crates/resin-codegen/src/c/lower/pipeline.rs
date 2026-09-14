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
        Instr::GpuComputePipeline {
            factory, shader, ..
        } => create(types, name, *factory, &[*shader], &args[0], result, out),
        Instr::GpuGraphicsPipeline {
            factory,
            vertex,
            fragment,
            ..
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
            projection,
            context,
            allocator,
            record: recorder,
            ..
        } => record(
            types,
            name,
            (*context, Some(*allocator), *recorder),
            Some(projection),
            args,
            result,
            out,
        ),
        Instr::GpuDraw {
            projection,
            context,
            allocator,
            record: recorder,
            ..
        } => record(
            types,
            name,
            (*context, *allocator, *recorder),
            projection.as_ref(),
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
    let span = Ty::byte_span();
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
        call(factory, &args)
    )
    .unwrap();
    let Ty::Result {
        value: pipeline, ..
    } = result
    else {
        unreachable!("pipeline result")
    };
    let metadata = resin_types::gpu_pipeline_contract(&types.module.types, pipeline)
        .expect("verified source pipeline contract");
    let token = format!(
        "(ResinGpuPipelineContract){{ .owner = {}, .root_type = {}u, .owner_type = {}u, .kind = {}u }}",
        owner_handle(
            types,
            &metadata.owner,
            &format!("{name}_factory.payload.v0")
        ),
        types.id(&metadata.root),
        types.id(&metadata.owner),
        pipeline_kind(metadata.kind)
    );
    writeln!(
        out,
        "  {} {name}_result = {{ .tag = {name}_factory.tag }};",
        types.name(result)
    )
    .unwrap();
    writeln!(
        out,
        "  if ({name}_factory.tag == 0u) {{ {name}_result.payload.v0.value.f0 = {token}; }}"
    )
    .unwrap();
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
    projection: Option<&resin_types::GpuProjectionPlan>,
    args: &[Slot],
    result: &Ty,
    out: &mut String,
) -> Result<String, Error> {
    let (context, allocator, recorder) = functions;
    let metadata = resin_types::gpu_pipeline_contract(&types.module.types, &args[1].ty)
        .expect("verified source pipeline");
    let (root, owner) = (&metadata.root, &metadata.owner);
    let draw = metadata.kind == resin_types::GpuPipelineKind::Graphics;
    let token = format!("({}).value.f0", args[1].expr);
    writeln!(out, "  if ({token}.root_type != {}u || {token}.owner_type != {}u || {token}.kind != {}u) resin_fail(\"GPU pipeline contract does not match its source wrapper\");", types.id(root), types.id(owner), pipeline_kind(metadata.kind)).unwrap();
    let native_owner = owner_value(types, owner, &format!("{token}.owner"));
    let Some(allocator) = allocator else {
        return Ok(record_call(
            types,
            recorder,
            args,
            owner,
            &native_owner,
            None,
            draw,
        ));
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
        call(context, &[types.copy(owner, &native_owner)])
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
        projection.expect("root projection"),
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
            &native_owner,
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
    native_owner: &str,
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
        types.copy(owner, native_owner),
        root,
    ];
    values.extend(args[3..].iter().map(|arg| types.copy(&arg.ty, &arg.expr)));
    call(recorder, &values)
}

fn call(function: FunctionId, args: &[String]) -> String {
    format!("r_fn{}({})", function.index(), args.join(", "))
}

fn pipeline_kind(kind: resin_types::GpuPipelineKind) -> u32 {
    match kind {
        resin_types::GpuPipelineKind::Compute => 0,
        resin_types::GpuPipelineKind::Graphics => 1,
    }
}

fn owner_handle(types: &Types<'_>, ty: &Ty, value: &str) -> String {
    match ty {
        Ty::StrongOwner => value.to_owned(),
        Ty::Defined { definition } => owner_handle(
            types,
            types.module.types[definition.index()].body().unwrap(),
            &format!("({value}).value"),
        ),
        Ty::Record { fields } => owner_handle(types, &fields[0].ty, &format!("({value}).f0")),
        _ => unreachable!("verified single shared owner facade"),
    }
}

fn owner_value(types: &Types<'_>, ty: &Ty, handle: &str) -> String {
    let payload = match ty {
        Ty::StrongOwner => return handle.to_owned(),
        Ty::Defined { definition } => format!(
            ".value = {}",
            owner_value(
                types,
                types.module.types[definition.index()].body().unwrap(),
                handle
            )
        ),
        Ty::Record { fields } => format!(".f0 = {}", owner_value(types, &fields[0].ty, handle)),
        _ => unreachable!("verified single shared owner facade"),
    };
    format!("({}){{ {payload} }}", types.name(ty))
}
