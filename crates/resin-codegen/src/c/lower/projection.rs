//! Build shader argument storage without exposing device addresses to host source.
use super::{Slot, types::Types};
use crate::Error;
use resin_types::prelude::*;
use std::fmt::Write;

pub(super) fn project(
    types: &Types<'_>,
    name: &str,
    allocator: FunctionId,
    plan: &resin_types::GpuProjectionPlan,
    args: &[Slot],
    result: &Ty,
    out: &mut String,
) -> Result<String, Error> {
    let root = &plan.target;
    let allocate = &types.module.functions[allocator.index()];
    let ok = types.tag(&Case::Ok);
    let err = types.tag(&Case::Err);
    let root_name = types.name(root);
    writeln!(out, "  {} {name}_result = {{0}};", types.name(result)).unwrap();
    writeln!(
        out,
        "  {} {name}_allocation = r_fn{}({}, sizeof({root_name}), _Alignof({root_name}), 0);",
        types.name(&allocate.result),
        allocator.index(),
        types.copy(&args[0].ty, &args[0].expr)
    )
    .unwrap();
    writeln!(out, "  if ({name}_allocation.tag == {ok}u) {{").unwrap();
    // An allocator is source code: validate the returned view against the
    // requested root, even if its body ignored the byte count or alignment.
    writeln!(out, "    ResinGpuPtr {name}_root_view = resin_gpu_ptr_offset({name}_allocation.payload.v{ok}, 0, sizeof({root_name}), _Alignof({root_name}));").unwrap();
    writeln!(out, "    (void)resin_gpu_ptr_host({name}_root_view, sizeof({root_name}), _Alignof({root_name}), 3u);").unwrap();
    writeln!(out, "    ResinArc *{name}_projection = resin_gpu_projection_new({name}_allocation.payload.v{ok});").unwrap();
    writeln!(
        out,
        "    resin_arc_release({name}_allocation.payload.v{ok}.owner);"
    )
    .unwrap();
    writeln!(
        out,
        "    {root_name} *{name}_root = resin_gpu_projection_root({name}_projection);"
    )
    .unwrap();
    project_value(
        types,
        plan,
        &args[1].expr,
        &format!("(*{name}_root)"),
        &format!("{name}_projection"),
        0,
        out,
    )?;
    writeln!(out, "    {name}_result.tag = {ok}u;").unwrap();
    writeln!(out, "    {name}_result.payload.v{ok} = {name}_projection;").unwrap();
    writeln!(out, "  }} else {{").unwrap();
    writeln!(out, "    {name}_result.tag = {err}u;").unwrap();
    writeln!(
        out,
        "    {name}_result.payload.v{err} = {name}_allocation.payload.v{err};"
    )
    .unwrap();
    writeln!(out, "  }}").unwrap();
    Ok(format!("{name}_result"))
}

fn representation<'a>(types: &'a Types<'_>, ty: &'a Ty, value: &str) -> (&'a Ty, String) {
    match ty {
        Ty::Defined { definition } => representation(
            types,
            types.module.types[definition.index()].body().unwrap(),
            &format!("({value}).value"),
        ),
        _ => (ty, value.to_owned()),
    }
}

fn project_value(
    types: &Types<'_>,
    plan: &resin_types::GpuProjectionPlan,
    source: &str,
    destination: &str,
    projection: &str,
    depth: usize,
    out: &mut String,
) -> Result<(), Error> {
    use resin_types::GpuProjectionOperation;
    if matches!(plan.operation, GpuProjectionOperation::Copy) {
        writeln!(out, "    {destination} = {source};").unwrap();
        return Ok(());
    }
    let (source_type, source) = representation(types, &plan.source, source);
    let (_, destination) = representation(types, &plan.target, destination);
    match &plan.operation {
        GpuProjectionOperation::Copy => unreachable!(),
        GpuProjectionOperation::Pointer { element } => {
            let ty = types.name(element);
            writeln!(out, "    {destination} = ({}) (uintptr_t) resin_gpu_projection_pointer({projection}, ({source}).f0, sizeof({ty}), _Alignof({ty}));", types.name(&plan.target)).unwrap();
        }
        GpuProjectionOperation::Sequence { element } => {
            let Ty::Record { fields } = source_type else {
                unreachable!("sequence source representation")
            };
            let (_, pointer) = representation(types, &fields[0].ty, &format!("({source}).f0"));
            let ty = types.name(element);
            writeln!(out, "    if (({source}).f1 > SIZE_MAX / sizeof({ty})) resin_fail(\"GPU projection length overflow\");").unwrap();
            let bytes = format!("(({source}).f1 * sizeof({ty}))");
            writeln!(out, "    ({destination}).f0 = ({ty} *) (uintptr_t) resin_gpu_projection_pointer({projection}, ({pointer}).f0, {bytes}, _Alignof({ty}));").unwrap();
            writeln!(out, "    ({destination}).f1 = ({source}).f1;").unwrap();
        }
        GpuProjectionOperation::Record { fields } => {
            for (index, field) in fields.iter().enumerate() {
                project_value(
                    types,
                    field,
                    &format!("({source}).f{index}"),
                    &format!("({destination}).f{index}"),
                    projection,
                    depth,
                    out,
                )?;
            }
        }
        GpuProjectionOperation::Array { element, length } => {
            let index = format!("r_projection_i{depth}");
            writeln!(
                out,
                "    for (size_t {index} = 0; {index} < {length}; ++{index}) {{"
            )
            .unwrap();
            project_value(
                types,
                element,
                &format!("({source}).items[{index}]"),
                &format!("({destination}).items[{index}]"),
                projection,
                depth + 1,
                out,
            )?;
            writeln!(out, "    }}").unwrap();
        }
    }
    Ok(())
}
