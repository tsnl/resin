//! Checked host access and ownership-preserving GPU allocation views.
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
        Instr::GpuElementLayout { element } => Ok(format!(
            "({}){{ .f0 = sizeof({}), .f1 = _Alignof({}) }}",
            types.name(result),
            types.name(element),
            types.name(element)
        )),
        Instr::GpuViewAllocate => native(types, name, args, result, out),
        Instr::GpuViewIndex { element } => {
            let index = format!("resin_index({}, {})", args[2].expr, args[1].expr);
            let offset = checked_bytes(types, element, &index, name, out);
            Ok(format!(
                "resin_gpu_ptr_offset({}, {offset}, sizeof({}), _Alignof({}))",
                args[0].expr,
                types.name(element),
                types.name(element)
            ))
        }
        Instr::GpuViewRange { element } => {
            writeln!(
                out,
                "  if ({} > {} || {} > {} - {}) resin_fail(\"GPU slice out of bounds\");",
                args[2].expr, args[1].expr, args[3].expr, args[1].expr, args[2].expr
            )
            .unwrap();
            let offset = checked_bytes(
                types,
                element,
                &args[2].expr,
                &format!("{name}_offset"),
                out,
            );
            let bytes = checked_bytes(types, element, &args[3].expr, name, out);
            Ok(format!(
                "resin_gpu_ptr_offset({}, {offset}, {bytes}, _Alignof({}))",
                args[0].expr,
                types.name(element)
            ))
        }
        Instr::GpuViewOffset => Ok(format!(
            "resin_gpu_ptr_offset({}, {}, {}, {})",
            args[0].expr, args[1].expr, args[2].expr, args[3].expr
        )),
        Instr::GpuViewRestrict => {
            writeln!(out, "  ResinGpuPtr {name}_view = {};", args[0].expr).unwrap();
            writeln!(out, "  {name}_view.access &= {};", args[1].expr).unwrap();
            Ok(format!("{name}_view"))
        }
        Instr::GpuViewLoad { element } => Ok(format!(
            "*({} *)resin_gpu_ptr_host({}, sizeof({}), _Alignof({}), 1u)",
            types.name(element),
            args[0].expr,
            types.name(element),
            types.name(element)
        )),
        Instr::GpuViewStore | Instr::GpuViewReplace => {
            let element = types.name(&args[1].ty);
            let access = if matches!(instr, Instr::GpuViewReplace) {
                3
            } else {
                2
            };
            writeln!(out, "  {element} *{name}_address = resin_gpu_ptr_host({}, sizeof({element}), _Alignof({element}), {access}u);", args[0].expr).unwrap();
            if access == 3 {
                writeln!(out, "  {element} {name}_previous = *{name}_address;").unwrap();
            }
            writeln!(out, "  *{name}_address = {};", args[1].expr).unwrap();
            Ok(if access == 3 {
                format!("{name}_previous")
            } else {
                "0".into()
            })
        }
        Instr::GpuViewCopyTo => {
            let Ty::Pointer { pointee } = &args[2].ty else {
                unreachable!("verified GPU copy")
            };
            let bytes = checked_bytes(types, pointee, &args[1].expr, name, out);
            writeln!(
                out,
                "  if ({} < {}) resin_fail(\"GPU copy destination is too short\");",
                args[3].expr, args[1].expr
            )
            .unwrap();
            writeln!(out, "  if ({bytes}) memmove({}, resin_gpu_ptr_host({}, {bytes}, _Alignof({}), 1u), {bytes});", args[2].expr, args[0].expr, types.name(pointee)).unwrap();
            Ok("0".into())
        }
        Instr::GpuViewCopyImage => Ok(format!(
            "resin_gpu_copy_image_to_span((ResinCommandBuffer *){}, (ResinImage *){}, (ResinGpuSpan){{ .data = {}, .length = {} }})",
            args[2].expr, args[3].expr, args[0].expr, args[1].expr
        )),

        Instr::GpuNew { allocator, element } | Instr::GpuAllocate { allocator, element } => {
            allocate(types, name, *allocator, element, args, result, out)
        }
        Instr::GpuAllocateNative => native(types, name, args, result, out),
        Instr::GpuReadOnly | Instr::GpuWriteOnly => {
            writeln!(
                out,
                "  {} {name}_view = {};",
                types.name(result),
                args[0].expr
            )
            .unwrap();
            let access = if matches!(result, Ty::GpuSpan { .. }) {
                "data.access"
            } else {
                "access"
            };
            let mask = if matches!(instr, Instr::GpuReadOnly) {
                1
            } else {
                2
            };
            writeln!(out, "  {name}_view.{access} &= {mask}u;").unwrap();
            Ok(format!("{name}_view"))
        }
        Instr::GpuSlice => slice(types, name, args, out),
        Instr::GpuCopyTo => {
            let Ty::GpuSpan { element } = &args[0].ty else {
                unreachable!()
            };
            let source = &args[0].expr;
            let target = &args[1].expr;
            let bytes = checked_bytes(types, element, &format!("({source}).length"), name, out);
            writeln!(out, "  if (({target}).f1 < ({source}).length) resin_fail(\"GPU copy destination is too short\");").unwrap();
            writeln!(out, "  if ({bytes}) memmove(({target}).f0, resin_gpu_ptr_host(({source}).data, {bytes}, _Alignof({}), 1u), {bytes});", types.name(element)).unwrap();
            Ok("0".into())
        }
        Instr::GpuComputePipeline { .. }
        | Instr::GpuGraphicsPipeline { .. }
        | Instr::GpuDispatch { .. }
        | Instr::GpuDraw { .. } => {
            super::pipeline::instruction(types, name, instr, args, result, out)
        }
        Instr::GpuArgumentsDispatch => Ok(format!(
            "resin_gpu_projected_dispatch((ResinCommandBuffer *){}, {}, {}, {}, {})",
            args[1].expr, args[0].expr, args[2].expr, args[3].expr, args[4].expr
        )),
        Instr::GpuArgumentsDraw => Ok(format!(
            "resin_gpu_projected_draw((ResinCommandBuffer *){}, {}, {})",
            args[1].expr, args[0].expr, args[2].expr
        )),
        Instr::GpuCopyImage => Ok(format!(
            "resin_gpu_copy_image_to_span((ResinCommandBuffer *){}, (ResinImage *){}, {})",
            args[1].expr, args[2].expr, args[0].expr
        )),
        _ => unreachable!("GPU operation dispatch"),
    }
}

fn native(
    types: &Types<'_>,
    name: &str,
    args: &[Slot],
    result: &Ty,
    out: &mut String,
) -> Result<String, Error> {
    let Ty::Record { fields } = result else {
        unreachable!()
    };
    let gpu = fields[0]
        .ty
        .without_none()
        .expect("verified GPU allocation result");
    let success = types.tag(&Case::Type(gpu));
    let none = types.tag(&Case::Type(Ty::None));
    writeln!(out, "  ResinGpuPtr {name}_pointer = {{0}};").unwrap();
    writeln!(out, "  int32_t {name}_status = resin_gpu_ptr_allocate((ResinGpu *){}, {}, {}, {}, {}, &{name}_pointer);", args[0].expr, args[1].expr, args[2].expr, args[3].expr, args[4].expr).unwrap();
    let value = format!(
        "({name}_status == 0 ? ({}){{ .tag = {success}u, .payload.v{success} = {name}_pointer }} : ({}){{ .tag = {none}u, .payload.v{none} = 0 }})",
        types.name(&fields[0].ty),
        types.name(&fields[0].ty)
    );
    Ok(format!(
        "({}){{ .f0 = {value}, .f1 = {name}_status }}",
        types.name(result)
    ))
}

fn allocate(
    types: &Types<'_>,
    name: &str,
    allocator: FunctionId,
    element: &Ty,
    args: &[Slot],
    result: &Ty,
    out: &mut String,
) -> Result<String, Error> {
    let initialized =
        matches!(result, Ty::Result { value, .. } if matches!(&**value, Ty::GpuPointer { .. }));
    let function = &types.module.functions[allocator.index()];
    let count = if initialized { "1" } else { &args[1].expr };
    let bytes = checked_bytes(types, element, count, name, out);
    writeln!(
        out,
        "  {} {name}_allocation = r_fn{}({}, {bytes}, _Alignof({}), 0);",
        types.name(&function.result),
        allocator.index(),
        args[0].expr,
        types.name(element)
    )
    .unwrap();
    writeln!(out, "  {} {name}_result = {{0}};", types.name(result)).unwrap();
    writeln!(
        out,
        "  {name}_result.tag = {name}_allocation.tag;\n  if ({name}_allocation.tag == 0u) {{"
    )
    .unwrap();
    if initialized {
        writeln!(out, "    *({} *)resin_gpu_ptr_host({name}_allocation.payload.v0, sizeof({}), _Alignof({}), 2u) = {};", types.name(element), types.name(element), types.name(element), args[1].expr).unwrap();
        writeln!(
            out,
            "    {name}_result.payload.v0 = {name}_allocation.payload.v0;"
        )
        .unwrap();
    } else {
        writeln!(out, "    {name}_result.payload.v0 = (ResinGpuSpan){{ .data = resin_gpu_ptr_offset({name}_allocation.payload.v0, 0, {bytes}, _Alignof({})), .length = {count} }};", types.name(element)).unwrap();
    }
    writeln!(
        out,
        "  }} else {{ {name}_result.payload.v1 = {name}_allocation.payload.v1; }}"
    )
    .unwrap();
    Ok(format!("{name}_result"))
}

fn checked_bytes(
    types: &Types<'_>,
    element: &Ty,
    count: &str,
    name: &str,
    out: &mut String,
) -> String {
    writeln!(out, "  if ((uint64_t)({count}) > SIZE_MAX / sizeof({})) resin_fail(\"GPU allocation or view size overflow\");", types.name(element)).unwrap();
    let bytes = format!("{name}_bytes");
    writeln!(
        out,
        "  size_t {bytes} = (size_t)({count}) * sizeof({});",
        types.name(element)
    )
    .unwrap();
    bytes
}

fn slice(types: &Types<'_>, name: &str, args: &[Slot], out: &mut String) -> Result<String, Error> {
    let source = &args[0].expr;
    let start = &args[1].expr;
    let length = &args[2].expr;
    let (element, pointer) = match &args[0].ty {
        Ty::GpuSpan { element } => {
            writeln!(out, "  if (({start}) > ({source}).length || ({length}) > ({source}).length - ({start})) resin_fail(\"GPU slice out of bounds\");").unwrap();
            (element, format!("({source}).data"))
        }
        Ty::GpuPointer { pointee } => (pointee, source.clone()),
        _ => unreachable!("verified GPU slice receiver"),
    };
    let bytes = checked_bytes(types, element, length, name, out);
    let offset = checked_bytes(types, element, start, &format!("{name}_offset"), out);
    Ok(format!(
        "(ResinGpuSpan){{ .data = resin_gpu_ptr_offset({pointer}, {offset}, {bytes}, _Alignof({})), .length = {length} }}",
        types.name(element)
    ))
}

pub(super) fn host_address(types: &Types<'_>, source: &Slot, required: u32) -> String {
    if let Ty::GpuPointer { pointee } = &source.ty {
        format!(
            "({} *)resin_gpu_ptr_host({}, sizeof({}), _Alignof({}), {required}u)",
            types.name(pointee),
            source.expr,
            types.name(pointee),
            types.name(pointee)
        )
    } else {
        types.unwrap(&source.ty, source.expr.clone())
    }
}

pub(super) fn project(
    types: &Types<'_>,
    source: &Slot,
    index: &str,
    dynamic: bool,
    result: &Ty,
    name: &str,
    out: &mut String,
) -> Result<Option<String>, Error> {
    let (base, element, count) = match &source.ty {
        Ty::GpuPointer { pointee } => (source.expr.clone(), &**pointee, None),
        Ty::GpuSpan { element } if dynamic => (
            format!("({}).data", source.expr),
            &**element,
            Some(format!("({}).length", source.expr)),
        ),
        _ => return Ok(None),
    };
    let Ty::GpuPointer { pointee } = result else {
        unreachable!("GPU address projection")
    };
    let offset = if dynamic {
        let checked_index = count.map_or_else(
            || index.to_string(),
            |count| format!("resin_index((uint64_t)({index}), {count})"),
        );
        checked_bytes(types, element, &checked_index, name, out)
    } else {
        let index: usize = index.parse().unwrap();
        let layout = crate::layout::layout(types.module, element)?;
        match types.shape(element) {
            Ty::Array { element, .. } => {
                (crate::layout::layout(types.module, element)?.size * index).to_string()
            }
            _ => layout.offsets[index].to_string(),
        }
    };
    Ok(Some(format!(
        "resin_gpu_ptr_offset({base}, {offset}, sizeof({}), _Alignof({}))",
        types.name(pointee),
        types.name(pointee)
    )))
}
