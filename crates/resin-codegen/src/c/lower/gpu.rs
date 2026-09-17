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
        Instr::GpuViewAllocate => native(types, name, args, result, out),
        Instr::GpuViewRange { element } => {
            writeln!(
                out,
                "  if ({} > {} || {} > {} - {}) resin_fail(\"GPU view out of bounds\");",
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
        Instr::GpuViewCopyFrom => {
            let Ty::Pointer { pointee } = &args[2].ty else {
                unreachable!("verified GPU copy")
            };
            let bytes = checked_bytes(types, pointee, &args[3].expr, name, out);
            writeln!(
                out,
                "  if ({} < {}) resin_fail(\"GPU copy destination is too short\");",
                args[1].expr, args[3].expr
            )
            .unwrap();
            writeln!(out, "  if ({bytes}) memmove(resin_gpu_ptr_host({}, {bytes}, _Alignof({}), 2u), {}, {bytes});", args[0].expr, types.name(pointee), args[2].expr).unwrap();
            Ok("0".into())
        }
        Instr::GpuViewCopyImage => Ok(format!(
            "resin_gpu_copy_image_to_span((ResinCommandBuffer *){}, (ResinImage *){}, (ResinGpuSpan){{ .data = {}, .length = {} }})",
            args[2].expr, args[3].expr, args[0].expr, args[1].expr
        )),

        Instr::GpuRayTracingPipeline { .. }
        | Instr::GpuComputePipeline { .. }
        | Instr::GpuGraphicsPipeline { .. }
        | Instr::GpuDispatch { .. }
        | Instr::GpuDraw { .. } => {
            super::pipeline::instruction(types, name, instr, args, result, out)
        }
        Instr::GpuArgumentsTraceRays => Ok(format!(
            "resin_gpu_projected_trace_rays((ResinCommandBuffer *){}, {}, {}, {}, {})",
            args[1].expr, args[0].expr, args[2].expr, args[3].expr, args[4].expr
        )),
        Instr::GpuArgumentsDispatch => Ok(format!(
            "resin_gpu_projected_dispatch((ResinCommandBuffer *){}, {}, {}, {}, {})",
            args[1].expr, args[0].expr, args[2].expr, args[3].expr, args[4].expr
        )),
        Instr::GpuArgumentsDraw => Ok(format!(
            "resin_gpu_projected_draw((ResinCommandBuffer *){}, {}, {})",
            args[1].expr, args[0].expr, args[2].expr
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
