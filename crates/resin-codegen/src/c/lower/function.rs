use crate::c::{CBody, CFunction, CStatement};
use resin_lir::FunctionTypes;
use resin_types::prelude::*;
use std::fmt::Write;

use crate::Error;
use resin_lir::{Instr, Terminator};

use super::{Slot, ops, types::Types, value::literal};

pub(super) fn lower(
    types: &Types<'_>,
    index: usize,
    flow: &FunctionTypes,
) -> Result<CFunction, Error> {
    let function = &types.module.functions[index];
    let signature = signature(types, index);
    if let Some(foreign) = &function.foreign {
        return Ok(CFunction {
            signature,
            body: CBody::Inline(super::foreign::lower(types, index, foreign)),
        });
    }
    let locals = locals(types, function, flow);
    let statements = lower_region(types, index, function.entry.index(), flow, None, None)?;
    Ok(CFunction {
        signature,
        body: CBody::Structured { locals, statements },
    })
}

pub(super) fn signature(types: &Types<'_>, index: usize) -> String {
    let function = &types.module.functions[index];
    let params = function.locals[..function.parameter_count]
        .iter()
        .enumerate()
        .map(|(i, local)| format!("{} r_arg{i}", types.name(&local.ty)))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{} r_fn{index}({})",
        types.name(&function.result),
        if params.is_empty() { "void" } else { &params }
    )
}

fn locals(types: &Types<'_>, function: &resin_lir::Function, flow: &FunctionTypes) -> String {
    let mut out = String::new();
    for (i, local) in function.locals.iter().enumerate() {
        writeln!(
            out,
            "  {} r_l{i} = {{0}}; (void)r_l{i};",
            types.name(&local.ty)
        )
        .unwrap();
    }
    for (i, local) in function.locals.iter().enumerate() {
        if local.ty.needs_drop(&types.module.types) {
            writeln!(
                out,
                "  bool r_live{i} = {};",
                if i < function.parameter_count {
                    "true"
                } else {
                    "false"
                }
            )
            .unwrap();
        }
    }
    for i in 0..function.parameter_count {
        writeln!(out, "  r_l{i} = r_arg{i};").unwrap();
    }
    for (block, inputs) in flow.inputs.iter().enumerate() {
        for (i, ty) in inputs.iter().enumerate() {
            writeln!(out, "  {} r_b{block}_{i};", types.name(ty)).unwrap();
            if tracks_initialization(types, ty) {
                writeln!(
                    out,
                    "  bool *r_b{block}_{i}_live = NULL; (void)r_b{block}_{i}_live;"
                )
                .unwrap();
            }
        }
    }
    out
}

// Continuations are a sequence, so only child regions recurse.
fn lower_region(
    types: &Types<'_>,
    index: usize,
    mut block_id: usize,
    flow: &FunctionTypes,
    exit_target: Option<ExitTarget>,
    loop_target: Option<(usize, usize)>,
) -> Result<Vec<CStatement>, Error> {
    let function = &types.module.functions[index];
    let mut statements = Vec::new();
    loop {
        let block = &function.blocks[block_id];
        let mut out = String::new();
        let mut stack = block_inputs(types, flow, block_id);
        let mut diverged = false;
        for (i, instr) in block.instrs.iter().enumerate() {
            if matches!(instr, Instr::Eliminate { .. }) {
                out.push_str("  abort();\n");
                diverged = true;
                break;
            }
            let args = stack.split_off(
                stack.len() - flow.operand_count(resin_lir::BlockId::from_index(block_id), i),
            );
            let result = flow.results[block_id][i].as_ref();
            let projected_value = projects_value(types, instr, &args);
            let name = format!("r_v{block_id}_{i}");
            let expr = instruction(types, function, &name, instr, &args, result, &mut out)
                .map_err(|error| Error::at(types.module, index, Some((block_id, i)), error))?;
            if let Some(ty) = result {
                writeln!(out, "  {} {name} = {}; (void){name};", types.name(ty), {
                    let expr = expr.unwrap();
                    if matches!(instr, Instr::Load) || projected_value {
                        types.copy(ty, &expr)
                    } else {
                        expr
                    }
                })
                .unwrap();
                stack.push(Slot {
                    ty: ty.clone(),
                    expr: name,
                    live: if let Instr::LocalAddress { local } = instr
                        && tracks_initialization(types, ty)
                    {
                        Some(format!("&r_live{}", local.index()))
                    } else {
                        None
                    },
                });
            }
            let consume = matches!(
                instr,
                Instr::IsVariant { .. }
                    | Instr::CallBuiltin { .. }
                    | Instr::OwnerData { .. }
                    | Instr::OwnerLength
                    | Instr::OwnerAllocate { .. }
                    | Instr::OwnerDowngrade
                    | Instr::OwnerUpgrade
                    | Instr::GpuViewAllocate
                    | Instr::GpuViewLoad { .. }
                    | Instr::GpuViewStore
                    | Instr::GpuViewReplace
                    | Instr::GpuViewCopyTo
                    | Instr::GpuViewCopyImage
                    | Instr::GpuComputePipeline { .. }
                    | Instr::GpuGraphicsPipeline { .. }
                    | Instr::GpuDispatch { .. }
                    | Instr::GpuDraw { .. }
                    | Instr::GpuArgumentsDispatch
                    | Instr::GpuArgumentsDraw
            ) || projected_value;
            if consume {
                for arg in &args {
                    types.drop_value(&arg.ty, &arg.expr, &mut out);
                }
            }
        }
        statements.push(CStatement::Text { source: out });
        if diverged {
            return Ok(statements);
        }
        let (exit, next) = lower_exit(
            types,
            index,
            block_id,
            flow,
            stack,
            exit_target,
            loop_target,
        )?;
        statements.extend(exit);
        let Some(next) = next else {
            return Ok(statements);
        };
        block_id = next;
    }
}

fn projects_value(types: &Types<'_>, instr: &Instr, args: &[Slot]) -> bool {
    match instr {
        Instr::AccessStatic { .. } => !matches!(types.shape(&args[0].ty), Ty::Pointer { .. }),
        Instr::AccessDynamic => matches!(types.shape(&args[0].ty), Ty::Array { .. }),
        _ => false,
    }
}

#[derive(Clone, Copy)]
enum ExitTarget {
    Merge { block: usize },
    LoopTest { body: usize, test: usize },
    Continue { condition: usize },
}

fn block_inputs(types: &Types<'_>, flow: &FunctionTypes, block: usize) -> Vec<Slot> {
    flow.inputs[block]
        .iter()
        .enumerate()
        .map(|(i, ty)| Slot {
            ty: ty.clone(),
            expr: format!("r_b{block}_{i}"),
            live: tracks_initialization(types, ty).then(|| format!("r_b{block}_{i}_live")),
        })
        .collect()
}

fn lower_exit(
    types: &Types<'_>,
    index: usize,
    block: usize,
    flow: &FunctionTypes,
    mut stack: Vec<Slot>,
    exit_target: Option<ExitTarget>,
    loop_target: Option<(usize, usize)>,
) -> Result<(Vec<CStatement>, Option<usize>), Error> {
    let mut statements = Vec::new();
    let next = match types.module.functions[index].blocks[block].terminator {
        Terminator::Break | Terminator::NextIteration => {
            let (condition, output) = loop_target.expect("verified loop exit");
            let exiting = matches!(
                types.module.functions[index].blocks[block].terminator,
                Terminator::Break
            );
            statements.push(CStatement::Text {
                source: transfer(types, if exiting { output } else { condition }, &stack),
            });
            statements.push(if exiting {
                CStatement::Break
            } else {
                CStatement::Continue
            });
            None
        }
        Terminator::Return => {
            statements.push(CStatement::Return {
                value: stack[0].expr.clone(),
            });
            None
        }
        Terminator::Merge => {
            let Some(ExitTarget::Merge { block }) = exit_target else {
                unreachable!("verified merge has a selection destination")
            };
            statements.push(CStatement::Text {
                source: transfer(types, block, &stack),
            });
            None
        }
        Terminator::LoopTest => {
            let Some(ExitTarget::LoopTest { body, test }) = exit_target else {
                unreachable!("verified loop test has a condition destination")
            };
            statements.push(CStatement::Text {
                source: loop_test(types, &stack, body, test),
            });
            None
        }
        Terminator::Continue => {
            let Some(ExitTarget::Continue { condition }) = exit_target else {
                unreachable!("verified continue has a loop destination")
            };
            statements.push(CStatement::Text {
                source: transfer(types, condition, &stack),
            });
            None
        }
        Terminator::If { then, els, next } => {
            let condition = stack.pop().unwrap();
            let target = next
                .map(|id| ExitTarget::Merge { block: id.index() })
                .or(exit_target.filter(|target| matches!(target, ExitTarget::Merge { .. })));
            statements.push(CStatement::If {
                condition: types.unwrap(&condition.ty, condition.expr),
                then: enter_region(
                    types,
                    index,
                    then.index(),
                    flow,
                    &stack,
                    target,
                    loop_target,
                )?,
                els: enter_region(types, index, els.index(), flow, &stack, target, loop_target)?,
            });
            next
        }
        Terminator::Loop {
            condition,
            body,
            next,
        } => {
            let condition = condition.index();
            let body = body.index();
            statements.push(CStatement::Text {
                source: transfer(types, condition, &stack),
            });
            statements.push(CStatement::Text {
                source: format!("  bool r_test{block} = false;\n"),
            });
            let mut repeated = lower_region(
                types,
                index,
                condition,
                flow,
                Some(ExitTarget::LoopTest { body, test: block }),
                None,
            )?;
            repeated.push(CStatement::If {
                condition: format!("!r_test{block}"),
                then: vec![CStatement::Break],
                els: vec![],
            });
            repeated.extend(lower_region(
                types,
                index,
                body,
                flow,
                Some(ExitTarget::Continue { condition }),
                Some((condition, body)),
            )?);
            statements.push(CStatement::Loop { body: repeated });
            // The condition writes these operands before its bool is tested, so
            // the false exit sees the final condition's values, even on a zero-trip loop.
            let output = block_inputs(types, flow, body);
            if let Some(next) = next {
                statements.push(CStatement::Text {
                    source: transfer(types, next.index(), &output),
                });
            } else {
                let Some(ExitTarget::Merge { block }) = exit_target else {
                    unreachable!("verified loop without a continuation merges a selection")
                };
                statements.push(CStatement::Text {
                    source: transfer(types, block, &output),
                });
            }
            next
        }
    };
    Ok((statements, next.map(|id| id.index())))
}

fn enter_region(
    types: &Types<'_>,
    index: usize,
    block: usize,
    flow: &FunctionTypes,
    stack: &[Slot],
    target: Option<ExitTarget>,
    loop_target: Option<(usize, usize)>,
) -> Result<Vec<CStatement>, Error> {
    let mut statements = vec![CStatement::Text {
        source: transfer(types, block, stack),
    }];
    statements.extend(lower_region(
        types,
        index,
        block,
        flow,
        target,
        loop_target,
    )?);
    Ok(statements)
}

fn loop_test(types: &Types<'_>, stack: &[Slot], body: usize, test: usize) -> String {
    let (condition, operands) = stack.split_last().unwrap();
    let mut source = transfer(types, body, operands);
    writeln!(
        source,
        "  r_test{test} = {};",
        types.unwrap(&condition.ty, condition.expr.clone())
    )
    .unwrap();
    source
}

/// Snapshot every operand before assigning destinations, including initialization
/// provenance for borrowed managed storage. A loop may permute its incoming slots.
fn transfer(types: &Types<'_>, target: usize, stack: &[Slot]) -> String {
    if stack.is_empty() {
        return String::new();
    }
    let mut source = String::from("  {\n");
    for (i, value) in stack.iter().enumerate() {
        writeln!(
            source,
            "    {} r_edge{i} = {};",
            types.name(&value.ty),
            value.expr
        )
        .unwrap();
        if tracks_initialization(types, &value.ty) {
            writeln!(
                source,
                "    bool *r_edge{i}_live = {};",
                value.live.as_deref().unwrap_or("NULL")
            )
            .unwrap();
        }
    }
    for (i, value) in stack.iter().enumerate() {
        writeln!(source, "    r_b{target}_{i} = r_edge{i};").unwrap();
        if tracks_initialization(types, &value.ty) {
            writeln!(source, "    r_b{target}_{i}_live = r_edge{i}_live;").unwrap();
        }
    }
    source.push_str("  }\n");
    source
}

fn tracks_initialization(types: &Types<'_>, ty: &Ty) -> bool {
    matches!(ty, Ty::Pointer { pointee } if pointee.needs_drop(&types.module.types))
}

fn instruction(
    types: &Types<'_>,
    function: &resin_lir::Function,
    temp: &str,
    instr: &Instr,
    args: &[Slot],
    result: Option<&Ty>,
    out: &mut String,
) -> Result<Option<String>, Error> {
    let expr = match instr {
        Instr::GpuViewAllocate
        | Instr::GpuViewRange { .. }
        | Instr::GpuViewOffset
        | Instr::GpuViewRestrict
        | Instr::GpuViewLoad { .. }
        | Instr::GpuViewStore
        | Instr::GpuViewReplace
        | Instr::GpuViewCopyTo
        | Instr::GpuViewCopyImage
        | Instr::GpuComputePipeline { .. }
        | Instr::GpuGraphicsPipeline { .. }
        | Instr::GpuDispatch { .. }
        | Instr::GpuDraw { .. }
        | Instr::GpuArgumentsDispatch
        | Instr::GpuArgumentsDraw => {
            super::gpu::instruction(types, temp, instr, args, result.unwrap(), out)?
        }
        Instr::ForgetLocal { local } => {
            if function.locals[local.index()]
                .ty
                .needs_drop(&types.module.types)
            {
                writeln!(out, "  r_live{} = false;", local.index()).unwrap();
            }
            return Ok(None);
        }
        Instr::OwnerData { .. } => format!(
            "({})resin_arc_data(*({}))",
            types.name(result.unwrap()),
            args[0].expr
        ),
        Instr::OwnerAllocate { element } => {
            let owner = allocate_span(types, temp, element, args, out);
            let ty = result.unwrap();
            let present = variant(types, ty, &Case::Type(ty.without_none().unwrap()), &owner);
            let absent = variant(types, ty, &Case::Type(Ty::None), "0");
            format!("({owner} ? {present} : {absent})")
        }
        Instr::OwnerLength => format!("resin_arc_span_length(*({}))", args[0].expr),
        Instr::OwnerDowngrade => {
            writeln!(out, "  resin_weak_retain(*({}));", args[0].expr).unwrap();
            format!("*({})", args[0].expr)
        }
        Instr::OwnerUpgrade => {
            writeln!(
                out,
                "  ResinArc *{temp}_upgraded = resin_weak_upgrade(*({}));",
                args[0].expr
            )
            .unwrap();
            let ty = result.unwrap();
            let present = variant(
                types,
                ty,
                &Case::Type(ty.without_none().unwrap()),
                &format!("{temp}_upgraded"),
            );
            let absent = variant(types, ty, &Case::Type(Ty::None), "0");
            format!("({temp}_upgraded ? {present} : {absent})")
        }
        Instr::WeakEmpty => "NULL".into(),
        Instr::TakeLocal { local } => {
            if function.locals[local.index()]
                .ty
                .needs_drop(&types.module.types)
            {
                writeln!(out, "  r_live{} = false;", local.index()).unwrap();
            }
            format!("r_l{}", local.index())
        }
        Instr::DropLocal { local } => {
            if !function.locals[local.index()]
                .ty
                .needs_drop(&types.module.types)
            {
                return Ok(None);
            }
            writeln!(out, "  if (r_live{}) {{", local.index()).unwrap();
            types.drop_value(
                &function.locals[local.index()].ty,
                &format!("r_l{}", local.index()),
                out,
            );
            writeln!(out, "    r_live{} = false; }}", local.index()).unwrap();
            return Ok(None);
        }
        Instr::SetLocal { local } => {
            if args[0].ty.needs_drop(&types.module.types) {
                writeln!(
                    out,
                    "  if (r_live{}) r_drop{}(&r_l{});",
                    local.index(),
                    types.id(&args[0].ty),
                    local.index()
                )
                .unwrap();
                writeln!(out, "  r_live{} = true;", local.index()).unwrap();
            }
            writeln!(out, "  r_l{} = {};", local.index(), args[0].expr).unwrap();
            return Ok(None);
        }
        Instr::MakeVariant { ty, tag } => {
            if matches!(tag, Case::Type(member) if member == ty) {
                args[0].expr.clone()
            } else {
                let tag = types.tag(tag);
                // Initialize the complete union storage before selecting a smaller
                // payload. Branch transfers may copy the entire Result value.
                writeln!(out, "  {} {temp}_variant = {{0}};", types.name(ty)).unwrap();
                writeln!(out, "  {temp}_variant.tag = {tag}u;").unwrap();
                writeln!(out, "  {temp}_variant.payload.v{tag} = {};", args[0].expr).unwrap();
                format!("{temp}_variant")
            }
        }
        Instr::ExcludeNone => {
            let condition = is_variant(types, &args[0].ty, &Case::Type(Ty::None), &args[0].expr);
            writeln!(
                out,
                "  if ({condition}) resin_fail(\"cannot unwrap None\");"
            )
            .unwrap();
            widen(types, &args[0].ty, result.unwrap(), &args[0].expr)
        }
        Instr::IsVariant { tag } => {
            let (ty, expr) = if let Ty::Pointer { pointee } = &args[0].ty {
                (pointee.as_ref(), format!("*({})", args[0].expr))
            } else {
                (&args[0].ty, args[0].expr.clone())
            };
            is_variant(types, ty, tag, &expr)
        }
        Instr::VariantPayload { tag } => {
            if matches!(tag, Case::Type(member) if member == &args[0].ty) {
                args[0].expr.clone()
            } else {
                let tag = types.tag(tag);
                writeln!(
                    out,
                    "  if (({}).tag != {tag}u) resin_fail(\"invalid union tag\");",
                    args[0].expr
                )
                .unwrap();
                format!("({}).payload.v{tag}", args[0].expr)
            }
        }
        Instr::Widen { ty } => widen(types, &args[0].ty, ty, &args[0].expr),
        Instr::Eliminate { .. } => unreachable!("diverging instruction ends the block"),
        Instr::NumericCast { ty } => {
            if let Some(invalid) = crate::numeric::invalid(
                &args[0].ty,
                ty,
                &args[0].expr,
                |from, value| format!("({})({value})", types.name(from)),
                if args[0].ty == Ty::Float32 {
                    "truncf"
                } else {
                    "trunc"
                },
            ) {
                writeln!(out, "  if ({invalid}) {{ fputs(\"numeric conversion out of range\\n\", stderr); abort(); }}").unwrap();
            }
            if args[0].ty == Ty::Float64 && *ty == Ty::Float32 {
                let x = &args[0].expr;
                format!(
                    "(({x}) > FLT_MAX ? INFINITY : ({x}) < -FLT_MAX ? -INFINITY : fabs({x}) < FLT_MIN ? copysignf(0.0f, (float)copysign(1.0, {x})) : (float)({x}))"
                )
            } else {
                format!("({})({})", types.name(ty), args[0].expr)
            }
        }
        Instr::PointerCast { ty } => format!("({})(uintptr_t)({})", types.name(ty), args[0].expr),
        Instr::Push { value } => literal(types, result.unwrap(), value),
        Instr::LocalAddress { local } => format!("&r_l{}", local.index()),
        Instr::Load | Instr::TransferLoad => {
            format!("*({})", types.unwrap(&args[0].ty, args[0].expr.clone()))
        }
        Instr::Replace => {
            let target = format!("*({})", types.unwrap(&args[0].ty, args[0].expr.clone()));
            let old = format!("{temp}_old");
            writeln!(out, "  {} {old} = {target};", types.name(&args[1].ty)).unwrap();
            writeln!(out, "  {target} = {};", args[1].expr).unwrap();
            old
        }
        Instr::Store => {
            let target = format!("*({})", types.unwrap(&args[0].ty, args[0].expr.clone()));
            if args[1].ty.needs_drop(&types.module.types) {
                if let Some(live) = &args[0].live {
                    writeln!(out, "  bool *{temp}_live = {live};").unwrap();
                    writeln!(
                        out,
                        "  if (!{temp}_live || *{temp}_live) r_drop{}(&({target}));",
                        types.id(&args[1].ty)
                    )
                    .unwrap();
                    writeln!(out, "  if ({temp}_live) *{temp}_live = true;").unwrap();
                } else {
                    types.drop_value(&args[1].ty, &target, out);
                }
            }
            writeln!(
                out,
                "  *({}) = {};",
                types.unwrap(&args[0].ty, args[0].expr.clone()),
                types.copy(&args[1].ty, &args[1].expr)
            )
            .unwrap();
            args[1].expr.clone()
        }
        Instr::Discard => {
            types.drop_value(&args[0].ty, &args[0].expr, out);
            writeln!(out, "  (void){};", args[0].expr).unwrap();
            return Ok(None);
        }
        Instr::Ascribe { ty } => {
            if ty == &args[0].ty {
                args[0].expr.clone()
            } else if is_view_conversion(&args[0].ty, ty) {
                format!(
                    "({}){{ ({}).f0, ({}).f1 }}",
                    types.name(ty),
                    args[0].expr,
                    args[0].expr
                )
            } else if let Ty::Defined { definition } = ty
                && types.module.types[definition.index()].body() == Some(&args[0].ty)
            {
                format!("({}){{ .value = {} }}", types.name(ty), args[0].expr)
            } else {
                format!("({}).value", args[0].expr)
            }
        }
        Instr::MakeRecord { .. } => {
            let values = args
                .iter()
                .map(|a| a.expr.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "({}){{ {} }}",
                types.name(result.unwrap()),
                if values.is_empty() { "0" } else { &values }
            )
        }
        Instr::MakeArray { .. } => {
            let values = args
                .iter()
                .map(|a| a.expr.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "({}){{ .items = {{ {} }} }}",
                types.name(result.unwrap()),
                if values.is_empty() { "0" } else { &values }
            )
        }
        Instr::AccessStatic { index } => project(types, &args[0], &index.to_string(), false)?,
        Instr::PointerRange => pointer_range(args, out),
        Instr::PointerBytes => pointer_bytes(types, args, result.unwrap(), out),
        Instr::PointerIndex => format!(
            "({} + resin_index({}, {}))",
            args[0].expr, args[2].expr, args[1].expr
        ),
        Instr::AccessDynamic => project(
            types,
            &args[0],
            &types.unwrap(&args[1].ty, args[1].expr.clone()),
            true,
        )?,
        Instr::Function { function } => format!(
            "({}){{ r_fn{} }}",
            types.name(result.unwrap()),
            function.index()
        ),
        Instr::Call { .. } => {
            let callee = types.unwrap(&args[0].ty, args[0].expr.clone());
            writeln!(
                out,
                "  if (!({callee}).call) resin_fail(\"calling an uninitialized function\");"
            )
            .unwrap();
            format!(
                "({callee}).call({})",
                args[1..]
                    .iter()
                    .map(|arg| arg.expr.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        Instr::CallBuiltin { name, result, .. } => ops::builtin(types, name, args, result)?,
    };
    Ok(Some(expr))
}

fn is_variant(types: &Types<'_>, ty: &Ty, case: &Case, value: &str) -> String {
    if matches!(case, Case::Type(member) if member == ty) {
        return "true".into();
    }
    format!("(({value}).tag == {}u)", types.tag(case))
}

fn variant(types: &Types<'_>, ty: &Ty, case: &Case, value: &str) -> String {
    if matches!(case, Case::Type(member) if member == ty) {
        return value.into();
    }
    let tag = types.tag(case);
    format!(
        "({}){{ .tag = {tag}u, .payload = {{ .v{tag} = {value} }} }}",
        types.name(ty)
    )
}

// Also used after the checked exclusion of None. Cases absent from the target
// cannot be selected on that path; payloads retain their module-wide identities.
fn widen(types: &Types<'_>, from: &Ty, to: &Ty, value: &str) -> String {
    if from == to {
        return value.into();
    }
    if matches!(to, Ty::Union { variants } if variants.contains(from)) {
        return variant(types, to, &Case::Type(from.clone()), value);
    }
    let initializer = if matches!(to, Ty::Defined { .. }) {
        ".value = {0}"
    } else {
        "0"
    };
    let mut expression = format!("({}){{ {initializer} }}", types.name(to));
    for (case, payload) in from.payloads().unwrap_or_default().into_iter().rev() {
        let Some(target) = to.payload(&case) else {
            continue;
        };
        let tag = types.tag(&case);
        let payload = widen(
            types,
            &payload,
            &target,
            &format!("({value}).payload.v{tag}"),
        );
        let constructed = variant(types, to, &case, &payload);
        expression = format!("(({value}).tag == {tag}u ? {constructed} : {expression})");
    }
    expression
}

fn project(types: &Types<'_>, source: &Slot, index: &str, dynamic: bool) -> Result<String, Error> {
    let mut ty = types.shape(&source.ty);
    let mut expr = types.unwrap(&source.ty, source.expr.clone());
    let pointer = if let Ty::Pointer { pointee } = ty {
        expr = types.unwrap(pointee, format!("*({expr})"));
        ty = types.shape(pointee);
        true
    } else {
        false
    };
    let expr = match ty {
        Ty::Str if dynamic => {
            return Ok(format!(
                "&(({expr}).f0[resin_index((uint64_t)({index}), ({expr}).f1)])"
            ));
        }
        Ty::Str | Ty::Record { .. } if !dynamic => format!("({expr}).f{index}"),
        Ty::Array { length, .. } => {
            let index = if dynamic {
                format!("resin_index((uint64_t)({index}), {length})")
            } else {
                index.into()
            };
            format!("({expr}).items[{index}]")
        }
        _ => return Err(Error(format!("unsupported projection through {ty:?}"))),
    };
    Ok(if pointer { format!("&({expr})") } else { expr })
}

// The argument remains owned until every element has received an ordinary copy.
// Copies cannot fail; allocation completes before any element is initialized.
fn allocate_span(
    types: &Types<'_>,
    temp: &str,
    element: &Ty,
    args: &[Slot],
    out: &mut String,
) -> String {
    let owner = format!("{temp}_allocated");
    let element_name = types.name(element);
    let destroy = if element.needs_drop(&types.module.types) {
        format!("r_drop{}", types.id(element))
    } else {
        "NULL".into()
    };
    let count = &args[0].expr;
    writeln!(out, "  ResinArc *{owner} = resin_arc_span_try_new({count}, sizeof({element_name}), _Alignof({element_name}), {destroy});").unwrap();
    writeln!(out, "  if ({owner}) {{").unwrap();
    writeln!(
        out,
        "    {element_name} *{temp}_data = resin_arc_data({owner});"
    )
    .unwrap();
    writeln!(
        out,
        "    uint64_t {temp}_length = resin_arc_span_length({owner});"
    )
    .unwrap();
    writeln!(
        out,
        "    for (uint64_t i = 0; i < {temp}_length; ++i) {temp}_data[i] = {};",
        types.copy(element, &args[1].expr)
    )
    .unwrap();
    out.push_str("  }\n");
    owner
}

fn is_view_conversion(from: &Ty, to: &Ty) -> bool {
    from.view_record().as_ref() == Some(to) || to.view_record().as_ref() == Some(from)
}

fn pointer_range(args: &[Slot], out: &mut String) -> String {
    let [data, capacity, start, count] = args else {
        unreachable!("verified pointer range")
    };
    writeln!(
        out,
        "  if ({0} > {1} || {2} > {1} - {0}) resin_fail(\"span slice out of bounds\");",
        start.expr, capacity.expr, count.expr
    )
    .unwrap();
    format!("({0} ? {1} + {0} : {1})", start.expr, data.expr)
}

fn pointer_bytes(types: &Types<'_>, args: &[Slot], result: &Ty, out: &mut String) -> String {
    let Ty::Pointer { pointee } = &args[0].ty else {
        unreachable!("verified numeric pointer")
    };
    let stride = format!("sizeof({})", types.name(pointee));
    writeln!(
        out,
        "  if ({} > UINT64_MAX / {stride}) resin_fail(\"span byte length overflow\");",
        args[1].expr
    )
    .unwrap();
    format!(
        "({}){{ (uint8_t *){}, {} * {stride} }}",
        types.name(result),
        args[0].expr,
        args[1].expr
    )
}
