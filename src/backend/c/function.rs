use std::fmt::Write;

use crate::{
    backend::Error,
    ir::{Case, Instr, Terminator, Ty, verify::FunctionTypes},
};

use super::{Slot, ops, types::Types, value::literal};

pub(super) fn emit(types: &Types<'_>, index: usize, flow: &FunctionTypes) -> Result<String, Error> {
    let function = &types.module.functions[index];
    if let Some(foreign) = &function.foreign {
        return Ok(super::foreign::emit(types, index, foreign));
    }
    let mut out = format!(
        "{} r_fn{index}({} r_arg) {{\n",
        types.name(&function.result),
        types.name(&function.locals[0].ty)
    );
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
                if i == 0 { "true" } else { "false" }
            )
            .unwrap();
        }
    }
    writeln!(out, "  r_l0 = r_arg;").unwrap();
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
    writeln!(out, "  goto r_b{};", function.entry.index()).unwrap();
    for (block_id, block) in function.blocks.iter().enumerate() {
        writeln!(out, "r_b{block_id}: {{").unwrap();
        let mut stack: Vec<_> = flow.inputs[block_id]
            .iter()
            .enumerate()
            .map(|(i, ty)| Slot {
                ty: ty.clone(),
                expr: format!("r_b{block_id}_{i}"),
                live: tracks_initialization(types, ty).then(|| format!("r_b{block_id}_{i}_live")),
            })
            .collect();
        let mut diverged = false;
        for (i, instr) in block.instrs.iter().enumerate() {
            if matches!(instr, Instr::Eliminate { .. }) {
                out.push_str("  abort();\n");
                diverged = true;
                break;
            }
            let args = stack.split_off(stack.len() - instr.stack_effect().pops);
            let result = flow.results[block_id][i].as_ref();
            let name = format!("r_v{block_id}_{i}");
            let expr = instruction(types, function, &name, instr, &args, result, &mut out)
                .map_err(|error| Error::at(types.module, index, Some((block_id, i)), error))?;
            if let Some(ty) = result {
                writeln!(out, "  {} {name} = {}; (void){name};", types.name(ty), {
                    let expr = expr.unwrap();
                    if matches!(instr, Instr::Load)
                        || matches!(instr, Instr::AccessStatic { .. })
                            && !matches!(args[0].ty, Ty::Pointer { .. })
                    {
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
                    | Instr::ArcData
                    | Instr::Downgrade
                    | Instr::Upgrade
            ) || matches!(instr, Instr::AccessStatic { .. })
                && !matches!(args[0].ty, Ty::Pointer { .. });
            if consume {
                for arg in &args {
                    types.drop_value(&arg.ty, &arg.expr, &mut out);
                }
            }
        }
        if diverged {
            out.push_str("}\n");
            continue;
        }
        match block.terminator {
            Terminator::Return => writeln!(out, "  return {};", stack[0].expr).unwrap(),
            Terminator::Break { target } => edge(types, target.index(), &stack, &mut out),
            Terminator::Branch { then, els } => {
                let condition = stack.pop().unwrap();
                writeln!(
                    out,
                    "  if ({}) {{",
                    types.unwrap(&condition.ty, condition.expr)
                )
                .unwrap();
                edge(types, then.index(), &stack, &mut out);
                out.push_str("  } else {\n");
                edge(types, els.index(), &stack, &mut out);
                out.push_str("  }\n");
            }
        }
        out.push_str("}\n");
    }
    out.push_str("}\n");
    Ok(out)
}

fn edge(types: &Types<'_>, target: usize, stack: &[Slot], out: &mut String) {
    out.push_str("  {\n");
    for (i, slot) in stack.iter().enumerate() {
        writeln!(
            out,
            "    {} r_edge{i} = {};",
            types.name(&slot.ty),
            slot.expr
        )
        .unwrap();
        if tracks_initialization(types, &slot.ty) {
            writeln!(
                out,
                "    bool *r_edge{i}_live = {};",
                slot.live.as_deref().unwrap_or("NULL")
            )
            .unwrap();
        }
    }
    for (i, slot) in stack.iter().enumerate() {
        writeln!(out, "    r_b{target}_{i} = r_edge{i};").unwrap();
        if tracks_initialization(types, &slot.ty) {
            writeln!(out, "    r_b{target}_{i}_live = r_edge{i}_live;").unwrap();
        }
    }
    writeln!(out, "    goto r_b{target};\n  }}").unwrap();
}

fn tracks_initialization(types: &Types<'_>, ty: &Ty) -> bool {
    matches!(ty, Ty::Pointer { pointee } if pointee.needs_drop(&types.module.types))
}

fn instruction(
    types: &Types<'_>,
    function: &crate::ir::Function,
    temp: &str,
    instr: &Instr,
    args: &[Slot],
    result: Option<&Ty>,
    out: &mut String,
) -> Result<Option<String>, Error> {
    let expr = match instr {
        Instr::ForgetLocal { local } => {
            if function.locals[local.index()]
                .ty
                .needs_drop(&types.module.types)
            {
                writeln!(out, "  r_live{} = false;", local.index()).unwrap();
            }
            return Ok(None);
        }
        Instr::ArcNew => {
            let ty = &args[0].ty;
            writeln!(
                out,
                "  ResinArc *{temp}_allocated = resin_arc_new(sizeof({}), _Alignof({}), r_drop{});",
                types.name(ty),
                types.name(ty),
                types.id(ty)
            )
            .unwrap();
            writeln!(
                out,
                "  *({} *)resin_arc_data({temp}_allocated) = {};",
                types.name(ty),
                args[0].expr
            )
            .unwrap();
            format!("{temp}_allocated")
        }
        Instr::ArcData => format!(
            "({})resin_arc_data({})",
            types.name(result.unwrap()),
            args[0].expr
        ),
        Instr::Downgrade => {
            writeln!(out, "  resin_weak_retain({});", args[0].expr).unwrap();
            args[0].expr.clone()
        }
        Instr::Upgrade => {
            writeln!(
                out,
                "  ResinArc *{temp}_upgraded = resin_weak_upgrade({});",
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
        Instr::WeakEmpty { .. } => "NULL".into(),
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
        Instr::MakeVariant { ty, tag } => variant(types, ty, tag, &args[0].expr),
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
            if let Some(invalid) = crate::backend::numeric::invalid(
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
        Instr::Shader { function, stage } => {
            let index = types
                .shaders
                .iter()
                .position(|shader| {
                    shader.function == *function && shader.stage.name() == stage.as_ref()
                })
                .ok_or_else(|| Error("shader needs SPIR-V compilation before C emission".into()))?;
            format!(
                "({}){{ (uint8_t *)r_spv{index}, sizeof(r_spv{index}) }}",
                types.name(result.unwrap())
            )
        }
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
            } else if matches!(ty, Ty::Span { .. }) || matches!(args[0].ty, Ty::Span { .. }) {
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
        Instr::Call => {
            let callee = types.unwrap(&args[0].ty, args[0].expr.clone());
            writeln!(
                out,
                "  if (!({callee}).call) resin_fail(\"calling an uninitialized function\");"
            )
            .unwrap();
            format!("({callee}).call({})", args[1].expr)
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
        Ty::Span { .. } if dynamic => {
            return Ok(format!(
                "&(({expr}).f0[resin_index((uint64_t)({index}), ({expr}).f1)])"
            ));
        }
        Ty::Span { .. } | Ty::Record { .. } if !dynamic => format!("({expr}).f{index}"),
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
