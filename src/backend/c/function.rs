use std::fmt::Write;

use crate::{
    backend::Error,
    ir::{Instr, Terminator, Ty, verify::FunctionTypes},
};

use super::{Slot, ops, types::Types, value::literal};

pub(super) fn emit(types: &Types<'_>, index: usize, flow: &FunctionTypes) -> Result<String, Error> {
    let function = &types.module.functions[index];
    let mut out = format!(
        "static {} r_fn{index}(void *r_env, {} r_arg) {{\n  (void)r_env;\n",
        types.name(&function.result),
        types.name(&function.locals[function.param.index()].ty)
    );
    for (i, local) in function.locals.iter().enumerate() {
        writeln!(
            out,
            "  {} r_l{i} = {{0}}; (void)r_l{i};",
            types.name(&local.ty)
        )
        .unwrap();
    }
    writeln!(out, "  r_l{} = r_arg;", function.param.index()).unwrap();
    for (block, inputs) in flow.inputs.iter().enumerate() {
        for (i, ty) in inputs.iter().enumerate() {
            writeln!(out, "  {} r_b{block}_{i};", types.name(ty)).unwrap();
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
            })
            .collect();
        for (i, instr) in block.instrs.iter().enumerate() {
            let args = stack.split_off(stack.len() - instr.stack_effect().pops);
            let result = flow.results[block_id][i].as_ref();
            let name = format!("r_v{block_id}_{i}");
            let expr = instruction(types, index, instr, &args, result, &name, &mut out).map_err(
                |error| {
                    Error(format!(
                        "function {index}, block {block_id}, instruction {i}: {error}"
                    ))
                },
            )?;
            if let Some(ty) = result {
                writeln!(
                    out,
                    "  {} {name} = {}; (void){name};",
                    types.name(ty),
                    expr.unwrap()
                )
                .unwrap();
                stack.push(Slot {
                    ty: ty.clone(),
                    expr: name,
                });
            }
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
    }
    for i in 0..stack.len() {
        writeln!(out, "    r_b{target}_{i} = r_edge{i};").unwrap();
    }
    writeln!(out, "    goto r_b{target};\n  }}").unwrap();
}

fn instruction(
    types: &Types<'_>,
    index: usize,
    instr: &Instr,
    args: &[Slot],
    result: Option<&Ty>,
    name: &str,
    out: &mut String,
) -> Result<Option<String>, Error> {
    let expr = match instr {
        Instr::Push { value } => literal(types, result.unwrap(), value),
        Instr::CurrentClosure => {
            format!("({}){{ r_fn{index}, r_env }}", types.name(result.unwrap()))
        }
        Instr::LocalAddress { local } => format!("&r_l{}", local.index()),
        Instr::GlobalAddress { global } => format!("&r_g{}", global.index()),
        Instr::NonLocalAddress { nonlocal } => {
            format!("&((r_env{index} *)r_env)->c{}", nonlocal.index())
        }
        Instr::Load => format!("*({})", types.unwrap(&args[0].ty, args[0].expr.clone())),
        Instr::Store => {
            writeln!(
                out,
                "  *({}) = {};",
                types.unwrap(&args[0].ty, args[0].expr.clone()),
                args[1].expr
            )
            .unwrap();
            args[1].expr.clone()
        }
        Instr::Discard => return Ok(None),
        Instr::Ascribe { ty } => {
            if ty == &args[0].ty {
                args[0].expr.clone()
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
        Instr::MakeClosure {
            function: target, ..
        } => {
            let target = target.index();
            let env = if args.is_empty() {
                "NULL".into()
            } else {
                let env = format!("{name}_env");
                writeln!(
                    out,
                    "  r_env{target} *{env} = r_alloc(sizeof(r_env{target}));"
                )
                .unwrap();
                for (i, arg) in args.iter().enumerate() {
                    writeln!(out, "  {env}->c{i} = {};", arg.expr).unwrap();
                }
                env
            };
            format!("({}){{ r_fn{target}, {env} }}", types.name(result.unwrap()))
        }
        Instr::Call => {
            let callee = types.unwrap(&args[0].ty, args[0].expr.clone());
            writeln!(
                out,
                "  if (!({callee}).call) r_fail(\"calling an uninitialized function\");"
            )
            .unwrap();
            format!("({callee}).call(({callee}).env, {})", args[1].expr)
        }
        Instr::CallBuiltin { name, result, .. } => ops::builtin(types, name, args, result)?,
    };
    Ok(Some(expr))
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
        Ty::Record { .. } if !dynamic => format!("({expr}).f{index}"),
        Ty::Array { length, .. } => {
            let index = if dynamic {
                format!("r_index((uint64_t)({index}), {length})")
            } else {
                index.into()
            };
            format!("({expr}).items[{index}]")
        }
        _ => return Err(Error(format!("unsupported projection through {ty:?}"))),
    };
    Ok(if pointer { format!("&({expr})") } else { expr })
}
