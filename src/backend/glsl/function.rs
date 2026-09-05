use std::fmt::Write;

use crate::{
    backend::Error,
    ir::{Function, Instr, Terminator, Ty, Value, verify::FunctionTypes},
};

use super::types::Types;

struct Slot {
    ty: Ty,
    expr: String,
}

pub(super) fn emit(
    types: &Types<'_>,
    function: &Function,
    flow: &FunctionTypes,
) -> Result<String, Error> {
    let mut out = format!(
        "{} r_entry({} arg) {{\n",
        types.name(&function.result),
        types.name(&function.locals[function.param.index()].ty)
    );
    for (i, local) in function.locals.iter().enumerate() {
        writeln!(
            out,
            "  {} r_l{i} = {};",
            types.name(&local.ty),
            types.zero(&local.ty)
        )
        .unwrap();
    }
    writeln!(out, "  r_l{} = arg;", function.param.index()).unwrap();
    for (b, inputs) in flow.inputs.iter().enumerate() {
        for (i, ty) in inputs.iter().enumerate() {
            writeln!(out, "  {} r_b{b}_{i};", types.name(ty)).unwrap();
        }
        for (i, ty) in flow.results[b].iter().enumerate() {
            if let Some(ty) = ty
                && !matches!(ty, Ty::Pointer { .. })
            {
                writeln!(out, "  {} r_v{b}_{i};", types.name(ty)).unwrap();
            }
        }
    }
    writeln!(
        out,
        "  int pc = {};\n  while (true) {{\n    switch (pc) {{",
        function.entry.index()
    )
    .unwrap();
    for (b, block) in function.blocks.iter().enumerate() {
        writeln!(out, "    case {b}: {{").unwrap();
        let mut stack: Vec<_> = flow.inputs[b]
            .iter()
            .enumerate()
            .map(|(i, ty)| Slot {
                ty: ty.clone(),
                expr: format!("r_b{b}_{i}"),
            })
            .collect();
        for (i, instr) in block.instrs.iter().enumerate() {
            let args = stack.split_off(stack.len() - instr.stack_effect().pops);
            let result = flow.results[b][i].as_ref();
            let expr = instruction(types, instr, &args, result, &mut out)
                .map_err(|error| Error(format!("shader block {b}, instruction {i}: {error}")))?;
            if let Some(ty) = result {
                let mut expr = expr.unwrap();
                if !matches!(ty, Ty::Pointer { .. }) {
                    let name = format!("r_v{b}_{i}");
                    writeln!(out, "      {name} = {expr};").unwrap();
                    expr = name;
                }
                stack.push(Slot {
                    ty: ty.clone(),
                    expr,
                });
            }
        }
        match block.terminator {
            Terminator::Return => writeln!(out, "      return {};", stack[0].expr).unwrap(),
            Terminator::Break { target } => edge(types, target.index(), &stack, &mut out),
            Terminator::Branch { then, els } => {
                let cond = stack.pop().unwrap();
                writeln!(out, "      if ({}) {{", types.unwrap(&cond.ty, cond.expr)).unwrap();
                edge(types, then.index(), &stack, &mut out);
                out.push_str("      } else {\n");
                edge(types, els.index(), &stack, &mut out);
                out.push_str("      }\n");
            }
        }
        out.push_str("    }\n");
    }
    writeln!(
        out,
        "    default: return {};\n    }}\n  }}\n}}",
        types.zero(&function.result)
    )
    .unwrap();
    Ok(out)
}

fn edge(types: &Types<'_>, target: usize, stack: &[Slot], out: &mut String) {
    out.push_str("      {\n");
    for (i, slot) in stack.iter().enumerate() {
        writeln!(
            out,
            "        {} edge{i} = {};",
            types.name(&slot.ty),
            slot.expr
        )
        .unwrap();
    }
    for i in 0..stack.len() {
        writeln!(out, "        r_b{target}_{i} = edge{i};").unwrap();
    }
    writeln!(out, "        pc = {target}; continue;\n      }}").unwrap();
}

fn instruction(
    types: &Types<'_>,
    instr: &Instr,
    args: &[Slot],
    result: Option<&Ty>,
    out: &mut String,
) -> Result<Option<String>, Error> {
    let expr = match instr {
        Instr::Push { value } => literal(types, result.unwrap(), value)?,
        Instr::LocalAddress { local } => format!("r_l{}", local.index()),
        Instr::Load => args[0].expr.clone(),
        Instr::Store => {
            writeln!(out, "      {} = {};", args[0].expr, args[1].expr).unwrap();
            args[1].expr.clone()
        }
        Instr::Discard => return Ok(None),
        Instr::Ascribe { ty } => {
            if ty == &args[0].ty {
                args[0].expr.clone()
            } else if let Ty::Defined { definition } = ty
                && types.module.types[definition.index()].body() == Some(&args[0].ty)
            {
                format!("{}({})", types.name(ty), args[0].expr)
            } else {
                format!("({}).value", args[0].expr)
            }
        }
        Instr::MakeRecord { .. } => {
            let args = args
                .iter()
                .map(|a| a.expr.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{}({})",
                types.name(result.unwrap()),
                if args.is_empty() { "0u" } else { &args }
            )
        }
        Instr::AccessStatic { index } => {
            let ty = if let Ty::Pointer { pointee } = &args[0].ty {
                pointee
            } else {
                &args[0].ty
            };
            if !matches!(types.shape(ty), Ty::Record { .. }) {
                return Err(Error("shader projection requires a record".into()));
            }
            format!("({}).f{index}", types.unwrap(ty, args[0].expr.clone()))
        }
        Instr::CallBuiltin { name, result, .. } => builtin(types, name, args, result)?,
        _ => return Err(Error(format!("shader profile does not support {instr:?}"))),
    };
    Ok(Some(expr))
}

fn builtin(types: &Types<'_>, name: &str, args: &[Slot], result: &Ty) -> Result<String, Error> {
    let unsupported = || Error(format!("unsupported shader builtin {name:?}"));
    let Some(first) = args.first() else {
        return Err(unsupported());
    };
    if args.iter().any(|a| a.ty != first.ty) {
        return Err(unsupported());
    }
    let ty = types.shape(&first.ty);
    let bool_result = matches!(
        name,
        "==" | "!=" | "<" | "<=" | ">" | ">=" | "!" | "&&" | "||"
    );
    if result != if bool_result { &Ty::Bool } else { &first.ty } {
        return Err(unsupported());
    }
    let values: Vec<_> = args
        .iter()
        .map(|a| types.unwrap(&a.ty, a.expr.clone()))
        .collect();
    let expr = match (name, values.as_slice()) {
        ("+", [a]) if ty.is_numeric() => a.clone(),
        ("-", [a]) if ty.is_numeric() => format!("-({a})"),
        ("~", [a]) if ty.is_integer() => format!("~({a})"),
        ("!", [a]) if ty == &Ty::Bool => format!("!({a})"),
        ("+" | "-" | "*", [a, b]) if ty.is_numeric() => format!("({a}) {name} ({b})"),
        ("/", [a, b]) if ty == &Ty::Float32 => format!("({a}) / ({b})"),
        ("&" | "|" | "^", [a, b]) if ty.is_integer() => format!("({a}) {name} ({b})"),
        ("==" | "!=", [a, b]) if ty.is_numeric() || ty == &Ty::Bool => {
            format!("({a}) {name} ({b})")
        }
        ("<" | "<=" | ">" | ">=", [a, b]) if ty.is_numeric() => format!("({a}) {name} ({b})"),
        ("&&" | "||", [a, b]) if ty == &Ty::Bool => format!("({a}) {name} ({b})"),
        _ => return Err(unsupported()),
    };
    Ok(types.wrap(result, expr))
}

fn literal(types: &Types<'_>, ty: &Ty, value: &Value) -> Result<String, Error> {
    Ok(match value {
        Value::Unit => "0u".into(),
        Value::Bool { value } => value.to_string(),
        Value::Int32 { value } => format!("int({}u)", *value as u32),
        Value::UInt32 { value } => format!("{value}u"),
        Value::Float32 { value } if value.is_finite() => format!("{value:e}"),
        Value::Record { value } => {
            let Ty::Record { fields } = ty else {
                unreachable!()
            };
            let args = fields
                .iter()
                .zip(&value.fields)
                .map(|(ty, field)| literal(types, &ty.ty, &field.value))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            format!(
                "{}({})",
                types.name(ty),
                if args.is_empty() { "0u" } else { &args }
            )
        }
        _ => return Err(Error("unsupported shader literal".into())),
    })
}
