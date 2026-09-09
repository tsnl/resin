//! Lower one stack operation to GLSL expressions and local statements.
use super::{Slot, types::Types};
use crate::Error;
use resin_lir::Instr;
use resin_types::prelude::*;
use std::fmt::Write;

pub(super) fn instruction(
    types: &mut Types<'_>,
    instr: &Instr,
    args: &[Slot],
    result: Option<&Ty>,
    out: &mut String,
) -> Result<Option<String>, Error> {
    let expr = match instr {
        Instr::ForgetLocal { .. } => return Ok(None),
        Instr::TransferLoad => dereference(types, &args[0])?,
        Instr::TakeLocal { local } => format!("r_l{}", local.index()),
        Instr::SetLocal { local } => {
            writeln!(out, "      r_l{} = {};", local.index(), args[0].expr).unwrap();
            return Ok(None);
        }
        Instr::MakeVariant { ty, tag } => variant(types, ty, tag, &args[0].expr),
        Instr::ExcludeNone => widen(types, &args[0].ty, result.unwrap(), &args[0].expr),
        Instr::IsVariant { tag } => {
            let (ty, expr) = if let Ty::Pointer { pointee } = &args[0].ty {
                (pointee.as_ref(), dereference(types, &args[0])?)
            } else {
                (&args[0].ty, args[0].expr.clone())
            };
            is_variant(types, ty, tag, &expr)
        }
        Instr::VariantPayload { tag } => {
            if matches!(tag, Case::Type(member) if member == &args[0].ty) {
                args[0].expr.clone()
            } else {
                format!("({}).v{}", args[0].expr, types.tag(tag))
            }
        }
        Instr::Widen { ty } => widen(types, &args[0].ty, ty, &args[0].expr),
        Instr::Function { function } => format!("r_fn{}", function.index()),
        Instr::Call => format!("{}({})", args[0].expr, args[1].expr),
        Instr::Push { value } => literal(types, result.unwrap(), value)?,
        Instr::LocalAddress { local } => format!("r_l{}", local.index()),
        Instr::Load => dereference(types, &args[0])?,
        Instr::Replace => {
            let target = dereference(types, &args[0])?;
            let old = format!("r_replaced_{}", out.len());
            writeln!(out, "      {} {old} = {target};", types.name(&args[1].ty)).unwrap();
            writeln!(out, "      {target} = {};", args[1].expr).unwrap();
            old
        }
        Instr::Store => {
            writeln!(
                out,
                "      {} = {};",
                dereference(types, &args[0])?,
                args[1].expr
            )
            .unwrap();
            args[1].expr.clone()
        }
        Instr::NumericCast { ty } => format!("{}({})", types.name(ty), args[0].expr),
        Instr::PointerCast { .. } => format!("uint64_t({})", args[0].expr),
        Instr::Discard => return Ok(None),
        Instr::Ascribe { ty } => {
            if ty == &args[0].ty {
                args[0].expr.clone()
            } else if matches!(ty, Ty::Span { .. }) || matches!(args[0].ty, Ty::Span { .. }) {
                format!(
                    "{}(({}).f0, ({}).f1)",
                    types.name(ty),
                    args[0].expr,
                    args[0].expr
                )
            } else if let Ty::Defined { definition } = ty
                && types.module.types[definition.index()].body() == Some(&args[0].ty)
            {
                format!("{}({})", types.name(ty), args[0].expr)
            } else {
                format!("({}).value", args[0].expr)
            }
        }
        Instr::MakeArray { elements, element } => format!(
            "{}({}[{elements}]({}))",
            types.name(result.unwrap()),
            types.name(element),
            args.iter()
                .map(|arg| arg.expr.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Instr::AccessDynamic => {
            let base = &args[0];
            let index = &args[1].expr;
            match types.shape(&base.ty) {
                Ty::Span { element } => {
                    let size = crate::layout::layout(types.module, element)?.size;
                    format!("({}).f0 + uint64_t({index}) * uint64_t({size})", base.expr)
                }
                Ty::Array { .. } => format!("({}).items[uint({index})]", base.expr),
                Ty::Pointer { .. } if base.local => {
                    format!("({}).items[uint({index})]", base.expr)
                }
                Ty::Pointer { pointee } => {
                    let Ty::Array { element, .. } = types.shape(pointee) else {
                        return Err(Error("array pointer required".into()));
                    };
                    let size = crate::layout::layout(types.module, element)?.size;
                    format!("({}) + uint64_t({index}) * uint64_t({size})", base.expr)
                }
                _ => return Err(Error("array or Span required".into())),
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
            if !matches!(types.shape(ty), Ty::Record { .. } | Ty::Span { .. }) {
                return Err(Error("shader projection requires a record".into()));
            }
            if matches!(args[0].ty, Ty::Pointer { .. }) && !args[0].local {
                let layout = crate::layout::layout(types.module, ty)?;
                format!("({}) + uint64_t({})", args[0].expr, layout.offsets[*index])
            } else {
                format!("({}).f{index}", types.unwrap(ty, args[0].expr.clone()))
            }
        }
        Instr::CallBuiltin { name, result, .. } => builtin(types, name, args, result)?,
        _ => return Err(Error(format!("shader profile does not support {instr:?}"))),
    };
    Ok(Some(expr))
}

pub(super) fn is_variant(types: &Types<'_>, ty: &Ty, case: &Case, value: &str) -> String {
    if matches!(case, Case::Type(member) if member == ty) {
        return "true".into();
    }
    format!("(({value}).tag == {}u)", types.tag(case))
}

fn variant(types: &Types<'_>, ty: &Ty, case: &Case, value: &str) -> String {
    if matches!(case, Case::Type(member) if member == ty) {
        return value.into();
    }
    let mut fields = vec![format!("{}u", types.tag(case))];
    fields.extend(ty.payloads().unwrap().iter().map(|(candidate, ty)| {
        if candidate == case {
            value.into()
        } else {
            types.zero(ty)
        }
    }));
    format!("{}({})", types.name(ty), fields.join(", "))
}

fn widen(types: &Types<'_>, from: &Ty, to: &Ty, value: &str) -> String {
    if from == to {
        return value.into();
    }
    if matches!(to, Ty::Union { variants } if variants.contains(from)) {
        return variant(types, to, &Case::Type(from.clone()), value);
    }
    let mut expression = types.zero(to);
    for (case, payload) in from.payloads().unwrap_or_default().into_iter().rev() {
        let Some(target) = to.payload(&case) else {
            continue;
        };
        let tag = types.tag(&case);
        let payload = widen(types, &payload, &target, &format!("({value}).v{tag}"));
        let constructed = variant(types, to, &case, &payload);
        expression = format!("(({value}).tag == {tag}u ? {constructed} : {expression})");
    }
    expression
}

fn dereference(types: &mut Types<'_>, slot: &Slot) -> Result<String, Error> {
    if slot.local {
        return Ok(slot.expr.clone());
    }
    let Ty::Pointer { pointee } = &slot.ty else {
        unreachable!()
    };
    Ok(format!("{}({}).value", types.buffer(pointee)?, slot.expr))
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
        ("==" | "!=", [a, b]) if ty.is_numeric() || matches!(ty, Ty::Bool | Ty::Pointer { .. }) => {
            format!("({a}) {name} ({b})")
        }
        ("<" | "<=" | ">" | ">=", [a, b]) if ty.is_numeric() => format!("({a}) {name} ({b})"),
        ("&&" | "||", [a, b]) if ty == &Ty::Bool => format!("({a}) {name} ({b})"),
        _ => return Err(unsupported()),
    };
    let expr = if result == &Ty::UInt8 {
        format!("uint8_t({expr})")
    } else {
        expr
    };
    Ok(types.wrap(result, expr))
}

fn literal(types: &Types<'_>, ty: &Ty, value: &Value) -> Result<String, Error> {
    Ok(match value {
        Value::None | Value::Unit => "0u".into(),
        Value::Bool { value } => value.to_string(),
        Value::Int32 { value } => format!("int({}u)", *value as u32),
        Value::UInt8 { value } => format!("uint8_t({value})"),
        Value::Bytes { .. } => return Err(Error("shader string literals need device-backed storage; pass a Span<ubyte> in the shader root".into())),
        Value::UInt32 { value } => format!("{value}u"),
        Value::UInt64 { value } => format!("{value}ul"),
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
