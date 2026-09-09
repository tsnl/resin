use crate::Error;
use resin_types::prelude::*;

use super::{Slot, types::Types};

pub(super) fn builtin(
    types: &Types<'_>,
    name: &str,
    args: &[Slot],
    result: &Ty,
) -> Result<String, Error> {
    if name == "string_from_str" {
        return super::formatting::from_str(types, args, result);
    }
    if name == "fmt" {
        return super::formatting::format(types, args, result);
    }
    if name == "print" {
        return super::formatting::emit(types, args, result);
    }
    let unsupported = || {
        Error(format!(
            "unsupported builtin {name:?}: {:?} -> {result:?}",
            args.iter().map(|a| &a.ty).collect::<Vec<_>>()
        ))
    };
    let Some(first) = args.first() else {
        return Err(unsupported());
    };

    if args.iter().any(|arg| arg.ty != first.ty) {
        return Err(unsupported());
    }
    let ty = types.shape(&first.ty);
    let values: Vec<_> = args
        .iter()
        .map(|arg| types.unwrap(&arg.ty, arg.expr.clone()))
        .collect();
    let comparison = matches!(
        name,
        "==" | "!=" | "<" | "<=" | ">" | ">=" | "!" | "&&" | "||"
    );
    if result != if comparison { &Ty::Bool } else { &first.ty } {
        return Err(unsupported());
    }
    let expr = match (name, values.as_slice()) {
        ("!", [a]) if ty == &Ty::Bool => format!("!({a})"),
        ("&&" | "||", [a, b]) if ty == &Ty::Bool => format!("({a}) {name} ({b})"),
        ("==" | "!=", [a, b])
            if ty.is_numeric() || matches!(ty, Ty::Bool | Ty::Type | Ty::Pointer { .. }) =>
        {
            format!("({a}) {name} ({b})")
        }
        ("<" | "<=" | ">" | ">=", [a, b]) if ty.is_numeric() => format!("({a}) {name} ({b})"),
        ("+", [a]) if ty.is_numeric() => a.clone(),
        ("-", [a]) if ty.is_integer() => {
            integer(types, ty, format!("UINT64_C(0) - (uint64_t)({a})"))
        }
        ("~", [a]) if ty.is_integer() => integer(types, ty, format!("~(uint64_t)({a})")),
        ("-", [a]) if ty.is_numeric() => format!("-({a})"),
        ("+" | "-" | "*" | "&" | "|" | "^", [a, b]) if ty.is_integer() => {
            integer(types, ty, format!("(uint64_t)({a}) {name} (uint64_t)({b})"))
        }
        ("/" | "%", [a, b]) if ty.is_integer() => {
            let sign = if signed(ty) { "i" } else { "u" };
            let op = if name == "/" { "div" } else { "mod" };
            integer(types, ty, format!("resin_{sign}{op}({a}, {b})"))
        }
        ("<<" | ">>", [a, b]) if ty.is_integer() => {
            let count = format!("resin_shift((uint64_t)({b}), {})", width(ty));
            let expr = if name == ">>" && signed(ty) {
                format!("({a}) < 0 ? ~((~(uint64_t)({a})) >> {count}) : (uint64_t)({a}) >> {count}")
            } else {
                format!("(uint64_t)({a}) {name} {count}")
            };
            integer(types, ty, expr)
        }
        ("+" | "-" | "*" | "/", [a, b]) if ty.is_numeric() => format!("({a}) {name} ({b})"),
        _ => return Err(unsupported()),
    };
    Ok(types.wrap(result, expr))
}

fn integer(types: &Types<'_>, ty: &Ty, expr: String) -> String {
    if signed(ty) {
        format!("resin_i{}({expr})", width(ty))
    } else {
        format!("({})({expr})", types.name(ty))
    }
}

fn signed(ty: &Ty) -> bool {
    matches!(ty, Ty::Int8 | Ty::Int16 | Ty::Int32 | Ty::Int64)
}

fn width(ty: &Ty) -> usize {
    match ty {
        Ty::Int8 | Ty::UInt8 => 8,
        Ty::Int16 | Ty::UInt16 => 16,
        Ty::Int32 | Ty::UInt32 => 32,
        Ty::Int64 | Ty::UInt64 => 64,
        _ => unreachable!(),
    }
}
