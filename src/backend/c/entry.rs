use std::fmt::Write;

use crate::{
    backend::Error,
    ir::{Entry, Ty},
};

use super::types::Types;

pub(super) fn emit(types: &Types<'_>) -> Result<String, Error> {
    let mut out = String::new();
    match types.module.entries.get("main") {
        Some(Entry::Function(id)) => {
            let function = &types.module.functions[id.index()];
            if function.foreign.is_some()
                || function.locals[function.param.index()].ty != Ty::Unit
                || !matches!(function.result, Ty::Unit | Ty::Int32)
            {
                return Err(Error("main must have type () -> int or () -> ()".into()));
            }
            if function.result == Ty::Unit {
                writeln!(out, "  r_fn{}(0);\n  return 0;", id.index()).unwrap();
            } else {
                writeln!(out, "  return r_fn{}(0);", id.index()).unwrap();
            }
        }
        Some(Entry::Global(id)) => {
            let global = &types.module.globals[id.index()];
            let Ty::Function { param, result } = types.shape(&global.ty) else {
                return Err(Error("main must be a function".into()));
            };
            if param.as_ref() != &Ty::Unit || !matches!(result.as_ref(), Ty::Unit | Ty::Int32) {
                return Err(Error("main must have type () -> int or () -> ()".into()));
            }
            let entry = types.unwrap(&global.ty, format!("r_g{}", id.index()));
            writeln!(
                out,
                "  if (!({entry}).call) resin_fail(\"main is not initialized\");"
            )
            .unwrap();
            if result.as_ref() == &Ty::Unit {
                writeln!(out, "  ({entry}).call(0);\n  return 0;").unwrap();
            } else {
                writeln!(out, "  return ({entry}).call(0);").unwrap();
            }
        }
        None => out.push_str("  return 0;\n"),
    }
    Ok(out)
}
