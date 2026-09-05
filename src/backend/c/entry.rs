use std::fmt::Write;

use crate::{backend::Error, ir::Ty};

use super::types::Types;

pub(super) fn emit(types: &Types<'_>) -> Result<String, Error> {
    let mut out = String::new();
    let entries: Vec<_> = types
        .module
        .globals
        .iter()
        .enumerate()
        .filter(|(_, global)| global.name.as_ref() == "main")
        .collect();
    let functions: Vec<_> = types
        .module
        .functions
        .iter()
        .enumerate()
        .filter(|(_, function)| function.name.as_deref() == Some("main"))
        .collect();
    if let [(index, function)] = functions.as_slice() {
        if !entries.is_empty()
            || function.foreign.is_some()
            || function.locals[function.param.index()].ty != Ty::Unit
            || !matches!(function.result, Ty::Unit | Ty::Int32)
        {
            return Err(Error("main must have type () -> int or () -> ()".into()));
        }
        if function.result == Ty::Unit {
            writeln!(out, "  r_fn{index}(0);\n  return 0;").unwrap();
        } else {
            writeln!(out, "  return r_fn{index}(0);").unwrap();
        }
    } else if !functions.is_empty() {
        return Err(Error("multiple functions named main".into()));
    } else {
        match entries.as_slice() {
            [] => out.push_str("  return 0;\n"),
            [(index, global)] => {
                let Ty::Function { param, result } = types.shape(&global.ty) else {
                    return Err(Error("main must be a function".into()));
                };
                if param.as_ref() != &Ty::Unit || !matches!(result.as_ref(), Ty::Unit | Ty::Int32) {
                    return Err(Error("main must have type () -> int or () -> ()".into()));
                }
                let entry = types.unwrap(&global.ty, format!("r_g{index}"));
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
            _ => return Err(Error("multiple globals named main".into())),
        }
    }
    Ok(out)
}
