use resin_types::prelude::*;
use std::fmt::Write;

use super::types::Types;

pub(super) fn lower(types: &Types<'_>, index: usize, foreign: &Foreign) -> String {
    let function = &types.module.functions[index];
    let args = foreign
        .params
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            let value = format!("r_arg{i}");
            if matches!(ty, Ty::Pointer { .. }) {
                format!("(void *){value}")
            } else {
                value
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let mut out = String::new();
    let call = format!("{}({args})", function.name.as_deref().unwrap());
    if function.result == Ty::Unit {
        writeln!(out, "  {call};\n  return 0;").unwrap();
    } else {
        writeln!(out, "  return ({}){call};", types.name(&function.result)).unwrap();
    }
    out
}
