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
            let value = if foreign.params.len() == 1 {
                "r_arg".into()
            } else {
                format!("r_arg.f{i}")
            };
            if matches!(ty, Ty::Pointer { .. }) {
                format!("(void *){value}")
            } else {
                value
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let mut out = "  (void)r_arg;\n".to_string();
    let call = format!("{}({args})", function.name.as_deref().unwrap());
    if function.result == Ty::Unit {
        writeln!(out, "  {call};\n  return 0;").unwrap();
    } else {
        writeln!(out, "  return ({}){call};", types.name(&function.result)).unwrap();
    }
    out
}
