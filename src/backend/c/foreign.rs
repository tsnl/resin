use std::fmt::Write;

use crate::ir::{Foreign, Ty};

use super::types::Types;

pub(super) fn emit(types: &Types<'_>, index: usize, foreign: &Foreign) -> String {
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
    let mut out = format!(
        "{} r_fn{index}({} r_arg) {{\n  (void)r_arg;\n",
        types.name(&function.result),
        types.name(&function.locals[function.param.index()].ty)
    );
    let call = format!("{}({args})", function.name.as_deref().unwrap());
    if function.result == Ty::Unit {
        writeln!(out, "  {call};\n  return 0;\n}}").unwrap();
    } else {
        writeln!(
            out,
            "  return ({}){call};\n}}",
            types.name(&function.result)
        )
        .unwrap();
    }
    out
}
