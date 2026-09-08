use super::FunctionTypes;
use crate::lir::{Instr, Module, Value};
use crate::types::TypeTable;

/// Complete the table after stack verification has resolved every operand.
/// The verified module shares this table across all shader and host emitters.
pub(super) fn collect(module: &Module, analysis: &[FunctionTypes]) -> TypeTable {
    let mut table = module.types.clone();
    for definition in module.types.iter() {
        let body = definition.body().expect("verified type definition");
        table.intern(body);
    }
    for (function, flow) in module.functions.iter().zip(analysis) {
        table.intern(&function.ty().expect("verified parameter"));
        for local in &function.locals {
            table.intern(&local.ty);
        }
        for ty in flow
            .inputs
            .iter()
            .flatten()
            .chain(flow.results.iter().flatten().flatten())
        {
            table.intern(ty);
        }
        for block in &function.blocks {
            for instr in &block.instrs {
                if let Instr::Push { value } = instr {
                    collect_value(&mut table, value);
                }
            }
        }
    }
    table
}
fn collect_value(table: &mut TypeTable, value: &Value) {
    match value {
        Value::Type { ty } => {
            table.intern(ty);
        }
        Value::Array { value } => {
            for v in &value.elements {
                collect_value(table, v);
            }
        }
        Value::Record { value } => {
            for f in &value.fields {
                collect_value(table, &f.value);
            }
        }
        _ => {}
    }
}
