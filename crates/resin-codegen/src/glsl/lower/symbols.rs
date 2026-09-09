//! Carry local places and direct functions across edges when predecessors agree.
//! Runtime operands, including dynamic indices, have already been assigned temporaries.
use super::{Slot, ops::instruction, types::Types};
use crate::Error;
use resin_lir::FunctionTypes;
use resin_lir::{Function, Instr, Terminator};

pub(super) fn inputs(
    types: &mut Types<'_>,
    function: &Function,
    flow: &FunctionTypes,
) -> Result<Vec<Vec<Slot>>, Error> {
    let mut inputs: Vec<Option<Vec<Slot>>> = vec![None; function.blocks.len()];
    inputs[function.entry.index()] = Some(Vec::new());
    let mut pending = vec![function.entry.index()];
    while let Some(b) = pending.pop() {
        let mut stack = inputs[b].clone().unwrap();
        for (i, instr) in function.blocks[b].instrs.iter().enumerate() {
            let args = stack
                .split_off(stack.len() - flow.operand_count(resin_lir::BlockId::from_index(b), i));
            if let Some(ty) = &flow.results[b][i] {
                let local = Slot::local_result(instr, &args);
                let expr = if local || matches!(instr, Instr::Function { .. }) {
                    instruction(types, instr, &args, Some(ty), &mut String::new())?.unwrap()
                } else {
                    format!("r_v{b}_{i}")
                };
                stack.push(Slot {
                    ty: ty.clone(),
                    expr,
                    local,
                });
            }
        }
        let targets = match function.blocks[b].terminator {
            Terminator::Return => vec![],
            Terminator::Break { target } => vec![target],
            Terminator::Branch { then, els } => {
                stack.pop();
                vec![then, els]
            }
        };
        for target in targets {
            let t = target.index();
            let incoming: Vec<_> = stack
                .iter()
                .enumerate()
                .map(|(i, slot)| {
                    if slot.symbolic() {
                        slot.clone()
                    } else {
                        Slot {
                            ty: slot.ty.clone(),
                            expr: format!("r_b{t}_{i}"),
                            local: false,
                        }
                    }
                })
                .collect();
            if let Some(previous) = &inputs[t] {
                if previous != &incoming {
                    return Err(Error(
                        "shader cannot merge distinct local addresses or function values".into(),
                    ));
                }
            } else {
                inputs[t] = Some(incoming);
                pending.push(t);
            }
        }
    }
    Ok(inputs.into_iter().map(Option::unwrap_or_default).collect())
}
