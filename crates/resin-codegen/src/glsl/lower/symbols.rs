//! Carry local places and direct functions through structured regions.
//! Runtime operands receive boundary temporaries; symbolic operands must agree
//! at joins and across loop iterations because GLSL cannot store local addresses.
use super::{Slot, ops::instruction, types::Types};
use crate::Error;
use resin_lir::{BlockId, Function, FunctionTypes, Instr, Terminator};

pub(super) fn inputs(
    types: &mut Types<'_>,
    function: &Function,
    flow: &FunctionTypes,
) -> Result<Vec<Vec<Slot>>, Error> {
    let mut regions = Regions {
        types,
        function,
        flow,
        inputs: vec![vec![]; function.blocks.len()],
    };
    regions.visit(function.entry, vec![])?;
    Ok(regions.inputs)
}

struct Regions<'a, 'm> {
    types: &'a mut Types<'m>,
    function: &'a Function,
    flow: &'a FunctionTypes,
    inputs: Vec<Vec<Slot>>,
}

impl Regions<'_, '_> {
    fn visit(
        &mut self,
        mut id: BlockId,
        mut incoming: Vec<Slot>,
    ) -> Result<Option<Vec<Slot>>, Error> {
        loop {
            let b = id.index();
            let mut stack: Vec<_> = incoming
                .into_iter()
                .enumerate()
                .map(|(i, slot)| {
                    if slot.symbolic() {
                        slot
                    } else {
                        Slot {
                            expr: format!("r_b{b}_{i}"),
                            ..slot
                        }
                    }
                })
                .collect();
            self.inputs[b] = stack.clone();
            for (i, instr) in self.function.blocks[b].instrs.iter().enumerate() {
                let args = stack.split_off(stack.len() - self.flow.operand_count(id, i));
                if let Some(ty) = &self.flow.results[b][i] {
                    let local = Slot::local_result(instr, &args);
                    let expr = if local || matches!(instr, Instr::Function { .. }) {
                        instruction(self.types, instr, &args, Some(ty), &mut String::new())?
                            .unwrap()
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
            let (next, output) = match self.function.blocks[b].terminator {
                Terminator::Return => return Ok(None),
                Terminator::Merge | Terminator::LoopTest | Terminator::Continue => {
                    return Ok(Some(stack));
                }
                Terminator::If { then, els, next } => {
                    stack.pop();
                    let then = self.visit(then, stack.clone())?;
                    let els = self.visit(els, stack)?;
                    let output = match (then, els) {
                        (Some(then), Some(els)) => {
                            agree(&then, &els)?;
                            Some(then)
                        }
                        (then, els) => then.or(els),
                    };
                    (next, output)
                }
                Terminator::Loop {
                    condition,
                    body,
                    next,
                } => {
                    let mut output = self.visit(condition, stack)?.unwrap();
                    output.pop();
                    if let Some(repeated) = self.visit(body, output.clone())? {
                        agree(&self.inputs[condition.index()], &repeated)?;
                    }
                    // False exits with the condition's operands, not the body's outputs.
                    (next, Some(output))
                }
            };
            let Some(next) = next else { return Ok(output) };
            incoming = output.unwrap();
            id = next;
        }
    }
}

fn agree(left: &[Slot], right: &[Slot]) -> Result<(), Error> {
    if left
        .iter()
        .zip(right)
        .any(|(a, b)| (a.symbolic() || b.symbolic()) && a != b)
    {
        return Err(Error(
            "shader cannot merge distinct local addresses or function values".into(),
        ));
    }
    Ok(())
}
