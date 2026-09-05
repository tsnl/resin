use std::collections::VecDeque;

use crate::ir::{BlockId, Function, FunctionId, Module, Terminator, Ty};

use super::error::Location;
use super::instructions::{check_instr, pop_one};
use super::types::{check_value, shape};
use super::{FunctionTypes, VerifyError, VerifyErrorKind};

pub(super) fn check_function(
    module: &Module,
    function_id: FunctionId,
    function: &Function,
) -> Result<FunctionTypes, VerifyError> {
    let entry = function.entry;
    let function_location = Location::function(function_id);

    check_value(&module.types, &function.result, function_location)?;
    for local in &function.locals {
        check_value(&module.types, &local.ty, function_location)?;
    }

    let location = Location::basic_block(function_id, entry);

    if let Some(foreign) = &function.foreign {
        if function.name.is_none()
            || !function.blocks.is_empty()
            || !foreign.valid(&function.result)
            || function
                .locals
                .get(function.param.index())
                .map(|local| &local.ty)
                != Some(&Ty::parameter(&foreign.params))
        {
            return Err(function_location.error(VerifyErrorKind::InvalidForeignSignature));
        }
        return Ok(FunctionTypes {
            inputs: Vec::new(),
            results: Vec::new(),
        });
    }

    if function.blocks.get(entry.index()).is_none() {
        return Err(location.error(VerifyErrorKind::InvalidBasicBlock {
            basic_block: entry.index(),
        }));
    }

    if function.locals.get(function.param.index()).is_none() {
        return Err(location.error(VerifyErrorKind::InvalidLocal {
            local: function.param.index(),
        }));
    }

    let mut entries = vec![None; function.blocks.len()];
    entries[entry.index()] = Some(Vec::new());
    let mut pending = VecDeque::from([entry]);
    let mut results = vec![Vec::new(); function.blocks.len()];

    while let Some(basic_block_id) = pending.pop_front() {
        let basic_block = &function.blocks[basic_block_id.index()];
        let mut stack = entries[basic_block_id.index()]
            .clone()
            .expect("queued basic blocks always have an inferred input stack");

        for (instruction, instr) in basic_block.instrs.iter().enumerate() {
            let location = Location::instruction(function_id, basic_block_id, instruction);
            check_instr(module, function, instr, &mut stack, location)?;
            if instr.stack_effect().pushes == 1 {
                check_value(&module.types, stack.last().unwrap(), location)?;
            }
            results[basic_block_id.index()]
                .push((instr.stack_effect().pushes == 1).then(|| stack.last().unwrap().clone()));
        }

        let location = Location::basic_block(function_id, basic_block_id);
        match basic_block.terminator {
            Terminator::Break { target } => propagate(
                function,
                target,
                stack,
                &mut entries,
                &mut pending,
                location,
            )?,
            Terminator::Branch { then, els } => {
                let condition = pop_one(&mut stack, location)?;
                let condition_shape = shape(&module.types, condition.clone(), location)?;
                if condition_shape != Ty::Bool {
                    return Err(location.error(VerifyErrorKind::TypeMismatch {
                        expected: Ty::Bool,
                        found: condition,
                    }));
                }
                propagate(
                    function,
                    then,
                    stack.clone(),
                    &mut entries,
                    &mut pending,
                    location,
                )?;
                propagate(function, els, stack, &mut entries, &mut pending, location)?;
            }
            Terminator::Return => {
                if stack.as_slice() != [function.result.clone()] {
                    return Err(location.error(VerifyErrorKind::InvalidReturnStack {
                        expected: function.result.clone(),
                        found: stack,
                    }));
                }
            }
        }
    }

    if let Some(basic_block) = entries.iter().position(Option::is_none) {
        return Err(
            Location::basic_block(function_id, BlockId::from_index(basic_block))
                .error(VerifyErrorKind::UnreachableBasicBlock),
        );
    }

    Ok(FunctionTypes {
        inputs: entries.into_iter().map(Option::unwrap).collect(),
        results,
    })
}

fn propagate(
    function: &Function,
    target: BlockId,
    stack: Vec<Ty>,
    entries: &mut [Option<Vec<Ty>>],
    pending: &mut VecDeque<BlockId>,
    location: Location,
) -> Result<(), VerifyError> {
    if function.blocks.get(target.index()).is_none() {
        return Err(location.error(VerifyErrorKind::InvalidBasicBlock {
            basic_block: target.index(),
        }));
    }

    match &entries[target.index()] {
        None => {
            entries[target.index()] = Some(stack);
            pending.push_back(target);
        }
        Some(expected) if expected != &stack => {
            return Err(location.error(VerifyErrorKind::ConflictingBasicBlockStack {
                expected: expected.clone(),
                found: stack,
            }));
        }
        Some(_) => {}
    }
    Ok(())
}
