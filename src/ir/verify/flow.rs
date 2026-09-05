use std::collections::VecDeque;

use crate::ir::{BlockId, Function, FunctionId, Module, Terminator, Ty};

use super::error::Location;
use super::instructions::{check_instr, pop_one};
use super::types::{check_type, shape};
use super::{VerifyError, VerifyErrorKind};

pub(super) fn check_function(
    module: &Module,
    function_id: FunctionId,
    function: &Function,
) -> Result<(), VerifyError> {
    let entry = function.entry;
    let function_location = Location::function(function_id);

    check_type(&module.types, &function.result, function_location)?;
    for local in &function.locals {
        check_type(&module.types, &local.ty, function_location)?;
    }
    for nonlocal in &function.nonlocals {
        check_type(&module.types, &nonlocal.ty, function_location)?;
    }

    let location = Location::basic_block(function_id, entry);

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

    while let Some(basic_block_id) = pending.pop_front() {
        let basic_block = &function.blocks[basic_block_id.index()];
        let mut stack = entries[basic_block_id.index()]
            .clone()
            .expect("queued basic blocks always have an inferred input stack");

        for (instruction, instr) in basic_block.instrs.iter().enumerate() {
            let location = Location::instruction(function_id, basic_block_id, instruction);
            check_instr(module, function, instr, &mut stack, location)?;
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

    Ok(())
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
