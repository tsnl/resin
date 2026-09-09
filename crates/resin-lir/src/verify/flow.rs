//! Verify a structured block tree and its stack contracts.

use resin_types::prelude::*;

use crate::{BlockId, Function, Module, Terminator};

use super::error::Location;
use super::instructions::{check_instr, pop_one};
use super::rules::{check_value, shape};
use crate::{FunctionTypes, VerifyError, VerifyErrorKind};

pub(super) fn check_function(
    module: &Module,
    function_id: FunctionId,
    function: &Function,
) -> Result<FunctionTypes, VerifyError> {
    let entry = function.entry;
    let function_location = Location::function(function_id);
    if function.locals.is_empty() {
        return Err(function_location.error(VerifyErrorKind::InvalidLocal { local: 0 }));
    }

    check_value(&module.types, &function.result, function_location)?;
    for local in &function.locals {
        check_value(&module.types, &local.ty, function_location)?;
    }

    if let Some(foreign) = &function.foreign {
        if function.name.is_none()
            || !function.blocks.is_empty()
            || !foreign.valid(&function.result)
            || function.locals[0].ty != Ty::parameter(&foreign.params)
        {
            return Err(function_location.error(VerifyErrorKind::InvalidForeignSignature));
        }
        return Ok(FunctionTypes {
            inputs: Vec::new(),
            results: Vec::new(),
            operand_counts: Vec::new(),
        });
    }

    let mut checker = Regions {
        module,
        function,
        function_id,
        entries: vec![None; function.blocks.len()],
        results: vec![Vec::new(); function.blocks.len()],
        operand_counts: vec![Vec::new(); function.blocks.len()],
    };
    checker.visit(entry, Vec::new(), Region::Function)?;
    if let Some(block) = checker.entries.iter().position(Option::is_none) {
        return Err(
            Location::basic_block(function_id, BlockId::from_index(block))
                .error(VerifyErrorKind::UnreachableBasicBlock),
        );
    }
    Ok(FunctionTypes {
        inputs: checker.entries.into_iter().map(Option::unwrap).collect(),
        results: checker.results,
        operand_counts: checker.operand_counts,
    })
}

/// Enter each owned block once. A return has no region result and does not
/// participate in joins; loops check their back-edge contract without a fixpoint.
/// Iterate continuations so only actual region nesting consumes call-stack depth.
struct Regions<'a> {
    module: &'a Module,
    function: &'a Function,
    function_id: FunctionId,
    entries: Vec<Option<Vec<Ty>>>,
    results: Vec<Vec<Option<Ty>>>,
    operand_counts: Vec<Vec<usize>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Region {
    Function,
    Selection,
    LoopCondition,
    LoopBody,
}

impl Regions<'_> {
    fn visit(
        &mut self,
        mut id: BlockId,
        mut stack: Vec<Ty>,
        region: Region,
    ) -> Result<Option<Vec<Ty>>, VerifyError> {
        loop {
            let location = Location::basic_block(self.function_id, id);
            let block = self.function.blocks.get(id.index()).ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidBasicBlock {
                    basic_block: id.index(),
                })
            })?;
            if self.entries[id.index()].is_some() {
                return Err(location.error(VerifyErrorKind::ReusedBasicBlock));
            }
            self.entries[id.index()] = Some(stack.clone());
            for (i, instr) in block.instrs.iter().enumerate() {
                let location = Location::instruction(self.function_id, id, i);
                check_instr(self.module, self.function, instr, &mut stack, location)?;
                let effect = super::stack_effect(instr);
                self.operand_counts[id.index()].push(effect.pops);
                if effect.pushes == 1 {
                    check_value(&self.module.types, stack.last().unwrap(), location)?;
                }
                self.results[id.index()]
                    .push((effect.pushes == 1).then(|| stack.last().unwrap().clone()));
            }
            let (next, output) = match block.terminator {
                Terminator::Merge => {
                    if region != Region::Selection {
                        return Err(location.error(VerifyErrorKind::UnexpectedMerge));
                    }
                    return Ok(Some(stack));
                }
                Terminator::LoopTest => {
                    if region != Region::LoopCondition {
                        return Err(location.error(VerifyErrorKind::UnexpectedLoopTest));
                    }
                    return Ok(Some(stack));
                }
                Terminator::Continue => {
                    if region != Region::LoopBody {
                        return Err(location.error(VerifyErrorKind::UnexpectedContinue));
                    }
                    return Ok(Some(stack));
                }
                Terminator::Return => {
                    if stack.as_slice() != [self.function.result.clone()] {
                        return Err(location.error(VerifyErrorKind::InvalidReturnStack {
                            expected: self.function.result.clone(),
                            found: stack,
                        }));
                    }
                    return Ok(None);
                }
                Terminator::If { then, els, next } => {
                    self.condition(&mut stack, location)?;
                    let then_stack = self.visit(then, stack.clone(), Region::Selection)?;
                    let else_stack = self.visit(els, stack, Region::Selection)?;
                    let output = join(then_stack, else_stack, location)?;
                    (next, output)
                }
                Terminator::Loop {
                    condition,
                    body,
                    next,
                } => {
                    let mut output = self
                        .visit(condition, stack.clone(), Region::LoopCondition)?
                        .ok_or_else(|| location.error(VerifyErrorKind::MissingLoopTest))?;
                    self.condition(&mut output, location)?;
                    same_stack(&stack, &output, location)?;
                    if let Some(repeated) = self.visit(body, output.clone(), Region::LoopBody)? {
                        same_stack(&stack, &repeated, location)?;
                    }
                    (next, Some(output))
                }
            };
            let Some(next) = next else {
                // Tail selections may forward a merge to their enclosing arm.
                // Testing or repeating a loop always needs its own explicit exit.
                if output.is_some() && region != Region::Selection {
                    return Err(location.error(VerifyErrorKind::MissingRegionContinuation));
                }
                return Ok(output);
            };
            stack = output.ok_or_else(|| location.error(VerifyErrorKind::MissingRegionResult))?;
            id = next;
        }
    }

    fn condition(&self, stack: &mut Vec<Ty>, location: Location) -> Result<(), VerifyError> {
        let condition = pop_one(stack, location)?;
        if shape(&self.module.types, condition.clone(), location)? != Ty::Bool {
            return Err(location.error(VerifyErrorKind::TypeMismatch {
                expected: Ty::Bool,
                found: condition,
            }));
        }
        Ok(())
    }
}

fn same_stack(expected: &[Ty], found: &[Ty], location: Location) -> Result<(), VerifyError> {
    if expected != found {
        return Err(location.error(VerifyErrorKind::ConflictingBasicBlockStack {
            expected: expected.to_vec(),
            found: found.to_vec(),
        }));
    }
    Ok(())
}

fn join(
    then: Option<Vec<Ty>>,
    els: Option<Vec<Ty>>,
    location: Location,
) -> Result<Option<Vec<Ty>>, VerifyError> {
    match (then, els) {
        (Some(then), Some(els)) => {
            same_stack(&then, &els, location)?;
            Ok(Some(then))
        }
        (then, els) => Ok(then.or(els)),
    }
}
