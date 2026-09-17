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
    if function.parameter_count > function.locals.len() {
        return Err(function_location.error(VerifyErrorKind::InvalidLocal {
            local: function.parameter_count - 1,
        }));
    }

    check_value(&module.types, &function.result, function_location)?;
    for local in &function.locals {
        check_value(&module.types, &local.ty, function_location)?;
    }

    if let Some(foreign) = &function.foreign {
        if function.name.is_none()
            || !function.blocks.is_empty()
            || !foreign.valid(&function.result)
            || function.parameter_count != foreign.params.len()
            || function
                .locals
                .iter()
                .take(function.parameter_count)
                .map(|local| &local.ty)
                .ne(foreign.params.iter())
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
        parallel_result: None,
        parallel_body: None,
        private_owners: private_owners(function, function_location)?,
    };
    checker.visit(entry, Vec::new(), Region::Function, None)?;
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
    parallel_result: Option<Ty>,
    parallel_body: Option<BlockId>,
    private_owners: Vec<Option<BlockId>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Region {
    Function,
    Selection,
    LoopCondition,
    LoopBody,
    Parallel,
}

impl Regions<'_> {
    fn visit(
        &mut self,
        mut id: BlockId,
        mut stack: Vec<Ty>,
        region: Region,
        loop_stack: Option<&[Ty]>,
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
                if let crate::Instr::LocalRef { local }
                | crate::Instr::TakeLocal { local }
                | crate::Instr::SetLocal { local }
                | crate::Instr::ForgetLocal { local }
                | crate::Instr::DropLocal { local }
                | crate::Instr::TakeField { local, .. }
                | crate::Instr::SetField { local, .. } = instr
                    && let Some(Some(owner)) = self.private_owners.get(local.index())
                    && self.parallel_body != Some(*owner)
                {
                    return Err(location.error(VerifyErrorKind::InvalidParallelRegion));
                }
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
                Terminator::Parallel {
                    ref operation,
                    ref private_locals,
                    body,
                    next,
                } => {
                    if self.function.profile != crate::Profile::Compute
                        || self.parallel_result.is_some()
                    {
                        return Err(location.error(VerifyErrorKind::InvalidParallelRegion));
                    }
                    let result = self.parallel(operation, private_locals, location)?;
                    self.parallel_result = Some(result.clone());
                    self.parallel_body = Some(body);
                    if let Some(output) = self.visit(body, vec![], Region::Parallel, None)? {
                        same_stack(&[result], &output, location)?;
                    }
                    self.parallel_result = None;
                    self.parallel_body = None;
                    (Some(next), Some(stack))
                }
                Terminator::ParallelYield => {
                    if region != Region::Parallel {
                        return Err(location.error(VerifyErrorKind::InvalidParallelRegion));
                    }
                    same_stack(&[self.parallel_result.clone().unwrap()], &stack, location)?;
                    return Ok(Some(stack));
                }
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
                Terminator::Break | Terminator::NextIteration => {
                    let Some(carried) = loop_stack else {
                        return Err(location.error(VerifyErrorKind::UnexpectedLoopExit));
                    };
                    same_stack(carried, &stack, location)?;
                    return Ok(None);
                }
                Terminator::Return => {
                    if self.parallel_result.is_some() {
                        return Err(location.error(VerifyErrorKind::InvalidParallelRegion));
                    }
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
                    let then_stack =
                        self.visit(then, stack.clone(), Region::Selection, loop_stack)?;
                    let else_stack = self.visit(els, stack, Region::Selection, loop_stack)?;
                    let output = join(then_stack, else_stack, location)?;
                    (next, output)
                }
                Terminator::Loop {
                    condition,
                    body,
                    next,
                } => {
                    let mut output = self
                        .visit(condition, stack.clone(), Region::LoopCondition, None)?
                        .ok_or_else(|| location.error(VerifyErrorKind::MissingLoopTest))?;
                    self.condition(&mut output, location)?;
                    same_stack(&stack, &output, location)?;
                    if let Some(repeated) =
                        self.visit(body, output.clone(), Region::LoopBody, Some(&stack))?
                    {
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

    fn parallel(
        &self,
        operation: &crate::ParallelOperation,
        private: &std::ops::Range<usize>,
        location: Location,
    ) -> Result<Ty, VerifyError> {
        use crate::ParallelOperation;
        let invalid = || location.error(VerifyErrorKind::InvalidParallelRegion);
        if private.start < self.function.parameter_count
            || private.start >= private.end
            || private.end > self.function.locals.len()
        {
            return Err(invalid());
        }
        let local = |id: LocalId, is_private| {
            if private.contains(&id.index()) != is_private
                || (!is_private
                    && self
                        .private_owners
                        .get(id.index())
                        .is_some_and(Option::is_some))
            {
                return Err(invalid());
            }
            self.function
                .locals
                .get(id.index())
                .map(|local| local.ty.clone())
                .ok_or_else(invalid)
        };
        let (input, output) = match operation {
            ParallelOperation::Map { input, output, .. }
            | ParallelOperation::Reduce { input, output, .. } => (*input, *output),
        };
        let Ty::Array { length, element } = local(input, false)? else {
            return Err(invalid());
        };
        match operation {
            ParallelOperation::Map {
                element: parameter, ..
            } => {
                if local(*parameter, true)? != *element {
                    return Err(invalid());
                }
                let Ty::Array {
                    length: result_length,
                    element: result,
                } = local(output, false)?
                else {
                    return Err(invalid());
                };
                if length != result_length {
                    return Err(invalid());
                }
                Ok(*result)
            }
            ParallelOperation::Reduce {
                identity,
                left,
                right,
                ..
            } => {
                if [
                    local(*identity, false)?,
                    local(*left, true)?,
                    local(*right, true)?,
                    local(output, false)?,
                ]
                .iter()
                .any(|ty| ty != element.as_ref())
                {
                    return Err(invalid());
                }
                Ok(*element)
            }
        }
    }
}

fn private_owners(
    function: &Function,
    location: Location,
) -> Result<Vec<Option<BlockId>>, VerifyError> {
    let mut owners = vec![None; function.locals.len()];
    for block in &function.blocks {
        if let Terminator::Parallel {
            ref private_locals,
            body,
            ..
        } = block.terminator
        {
            let locals = owners
                .get_mut(private_locals.clone())
                .ok_or_else(|| location.error(VerifyErrorKind::InvalidParallelRegion))?;
            for owner in locals {
                if owner.replace(body).is_some() {
                    return Err(location.error(VerifyErrorKind::InvalidParallelRegion));
                }
            }
        }
    }
    Ok(owners)
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
