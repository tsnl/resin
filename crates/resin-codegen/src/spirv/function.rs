//! Preserve region ownership as selection merges, loop merges, and backedges.
use super::{
    Context, ops,
    symbols::{self, Slot},
};
use crate::Error;
use resin_lir::{BlockId, Function, FunctionTypes, Instr, Terminator};
use resin_types::prelude::*;
use rspirv::spirv::{FunctionControl, LoopControl, SelectionControl, StorageClass, Word};
use std::collections::HashMap;

pub(super) fn lower(
    context: &mut Context<'_>,
    function: &Function,
    flow: &FunctionTypes,
    index: usize,
) -> Result<(), Error> {
    let result = context.ty(&function.result)?;
    let parameter = context.ty(&function.locals[0].ty)?;
    let signature = context.builder.type_function(result, [parameter]);
    if let Some(name) = &function.name {
        context
            .builder
            .name(context.functions[index], name.as_ref());
    }
    context
        .builder
        .begin_function(
            result,
            Some(context.functions[index]),
            FunctionControl::NONE,
            signature,
        )
        .unwrap();
    let argument = context.builder.function_parameter(parameter).unwrap();
    context.builder.begin_block(None).unwrap();
    let locals = local_variables(context, function)?;
    let stacks = region_outputs(function, flow);
    let destinations = destinations(context, flow, &stacks.outputs, &stacks.tests)?;
    let arrays = array_variables(context, function, flow)?;
    context
        .builder
        .store(locals[0], argument, None, [])
        .unwrap();
    let mut lowering = FunctionLowering {
        context,
        function,
        flow,
        index,
        locals,
        arrays,
        destinations,
        has_loop_test: stacks.tests.iter().map(Option::is_some).collect(),
    };
    lowering.region(function.entry.index(), vec![], None)?;
    lowering.context.builder.end_function().unwrap();
    Ok(())
}

struct Destination {
    label: Word,
    variables: Vec<Option<Word>>,
    incoming: Option<Vec<Slot>>,
}

#[derive(Clone, Copy)]
enum ExitTarget {
    Merge { destination: usize },
    LoopTest { destination: usize },
    Continue { destination: usize },
}

struct FunctionLowering<'a, 'm> {
    context: &'a mut Context<'m>,
    function: &'a Function,
    flow: &'a FunctionTypes,
    index: usize,
    locals: Vec<Word>,
    arrays: HashMap<Ty, Word>,
    destinations: Vec<Destination>,
    has_loop_test: Vec<bool>,
}

fn variable(context: &mut Context<'_>, ty: &Ty, initial: Option<Word>) -> Result<Word, Error> {
    let pointer = context.pointer_type(StorageClass::Function, ty)?;
    Ok(context
        .builder
        .variable(pointer, None, StorageClass::Function, initial))
}

fn local_variables(context: &mut Context<'_>, function: &Function) -> Result<Vec<Word>, Error> {
    function
        .locals
        .iter()
        .map(|local| {
            let zero = context.zero(&local.ty)?;
            let id = variable(context, &local.ty, Some(zero))?;
            if let Some(name) = &local.name {
                context.builder.name(id, name.as_ref());
            }
            Ok(id)
        })
        .collect()
}

fn destinations(
    context: &mut Context<'_>,
    flow: &FunctionTypes,
    outputs: &[Option<Vec<Ty>>],
    tests: &[Option<Vec<Ty>>],
) -> Result<Vec<Destination>, Error> {
    flow.inputs
        .iter()
        .map(Vec::as_slice)
        .chain(
            outputs
                .iter()
                .chain(tests)
                .map(|output| output.as_deref().unwrap_or(&[])),
        )
        .map(|types| {
            let variables = types
                .iter()
                .map(|ty| {
                    if matches!(ty, Ty::Function { .. }) {
                        Ok(None)
                    } else {
                        variable(context, ty, None).map(Some)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Destination {
                label: 0,
                variables,
                incoming: None,
            })
        })
        .collect()
}

fn array_variables(
    context: &mut Context<'_>,
    function: &Function,
    flow: &FunctionTypes,
) -> Result<HashMap<Ty, Word>, Error> {
    let mut arrays = HashMap::new();
    for ty in function
        .locals
        .iter()
        .map(|local| &local.ty)
        .chain(flow.inputs.iter().flatten())
        .chain(flow.results.iter().flatten().flatten())
    {
        if matches!(context.shape(ty), Ty::Array { .. }) && !arrays.contains_key(ty) {
            arrays.insert(ty.clone(), variable(context, ty, None)?);
        }
    }
    Ok(arrays)
}

impl FunctionLowering<'_, '_> {
    // The continuation is a sequence. Recurse only for owned child regions.
    fn region(
        &mut self,
        mut block: usize,
        mut stack: Vec<Slot>,
        exit: Option<ExitTarget>,
    ) -> Result<(), Error> {
        loop {
            if self.instructions(block, &mut stack)? {
                return Ok(());
            }
            match self.function.blocks[block].terminator {
                Terminator::If { then, els, next } => {
                    let Some(values) = self.selection(block, then.index(), els.index(), stack)?
                    else {
                        return Ok(());
                    };
                    stack = values;
                    if let Some(next) = next {
                        block = next.index();
                    } else {
                        return self.exit(stack, exit.unwrap());
                    }
                }
                Terminator::Loop {
                    condition,
                    body,
                    next,
                } => {
                    let Some(values) =
                        self.loop_region(block, condition.index(), body.index(), stack)?
                    else {
                        return Ok(());
                    };
                    stack = values;
                    if let Some(next) = next {
                        block = next.index();
                    } else {
                        return self.exit(stack, exit.unwrap());
                    }
                }
                Terminator::Return => {
                    if stack[0].local.is_some() {
                        return Err(Error::at(
                            self.context.module,
                            self.index,
                            Some((block, self.function.blocks[block].instrs.len())),
                            Error("shader cannot return a local address".into()),
                        ));
                    }
                    self.context.builder.ret_value(stack[0].id).unwrap();
                    return Ok(());
                }
                Terminator::Merge | Terminator::LoopTest | Terminator::Continue => {
                    return self.exit(stack, exit.unwrap());
                }
            }
        }
    }

    fn instructions(&mut self, block: usize, stack: &mut Vec<Slot>) -> Result<bool, Error> {
        for (index, instruction) in self.function.blocks[block].instrs.iter().enumerate() {
            if matches!(instruction, Instr::Eliminate { .. }) {
                self.abort()?;
                return Ok(true);
            }
            let count = self.flow.operand_count(BlockId::from_index(block), index);
            let args = stack.split_off(stack.len() - count);
            let result = self.flow.results[block][index].as_ref();
            let emitted = self
                .instruction(instruction, &args, result)
                .map_err(|error| {
                    Error::at(self.context.module, self.index, Some((block, index)), error)
                })?;
            if let Some(value) = emitted {
                stack.push(value);
            }
        }
        Ok(false)
    }

    fn instruction(
        &mut self,
        instruction: &Instr,
        args: &[Slot],
        result: Option<&Ty>,
    ) -> Result<Option<Slot>, Error> {
        symbols::check(self.context.module, instruction, args, result)?;
        if let Some(invalid) = ops::invalid(self.context, instruction, args)? {
            self.check(invalid, true)?;
        }
        let value = ops::instruction(
            self.context,
            instruction,
            args,
            result,
            &self.locals,
            &self.arrays,
        )?;
        if matches!(instruction, Instr::Call) {
            let failed = ops::load(self.context, &Ty::Bool, self.context.failed)?;
            self.check(failed, false)?;
        }
        Ok(value)
    }

    fn check(&mut self, condition: Word, mark_failed: bool) -> Result<(), Error> {
        let failed = self.context.builder.id();
        let next = self.context.builder.id();
        self.context.builder.name(failed, "checked.failure");
        self.context.builder.name(next, "checked.continue");
        self.context
            .builder
            .selection_merge(next, SelectionControl::NONE)
            .unwrap();
        self.context
            .builder
            .branch_conditional(condition, failed, next, [])
            .unwrap();
        self.context.builder.begin_block(Some(failed)).unwrap();
        if mark_failed {
            self.abort()?;
        } else {
            self.return_zero()?;
        }
        self.context.builder.begin_block(Some(next)).unwrap();
        Ok(())
    }

    fn abort(&mut self) -> Result<(), Error> {
        let failed = self.context.constant_bool(true);
        self.context
            .builder
            .store(self.context.failed, failed, None, [])
            .unwrap();
        self.return_zero()
    }

    fn return_zero(&mut self) -> Result<(), Error> {
        let zero = self.context.zero(&self.function.result)?;
        self.context.builder.ret_value(zero).unwrap();
        Ok(())
    }

    fn selection(
        &mut self,
        block: usize,
        then: usize,
        els: usize,
        mut stack: Vec<Slot>,
    ) -> Result<Option<Vec<Slot>>, Error> {
        let condition = stack.pop().unwrap().id;
        let yes = self.context.builder.id();
        let no = self.context.builder.id();
        let merge = self.context.builder.id();
        self.context.builder.name(yes, "if.then");
        self.context.builder.name(no, "if.else");
        self.context.builder.name(merge, "if.merge");
        let destination = self.function.blocks.len() + block;
        self.destinations[destination].label = merge;
        self.context
            .builder
            .selection_merge(merge, SelectionControl::NONE)
            .unwrap();
        self.context
            .builder
            .branch_conditional(condition, yes, no, [])
            .unwrap();
        let exit = Some(ExitTarget::Merge { destination });
        self.context.builder.begin_block(Some(yes)).unwrap();
        self.region(then, stack.clone(), exit)?;
        self.context.builder.begin_block(Some(no)).unwrap();
        self.region(els, stack, exit)?;
        self.context.builder.begin_block(Some(merge)).unwrap();
        if self.destinations[destination].incoming.is_none() {
            self.context.builder.unreachable().unwrap();
            Ok(None)
        } else {
            self.read(destination).map(Some)
        }
    }

    fn loop_region(
        &mut self,
        block: usize,
        condition: usize,
        body: usize,
        stack: Vec<Slot>,
    ) -> Result<Option<Vec<Slot>>, Error> {
        // Verification follows Eliminate's typed continuation. Emission stops
        // there: an always-diverging condition executes once and never loops.
        if !self.has_loop_test[condition] {
            self.region(condition, stack, None)?;
            return Ok(None);
        }
        let header = self.context.builder.id();
        let condition_label = self.context.builder.id();
        let body_label = self.context.builder.id();
        let continue_label = self.context.builder.id();
        let test_label = self.context.builder.id();
        let merge = self.context.builder.id();
        self.context.builder.name(header, "loop.header");
        self.context.builder.name(condition_label, "loop.condition");
        self.context.builder.name(body_label, "loop.body");
        self.context.builder.name(continue_label, "loop.continue");
        self.context.builder.name(test_label, "loop.test");
        self.context.builder.name(merge, "loop.merge");
        let test_destination = 2 * self.function.blocks.len() + condition;
        let merge_destination = self.function.blocks.len() + block;
        self.destinations[condition].label = continue_label;
        self.destinations[test_destination].label = test_label;
        self.destinations[merge_destination].label = merge;
        self.transfer(condition, &stack)?;
        self.context.builder.branch(header).unwrap();
        self.context.builder.begin_block(Some(header)).unwrap();
        let carried = self.read(condition)?;
        self.context
            .builder
            .loop_merge(merge, continue_label, LoopControl::NONE, [])
            .unwrap();
        self.context.builder.branch(condition_label).unwrap();
        self.context
            .builder
            .begin_block(Some(condition_label))
            .unwrap();
        self.region(
            condition,
            carried,
            Some(ExitTarget::LoopTest {
                destination: test_destination,
            }),
        )?;
        self.context.builder.begin_block(Some(test_label)).unwrap();
        let mut tested = self.read(test_destination)?;
        let test = tested.pop().unwrap().id;
        self.transfer(merge_destination, &tested)?;
        self.context
            .builder
            .branch_conditional(test, body_label, merge, [])
            .unwrap();
        self.context.builder.begin_block(Some(body_label)).unwrap();
        self.region(
            body,
            tested,
            Some(ExitTarget::Continue {
                destination: condition,
            }),
        )?;
        self.context
            .builder
            .begin_block(Some(continue_label))
            .unwrap();
        self.context.builder.branch(header).unwrap();
        self.context.builder.begin_block(Some(merge)).unwrap();
        self.read(merge_destination).map(Some)
    }

    fn exit(&mut self, stack: Vec<Slot>, exit: ExitTarget) -> Result<(), Error> {
        let destination = match exit {
            ExitTarget::Merge { destination }
            | ExitTarget::LoopTest { destination }
            | ExitTarget::Continue { destination } => destination,
        };
        self.transfer(destination, &stack)?;
        self.context
            .builder
            .branch(self.destinations[destination].label)
            .unwrap();
        Ok(())
    }

    fn transfer(&mut self, destination: usize, stack: &[Slot]) -> Result<(), Error> {
        let target = &mut self.destinations[destination];
        if let Some(previous) = &target.incoming {
            symbols::agree(previous, stack)?;
        } else {
            target.incoming = Some(stack.to_vec());
        }
        for (slot, variable) in stack.iter().zip(&target.variables) {
            if !slot.symbolic() {
                self.context
                    .builder
                    .store(variable.unwrap(), slot.id, None, [])
                    .unwrap();
            }
        }
        Ok(())
    }

    fn read(&mut self, destination: usize) -> Result<Vec<Slot>, Error> {
        let target = &self.destinations[destination];
        let mut stack = target
            .incoming
            .clone()
            .expect("verified region has an incoming path");
        for (slot, variable) in stack.iter_mut().zip(&target.variables) {
            if !slot.symbolic() {
                slot.id = ops::load(self.context, &slot.ty, variable.unwrap())?;
            }
        }
        Ok(stack)
    }
}

// Determine join storage before emitting the function entry's OpVariable list.
struct RegionStacks {
    outputs: Vec<Option<Vec<Ty>>>,
    tests: Vec<Option<Vec<Ty>>>,
}

fn region_outputs(function: &Function, flow: &FunctionTypes) -> RegionStacks {
    let mut outputs = vec![None; function.blocks.len()];
    let mut tests = vec![None; function.blocks.len()];
    output(
        function,
        flow,
        function.entry.index(),
        &mut outputs,
        &mut tests,
    );
    RegionStacks { outputs, tests }
}

fn output(
    function: &Function,
    flow: &FunctionTypes,
    mut block: usize,
    outputs: &mut [Option<Vec<Ty>>],
    tests: &mut [Option<Vec<Ty>>],
) -> Option<Vec<Ty>> {
    loop {
        let mut stack = flow.inputs[block].clone();
        for (index, instruction) in function.blocks[block].instrs.iter().enumerate() {
            if matches!(instruction, Instr::Eliminate { .. }) {
                return None;
            }
            stack.truncate(stack.len() - flow.operand_count(BlockId::from_index(block), index));
            stack.extend(flow.results[block][index].clone());
        }
        let (next, result) = match function.blocks[block].terminator {
            Terminator::Return => (None, None),
            Terminator::Merge | Terminator::LoopTest | Terminator::Continue => (None, Some(stack)),
            Terminator::If { then, els, next } => {
                let yes = output(function, flow, then.index(), outputs, tests);
                let no = output(function, flow, els.index(), outputs, tests);
                (next, yes.or(no))
            }
            Terminator::Loop {
                condition,
                body,
                next,
            } => {
                let result = output(function, flow, condition.index(), outputs, tests);
                tests[condition.index()] = result.clone();
                let result = result.map(|mut stack| {
                    output(function, flow, body.index(), outputs, tests);
                    stack.pop();
                    stack
                });
                (next, result)
            }
        };
        outputs[block] = result.clone();
        result.as_ref()?;
        let Some(next) = next else {
            return result;
        };
        block = next.index();
    }
}
