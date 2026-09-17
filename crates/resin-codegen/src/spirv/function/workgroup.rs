//! Scalar regions execute on lane zero; array regions distribute independent jobs.
//! Every boundary publishes results and converges failures before any lane returns.
use super::*;
use crate::spirv::{build_error, symbols::LocalIndex};
use resin_lir::ParallelOperation;
use rspirv::spirv::Scope;

pub(super) struct Group {
    pub lane: Word,
    pub failed: Word,
    scratch: HashMap<LocalId, Word>,
}

impl Group {
    pub fn prepare(context: &mut Context<'_>, function: &Function) -> Result<Option<Self>, Error> {
        if !has_parallel(function) {
            return Ok(None);
        }
        let failed = context.shared_variable(&Ty::UInt32)?;
        let mut scratch = HashMap::new();
        for block in &function.blocks {
            if let Terminator::Parallel {
                operation: ParallelOperation::Reduce { input, output, .. },
                ..
            } = block.terminator
            {
                let Ty::Array { length, element } = &function.locals[input.index()].ty else {
                    unreachable!()
                };
                let ty = Ty::Array {
                    length: length
                        .checked_add(1)
                        .ok_or_else(|| Error::unsupported("reduction array is too large".into()))?,
                    element: element.clone(),
                };
                scratch.insert(output, context.shared_variable(&ty)?);
            }
        }
        let uint = context.ty(&Ty::UInt32)?;
        let vector = context.builder.type_vector(uint, 3);
        let input = context.compute.as_ref().unwrap().lane;
        let value = context
            .builder
            .load(vector, None, input, None, [])
            .map_err(build_error)?;
        let lane = context
            .builder
            .composite_extract(uint, None, value, [0])
            .map_err(build_error)?;
        Ok(Some(Self {
            lane,
            failed,
            scratch,
        }))
    }
}

pub(super) struct Leader {
    pub merge: Word,
    header: Word,
    continue_label: Word,
}

pub(super) fn leader(context: &mut Context<'_>, lane: Word) -> Result<Leader, Error> {
    let bool_type = context.ty(&Ty::Bool)?;
    let zero = context.constant_u32(0);
    let selected = context
        .builder
        .i_equal(bool_type, None, lane, zero)
        .map_err(build_error)?;
    let before = current_label(context);
    let header = context.builder.id();
    let body = context.builder.id();
    let merge = context.builder.id();
    let continue_label = context.builder.id();
    let done = context.constant_bool(false);
    context.builder.branch(header).map_err(build_error)?;
    context
        .builder
        .begin_block(Some(header))
        .map_err(build_error)?;
    let active = context
        .builder
        .phi(
            bool_type,
            None,
            [(selected, before), (done, continue_label)],
        )
        .map_err(build_error)?;
    // A one-iteration loop permits checked failures in nested selections to
    // exit this whole scalar phase through a structured loop merge.
    context
        .builder
        .loop_merge(merge, continue_label, LoopControl::NONE, [])
        .map_err(build_error)?;
    context
        .builder
        .branch_conditional(active, body, merge, [])
        .map_err(build_error)?;
    context
        .builder
        .begin_block(Some(body))
        .map_err(build_error)?;
    Ok(Leader {
        merge,
        header,
        continue_label,
    })
}

pub(super) fn end_leader(context: &mut Context<'_>, leader: Leader) -> Result<(), Error> {
    context
        .builder
        .branch(leader.continue_label)
        .map_err(build_error)?;
    finish_leader(context, leader)
}

fn finish_leader(context: &mut Context<'_>, leader: Leader) -> Result<(), Error> {
    context
        .builder
        .begin_block(Some(leader.continue_label))
        .map_err(build_error)?;
    context.builder.branch(leader.header).map_err(build_error)?;
    context
        .builder
        .begin_block(Some(leader.merge))
        .map_err(build_error)?;
    Ok(())
}

fn current_label(context: &Context<'_>) -> Word {
    let function = context.builder.selected_function().unwrap();
    let block = context.builder.selected_block().unwrap();
    context.builder.module_ref().functions[function].blocks[block]
        .label
        .as_ref()
        .unwrap()
        .result_id
        .unwrap()
}

impl FunctionLowering<'_, '_> {
    pub(super) fn scalar_instructions(
        &mut self,
        block: usize,
        stack: &mut Vec<Slot>,
    ) -> Result<bool, Error> {
        if self.function.blocks[block].instrs.is_empty() {
            return Ok(false);
        }
        let merge = leader(self.context, self.group.as_ref().unwrap().lane)?;
        self.failure_target = Some(merge.merge);
        let ended = self.instructions(block, stack)?;
        self.failure_target = None;
        let mut broadcasts = vec![];
        if !ended {
            for slot in stack.iter() {
                broadcasts.push(self.publish(slot)?);
            }
        }
        let aliases: std::collections::BTreeMap<_, _> = self
            .aliases
            .iter()
            .map(|(&root, slot)| (root, slot.clone()))
            .collect();
        let mut alias_broadcasts = vec![];
        if !ended {
            for (&root, slot) in &aliases {
                alias_broadcasts.push((root, self.publish(slot)?));
            }
            self.context
                .builder
                .branch(merge.continue_label)
                .map_err(build_error)?;
        }
        finish_leader(self.context, merge)?;
        self.converge()?;
        if ended {
            self.context.builder.unreachable().map_err(build_error)?;
            return Ok(true);
        }
        for (slot, variables) in stack.iter_mut().zip(broadcasts) {
            self.receive(slot, &variables)?;
        }
        for (root, variables) in alias_broadcasts {
            let mut slot = aliases[&root].clone();
            self.receive(&mut slot, &variables)?;
            self.aliases.insert(root, slot);
        }
        // A subsequent loop iteration may reuse these broadcast variables.
        self.context.barrier()?;
        Ok(false)
    }

    // Logical references keep a static root; only their dynamic indices need
    // publishing. Never reinterpret a Workgroup pointer as a device address.
    fn publish(&mut self, slot: &Slot) -> Result<Vec<Word>, Error> {
        let values = if let Some(local) = &slot.local {
            local
                .indices
                .iter()
                .filter_map(|index| match index {
                    LocalIndex::Dynamic { id, ty } => Some((ty, *id)),
                    _ => None,
                })
                .collect::<Vec<_>>()
        } else if slot.symbolic() {
            vec![]
        } else {
            vec![(&slot.ty, slot.id)]
        };
        values
            .into_iter()
            .map(|(ty, id)| {
                let variable = self.context.shared_variable(ty)?;
                self.context
                    .builder
                    .store(variable, id, None, [])
                    .map_err(build_error)?;
                Ok(variable)
            })
            .collect()
    }

    fn receive(&mut self, slot: &mut Slot, variables: &[Word]) -> Result<(), Error> {
        let mut variables = variables.iter();
        if let Some(local) = &mut slot.local {
            for index in &mut local.indices {
                if let LocalIndex::Dynamic { id, ty } = index {
                    *id = ops::load(self.context, ty, *variables.next().unwrap())?;
                }
            }
        } else if !slot.symbolic() {
            slot.id = ops::load(self.context, &slot.ty, *variables.next().unwrap())?;
        }
        Ok(())
    }

    fn converge(&mut self) -> Result<(), Error> {
        let failed = ops::load(self.context, &Ty::Bool, self.context.failed)?;
        let uint = self.context.ty(&Ty::UInt32)?;
        let zero = self.context.constant_u32(0);
        let one = self.context.constant_u32(1);
        let value = self
            .context
            .builder
            .select(uint, None, failed, one, zero)
            .map_err(build_error)?;
        let scope = self.context.constant_u32(Scope::Workgroup as u32);
        let relaxed = self.context.constant_u32(0);
        let flag = self.group.as_ref().unwrap().failed;
        self.context
            .builder
            .atomic_or(uint, None, flag, scope, relaxed, value)
            .map_err(build_error)?;
        self.context.barrier()?;
        let value = ops::load(self.context, &Ty::UInt32, flag)?;
        // Every lane must read this round's flag before a faster lane can
        // publish a failure from the next round.
        self.context.barrier()?;
        let bool_type = self.context.ty(&Ty::Bool)?;
        let failed = self
            .context
            .builder
            .i_not_equal(bool_type, None, value, zero)
            .map_err(build_error)?;
        self.check(failed, true)
    }

    pub(super) fn parallel(
        &mut self,
        operation: &ParallelOperation,
        body: usize,
    ) -> Result<(), Error> {
        let (input, output) = match operation {
            ParallelOperation::Map { input, output, .. }
            | ParallelOperation::Reduce { input, output, .. } => (*input, *output),
        };
        let Ty::Array { length, element } = self.function.locals[input.index()].ty.clone() else {
            unreachable!()
        };
        let count = u32::try_from(length)
            .map_err(|_| Error::unsupported("parallel array is too large".into()))?;
        self.cooperative = false;
        match operation {
            ParallelOperation::Map { .. } => {
                let stride = self.context.constant_u32(1);
                self.jobs(body, count, stride, operation)?;
            }
            ParallelOperation::Reduce { identity, .. } => {
                let count = count
                    .checked_add(1)
                    .ok_or_else(|| Error::unsupported("parallel array is too large".into()))?;
                let scratch = self.group.as_ref().unwrap().scratch[&output];
                let merge = leader(self.context, self.group.as_ref().unwrap().lane)?;
                let zero = self.context.constant_u32(0);
                let identity = ops::load(self.context, &element, self.locals[identity.index()])?;
                self.store_element(scratch, &element, zero, identity)?;
                // Copy source-known elements once before the logarithmic tree.
                for index in 0..length {
                    let source = self.context.constant_u32(index as u32);
                    let value = self.load_element(self.locals[input.index()], &element, source)?;
                    let target = self.context.constant_u32(index as u32 + 1);
                    self.store_element(scratch, &element, target, value)?;
                }
                end_leader(self.context, merge)?;
                self.context.barrier()?;
                self.reduce_rounds(body, count, operation)?;
                let merge = leader(self.context, self.group.as_ref().unwrap().lane)?;
                let value = self.load_element(scratch, &element, zero)?;
                self.context
                    .builder
                    .store(self.locals[output.index()], value, None, [])
                    .map_err(build_error)?;
                end_leader(self.context, merge)?;
                self.context.barrier()?;
            }
        }
        self.cooperative = true;
        Ok(())
    }

    fn reduce_rounds(
        &mut self,
        body: usize,
        count: u32,
        operation: &ParallelOperation,
    ) -> Result<(), Error> {
        // The outer loop is uniform. Each round pairs disjoint strided slots,
        // retaining unpaired values; no lane reads a slot another pair writes.
        let uint = self.context.ty(&Ty::UInt32)?;
        let one = self.context.constant_u32(1);
        let limit = self.context.constant_u32(count);
        let before = self.label();
        let header = self.context.builder.id();
        let active = self.context.builder.id();
        let continue_label = self.context.builder.id();
        let merge = self.context.builder.id();
        let doubled = self.context.builder.id();
        self.context.builder.branch(header).map_err(build_error)?;
        self.context
            .builder
            .begin_block(Some(header))
            .map_err(build_error)?;
        let stride = self
            .context
            .builder
            .phi(uint, None, [(one, before), (doubled, continue_label)])
            .map_err(build_error)?;
        let bool_type = self.context.ty(&Ty::Bool)?;
        let test = self
            .context
            .builder
            .u_less_than(bool_type, None, stride, limit)
            .map_err(build_error)?;
        self.context
            .builder
            .loop_merge(merge, continue_label, LoopControl::NONE, [])
            .map_err(build_error)?;
        self.context
            .builder
            .branch_conditional(test, active, merge, [])
            .map_err(build_error)?;
        self.context
            .builder
            .begin_block(Some(active))
            .map_err(build_error)?;
        self.jobs(body, count, stride, operation)?;
        self.context
            .builder
            .branch(continue_label)
            .map_err(build_error)?;
        self.context
            .builder
            .begin_block(Some(continue_label))
            .map_err(build_error)?;
        self.context
            .builder
            .i_add(uint, Some(doubled), stride, stride)
            .map_err(build_error)?;
        self.context.builder.branch(header).map_err(build_error)?;
        self.context
            .builder
            .begin_block(Some(merge))
            .map_err(build_error)?;
        Ok(())
    }

    fn jobs(
        &mut self,
        body: usize,
        count: u32,
        stride: Word,
        operation: &ParallelOperation,
    ) -> Result<(), Error> {
        let (input, output) = match operation {
            ParallelOperation::Map { input, output, .. }
            | ParallelOperation::Reduce { input, output, .. } => (*input, *output),
        };
        let Ty::Array { element, .. } = &self.function.locals[input.index()].ty else {
            unreachable!()
        };
        let (left, right) = match operation {
            ParallelOperation::Map { element, .. } => (*element, None),
            ParallelOperation::Reduce { left, right, .. } => (*left, Some(*right)),
        };
        let uint = self.context.ty(&Ty::UInt32)?;
        let lane = self.group.as_ref().unwrap().lane;
        let width = self.context.compute.as_ref().unwrap().width;
        let spacing = if right.is_some() {
            self.context
                .builder
                .i_add(uint, None, stride, stride)
                .map_err(build_error)?
        } else {
            stride
        };
        let start = self
            .context
            .builder
            .i_mul(uint, None, lane, spacing)
            .map_err(build_error)?;
        let step = self
            .context
            .builder
            .i_mul(uint, None, width, spacing)
            .map_err(build_error)?;
        let limit = self.context.constant_u32(count);
        let before = self.label();
        let header = self.context.builder.id();
        let active = self.context.builder.id();
        let continue_label = self.context.builder.id();
        let merge = self.context.builder.id();
        let advanced = self.context.builder.id();
        self.context.builder.branch(header).map_err(build_error)?;
        self.context
            .builder
            .begin_block(Some(header))
            .map_err(build_error)?;
        let index = self
            .context
            .builder
            .phi(uint, None, [(start, before), (advanced, continue_label)])
            .map_err(build_error)?;
        let last = if right.is_some() {
            self.context
                .builder
                .i_add(uint, None, index, stride)
                .map_err(build_error)?
        } else {
            index
        };
        let bool_type = self.context.ty(&Ty::Bool)?;
        let bounded = self
            .context
            .builder
            .u_less_than(bool_type, None, last, limit)
            .map_err(build_error)?;
        let failed = ops::load(self.context, &Ty::Bool, self.context.failed)?;
        let ready = self
            .context
            .builder
            .logical_not(bool_type, None, failed)
            .map_err(build_error)?;
        let test = self
            .context
            .builder
            .logical_and(bool_type, None, bounded, ready)
            .map_err(build_error)?;
        self.context
            .builder
            .loop_merge(merge, continue_label, LoopControl::NONE, [])
            .map_err(build_error)?;
        self.context
            .builder
            .branch_conditional(test, active, merge, [])
            .map_err(build_error)?;
        self.context
            .builder
            .begin_block(Some(active))
            .map_err(build_error)?;
        let source = if right.is_some() {
            self.group.as_ref().unwrap().scratch[&output]
        } else {
            self.locals[input.index()]
        };
        let value = self.load_element(source, element, index)?;
        self.context
            .builder
            .store(self.locals[left.index()], value, None, [])
            .map_err(build_error)?;
        let (target, result_type) = if let Some(right) = right {
            let value = self.load_element(source, element, last)?;
            self.context
                .builder
                .store(self.locals[right.index()], value, None, [])
                .map_err(build_error)?;
            (source, *element.clone())
        } else {
            let Ty::Array { element, .. } = &self.function.locals[output.index()].ty else {
                unreachable!()
            };
            (self.locals[output.index()], *element.clone())
        };
        let pointer = self.element_pointer(target, &result_type, index)?;
        self.iteration_output = Some(pointer);
        self.failure_target = Some(continue_label);
        self.region(body, vec![], None)?;
        self.failure_target = None;
        self.iteration_output = None;
        self.context
            .builder
            .begin_block(Some(continue_label))
            .map_err(build_error)?;
        self.context
            .builder
            .i_add(uint, Some(advanced), index, step)
            .map_err(build_error)?;
        self.context.builder.branch(header).map_err(build_error)?;
        self.context
            .builder
            .begin_block(Some(merge))
            .map_err(build_error)?;
        self.converge()
    }

    fn label(&self) -> Word {
        current_label(self.context)
    }

    fn element_pointer(&mut self, array: Word, element: &Ty, index: Word) -> Result<Word, Error> {
        let pointer = self
            .context
            .pointer_type(StorageClass::Workgroup, element)?;
        self.context
            .builder
            .access_chain(pointer, None, array, [index])
            .map_err(build_error)
    }

    fn load_element(&mut self, array: Word, element: &Ty, index: Word) -> Result<Word, Error> {
        let pointer = self.element_pointer(array, element, index)?;
        ops::load(self.context, element, pointer)
    }

    fn store_element(
        &mut self,
        array: Word,
        element: &Ty,
        index: Word,
        value: Word,
    ) -> Result<(), Error> {
        let pointer = self.element_pointer(array, element, index)?;
        self.context
            .builder
            .store(pointer, value, None, [])
            .map_err(build_error)
    }
}
