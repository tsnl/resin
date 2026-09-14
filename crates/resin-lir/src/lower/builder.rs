use resin_types::prelude::*;
use std::sync::Arc;

use crate::{BasicBlock, BlockId, Function, Instr, Local, Terminator};

pub(super) struct FunctionBuilder {
    function: Function,
    current: BlockId,
    terminated: Vec<bool>,
    inputs: Vec<BlockInput>,
    // Each operand records how many locals existed when it was produced.
    // Local IDs give registration order; the stack gives operand order.
    stack_boundaries: Vec<usize>,
}

struct BlockInput {
    inherited: Vec<usize>,
    produced: usize,
}

impl FunctionBuilder {
    pub(super) fn new(name: Option<Arc<str>>, profile: crate::Profile) -> Self {
        Self {
            function: Function {
                name,
                profile,
                foreign: None,
                result: Ty::Unit,
                parameter_count: 0,
                locals: vec![],
                entry: BlockId::from_index(0),
                blocks: vec![BasicBlock {
                    name: Some("entry".into()),
                    instrs: Vec::new(),
                    terminator: Terminator::Return,
                }],
            },
            current: BlockId::from_index(0),
            terminated: vec![false],
            inputs: vec![BlockInput {
                inherited: vec![],
                produced: 0,
            }],
            stack_boundaries: vec![],
        }
    }

    pub(super) fn finish(self) -> Function {
        self.function
    }

    pub(super) fn parameter(&mut self, name: Option<Arc<str>>, ty: Ty) -> LocalId {
        assert_eq!(self.function.parameter_count, self.function.locals.len());
        self.function.parameter_count += 1;
        self.local(ty, name)
    }

    pub(super) fn result(&mut self, ty: Ty) {
        self.function.result = ty;
    }

    pub(super) fn local_type(&self, id: LocalId) -> &Ty {
        &self.function.locals[id.index()].ty
    }

    pub(super) fn result_type(&self) -> &Ty {
        &self.function.result
    }
    pub(super) fn stack_len(&self) -> usize {
        self.stack_boundaries.len()
    }

    pub(super) fn stack_is_newer_than(&self, local: LocalId) -> bool {
        self.stack_boundaries
            .last()
            .is_some_and(|boundary| *boundary > local.index())
    }

    pub(super) fn local(&mut self, ty: Ty, name: Option<Arc<str>>) -> LocalId {
        let id = LocalId::from_index(self.function.locals.len());
        self.function.locals.push(Local { name, ty });
        id
    }

    pub(super) fn position(&self) -> (BlockId, usize) {
        (
            self.current,
            self.function.blocks[self.current.index()].instrs.len(),
        )
    }

    pub(super) fn emit(&mut self, instr: Instr) {
        let current = self.current.index();
        debug_assert!(!self.terminated[current]);
        let effect = crate::verify::stack_effect(&instr);
        let remaining = self
            .stack_boundaries
            .len()
            .checked_sub(effect.pops)
            .expect("valid generated stack");
        self.stack_boundaries.truncate(remaining);
        self.push_results(effect.pushes);
        self.function.blocks[current].instrs.push(instr);
    }

    pub(super) fn terminate(&mut self, terminator: Terminator) {
        let current = self.current.index();
        debug_assert!(!self.terminated[current]);
        self.function.blocks[current].terminator = terminator;
        self.terminated[current] = true;
    }

    pub(super) fn switch(&mut self, block: BlockId) {
        self.current = block;
        self.stack_boundaries = self.inputs[block.index()].inherited.clone();
        self.push_results(self.inputs[block.index()].produced);
    }

    pub(super) fn new_block(&mut self, hint: &str, inherited: usize, produced: usize) -> BlockId {
        let id = BlockId::from_index(self.function.blocks.len());
        self.function.blocks.push(BasicBlock {
            name: Some(self.unique_block_name(hint)),
            instrs: Vec::new(),
            terminator: Terminator::Return,
        });
        self.terminated.push(false);
        self.inputs.push(BlockInput {
            inherited: self.stack_boundaries[..inherited].to_vec(),
            produced,
        });
        id
    }

    fn push_results(&mut self, count: usize) {
        // Replacing an operand or merging a region produces a new value. Only
        // inherited stack prefixes keep their order relative to existing locals.
        self.stack_boundaries
            .extend(std::iter::repeat_n(self.function.locals.len(), count));
    }

    fn unique_block_name(&self, hint: &str) -> Arc<str> {
        if !self
            .function
            .blocks
            .iter()
            .any(|block| block.name.as_deref() == Some(hint))
        {
            return hint.into();
        }
        let mut suffix = 1;
        loop {
            let candidate = format!("{hint}.{suffix}");
            if !self
                .function
                .blocks
                .iter()
                .any(|block| block.name.as_deref() == Some(candidate.as_str()))
            {
                return candidate.into();
            }
            suffix += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_a_function_without_syntax_or_scopes() {
        let mut builder = FunctionBuilder::new(Some("identity".into()), crate::Profile::Host);
        builder.parameter(Some("value".into()), Ty::Int32);
        builder.result(Ty::Int32);
        assert_eq!(builder.local(Ty::Bool, Some("temporary".into())).index(), 1);
        let body = BlockId::from_index(0);
        builder.emit(Instr::LocalAddress {
            local: LocalId::from_index(0),
        });
        builder.emit(Instr::Load);
        builder.terminate(Terminator::Return);
        let function = builder.finish();
        assert_eq!(function.blocks.len(), 1);
        assert_eq!(function.locals[0].name.as_deref(), Some("value"));
        assert_eq!(function.blocks[body.index()].terminator, Terminator::Return);
        assert_eq!(function.blocks[body.index()].instrs.len(), 2);
    }

    #[test]
    fn block_names_stay_unique_when_hints_collide() {
        let mut builder = FunctionBuilder::new(None, crate::Profile::Host);
        for hint in ["body", "body", "body.1", "body"] {
            builder.new_block(hint, 0, 0);
        }
        let function = builder.finish();
        let names: Vec<_> = function
            .blocks
            .iter()
            .map(|block| block.name.as_deref())
            .collect();
        assert_eq!(
            names,
            [
                Some("entry"),
                Some("body"),
                Some("body.1"),
                Some("body.1.1"),
                Some("body.2"),
            ]
        );
    }

    #[test]
    fn cleanup_order_distinguishes_inherited_replaced_and_merged_values() {
        let mut builder = FunctionBuilder::new(None, crate::Profile::Host);
        builder.emit(Instr::Push { value: Value::Unit });
        let local = builder.local(Ty::Unit, None);
        let inherited = builder.new_block("inherited", 1, 0);
        assert!(!builder.stack_is_newer_than(local));

        builder.emit(Instr::Ascribe { ty: Ty::Unit });
        assert!(builder.stack_is_newer_than(local));
        builder.switch(inherited);
        assert!(!builder.stack_is_newer_than(local));

        let merged = builder.new_block("merged", 1, 1);
        let later_local = builder.local(Ty::Unit, None);
        builder.switch(merged);
        assert_eq!(builder.stack_len(), 2);
        assert!(builder.stack_is_newer_than(later_local));
        builder.emit(Instr::Discard);
        assert!(!builder.stack_is_newer_than(local));
        builder.emit(Instr::Discard);
        assert!(!builder.stack_is_newer_than(local));
    }
}
