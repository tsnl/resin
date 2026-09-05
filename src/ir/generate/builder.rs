use std::sync::Arc;

use crate::ir::{
    BasicBlock, BlockId, Function, Instr, Local, LocalId, NonLocal, NonLocalId, Terminator, Ty,
};

pub(super) struct FunctionBuilder {
    function: Function,
    current: BlockId,
    terminated: Vec<bool>,
}

impl FunctionBuilder {
    pub(super) fn new(name: Option<Arc<str>>) -> Self {
        Self {
            function: Function {
                name,
                nonlocals: Vec::new(),
                param: LocalId::from_index(0),
                result: Ty::Unit,
                locals: vec![Local {
                    name: None,
                    ty: Ty::Unit,
                }],
                entry: BlockId::from_index(0),
                blocks: vec![BasicBlock {
                    name: Some("entry".into()),
                    instrs: Vec::new(),
                    terminator: Terminator::Return,
                }],
            },
            current: BlockId::from_index(0),
            terminated: vec![false],
        }
    }

    pub(super) fn finish(self) -> Function {
        self.function
    }

    pub(super) fn name(&self) -> Option<Arc<str>> {
        self.function.name.clone()
    }

    pub(super) fn parameter(&mut self, name: Option<Arc<str>>, ty: Ty) {
        self.function.locals[self.function.param.index()] = Local { name, ty };
    }

    pub(super) fn result(&mut self, ty: Ty) {
        self.function.result = ty;
    }

    pub(super) fn local(&mut self, ty: Ty, name: Option<Arc<str>>) -> LocalId {
        let id = LocalId::from_index(self.function.locals.len());
        self.function.locals.push(Local { name, ty });
        id
    }

    pub(super) fn local_name(&self, id: LocalId) -> Option<Arc<str>> {
        self.function.locals[id.index()].name.clone()
    }

    pub(super) fn set_local_type(&mut self, id: LocalId, ty: Ty) {
        self.function.locals[id.index()].ty = ty;
    }

    pub(super) fn nonlocal(&mut self, ty: Ty, name: Option<Arc<str>>) -> NonLocalId {
        let id = NonLocalId::from_index(self.function.nonlocals.len());
        self.function.nonlocals.push(NonLocal { name, ty });
        id
    }

    pub(super) fn emit(&mut self, instr: Instr) {
        let current = self.current.index();
        debug_assert!(!self.terminated[current]);
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
    }

    pub(super) fn new_block(&mut self, hint: &str) -> BlockId {
        let id = BlockId::from_index(self.function.blocks.len());
        self.function.blocks.push(BasicBlock {
            name: Some(self.unique_block_name(hint)),
            instrs: Vec::new(),
            terminator: Terminator::Return,
        });
        self.terminated.push(false);
        id
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
    use crate::ir::{Module, verify};

    #[test]
    fn builds_a_function_without_syntax_or_scopes() {
        let mut builder = FunctionBuilder::new(Some("identity".into()));
        builder.parameter(Some("value".into()), Ty::Int32);
        builder.result(Ty::Int32);
        let body = builder.new_block("body");
        builder.terminate(Terminator::Break { target: body });
        builder.switch(body);
        builder.emit(Instr::LocalAddress {
            local: LocalId::from_index(0),
        });
        builder.emit(Instr::Load);
        builder.terminate(Terminator::Return);
        let function = builder.finish();
        assert_eq!(function.blocks.len(), 2);
        assert_eq!(function.locals[0].name.as_deref(), Some("value"));
        verify(&Module {
            functions: vec![function],
            ..Default::default()
        })
        .unwrap();
    }

    #[test]
    fn block_names_stay_unique_when_hints_collide() {
        let mut builder = FunctionBuilder::new(None);
        for hint in ["body", "body", "body.1", "body"] {
            builder.new_block(hint);
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
}
