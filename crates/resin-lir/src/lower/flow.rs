//! Preserve source control-flow regions while making storage and cleanup explicit.
use super::LowerError;

use crate::lower::concrete::{Statement, Term};
use crate::{Instr, Terminator};
use resin_types::prelude::*;

use super::FunctionLowering;

impl FunctionLowering<'_> {
    pub(super) fn gen_while(&mut self, cond: &Term, body: &Term) -> Result<Ty, LowerError> {
        if cond.exits() {
            self.gen_term(cond, Some(&Ty::Bool))?;
            return Ok(Ty::Unit);
        }
        let height = self.function.stack_len();
        let condition = self.new_block("while.cond", height, 0);
        let body_block = self.new_block("while.body", height, 0);
        let exit = self.new_block("while.exit", height, 0);
        self.terminate(Terminator::Loop {
            condition,
            body: body_block,
            next: Some(exit),
        });

        self.switch(condition);
        self.gen_term(cond, Some(&Ty::Bool))?;
        self.terminate(Terminator::LoopTest);

        self.switch(body_block);
        self.gen_term(body, None)?;
        self.emit(Instr::Discard);
        self.terminate(Terminator::Continue);

        self.switch(exit);
        self.emit(Instr::Push { value: Value::Unit });
        Ok(Ty::Unit)
    }

    pub(super) fn gen_if(
        &mut self,
        cond: &Term,
        then: &Term,
        els: &Term,
        expected: &Ty,
    ) -> Result<Ty, LowerError> {
        self.gen_term(cond, Some(&Ty::Bool))?;
        if self.function.terminated() {
            return Ok(expected.clone());
        }
        let height = self.function.stack_len() - 1;
        let then_block = self.new_block("then", height, 0);
        let else_block = self.new_block("else", height, 0);
        let join_block =
            (!(then.exits() && els.exits())).then(|| self.new_block("join", height, 1));
        self.terminate(Terminator::If {
            then: then_block,
            els: else_block,
            next: join_block,
        });

        self.switch(then_block);
        let _ = self.gen_term(then, Some(expected))?;
        self.terminate(Terminator::Merge);

        self.switch(else_block);
        let _ = self.gen_term(els, Some(expected))?;
        self.terminate(Terminator::Merge);

        if let Some(join_block) = join_block {
            self.switch(join_block);
        }
        Ok(expected.clone())
    }

    pub(super) fn gen_block(
        &mut self,
        stmts: &[Statement],
        tail: &Term,
        expected: &Ty,
    ) -> Result<Ty, LowerError> {
        self.enter_scope();
        for stmt in stmts {
            self.lower_statement(stmt)?;
        }
        let ty = self.gen_term(tail, Some(expected))?;
        self.cleanup(self.owned.len() - 1, &ty);
        self.owned.pop();
        Ok(ty)
    }
}

impl FunctionLowering<'_> {
    pub(super) fn gen_return(&mut self, value: &Term, expected: &Ty) -> Result<Ty, LowerError> {
        let result = self.function.result_type().clone();
        self.gen_term(value, Some(&result))?;
        self.cleanup(0, &result);
        self.terminate(Terminator::Return);
        Ok(expected.clone())
    }
}
