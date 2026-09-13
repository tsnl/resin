//! Preserve source control-flow regions while making storage and cleanup explicit.
use super::LowerError;

use crate::lower::concrete::{Statement, Term};
use crate::{Instr, Terminator};
use resin_types::prelude::*;

use super::FunctionLowering;

impl FunctionLowering<'_> {
    pub(super) fn gen_while(&mut self, cond: &Term, body: &Term) -> Result<Ty, LowerError> {
        let height = self.function.stack_len();
        let condition = self.new_block("while.cond", height);
        let body_block = self.new_block("while.body", height);
        let exit = self.new_block("while.exit", height);
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
        let height = self.function.stack_len() - 1;
        let then_block = self.new_block("then", height);
        let else_block = self.new_block("else", height);
        let join_block = self.new_block("join", height + 1);
        self.terminate(Terminator::If {
            then: then_block,
            els: else_block,
            next: Some(join_block),
        });

        self.switch(then_block);
        let _ = self.gen_term(then, Some(expected))?;
        self.terminate(Terminator::Merge);

        self.switch(else_block);
        let _ = self.gen_term(els, Some(expected))?;
        self.terminate(Terminator::Merge);

        self.switch(join_block);
        Ok(expected.clone())
    }

    pub(super) fn gen_block(
        &mut self,
        stmts: &[Statement],
        tail: &Term,
        expected: &Ty,
    ) -> Result<Ty, LowerError> {
        self.owned.push(vec![]);
        for stmt in stmts {
            self.lower_statement(stmt)?;
        }
        let ty = self.gen_term(tail, Some(expected))?;
        self.cleanup(self.owned.len() - 1, &ty);
        self.owned.pop();
        Ok(ty)
    }
}
