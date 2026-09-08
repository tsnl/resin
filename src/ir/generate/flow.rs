//! Emit blocks and branches for source-level control-flow expressions.

use super::typed::{Statement, Term};
use crate::ir::{Instr, Terminator, Ty, Value};

use super::{GenerateError, Generator};

impl Generator {
    pub(super) fn gen_while(&mut self, cond: &Term, body: &Term) -> Result<Ty, GenerateError> {
        let condition = self.new_block("while.cond");
        let body_block = self.new_block("while.body");
        let exit = self.new_block("while.exit");
        self.terminate(Terminator::Break { target: condition });

        self.switch(condition);
        self.gen_term(cond, None)?;
        let after_condition = self.environment.clone();
        self.terminate(Terminator::Branch {
            then: body_block,
            els: exit,
        });

        self.switch(body_block);
        self.gen_term(body, None)?;
        self.emit(Instr::Discard);
        self.terminate(Terminator::Break { target: condition });

        self.environment = after_condition;
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
    ) -> Result<Ty, GenerateError> {
        self.gen_term(cond, None)?;
        let then_block = self.new_block("then");
        let else_block = self.new_block("else");
        let join_block = self.new_block("join");
        self.terminate(Terminator::Branch {
            then: then_block,
            els: else_block,
        });

        let before = self.environment.clone();
        self.switch(then_block);
        let _ = self.gen_term(then, Some(expected))?;
        self.terminate(Terminator::Break { target: join_block });
        let after_then = self.environment.clone();

        self.environment = before;
        self.switch(else_block);
        let _ = self.gen_term(els, Some(expected))?;
        self.terminate(Terminator::Break { target: join_block });
        self.environment.intersect_initialization(&after_then);

        self.switch(join_block);
        Ok(expected.clone())
    }

    pub(super) fn gen_block(
        &mut self,
        stmts: &[Statement],
        tail: &Term,
        expected: &Ty,
    ) -> Result<Ty, GenerateError> {
        self.owned.push(vec![]);
        for stmt in stmts {
            self.lower_statement(stmt)?;
        }
        let ty = self.gen_term(tail, Some(expected))?;
        self.cleanup(self.owned.len() - 1, &ty);
        self.owned.pop();
        Ok(ty)
    }

    pub(super) fn gen_short_circuit(
        &mut self,
        name: &str,
        args: &[Term],
    ) -> Result<Ty, GenerateError> {
        self.gen_term(&args[0], None)?;
        let then_block = self.new_block("then");
        let else_block = self.new_block("else");
        let join_block = self.new_block("join");
        self.terminate(Terminator::Branch {
            then: then_block,
            els: else_block,
        });
        let before_right = self.environment.clone();
        if name == "&&" {
            self.switch(then_block);
            self.gen_term(&args[1], None)?;
            self.terminate(Terminator::Break { target: join_block });
            self.switch(else_block);
            self.emit(Instr::Push {
                value: Value::Bool { value: false },
            });
            self.terminate(Terminator::Break { target: join_block });
        } else {
            self.switch(then_block);
            self.emit(Instr::Push {
                value: Value::Bool { value: true },
            });
            self.terminate(Terminator::Break { target: join_block });
            self.switch(else_block);
            self.gen_term(&args[1], None)?;
            self.terminate(Terminator::Break { target: join_block });
        }
        self.switch(join_block);
        self.environment.intersect_initialization(&before_right);
        Ok(Ty::Bool)
    }
}
