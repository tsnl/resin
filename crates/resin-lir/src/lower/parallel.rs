//! Serial reference schedule for fixed-size parallel regions. Keep these nodes
//! until specialization; a cooperative schedule must be chosen before storage.
use super::concrete::{ParallelParameter, Term};
use super::{FunctionLowering, LowerError, ValueBinding};
use crate::{Instr, ParallelOperation, Profile, Terminator};
use resin_types::prelude::*;

impl FunctionLowering<'_> {
    pub(super) fn gen_parallel_map(
        &mut self,
        input: &Term,
        element: &ParallelParameter,
        body: &Term,
        expected: &Ty,
    ) -> Result<Ty, LowerError> {
        if self.cooperative() {
            return self.gen_cooperative(input, None, &[element], body, expected);
        }
        let Ty::Array {
            length,
            element: result,
        } = expected
        else {
            return Err(LowerError::invalid_hir(
                body.span,
                "parallel map result must be an array",
            ));
        };
        self.enter_scope();
        self.gen_term(input, None)?;
        let array = self.save_top(&input.ty);
        // Initially expand the source-known extent. This keeps partial results
        // on the ordinary ownership stack, with per-iteration scope cleanup.
        for index in 0..*length {
            self.enter_scope();
            self.parallel_element(array, index);
            self.parallel_binding(element);
            self.gen_term(body, Some(result))?;
            self.cleanup(self.owned.len() - 1, result);
            self.owned.pop();
        }
        self.emit(Instr::MakeArray {
            elements: *length,
            element: *result.clone(),
        });
        self.cleanup(self.owned.len() - 1, expected);
        self.owned.pop();
        Ok(expected.clone())
    }

    pub(super) fn gen_parallel_reduce(
        &mut self,
        input: &Term,
        identity: &Term,
        parameters: [&ParallelParameter; 2],
        body: &Term,
    ) -> Result<Ty, LowerError> {
        if self.cooperative() {
            return self.gen_cooperative(input, Some(identity), &parameters, body, &identity.ty);
        }
        let Ty::Array { length, .. } = &input.ty else {
            return Err(LowerError::invalid_hir(
                input.span,
                "parallel reduction input must be an array",
            ));
        };
        self.enter_scope();
        self.gen_term(input, None)?;
        let array = self.save_top(&input.ty);
        self.gen_term(identity, None)?;
        let accumulator = self.save_top(&identity.ty);
        for index in 0..*length {
            self.enter_scope();
            self.emit(Instr::TakeLocal { local: accumulator });
            self.parallel_binding(parameters[0]);
            self.parallel_element(array, index);
            self.parallel_binding(parameters[1]);
            self.gen_term(body, Some(&identity.ty))?;
            self.cleanup(self.owned.len() - 1, &identity.ty);
            self.owned.pop();
            self.emit(Instr::SetLocal { local: accumulator });
        }
        self.emit(Instr::TakeLocal { local: accumulator });
        self.cleanup(self.owned.len() - 1, &identity.ty);
        self.owned.pop();
        Ok(identity.ty.clone())
    }

    fn parallel_element(&mut self, array: LocalId, index: usize) {
        self.emit(Instr::LocalRef { local: array });
        self.emit(Instr::AccessStatic { index });
        self.emit(Instr::Load);
    }

    fn parallel_binding(&mut self, parameter: &ParallelParameter) {
        let local = self.parallel_parameter(parameter);
        self.emit(Instr::SetLocal { local });
    }

    fn parallel_parameter(&mut self, parameter: &ParallelParameter) -> LocalId {
        let local = self.alloc_local(parameter.ty.clone(), Some(parameter.name.val.clone()));
        self.bindings.insert(
            parameter.binding,
            ValueBinding {
                local,
                ty: parameter.ty.clone(),
            },
        );
        local
    }

    fn cooperative(&self) -> bool {
        self.function.profile() == Profile::Compute && self.parallel_depth == 0
    }

    fn gen_cooperative(
        &mut self,
        input: &Term,
        identity: Option<&Term>,
        parameters: &[&ParallelParameter],
        body: &Term,
        expected: &Ty,
    ) -> Result<Ty, LowerError> {
        self.enter_scope();
        self.gen_term(input, None)?;
        let input = self.save_top(&input.ty);
        let identity = identity
            .map(|value| {
                self.gen_term(value, None)?;
                Ok::<_, LowerError>(self.save_top(&value.ty))
            })
            .transpose()?;
        let output = self.alloc_local(expected.clone(), None);
        let height = self.function.stack_len();
        let parent = self.function.position().0;
        let iteration = self.new_block("parallel.body", 0, 0);
        let next = self.new_block("parallel.next", height, 0);
        self.switch(iteration);
        self.enter_scope();
        let first = self.function.local_count();
        let left = self.parallel_parameter(parameters[0]);
        let operation = if let Some(identity) = identity {
            let right = self.parallel_parameter(parameters[1]);
            ParallelOperation::Reduce {
                input,
                identity,
                left,
                right,
                output,
            }
        } else {
            ParallelOperation::Map {
                input,
                element: left,
                output,
            }
        };
        self.parallel_depth += 1;
        self.gen_term(body, Some(&body.ty))?;
        self.parallel_depth -= 1;
        self.cleanup(self.owned.len() - 1, &body.ty);
        self.owned.pop();
        self.terminate(Terminator::ParallelYield);
        let end = self.function.local_count();
        self.switch(parent);
        self.terminate(Terminator::Parallel {
            operation,
            private_locals: first..end,
            body: iteration,
            next,
        });
        self.switch(next);
        self.emit(Instr::TakeLocal { local: output });
        self.cleanup(self.owned.len() - 1, expected);
        self.owned.pop();
        Ok(expected.clone())
    }
}
