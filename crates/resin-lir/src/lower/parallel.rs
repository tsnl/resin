//! Serial reference schedule for fixed-size parallel regions. Keep these nodes
//! until specialization; a cooperative schedule must be chosen before storage.
use super::concrete::{ParallelParameter, Term};
use super::{FunctionLowering, LowerError, ValueBinding};
use crate::Instr;
use resin_types::prelude::*;

impl FunctionLowering<'_> {
    pub(super) fn gen_parallel_map(
        &mut self,
        input: &Term,
        element: &ParallelParameter,
        body: &Term,
        expected: &Ty,
    ) -> Result<Ty, LowerError> {
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
        let local = self.alloc_local(parameter.ty.clone(), Some(parameter.name.val.clone()));
        self.emit(Instr::SetLocal { local });
        self.bindings.insert(
            parameter.binding,
            ValueBinding {
                local,
                ty: parameter.ty.clone(),
            },
        );
    }
}
