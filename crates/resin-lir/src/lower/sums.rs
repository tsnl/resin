use crate::{Case, Instr, Terminator, Ty};
use resin_common::source::Span;
use resin_hir::{MatchArm, Term};

use super::scope::{Initialization, ValueBinding};
use super::{GenerateError, Generator};

impl Generator {
    pub(super) fn coerce(&mut self, span: Span, from: Ty, to: &Ty) -> Result<Ty, GenerateError> {
        if &from != to {
            if !from.widens_to(to) {
                return self
                    .typer
                    .same(to, &from)
                    .map(|()| to.clone())
                    .map_err(|e| GenerateError::typing(span, e));
            }
            self.emit(if from == Ty::union([]) {
                Instr::Eliminate { result: to.clone() }
            } else {
                Instr::Widen { ty: to.clone() }
            });
        }
        Ok(to.clone())
    }

    pub(super) fn gen_result(
        &mut self,
        span: Span,
        failure: bool,
        arg: &Term,
        expected: &Ty,
    ) -> Result<Ty, GenerateError> {
        let ty @ Ty::Result {
            value,
            error: errors,
        } = expected
        else {
            return Err(GenerateError::inference(
                span,
                "cannot infer Result; annotate its value and error types",
            ));
        };
        self.gen_term(arg, Some(if failure { errors } else { value }))?;
        self.emit(Instr::MakeVariant {
            ty: ty.clone(),
            tag: if failure { Case::Err } else { Case::Ok },
        });
        Ok(ty.clone())
    }

    pub(super) fn gen_try(&mut self, span: Span, term: &Term) -> Result<Ty, GenerateError> {
        let ty = self.gen_term(term, None)?;
        let Ty::Result {
            value,
            error: errors,
        } = &ty
        else {
            return Err(GenerateError::inference(
                span,
                "postfix ? requires a Result value",
            ));
        };
        let result = self.function().result_type().clone();
        let Ty::Result { error: target, .. } = &result else {
            return Err(GenerateError::inference(
                span,
                "postfix ? requires a Result return type",
            ));
        };
        if !errors.widens_to(target) {
            return Err(GenerateError::inference(
                span,
                "the return type does not include every propagated error",
            ));
        }
        let saved = self.save_top(&ty);
        self.emit(Instr::LocalAddress { local: saved });
        self.emit(Instr::IsVariant { tag: Case::Ok });
        let success = self.new_block("try.ok");
        let failure = self.new_block("try.err");
        self.terminate(Terminator::Branch {
            then: success,
            els: failure,
        });
        self.switch(failure);
        for _ in 0..self.function().stack_len() {
            self.emit(Instr::Discard);
        }
        self.emit(Instr::TakeLocal { local: saved });
        self.emit(Instr::VariantPayload { tag: Case::Err });
        self.coerce(span, *errors.clone(), target)?;
        self.emit(Instr::MakeVariant {
            ty: result.clone(),
            tag: Case::Err,
        });
        let before_cleanup = self.environment.clone();
        self.cleanup(0, &result);
        self.terminate(Terminator::Return);
        self.environment = before_cleanup;
        self.switch(success);
        self.emit(Instr::TakeLocal { local: saved });
        self.emit(Instr::VariantPayload { tag: Case::Ok });
        Ok(*value.clone())
    }

    pub(super) fn gen_match(
        &mut self,
        term: &Term,
        arms: &[MatchArm],
        expected: &Ty,
    ) -> Result<Ty, GenerateError> {
        let ty = self.gen_term(term, None)?;
        let saved = self.save_top(&ty);
        let before = self.environment.clone();
        let mut after = None;
        let mut result = Some(expected.clone());
        let join = self.new_block("match.join");
        for (i, arm) in arms.iter().enumerate() {
            let tag = &arm.tag;
            let next = if i + 1 < arms.len() {
                self.emit(Instr::LocalAddress { local: saved });
                self.emit(Instr::IsVariant { tag: tag.clone() });
                let body = self.new_block("match.arm");
                let next = self.new_block("match.next");
                self.terminate(Terminator::Branch {
                    then: body,
                    els: next,
                });
                self.switch(body);
                Some(next)
            } else {
                None
            };
            self.environment = before.clone();
            self.owned.push(vec![]);
            self.emit(Instr::TakeLocal { local: saved });
            self.emit(Instr::VariantPayload { tag: tag.clone() });
            let payload = ty.payload(tag).unwrap();
            let local = self.save_top(&payload);
            if let Some(binding) = arm.binding {
                self.environment.bind(
                    binding,
                    ValueBinding {
                        local,
                        ty: payload,
                        initialization: Initialization::Initialized,
                    },
                );
            }
            result = Some(self.gen_term(&arm.body, result.as_ref())?);
            self.cleanup(self.owned.len() - 1, result.as_ref().unwrap());
            self.owned.pop();
            if let Some(previous) = &after {
                self.environment.intersect_initialization(previous);
            }
            after = Some(self.environment.clone());
            self.terminate(Terminator::Break { target: join });
            if let Some(next) = next {
                self.switch(next);
            }
        }
        self.environment = after.unwrap();
        self.switch(join);
        Ok(result.unwrap())
    }
}
