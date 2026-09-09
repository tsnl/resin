use super::LowerError;
use crate::{Instr, Terminator};
use resin_hir::{MatchArm, Term};
use resin_source::prelude::*;
use resin_types::prelude::*;

use super::Generator;
use super::{Initialization, ValueBinding};

impl Generator {
    pub(super) fn coerce(&mut self, span: Span, from: Ty, to: &Ty) -> Result<Ty, LowerError> {
        if &from != to {
            if !from.widens_to(to) {
                return self
                    .typer
                    .same(to, &from)
                    .map(|()| to.clone())
                    .map_err(|e| LowerError::typing(span, e));
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
    ) -> Result<Ty, LowerError> {
        let ty @ Ty::Result {
            value,
            error: errors,
        } = expected
        else {
            return Err(LowerError::invalid_hir(
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

    pub(super) fn gen_try(&mut self, span: Span, term: &Term) -> Result<Ty, LowerError> {
        let ty = self.gen_term(term, None)?;
        let Ty::Result {
            value,
            error: errors,
        } = &ty
        else {
            return Err(LowerError::invalid_hir(
                span,
                "postfix ? requires a Result value",
            ));
        };
        let result = self.function().result_type().clone();
        let Ty::Result { error: target, .. } = &result else {
            return Err(LowerError::invalid_hir(
                span,
                "postfix ? requires a Result return type",
            ));
        };
        if !errors.widens_to(target) {
            return Err(LowerError::invalid_hir(
                span,
                "the return type does not include every propagated error",
            ));
        }
        let saved = self.save_top(&ty);
        self.emit(Instr::LocalAddress { local: saved });
        self.emit(Instr::IsVariant { tag: Case::Ok });
        let height = self.function().stack_len() - 1;
        let success = self.new_block("try.ok", height);
        let failure = self.new_block("try.err", height);
        let next = self.new_block("try.next", height + 1);
        self.terminate(Terminator::If {
            then: success,
            els: failure,
            next: Some(next),
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
        let before_cleanup = self.bindings.clone();
        self.cleanup(0, &result);
        self.terminate(Terminator::Return);
        self.bindings = before_cleanup;
        self.switch(success);
        self.emit(Instr::TakeLocal { local: saved });
        self.emit(Instr::VariantPayload { tag: Case::Ok });
        self.terminate(Terminator::Merge);
        self.switch(next);
        Ok(*value.clone())
    }

    pub(super) fn gen_match(
        &mut self,
        term: &Term,
        arms: &[MatchArm],
        expected: &Ty,
    ) -> Result<Ty, LowerError> {
        let ty = self.gen_term(term, None)?;
        let saved = self.save_top(&ty);
        let before = self.bindings.clone();
        let mut after = None;
        let mut result = Some(expected.clone());
        let height = self.function().stack_len();
        let join = (arms.len() > 1).then(|| self.new_block("match.join", height + 1));
        for (i, arm) in arms.iter().enumerate() {
            let tag = &arm.tag;
            let next = if i + 1 < arms.len() {
                self.emit(Instr::LocalAddress { local: saved });
                self.emit(Instr::IsVariant { tag: tag.clone() });
                let body = self.new_block("match.arm", height);
                let next = self.new_block("match.next", height);
                self.terminate(Terminator::If {
                    then: body,
                    els: next,
                    next: if i == 0 { join } else { None },
                });
                self.switch(body);
                Some(next)
            } else {
                None
            };
            self.bindings = before.clone();
            self.owned.push(vec![]);
            self.emit(Instr::TakeLocal { local: saved });
            self.emit(Instr::VariantPayload { tag: tag.clone() });
            let payload = ty.payload(tag).unwrap();
            let local = self.save_top(&payload);
            if let Some(binding) = arm.binding {
                self.bindings.insert(
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
                self.intersect_initialization(previous);
            }
            after = Some(self.bindings.clone());
            if join.is_some() {
                self.terminate(Terminator::Merge);
            }
            if let Some(next) = next {
                self.switch(next);
            }
        }
        self.bindings = after.unwrap();
        if let Some(join) = join {
            self.switch(join);
        }
        Ok(result.unwrap())
    }
}
