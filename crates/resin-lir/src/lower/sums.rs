use super::LowerError;
use crate::lower::concrete::{MatchArm, Term};
use crate::{Instr, Terminator};
use resin_source::prelude::*;
use resin_types::prelude::*;

use super::FunctionLowering;
use super::ValueBinding;

impl FunctionLowering<'_> {
    pub(super) fn coerce(&mut self, span: Span, from: Ty, to: &Ty) -> Result<Ty, LowerError> {
        if self.function.terminated() {
            return Ok(to.clone());
        }
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

    pub(super) fn gen_try(&mut self, span: Span, term: &Term) -> Result<Ty, LowerError> {
        let ty = self.gen_term(term, None)?;
        if self.function.terminated() {
            return Ok(ty);
        }
        self.gen_try_union(span, ty)
    }

    fn gen_try_union(&mut self, span: Span, ty: Ty) -> Result<Ty, LowerError> {
        let members = ty.members();
        let success = Ty::union_of(
            members
                .iter()
                .filter(|ty| !matches!(ty, Ty::Error { .. }))
                .cloned(),
        );
        if success == ty {
            return Ok(ty);
        }
        let result = self.function.result_type().clone();
        for member in &members {
            if matches!(member, Ty::Error { .. }) && !member.widens_to(&result) {
                return Err(LowerError::invalid_hir(
                    span,
                    "the return type does not include every propagated Err",
                ));
            }
        }
        let saved = self.save_top(&ty);
        let height = self.function.stack_len();
        let join = (success != Ty::union([]) && members.len() > 1)
            .then(|| self.new_block("try.value", height, 1));
        for (index, member) in members.iter().enumerate() {
            let case = Case::Type(member.clone());
            let next = if index + 1 < members.len() {
                self.emit(Instr::LocalRef { local: saved });
                self.emit(Instr::IsVariant { tag: case.clone() });
                let body = self.new_block("try.member", height, 0);
                let next = self.new_block("try.remaining", height, 0);
                self.terminate(Terminator::If {
                    then: body,
                    els: next,
                    next: if index == 0 { join } else { None },
                });
                self.switch(body);
                Some(next)
            } else {
                None
            };
            self.emit(Instr::TakeLocal { local: saved });
            self.emit(Instr::VariantPayload { tag: case });
            if matches!(member, Ty::Error { .. }) {
                self.coerce(span, member.clone(), &result)?;
                self.cleanup(0, &result);
                self.terminate(Terminator::Return);
            } else {
                self.coerce(span, member.clone(), &success)?;
                if join.is_some() {
                    self.terminate(Terminator::Merge);
                }
            }
            if let Some(next) = next {
                self.switch(next);
            }
        }
        if let Some(join) = join {
            self.switch(join);
        }
        Ok(success)
    }

    pub(super) fn gen_match(
        &mut self,
        term: &Term,
        arms: &[MatchArm],
        expected: &Ty,
    ) -> Result<Ty, LowerError> {
        let ty = self.gen_term(term, None)?;
        if self.function.terminated() {
            return Ok(expected.clone());
        }
        let saved = self.save_top(&ty);
        let mut result = Some(expected.clone());
        let height = self.function.stack_len();
        let join = (arms.len() > 1 && arms.iter().any(|arm| !arm.body.exits()))
            .then(|| self.new_block("match.join", height, 1));
        for (i, arm) in arms.iter().enumerate() {
            let tag = &arm.tag;
            let next = if i + 1 < arms.len() {
                self.emit(Instr::LocalRef { local: saved });
                self.emit(Instr::IsVariant { tag: tag.clone() });
                let body = self.new_block("match.arm", height, 0);
                let next = self.new_block("match.next", height, 0);
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
            self.enter_scope();
            self.emit(Instr::TakeLocal { local: saved });
            self.emit(Instr::VariantPayload { tag: tag.clone() });
            let mut payload = ty.payload(tag).unwrap();
            if let Some(target) = &arm.error_payload {
                let Ty::Error { payload: inner } = payload else {
                    unreachable!("Err pattern member");
                };
                self.emit(Instr::Ascribe { ty: *inner.clone() });
                self.coerce(term.span, *inner, target)?;
                payload = target.clone();
            }
            let local = self.save_top(&payload);
            if let Some(binding) = arm.binding {
                self.bindings
                    .insert(binding, ValueBinding { local, ty: payload });
            }
            result = Some(self.gen_term(&arm.body, result.as_ref())?);
            self.cleanup(self.owned.len() - 1, result.as_ref().unwrap());
            self.owned.pop();
            if join.is_some() {
                self.terminate(Terminator::Merge);
            }
            if let Some(next) = next {
                self.switch(next);
            }
        }
        if let Some(join) = join {
            self.switch(join);
        }
        Ok(result.unwrap())
    }
}
