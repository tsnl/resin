use super::plan::{MatchArm, Term};
use crate::ast::Span;
use crate::ir::{Case, Instr, Terminator, Ty};

use super::scope::{Initialization, ValueBinding, ValueBindingKind};
use super::{GenerateError, Generator, plan::error};

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
            return Err(error(
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
            return Err(error(span, "postfix ? requires a Result value"));
        };
        let result = self.function().result_type().clone();
        let Ty::Result { error: target, .. } = &result else {
            return Err(error(span, "postfix ? requires a Result return type"));
        };
        if !errors.widens_to(target) {
            return Err(error(
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
        let before_cleanup = self.scopes.clone();
        self.cleanup(0, &result);
        self.terminate(Terminator::Return);
        self.scopes = before_cleanup;
        self.switch(success);
        self.emit(Instr::TakeLocal { local: saved });
        self.emit(Instr::VariantPayload { tag: Case::Ok });
        Ok(*value.clone())
    }

    pub(super) fn gen_match(
        &mut self,
        span: Span,
        term: &Term,
        arms: &[MatchArm],
        expected: &Ty,
    ) -> Result<Ty, GenerateError> {
        let ty = self.gen_term(term, None)?;
        let tags = match &ty {
            Ty::Result { .. } => vec![Case::Ok, Case::Err],
            _ => ty.members().into_iter().map(Case::Type).collect(),
        };
        let mut patterns = vec![];
        for arm in arms {
            let tag = match (&arm.variant, &ty) {
                (None, Ty::Result { .. }) => {
                    if arm.failure {
                        Case::Err
                    } else {
                        Case::Ok
                    }
                }
                (Some(ann), ty) if !matches!(ty, Ty::Result { .. }) => {
                    let member = ann.resolve(self)?;
                    if matches!(member, Ty::Union { .. }) {
                        return Err(error(
                            ann.span,
                            "union patterns must name a single member type",
                        ));
                    }
                    Case::Type(member)
                }
                _ => {
                    return Err(error(
                        arm.body.span,
                        "pattern does not belong to this match type",
                    ));
                }
            };
            if !tags.contains(&tag) || patterns.contains(&tag) {
                return Err(error(arm.body.span, "unknown or duplicate match variant"));
            }
            patterns.push(tag);
        }
        if tags.len() != patterns.len() || tags.is_empty() {
            return Err(error(span, "match must cover every variant exactly once"));
        }
        let saved = self.save_top(&ty);
        let before = self.scopes.clone();
        let mut after = None;
        let mut result = Some(expected.clone());
        let join = self.new_block("match.join");
        for (i, (arm, tag)) in arms.iter().zip(patterns).enumerate() {
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
            self.scopes = before.clone();
            self.scopes.push();
            self.owned.push(vec![]);
            self.emit(Instr::TakeLocal { local: saved });
            self.emit(Instr::VariantPayload { tag: tag.clone() });
            let payload = ty.payload(&tag).unwrap();
            let local = self.save_top(&payload);
            if let Some(name) = &arm.name {
                self.bind_value(
                    name,
                    ValueBinding {
                        shader: false,
                        kind: ValueBindingKind::Local(local),
                        ty: Some(payload),
                        initialization: Initialization::Initialized,
                    },
                )?;
            }
            result = Some(self.gen_term(&arm.body, result.as_ref())?);
            self.cleanup(self.owned.len() - 1, result.as_ref().unwrap());
            self.owned.pop();
            self.scopes.pop();
            if let Some(previous) = &after {
                self.scopes.intersect_initialization(previous);
            }
            after = Some(self.scopes.clone());
            self.terminate(Terminator::Break { target: join });
            if let Some(next) = next {
                self.switch(next);
            }
        }
        self.scopes = after.unwrap();
        self.switch(join);
        Ok(result.unwrap())
    }
}
