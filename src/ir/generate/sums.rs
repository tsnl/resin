use crate::ast::{MatchArm, MatchVariant, Span, Term};
use crate::ir::{Instr, Terminator, Ty, Value};

use super::scope::{Initialization, ValueBinding, ValueBindingKind};
use super::{GenerateError, Generator, check::error};

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
            self.emit(Instr::Widen { ty: to.clone() });
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
            tag: u32::from(failure),
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
        self.load_local(saved);
        self.emit(Instr::VariantTag);
        self.emit(Instr::Push {
            value: Value::UInt32 { value: 0 },
        });
        self.emit(Instr::CallBuiltin {
            name: "==".into(),
            params: vec![Ty::UInt32, Ty::UInt32],
            result: Ty::Bool,
        });
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
        self.load_local(saved);
        self.emit(Instr::VariantPayload { tag: 1 });
        self.coerce(span, *errors.clone(), target)?;
        self.emit(Instr::MakeVariant {
            ty: result.clone(),
            tag: 1,
        });
        let before_cleanup = self.scopes.clone();
        self.cleanup(0, &result)?;
        self.terminate(Terminator::Return);
        self.scopes = before_cleanup;
        self.switch(success);
        self.load_local(saved);
        self.emit(Instr::VariantPayload { tag: 0 });
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
            Ty::Result { .. } => vec![0, 1],
            _ => ty
                .variants()
                .ok_or_else(|| error(span, "match requires a Result or a union of structs"))?
                .into_iter()
                .map(|id| id.tag())
                .collect(),
        };
        let mut patterns = vec![];
        for arm in arms {
            let tag = match &arm.variant {
                MatchVariant::Ok if matches!(ty, Ty::Result { .. }) => 0,
                MatchVariant::Err if matches!(ty, Ty::Result { .. }) => 1,
                MatchVariant::Type(ann) if !matches!(ty, Ty::Result { .. }) => {
                    let Ty::Defined { definition } = self.evaluator().ty(ann)? else {
                        return Err(error(ann.span, "union patterns must name a single struct"));
                    };
                    definition.tag()
                }
                _ => {
                    return Err(error(
                        arm.name.span,
                        "pattern does not belong to this match type",
                    ));
                }
            };
            if !tags.contains(&tag) || patterns.contains(&tag) {
                return Err(error(arm.name.span, "unknown or duplicate match variant"));
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
                self.load_local(saved);
                self.emit(Instr::VariantTag);
                self.emit(Instr::Push {
                    value: Value::UInt32 { value: tag },
                });
                self.emit(Instr::CallBuiltin {
                    name: "==".into(),
                    params: vec![Ty::UInt32, Ty::UInt32],
                    result: Ty::Bool,
                });
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
            self.load_local(saved);
            self.emit(Instr::VariantPayload { tag });
            let payload = ty.payload(tag).unwrap();
            let local = self.save_top(&payload);
            self.bind_value(
                &arm.name,
                ValueBinding {
                    shader: false,
                    kind: ValueBindingKind::Local(local),
                    ty: Some(payload),
                    initialization: Initialization::Initialized,
                },
            )?;
            result = Some(self.gen_term(&arm.body, result.as_ref())?);
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
