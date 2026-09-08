use crate::ast::{Ident, Span, Term, TermKind};
use crate::ir::{Conv, Instr, Ty, TypeError, TypeErrorKind};

use super::scope::{Initialization, ValueBindingKind};
use super::{GenerateError, GenerateErrorKind, Generator};

pub(super) enum Operand {
    Value(Ty),
    Place(Ty),
}

impl Generator {
    pub(super) fn gen_assign(&mut self, place: &Term, value: &Term) -> Result<Ty, GenerateError> {
        let place_ty = self.gen_place(place)?;
        let Ty::Pointer { pointee } = place_ty else {
            return Err(GenerateError::typing(
                place.span,
                TypeError {
                    kind: TypeErrorKind::ExpectedPointer { found: place_ty },
                },
            ));
        };
        let value_ty = self.gen_term(value, Some(&pointee))?;
        let ty = self
            .typer
            .type_assign(&pointee, &value_ty)
            .map_err(|err| GenerateError::typing(place.span, err))?;
        self.emit(Instr::Store);
        if let TermKind::Var { name } = &place.val {
            self.scopes
                .lookup_value_mut(&name.val)
                .expect("assigned binding")
                .initialization = Initialization::Initialized;
        }
        Ok(ty)
    }

    pub(super) fn gen_field_value(
        &mut self,
        span: Span,
        base: &Term,
        name: &Ident,
    ) -> Result<Ty, GenerateError> {
        if name.val.as_ref() == "spirv"
            && let TermKind::Var {
                name: function_name,
            } = &base.val
        {
            let binding = self.resolve_value(function_name)?;
            if let ValueBindingKind::Function(function) = binding.kind {
                let entry =
                    self.module
                        .shaders
                        .get_mut(&function)
                        .ok_or_else(|| GenerateError {
                            span,
                            kind: GenerateErrorKind::InvalidShader {
                                message: "`.spirv` requires a function with a shader decorator"
                                    .into(),
                            },
                        })?;
                entry.embedded = true;
                let stage = entry.stage.clone();
                self.scopes
                    .record_fields(name, &Ty::shader_properties(), &self.typer);
                self.emit(Instr::Shader { function, stage });
                return Ok(Ty::shader());
            }
        }
        self.check_place_initialized(base)?;
        let base = self.gen_operand(base)?;
        match self.gen_field_operand(span, base, name)? {
            Operand::Place(ty) => {
                self.emit(Instr::Load);
                Ok(ty)
            }
            Operand::Value(ty) => Ok(ty),
        }
    }

    pub(super) fn gen_place(&mut self, term: &Term) -> Result<Ty, GenerateError> {
        match self.gen_operand(term)? {
            Operand::Place(ty) => Ok(Ty::Pointer {
                pointee: Box::new(ty),
            }),
            Operand::Value(_) => Err(GenerateError {
                span: term.span,
                kind: GenerateErrorKind::NotAPlace,
            }),
        }
    }

    // Lower once, preserving an address when available. Speculatively generating
    // a place and then retrying as a value can evaluate side effects twice.
    pub(super) fn gen_operand(&mut self, term: &Term) -> Result<Operand, GenerateError> {
        match &term.val {
            TermKind::Var { name } => {
                let binding = self.resolve_binding(name, false)?;
                if matches!(binding.kind, ValueBindingKind::Function(_)) {
                    return self.gen_var(name).map(Operand::Value);
                }
                let ty = self.binding_ty(name, &binding)?;
                self.emit_binding_address(&binding);
                Ok(Operand::Place(ty))
            }
            TermKind::Field { base, name }
                if name.val.as_ref() == "spirv"
                    && matches!(&base.val, TermKind::Var { name } if self.scopes.lookup_value(&name.val).is_some_and(|binding| matches!(binding.kind, ValueBindingKind::Function(_)))) =>
            {
                self.gen_term(term, None).map(Operand::Value)
            }
            TermKind::Field { base, name } => {
                self.check_place_initialized(base)?;
                let base = self.gen_operand(base)?;
                self.gen_field_operand(term.span, base, name)
            }
            TermKind::Deref { pointer } => {
                let checked =
                    self.checked.expressions[&std::ptr::from_ref(pointer.as_ref())].clone();
                let pointer_ty = if matches!(checked, Ty::Arc { .. }) {
                    self.hold_arc_address(pointer)?
                } else {
                    self.gen_term(pointer, None)?
                };
                let pointee = self
                    .typer
                    .type_deref(&pointer_ty)
                    .map_err(|err| GenerateError::typing(term.span, err))?;
                Ok(Operand::Place(pointee))
            }
            _ => self.gen_term(term, None).map(Operand::Value),
        }
    }

    fn gen_field_operand(
        &mut self,
        span: Span,
        base: Operand,
        name: &Ident,
    ) -> Result<Operand, GenerateError> {
        let (mut base_ty, mut is_place) = match base {
            Operand::Value(ty) => (ty, false),
            Operand::Place(ty) => (ty, true),
        };
        loop {
            if let Ty::Pointer { pointee } = &base_ty {
                let pointee = *pointee.clone();
                if is_place {
                    self.emit(Instr::Load);
                }
                base_ty = pointee;
                is_place = true;
                continue;
            }
            let Ty::Arc { pointee } = &base_ty else {
                break;
            };
            let pointee = *pointee.clone();
            if is_place {
                self.emit(Instr::Load);
            }
            let owner = self.save_top(&base_ty);
            self.load_local(owner);
            self.emit(Instr::ArcData);
            base_ty = pointee;
            is_place = true;
        }
        self.scopes.record_fields(name, &base_ty, &self.typer);
        let access = self
            .typer
            .type_field(&base_ty, &name.val)
            .map_err(|err| GenerateError::typing(span, err))?;
        if is_place {
            self.emit_place_conv(&access.steps);
        } else if let Some(last_deref) = access
            .steps
            .iter()
            .rposition(|step| matches!(step, Conv::Deref))
        {
            // Keep the last address so pointer-valued expressions have writable fields.
            self.emit_value_conv(&access.steps[..last_deref]);
            is_place = true;
        }
        // AccessStatic projects nominal layouts itself. Preserve the value's
        // nominal type so consuming a temporary invokes its destruction hook.
        self.emit(Instr::AccessStatic {
            index: access.index,
        });
        Ok(if is_place {
            Operand::Place(access.ty)
        } else {
            Operand::Value(access.ty)
        })
    }

    fn emit_place_conv(&mut self, steps: &[Conv]) {
        for step in steps {
            if matches!(step, Conv::Deref) {
                self.emit(Instr::Load);
            }
        }
    }

    pub(super) fn check_place_initialized(&self, term: &Term) -> Result<(), GenerateError> {
        match &term.val {
            TermKind::Var { name } => {
                self.resolve_value(name)?;
            }
            TermKind::Field { base, .. } => self.check_place_initialized(base)?,
            _ => {}
        }
        Ok(())
    }
}
