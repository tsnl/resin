use crate::ast::{Ident, Span, Term, TermKind};
use crate::ir::{Conv, Instr, Ty, TypeError, TypeErrorKind};

use super::scope::{Initialization, ValueBindingKind};
use super::{GenerateError, GenerateErrorKind, Generator};

enum Operand {
    Value(Ty),
    Place(Ty),
}

impl Generator {
    pub(super) fn gen_assign(&mut self, place: &Term, value: &Term) -> Result<Ty, GenerateError> {
        let place_ty = self.gen_place(place)?;
        let converted = self
            .typer
            .as_pointer(&place_ty)
            .map_err(|err| GenerateError::typing(place.span, err))?;
        self.emit_value_conv(&converted.steps);
        let Ty::Pointer { pointee } = converted.ty else {
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
    fn gen_operand(&mut self, term: &Term) -> Result<Operand, GenerateError> {
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
            TermKind::Field { base, name } => {
                self.check_place_initialized(base)?;
                let base = self.gen_operand(base)?;
                self.gen_field_operand(term.span, base, name)
            }
            TermKind::Deref { pointer } => {
                let pointer_ty = self.gen_term(pointer, None)?;
                let converted = self
                    .typer
                    .as_pointer(&pointer_ty)
                    .map_err(|err| GenerateError::typing(term.span, err))?;
                self.emit_value_conv(&converted.steps);
                let Ty::Pointer { pointee } = converted.ty else {
                    unreachable!("as_pointer returns a pointer")
                };
                Ok(Operand::Place(*pointee))
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
        let (base_ty, mut is_place) = match base {
            Operand::Value(ty) => (ty, false),
            Operand::Place(ty) => (ty, true),
        };
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
        } else {
            self.emit_value_conv(&access.steps);
        }
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

    fn check_place_initialized(&self, term: &Term) -> Result<(), GenerateError> {
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
