use super::{ErrorKind, LowerError};
use crate::Instr;
use resin_hir::{Term, TermKind};
use resin_types::prelude::*;

use super::Generator;
use super::Initialization;

pub(super) enum Operand {
    Value(Ty),
    Place(Ty),
}

impl Generator {
    pub(super) fn gen_assign(&mut self, place: &Term, value: &Term) -> Result<Ty, LowerError> {
        let place_ty = self.gen_place(place)?;
        let Ty::Pointer { pointee } = place_ty else {
            return Err(LowerError::typing(
                place.span,
                TypeError {
                    kind: TypeErrorKind::ExpectedPointer { found: place_ty },
                },
            ));
        };
        let ty = self.gen_term(value, Some(&pointee))?;
        self.emit(Instr::Store);
        if let TermKind::Local { binding: id, .. } = &place.kind {
            self.bindings
                .get_mut(id)
                .expect("assigned binding")
                .initialization = Initialization::Initialized;
        }
        Ok(ty)
    }

    pub(super) fn gen_field_value(
        &mut self,
        base: &Term,
        access: &FieldAccess,
    ) -> Result<Ty, LowerError> {
        self.check_place_initialized(base)?;
        let base = self.gen_operand(base)?;
        match self.gen_field_operand(base, access)? {
            Operand::Place(ty) => {
                self.emit(Instr::Load);
                Ok(ty)
            }
            Operand::Value(ty) => Ok(ty),
        }
    }

    pub(super) fn gen_place(&mut self, term: &Term) -> Result<Ty, LowerError> {
        match self.gen_operand(term)? {
            Operand::Place(ty) => Ok(Ty::Pointer {
                pointee: Box::new(ty),
            }),
            Operand::Value(_) => Err(LowerError {
                span: term.span,
                kind: ErrorKind::NotAPlace,
            }),
        }
    }

    // Lower once, preserving an address when available. Speculatively generating
    // a place and then retrying as a value can evaluate side effects twice.
    pub(super) fn gen_operand(&mut self, term: &Term) -> Result<Operand, LowerError> {
        match &term.kind {
            TermKind::Local { binding: id, name } => {
                let binding = self.resolve_binding(*id, name, false)?;
                let ty = binding.ty;
                self.emit(Instr::LocalAddress {
                    local: binding.local,
                });
                Ok(Operand::Place(ty))
            }
            TermKind::Field { base, access } => {
                self.check_place_initialized(base)?;
                let base = self.gen_operand(base)?;
                self.gen_field_operand(base, access)
            }
            TermKind::Deref { pointer } => {
                let checked = &pointer.ty;
                if matches!(checked, Ty::Arc { .. }) {
                    self.hold_arc_address(pointer)?
                } else {
                    self.gen_term(pointer, None)?
                };
                let pointee = term.ty.clone();
                Ok(Operand::Place(pointee))
            }
            _ => self.gen_term(term, None).map(Operand::Value),
        }
    }

    fn gen_field_operand(
        &mut self,
        base: Operand,
        access: &FieldAccess,
    ) -> Result<Operand, LowerError> {
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
            Operand::Place(access.ty.clone())
        } else {
            Operand::Value(access.ty.clone())
        })
    }

    fn emit_place_conv(&mut self, steps: &[Conv]) {
        for step in steps {
            if matches!(step, Conv::Deref) {
                self.emit(Instr::Load);
            }
        }
    }

    pub(super) fn check_place_initialized(&self, term: &Term) -> Result<(), LowerError> {
        match &term.kind {
            TermKind::Local { binding: id, name } => {
                self.resolve_value(*id, name)?;
            }
            TermKind::Field { base, .. } => self.check_place_initialized(base)?,
            _ => {}
        }
        Ok(())
    }
}
