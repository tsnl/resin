use super::{ErrorKind, LowerError};
use crate::Instr;
use crate::lower::concrete::{Term, TermKind};
use resin_types::prelude::*;

use super::FunctionLowering;

pub(super) enum Operand {
    Value(Ty),
    Place { ty: Ty },
}

impl FunctionLowering<'_> {
    fn local_path(&self, term: &Term) -> Result<Option<(LocalId, Vec<usize>)>, LowerError> {
        match &term.kind {
            TermKind::Local { binding, name } => {
                Ok(Some((self.resolve_binding(*binding, name)?.local, vec![])))
            }
            TermKind::Field { base, access }
                if !matches!(base.ty, Ty::Pointer { .. })
                    && !access.steps.iter().any(|step| matches!(step, Conv::Deref)) =>
            {
                let Some((local, mut path)) = self.local_path(base)? else {
                    return Ok(None);
                };
                path.push(access.index);
                Ok(Some((local, path)))
            }
            _ => Ok(None),
        }
    }

    pub(super) fn gen_move(&mut self, place: &Term) -> Result<Ty, LowerError> {
        let Some((local, path)) = self.local_path(place)? else {
            return Err(LowerError::invalid_hir(
                place.span,
                "move requires an owned local or field",
            ));
        };
        self.emit(if path.is_empty() {
            Instr::TakeLocal { local }
        } else {
            Instr::TakeField { local, path }
        });
        Ok(place.ty.clone())
    }

    pub(super) fn gen_assign(&mut self, place: &Term, value: &Term) -> Result<Ty, LowerError> {
        if let Some((local, path)) = self.local_path(place)? {
            self.gen_term(value, Some(&place.ty))?;
            self.emit(if path.is_empty() {
                Instr::SetLocal { local }
            } else {
                Instr::SetField { local, path }
            });
            self.emit(Instr::Push { value: Value::Unit });
            return Ok(Ty::Unit);
        }
        let place_ty = self.gen_place(place)?;
        let Ty::Pointer { pointee } = place_ty else {
            unreachable!("place produces a pointer");
        };
        self.gen_term(value, Some(&pointee))?;
        self.emit(Instr::Store);
        Ok(Ty::Unit)
    }

    pub(super) fn gen_field_value(
        &mut self,
        base: &Term,
        access: &FieldAccess,
    ) -> Result<Ty, LowerError> {
        let base = self.gen_operand(base)?;
        match self.gen_field_operand(base, access)? {
            Operand::Place { ty, .. } => {
                self.emit(Instr::Load);
                Ok(ty)
            }
            Operand::Value(ty) => Ok(ty),
        }
    }

    pub(super) fn gen_place(&mut self, term: &Term) -> Result<Ty, LowerError> {
        match self.gen_operand(term)? {
            Operand::Place { ty } => Ok(Ty::Pointer {
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
                let binding = self.resolve_binding(*id, name)?;
                let ty = binding.ty;
                self.emit(Instr::LocalAddress {
                    local: binding.local,
                });
                Ok(Operand::Place { ty })
            }
            TermKind::Field { base, access } => {
                let base = self.gen_operand(base)?;
                self.gen_field_operand(base, access)
            }
            TermKind::Deref { pointer } => {
                self.gen_term(pointer, None)?;
                let pointee = term.ty.clone();
                Ok(Operand::Place { ty: pointee })
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
            Operand::Place { ty } => (ty, true),
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
            break;
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
            Operand::Place {
                ty: access.ty.clone(),
            }
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
}
