use super::scope::{Initialization, ValueBinding};
use super::{GenerateError, GenerateErrorKind, Generator};
use crate::hir::{BindingId, Term};
use crate::source::Ident;
use crate::{Instr, Ty};

impl Generator {
    pub(super) fn gen_define(
        &mut self,
        id: BindingId,
        name: &Ident,
        init: &Term,
    ) -> Result<(), GenerateError> {
        let local = self.alloc_local(init.ty.clone(), Some(name.val.clone()));
        self.environment.bind(
            id,
            ValueBinding {
                local,
                ty: init.ty.clone(),
                initialization: Initialization::Initializing,
            },
        );
        self.gen_term(init, None)?;
        self.environment.binding_mut(id).unwrap().initialization = Initialization::Initialized;
        self.emit(Instr::SetLocal { local });
        Ok(())
    }

    pub(super) fn gen_declare(
        &mut self,
        id: BindingId,
        name: &Ident,
        ty: Ty,
    ) -> Result<(), GenerateError> {
        let local = self.alloc_local(ty.clone(), Some(name.val.clone()));
        self.environment.bind(
            id,
            ValueBinding {
                local,
                ty,
                initialization: Initialization::Uninitialized,
            },
        );
        Ok(())
    }

    pub(super) fn gen_var(&mut self, id: BindingId, name: &Ident) -> Result<Ty, GenerateError> {
        let binding = self.resolve_value(id, name)?;
        self.load_local(binding.local);
        Ok(binding.ty)
    }

    pub(super) fn resolve_value(
        &self,
        id: BindingId,
        name: &Ident,
    ) -> Result<ValueBinding, GenerateError> {
        self.resolve_binding(id, name, true)
    }

    pub(super) fn resolve_binding(
        &self,
        id: BindingId,
        name: &Ident,
        read: bool,
    ) -> Result<ValueBinding, GenerateError> {
        let binding = self
            .environment
            .binding(id)
            .cloned()
            .ok_or_else(|| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::UnboundValue {
                    name: name.val.clone(),
                },
            })?;
        let kind = match binding.initialization {
            Initialization::Initializing => GenerateErrorKind::EagerRecursion {
                name: name.val.clone(),
            },
            Initialization::Uninitialized if read => GenerateErrorKind::UninitializedValue {
                name: name.val.clone(),
            },
            _ => return Ok(binding),
        };
        Err(GenerateError {
            span: name.span,
            kind,
        })
    }
}
