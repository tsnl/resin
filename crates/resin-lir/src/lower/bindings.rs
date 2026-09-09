use super::Generator;
use super::{ErrorKind, LowerError};
use super::{Initialization, ValueBinding};
use crate::Instr;
use resin_hir::{BindingId, Term};
use resin_source::prelude::*;
use resin_types::prelude::*;

impl Generator {
    pub(super) fn gen_define(
        &mut self,
        id: BindingId,
        name: &Ident,
        init: &Term,
    ) -> Result<(), LowerError> {
        let local = self.alloc_local(init.ty.clone(), Some(name.val.clone()));
        self.bindings.insert(
            id,
            ValueBinding {
                local,
                ty: init.ty.clone(),
                initialization: Initialization::Initializing,
            },
        );
        self.gen_term(init, None)?;
        self.bindings.get_mut(&id).unwrap().initialization = Initialization::Initialized;
        self.emit(Instr::SetLocal { local });
        Ok(())
    }

    pub(super) fn gen_declare(
        &mut self,
        id: BindingId,
        name: &Ident,
        ty: Ty,
    ) -> Result<(), LowerError> {
        let local = self.alloc_local(ty.clone(), Some(name.val.clone()));
        self.bindings.insert(
            id,
            ValueBinding {
                local,
                ty,
                initialization: Initialization::Uninitialized,
            },
        );
        Ok(())
    }

    pub(super) fn gen_var(&mut self, id: BindingId, name: &Ident) -> Result<Ty, LowerError> {
        let binding = self.resolve_value(id, name)?;
        self.load_local(binding.local);
        Ok(binding.ty)
    }

    pub(super) fn resolve_value(
        &self,
        id: BindingId,
        name: &Ident,
    ) -> Result<ValueBinding, LowerError> {
        self.resolve_binding(id, name, true)
    }

    pub(super) fn resolve_binding(
        &self,
        id: BindingId,
        name: &Ident,
        read: bool,
    ) -> Result<ValueBinding, LowerError> {
        let binding = self.bindings.get(&id).cloned().ok_or_else(|| LowerError {
            span: name.span,
            kind: ErrorKind::UnboundValue {
                name: name.val.clone(),
            },
        })?;
        let kind = match binding.initialization {
            Initialization::Initializing => ErrorKind::EagerRecursion {
                name: name.val.clone(),
            },
            Initialization::Uninitialized if read => ErrorKind::UninitializedValue {
                name: name.val.clone(),
            },
            _ => return Ok(binding),
        };
        Err(LowerError {
            span: name.span,
            kind,
        })
    }
}
