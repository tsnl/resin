use super::FunctionLowering;
use super::ValueBinding;
use super::{ErrorKind, LowerError};
use crate::Instr;
use crate::lower::concrete::Term;
use resin_hir::BindingId;
use resin_source::prelude::*;
use resin_types::prelude::*;

impl FunctionLowering<'_> {
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
            },
        );
        self.gen_expression(init, None)?;
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
        self.bindings.insert(id, ValueBinding { local, ty });
        Ok(())
    }

    pub(super) fn gen_var(&mut self, id: BindingId, name: &Ident) -> Result<Ty, LowerError> {
        let binding = self.resolve_binding(id, name)?;
        self.load_local(binding.local);
        Ok(binding.ty)
    }

    pub(super) fn resolve_binding(
        &self,
        id: BindingId,
        name: &Ident,
    ) -> Result<ValueBinding, LowerError> {
        self.bindings.get(&id).cloned().ok_or_else(|| LowerError {
            span: name.span,
            kind: ErrorKind::UnboundValue {
                name: name.val.clone(),
            },
        })
    }
}
