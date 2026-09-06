use std::sync::Arc;

use crate::ast::{Ident, Term, Type};
use crate::ir::{Instr, Ty, TypeId};

use super::scope::{Initialization, ValueBinding, ValueBindingKind};
use super::{GenerateError, GenerateErrorKind, Generator};

impl Generator {
    pub(super) fn gen_define(&mut self, name: &Ident, init: &Term) -> Result<(), GenerateError> {
        let placeholder = Ty::Unit;
        let depth = self.current_depth();
        let binding = if depth == 0 {
            let global = self.alloc_global(placeholder, name.val.clone());
            ValueBinding {
                kind: ValueBindingKind::Global(global),
                ty: None,
                initialization: Initialization::Initializing,
                depth,
            }
        } else {
            let local = self.alloc_local(placeholder, Some(name.val.clone()));
            ValueBinding {
                kind: ValueBindingKind::Local(local),
                ty: None,
                initialization: Initialization::Initializing,
                depth,
            }
        };
        self.bind_value(name, binding.clone())?;
        self.emit_binding_address(&binding);
        let ty = self.gen_term(init, None)?;
        self.complete_value(&name.val, ty)?;
        self.emit(Instr::Store);
        self.emit(Instr::Discard);
        Ok(())
    }

    pub(super) fn gen_define_type(
        &mut self,
        name: &Ident,
        init: &Type,
    ) -> Result<(), GenerateError> {
        let definition = self.typer.reserve_type(name.val.clone());
        self.bind_type(name, definition)?;
        let body = self.evaluator().ty(init)?;
        self.typer
            .define_type(definition, body)
            .map_err(|err| GenerateError::typing(init.span, err))
    }

    pub(super) fn gen_declare(&mut self, name: &Ident, ann: &Type) -> Result<(), GenerateError> {
        let ty = self.evaluator().ty(ann)?;
        let depth = self.current_depth();
        let binding = if depth == 0 {
            let global = self.alloc_global(ty.clone(), name.val.clone());
            ValueBinding {
                kind: ValueBindingKind::Global(global),
                ty: Some(ty),
                initialization: Initialization::Uninitialized,
                depth,
            }
        } else {
            let local = self.alloc_local(ty.clone(), Some(name.val.clone()));
            ValueBinding {
                kind: ValueBindingKind::Local(local),
                ty: Some(ty),
                initialization: Initialization::Uninitialized,
                depth,
            }
        };
        self.bind_value(name, binding)
    }

    pub(super) fn gen_var(&mut self, name: &Ident) -> Result<Ty, GenerateError> {
        let binding = self.resolve_value(name)?;
        let ty = self.binding_ty(name, &binding)?;
        if let ValueBindingKind::Function(function) = binding.kind {
            self.emit(Instr::Function { function });
            return Ok(ty);
        }
        self.emit_binding_address(&binding);
        self.emit(Instr::Load);
        Ok(ty)
    }

    pub(super) fn emit_binding_address(&mut self, binding: &ValueBinding) {
        match binding.kind {
            ValueBindingKind::Global(global) => self.emit(Instr::GlobalAddress { global }),
            ValueBindingKind::Local(local) => self.emit(Instr::LocalAddress { local }),
            ValueBindingKind::Function(_) => unreachable!("functions are immutable values"),
        }
    }

    pub(super) fn resolve_value(&self, name: &Ident) -> Result<ValueBinding, GenerateError> {
        self.resolve_binding(name, true)
    }

    pub(super) fn resolve_binding(
        &self,
        name: &Ident,
        read: bool,
    ) -> Result<ValueBinding, GenerateError> {
        self.scopes.record_reference(name, false);
        let binding = self
            .scopes
            .lookup_value(&name.val)
            .cloned()
            .ok_or_else(|| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::UnboundValue {
                    name: name.val.clone(),
                },
            })?;
        if binding.initialization == Initialization::Initializing
            && binding.depth == self.current_depth()
        {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::EagerRecursion {
                    name: name.val.clone(),
                },
            });
        }
        if binding.initialization != Initialization::Initialized
            && read
            && binding.depth == self.current_depth()
        {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::UninitializedValue {
                    name: name.val.clone(),
                },
            });
        }
        Ok(binding)
    }

    pub(super) fn bind_value(
        &mut self,
        name: &Ident,
        binding: ValueBinding,
    ) -> Result<(), GenerateError> {
        Self::check_binding_name(name)?;
        let ty = binding.ty.clone();
        self.scopes
            .define_value(name.val.clone(), binding)
            .map_err(|dup| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateValue { name: dup },
            })?;
        self.scopes
            .record_definition(name, false, ty.as_ref(), &self.typer);
        Ok(())
    }

    pub(super) fn check_binding_name(name: &Ident) -> Result<(), GenerateError> {
        if matches!(name.val.as_ref(), "print" | "shader") {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::ReservedBuiltin {
                    name: name.val.clone(),
                },
            });
        }
        Ok(())
    }

    fn bind_type(&mut self, name: &Ident, definition: TypeId) -> Result<(), GenerateError> {
        self.scopes
            .define_type(name.val.clone(), definition)
            .map_err(|dup| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateType { name: dup },
            })?;
        self.scopes
            .record_definition(name, true, Some(&Ty::Defined { definition }), &self.typer);
        Ok(())
    }

    fn complete_value(&mut self, name: &Arc<str>, ty: Ty) -> Result<(), GenerateError> {
        self.scopes.record_binding_type(name, &ty, &self.typer);
        let (kind, depth) = {
            let binding = self.scopes.lookup_value_mut(name).expect("defined binding");
            binding.ty = Some(ty.clone());
            binding.initialization = Initialization::Initialized;
            (binding.kind, binding.depth)
        };
        match kind {
            ValueBindingKind::Global(global) => {
                self.module.globals[global.index()].ty = ty;
            }
            ValueBindingKind::Local(local) => {
                self.functions[depth].set_local_type(local, ty);
            }
            ValueBindingKind::Function(_) => {
                unreachable!("cannot define a recursive-name binding")
            }
        }
        Ok(())
    }

    pub(super) fn binding_ty(
        &self,
        name: &Ident,
        binding: &ValueBinding,
    ) -> Result<Ty, GenerateError> {
        binding.ty.clone().ok_or_else(|| GenerateError {
            span: name.span,
            kind: GenerateErrorKind::NeedsTypeAnnotation {
                name: name.val.clone(),
            },
        })
    }
}
