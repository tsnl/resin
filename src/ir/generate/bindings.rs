use std::sync::Arc;

use super::plan::Term;
use crate::ast::{Ident, Type};
use crate::ir::{Instr, Ty, TypeId, typecheck::check_binding_name};

use super::scope::{Initialization, ValueBinding, ValueBindingKind};
use super::{GenerateError, GenerateErrorKind, Generator};

impl Generator {
    pub(super) fn gen_define(&mut self, name: &Ident, init: &Term) -> Result<(), GenerateError> {
        let local = self.alloc_local(Ty::Unit, Some(name.val.clone()));
        let binding = ValueBinding {
            shader: false,
            kind: ValueBindingKind::Local(local),
            ty: None,
            initialization: Initialization::Initializing,
        };
        self.bind_value(name, binding.clone())?;
        let ty = self.gen_term(init, None)?;
        self.complete_value(&name.val, ty);
        self.emit(Instr::SetLocal { local });
        Ok(())
    }

    pub(super) fn gen_define_type(
        &mut self,
        name: &Ident,
        init: &Type,
    ) -> Result<(), GenerateError> {
        let ty = self.evaluator().ty(init)?;
        self.bind_alias(name, init.span, ty)
    }

    pub(super) fn bind_alias(
        &mut self,
        name: &Ident,
        span: crate::ast::Span,
        ty: Ty,
    ) -> Result<(), GenerateError> {
        self.scopes
            .define_alias(name.val.clone(), ty.clone())
            .map_err(|name| GenerateError {
                span,
                kind: GenerateErrorKind::DuplicateType { name },
            })?;
        self.scopes
            .record_definition(name, true, Some(&ty), &self.typer);
        Ok(())
    }

    pub(super) fn gen_struct(&mut self, name: &Ident, init: &Type) -> Result<(), GenerateError> {
        let definition = self.typer.declare_type(
            name.val.clone(),
            crate::ir::typecheck::SourceOrigin {
                module: self.source_module,
                span: name.span,
            },
        );
        self.bind_type(name, definition)?;
        let body = self.evaluator().ty(init)?;
        self.typer
            .define_type(definition, body)
            .map_err(|err| GenerateError::typing(init.span, err))
    }

    pub(super) fn gen_declare(&mut self, name: &Ident, ty: Ty) -> Result<(), GenerateError> {
        let local = self.alloc_local(ty.clone(), Some(name.val.clone()));
        let binding = ValueBinding {
            shader: false,
            kind: ValueBindingKind::Local(local),
            ty: Some(ty),
            initialization: Initialization::Uninitialized,
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
        if binding.initialization == Initialization::Initializing {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::EagerRecursion {
                    name: name.val.clone(),
                },
            });
        }
        if binding.initialization != Initialization::Initialized && read {
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
        check_binding_name(name)?;
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

    pub(super) fn bind_type(
        &mut self,
        name: &Ident,
        definition: TypeId,
    ) -> Result<(), GenerateError> {
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

    fn complete_value(&mut self, name: &Arc<str>, ty: Ty) {
        self.scopes.record_binding_type(name, &ty, &self.typer);
        let kind = {
            let binding = self.scopes.lookup_value_mut(name).expect("defined binding");
            binding.ty = Some(ty.clone());
            binding.initialization = Initialization::Initialized;
            binding.kind
        };
        match kind {
            ValueBindingKind::Local(local) => {
                self.function().set_local_type(local, ty);
            }
            ValueBindingKind::Function(_) => {
                unreachable!("cannot define a recursive-name binding")
            }
        }
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
