use super::typed::Term;
use crate::ast::{Ident, SourceFile, StmtKind, Type};
use crate::ir::{Instr, Ty, TyperContext};

use super::scope::{DeclarationId, Initialization, Scopes, ValueBinding, ValueBindingKind};
use super::{GenerateError, GenerateErrorKind, Generator};

impl Generator {
    pub(super) fn gen_define(
        &mut self,
        declaration: DeclarationId,
        name: &Ident,
        init: &Term,
    ) -> Result<(), GenerateError> {
        let local = self.alloc_local(Ty::Unit, Some(name.val.clone()));
        let binding = ValueBinding {
            shader: false,
            kind: ValueBindingKind::Local(local),
            ty: None,
            initialization: Initialization::Initializing,
        };
        self.environment.bind(declaration, binding);
        let ty = self.gen_term(init, None)?;
        self.complete_value(declaration, ty);
        self.emit(Instr::SetLocal { local });
        Ok(())
    }

    pub(super) fn gen_declare(
        &mut self,
        declaration: DeclarationId,
        name: &Ident,
        ty: Ty,
    ) -> Result<(), GenerateError> {
        let local = self.alloc_local(ty.clone(), Some(name.val.clone()));
        let binding = ValueBinding {
            shader: false,
            kind: ValueBindingKind::Local(local),
            ty: Some(ty),
            initialization: Initialization::Uninitialized,
        };
        self.environment.bind(declaration, binding);
        Ok(())
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
        let binding = self
            .environment
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

    fn complete_value(&mut self, declaration: DeclarationId, ty: Ty) {
        let kind = {
            let binding = self
                .environment
                .binding_mut(declaration)
                .expect("defined binding");
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

impl Scopes {
    pub(super) fn prepare(
        &mut self,
        file: &SourceFile,
        typer: &mut TyperContext,
        source_module: crate::ir::typecheck::SourceModuleId,
    ) -> Vec<GenerateError> {
        let mut errors = Vec::new();
        for stmt in file.declarations() {
            let result = match &stmt.val {
                StmtKind::Define { .. } | StmtKind::Declare { .. } | StmtKind::Expr { .. } => {
                    Err(GenerateError {
                        span: stmt.span,
                        kind: GenerateErrorKind::InvalidModuleItem,
                    })
                }
                StmtKind::ForeignType { name } => self
                    .define_alias(
                        name,
                        Ty::Foreign {
                            name: name.val.clone(),
                        },
                    )
                    .map(|_| ())
                    .map_err(|name| GenerateError {
                        span: stmt.span,
                        kind: GenerateErrorKind::DuplicateType { name },
                    }),
                _ => Ok(()),
            };
            if let Err(error) = result {
                errors.push(error);
            }
        }
        for stmt in file.declarations() {
            let result = match &stmt.val {
                StmtKind::DefineType { name, init } => match self.annotation(init, typer) {
                    Ok(ty) => {
                        self.define_alias(name, ty)
                            .map(|_| ())
                            .map_err(|name| GenerateError {
                                span: init.span,
                                kind: GenerateErrorKind::DuplicateType { name },
                            })
                    }
                    Err(error) => {
                        self.define_invalid_type(name);
                        Err(error)
                    }
                },
                StmtKind::Struct { name, body } => (|| {
                    let id = typer.declare_type(
                        name.val.clone(),
                        crate::ir::typecheck::SourceOrigin {
                            module: source_module,
                            span: name.span,
                        },
                    );
                    self.define_type(name, id).map_err(|name| GenerateError {
                        span: stmt.span,
                        kind: GenerateErrorKind::DuplicateType { name },
                    })?;
                    let ty = self.annotation(body, typer)?;
                    typer
                        .define_type(id, ty)
                        .map_err(|error| GenerateError::typing(body.span, error))
                })(),
                _ => Ok(()),
            };
            if let Err(error) = result {
                errors.push(error);
            }
        }
        errors
    }
    fn annotation(&mut self, ann: &Type, typer: &TyperContext) -> Result<Ty, GenerateError> {
        self.push_at(ann.span);
        let result = super::eval::Evaluator {
            scopes: self.view(),
            typer,
        }
        .ty(ann);
        self.pop();
        result
    }
}
