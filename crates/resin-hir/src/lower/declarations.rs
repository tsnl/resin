use super::scope::Scopes;
use crate::lower::context::Context;
use resin_ast::{SourceFile, StmtKind, Type};
use resin_common::prelude::*;

impl Scopes {
    pub(super) fn prepare(
        &mut self,
        file: &SourceFile,
        typer: &mut Context,
        source_module: crate::lower::namespaces::SourceModuleId,
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
                        crate::lower::namespaces::SourceOrigin {
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
    fn annotation(&mut self, ann: &Type, typer: &Context) -> Result<Ty, GenerateError> {
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
