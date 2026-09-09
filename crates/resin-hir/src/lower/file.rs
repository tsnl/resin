//! Publish checked declarations, then elaborate bodies against their stable identities.
use super::check::CheckedFile;
use super::{Generator, functions, typed};
use resin_ast::{SourceFile, StmtKind};
use resin_common::prelude::*;
use std::collections::BTreeSet;

impl Generator {
    pub(super) fn declare_checked_functions(&mut self, file: &SourceFile, checked: &CheckedFile) {
        let mut declared = BTreeSet::new();
        for stmt in file.declarations() {
            let Some(name) = function_name(&stmt.val) else {
                continue;
            };
            let Some(signature) = checked.signatures.get(&name.val) else {
                continue;
            };
            if declared.insert(&name.val)
                && let Err(error) = self.declare_checked(&stmt.val, name, signature)
            {
                self.errors.push(error);
            }
        }
    }

    fn declare_checked(
        &mut self,
        stmt: &StmtKind,
        name: &Ident,
        signature: &typed::Signature,
    ) -> Result<(), GenerateError> {
        match stmt {
            StmtKind::Function { decorators, .. } => {
                if let Some(id) = self.function_identity(name, signature)? {
                    for decorator in decorators {
                        self.declare_shader(id, decorator)?;
                    }
                }
            }
            StmtKind::ForeignFunction { header, .. } => {
                self.declare_foreign(header, name, signature)?;
            }
            _ => unreachable!("function declaration"),
        }
        Ok(())
    }

    fn function_identity(
        &mut self,
        name: &Ident,
        signature: &typed::Signature,
    ) -> Result<Option<FunctionId>, GenerateError> {
        if name.val.contains('.') {
            return Ok(signature
                .declaration
                .and_then(|id| self.environment.binding(id)));
        }
        self.declare_function(name, signature).map(Some)
    }

    fn declare_shader(&mut self, id: FunctionId, decorator: &Ident) -> Result<(), GenerateError> {
        let stage = shader_stage(decorator)?;
        if self.module.shaders.contains_key(&id) {
            return Err(shader_error(
                decorator,
                "a function can have only one shader decorator",
            ));
        }
        let signature = &self.module.functions[id.index()].signature;
        resin_common::types::shader::validate(
            &self.typer,
            &signature.parameter_type(),
            &signature.result.ty,
            false,
            stage,
        )
        .map_err(|message| shader_error(decorator, &message))?;
        self.module.shaders.insert(
            id,
            resin_common::types::shader::ShaderEntry {
                stage: stage.into(),
                embedded: false,
            },
        );
        Ok(())
    }

    pub(super) fn elaborate_functions(&mut self, file: &SourceFile, checked: &CheckedFile) {
        for stmt in file.declarations() {
            let StmtKind::Function { name, .. } = &stmt.val else {
                continue;
            };
            let Some(body) = checked.bodies.get(&name.val) else {
                continue;
            };
            let Some(signature) = checked.signatures.get(&name.val) else {
                continue;
            };
            let Some(id) = self.environment.lookup_value(&name.val) else {
                continue;
            };
            self.elaborate_function(id, signature, body);
        }
    }

    fn elaborate_function(
        &mut self,
        id: FunctionId,
        signature: &typed::Signature,
        body: &typed::Term,
    ) {
        match self.elaborate(body) {
            Ok(body) => {
                self.module.functions[id.index()].signature = functions::signature(signature);
                self.module.functions[id.index()].body = Some(body);
            }
            Err(error) if !self.errors.contains(&error) => self.errors.push(error),
            Err(_) => {}
        }
    }
}

fn function_name(stmt: &StmtKind) -> Option<&Ident> {
    match stmt {
        StmtKind::Function { name, .. } | StmtKind::ForeignFunction { name, .. } => Some(name),
        _ => None,
    }
}

fn shader_stage(decorator: &Ident) -> Result<&'static str, GenerateError> {
    match decorator.val.as_ref() {
        "compute_shader" => Ok("compute"),
        "vertex_shader" => Ok("vertex"),
        "fragment_shader" => Ok("fragment"),
        _ => Err(shader_error(decorator, "unknown decorator")),
    }
}

fn shader_error(decorator: &Ident, message: &str) -> GenerateError {
    GenerateError {
        span: decorator.span,
        kind: GenerateErrorKind::InvalidShader {
            message: message.into(),
        },
    }
}
