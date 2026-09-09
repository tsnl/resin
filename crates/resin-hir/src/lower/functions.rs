use super::{Generator, typed};
use crate::{Annotation, Function, Parameter, Signature};
use resin_common::prelude::*;

pub(super) fn signature(source: &typed::Signature) -> Signature {
    Signature {
        params: source
            .params
            .iter()
            .enumerate()
            .map(|(index, (name, a))| Parameter {
                binding: source.parameters.get(index).copied().flatten(),
                name: name.clone(),
                annotation: annotation(a),
            })
            .collect(),
        result: annotation(&source.result),
    }
}
pub(super) fn annotation(source: &typed::Annotation) -> Annotation {
    Annotation {
        ty: source.ty.clone(),
        span: source.span,
    }
}

impl Generator {
    pub(super) fn declare_function(
        &mut self,
        name: &Ident,
        source: &typed::Signature,
    ) -> Result<FunctionId, GenerateError> {
        check_parameters(source)?;
        let id = FunctionId::from_index(self.module.functions.len());
        let params = source.params.iter().map(|(_, a)| a.ty.clone()).collect();
        self.typer
            .register_function(id, params, source.result.ty.clone());
        self.module.origins.functions.insert(
            id,
            SourceLocation {
                path: self.source_path.clone(),
                span: name.span,
            },
        );
        self.module.functions.push(Function {
            name: name.val.clone(),
            signature: signature(source),
            body: None,
            foreign: None,
        });
        self.environment
            .bind(source.declaration.expect("checked function"), id);
        Ok(id)
    }
    pub(super) fn declare_foreign(
        &mut self,
        header: &str,
        name: &Ident,
        signature: &typed::Signature,
    ) -> Result<FunctionId, GenerateError> {
        let id = self.declare_function(name, signature)?;
        let params = self.typer.declared_function(id).params.clone();
        let foreign = Foreign {
            header: header.into(),
            params,
        };
        if !foreign.valid(&signature.result.ty) {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::InvalidForeignSignature,
            });
        }
        self.module.functions[id.index()].foreign = Some(foreign);
        Ok(id)
    }
}

fn check_parameters(signature: &typed::Signature) -> Result<(), GenerateError> {
    let mut names = std::collections::HashSet::new();
    for (name, _) in &signature.params {
        crate::lower::infer::check_binding_name(name)?;
        if !names.insert(&name.val) {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateValue {
                    name: name.val.clone(),
                },
            });
        }
    }
    Ok(())
}
