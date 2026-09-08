//! Register inherent methods in the defining type's source namespace.
use super::{GenerateError, GenerateErrorKind, Generator, eval::Evaluator};
use super::{
    scope::{DeclarationId, Scopes},
    semantic::DefinitionKind,
    typed::{Annotation, Signature},
};
use crate::ast::{self, Ident, SourceFile, StmtKind};
use crate::lower::infer::types::Type;
use crate::types::{FunctionId, Ty, TypeId};
use std::{collections::BTreeMap, sync::Arc};

type Declarations = BTreeMap<Arc<str>, DeclarationId>;

impl Generator {
    pub(super) fn declare_methods(
        &mut self,
        file: &SourceFile,
        scopes: &mut Scopes,
    ) -> Declarations {
        let mut declarations = BTreeMap::new();
        for stmt in file.declarations() {
            if let Err(error) = self.declare_method(&stmt.val, scopes, &mut declarations) {
                self.errors.push(error);
            }
        }
        declarations
    }

    fn declare_method(
        &mut self,
        stmt: &StmtKind,
        scopes: &mut Scopes,
        declarations: &mut Declarations,
    ) -> Result<(), GenerateError> {
        let StmtKind::Function {
            receiver: Some(receiver),
            name,
            params,
            result,
            decorators,
            ..
        } = stmt
        else {
            return Ok(());
        };
        let declaration = reserve_method(name, scopes, declarations)?;
        let definition = self.method_owner(receiver, scopes)?;
        scopes.record_method_definition(definition, declaration);
        if !decorators.is_empty() {
            return Err(GenerateError::inference(
                name.span,
                "methods cannot be shader entries",
            ));
        }
        let evaluator = Evaluator {
            scopes: scopes.view(),
            typer: &self.typer,
        };
        let signature = method_signature(&evaluator, declaration, params, result)?;
        let function = self.declare_function(name, &signature)?;
        self.register_method(definition, name, function)
    }

    fn method_owner(&self, receiver: &Ident, scopes: &Scopes) -> Result<TypeId, GenerateError> {
        let evaluator = Evaluator {
            scopes: scopes.view(),
            typer: &self.typer,
        };
        let Ty::Defined { definition } = evaluator.type_name(receiver)? else {
            return Err(GenerateError::inference(
                receiver.span,
                "impl requires a nominal struct type",
            ));
        };
        if self
            .typer
            .type_origin(definition)
            .map(|origin| origin.module)
            != Some(self.source_module)
        {
            return Err(GenerateError::inference(
                receiver.span,
                "impl requires a type defined in this module",
            ));
        }
        Ok(definition)
    }

    fn register_method(
        &mut self,
        owner: TypeId,
        name: &Ident,
        function: FunctionId,
    ) -> Result<(), GenerateError> {
        let short = name.val.rsplit('.').next().unwrap();
        if !self.typer.define_method(owner, short.into(), function) {
            return Err(GenerateError::inference(name.span, "duplicate method"));
        }
        if short == "drop" {
            self.register_drop(owner, name, function)?;
        }
        Ok(())
    }

    fn register_drop(
        &mut self,
        owner: TypeId,
        name: &Ident,
        function: FunctionId,
    ) -> Result<(), GenerateError> {
        let declaration = self.typer.declared_function(function);
        let pointer = Ty::Pointer {
            pointee: Box::new(Ty::Defined { definition: owner }),
        };
        if declaration.params != [pointer] || declaration.result != Ty::Unit {
            return Err(GenerateError::inference(
                name.span,
                "drop must have signature drop(receiver: Ptr<T>) -> ()",
            ));
        }
        self.typer.define_drop(owner, function);
        Ok(())
    }
}

fn reserve_method(
    name: &Ident,
    scopes: &mut Scopes,
    declarations: &mut Declarations,
) -> Result<DeclarationId, GenerateError> {
    if declarations.contains_key(&name.val) {
        return Err(duplicate(name));
    }
    let declaration = scopes
        .define_inferred(name, Type::Invalid, DefinitionKind::Function)
        .map_err(|_| duplicate(name))?;
    declarations.insert(name.val.clone(), declaration);
    Ok(declaration)
}

fn duplicate(name: &Ident) -> GenerateError {
    GenerateError {
        span: name.span,
        kind: GenerateErrorKind::DuplicateValue {
            name: name.val.clone(),
        },
    }
}

fn method_signature(
    evaluator: &Evaluator<'_>,
    declaration: DeclarationId,
    params: &[(Ident, ast::Type)],
    result: &ast::Type,
) -> Result<Signature, GenerateError> {
    Ok(Signature {
        declaration: Some(declaration),
        parameters: vec![],
        params: params
            .iter()
            .map(|(name, ty)| Ok((name.clone(), annotation(evaluator, ty)?)))
            .collect::<Result<_, GenerateError>>()?,
        result: annotation(evaluator, result)?,
    })
}

fn annotation(evaluator: &Evaluator<'_>, source: &ast::Type) -> Result<Annotation, GenerateError> {
    Ok(Annotation {
        ty: evaluator.ty(source)?,
        span: source.span,
    })
}
