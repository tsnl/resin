use super::{GenerateError, Planner, Result, Type};
use crate::ir::typecheck::check_binding_name;
use crate::{
    ast::{self, Ident},
    ir::GenerateErrorKind,
};
use std::collections::BTreeMap;
impl Planner<'_> {
    pub fn annotation(&mut self, ann: &ast::Type, infer: bool) -> Result<Type> {
        super::super::annotation::Decoder {
            solver: &mut self.typing.solver,
            holes: &mut self.holes,
            resolve: &mut |name| {
                if name.val.as_ref() == "String" {
                    return Ok(self
                        .typing
                        .typer
                        .string_type
                        .clone()
                        .expect("builtin String"));
                }
                self.references.push(name.clone());
                self.scopes
                    .lookup_type(&name.val)
                    .ok_or_else(|| GenerateError {
                        span: name.span,
                        kind: GenerateErrorKind::UnboundType {
                            name: name.val.clone(),
                        },
                    })
            },
        }
        .decode(ann, infer)
    }

    pub fn push(&mut self) {
        self.scopes.push();
        self.locals.push(BTreeMap::new());
    }
    pub fn pop(&mut self) {
        self.scopes.pop();
        self.locals.pop();
    }

    pub fn bind(&mut self, name: &Ident, ty: Type) -> Result<()> {
        check_binding_name(name)?;
        let frame = self.locals.last_mut().unwrap();
        if frame.insert(name.val.clone(), ty).is_some() {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateValue {
                    name: name.val.clone(),
                },
            });
        }
        Ok(())
    }

    pub fn value(&mut self, name: &Ident) -> Result<Type> {
        for frame in self.locals.iter().rev() {
            if let Some(ty) = frame.get(&name.val) {
                return Ok(ty.clone());
            }
        }
        if let Some(ty) = self.functions.get(&name.val) {
            self.dependencies.insert(name.val.clone());
            return Ok(ty.clone());
        }
        self.scopes
            .lookup_value(&name.val)
            .and_then(|b| b.ty.clone())
            .map(Type::from)
            .ok_or_else(|| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::UnboundValue {
                    name: name.val.clone(),
                },
            })
    }
}
