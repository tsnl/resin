use super::{GenerateError, Planner, Result, Type};
use crate::ir::typecheck::check_binding_name;
use crate::{
    ast::{self, Ident},
    ir::GenerateErrorKind,
};
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
                        .expect("builtin String")
                        .into());
                }
                self.scopes.resolve_type(name)
            },
        }
        .decode(ann, infer)
    }

    pub fn bind(&mut self, name: &Ident, ty: Type) -> Result<()> {
        check_binding_name(name)?;
        self.scopes
            .define_inferred(name, ty, false)
            .map_err(|duplicate| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateValue { name: duplicate },
            })
    }

    pub fn value(&mut self, name: &Ident) -> Result<Type> {
        self.scopes
            .lookup_inferred(&name.val)
            .map(|(ty, function)| {
                if function {
                    self.dependencies.insert(name.val.clone());
                }
                ty
            })
            .ok_or_else(|| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::UnboundValue {
                    name: name.val.clone(),
                },
            })
    }
}
