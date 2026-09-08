use super::super::semantic::DefinitionKind;
use super::{GenerateError, Planner, Result, Type};
use crate::ir::typecheck::check_binding_name;
use crate::{ast::Ident, ir::GenerateErrorKind};
impl Planner<'_> {
    pub fn bind(
        &mut self,
        name: &Ident,
        ty: Type,
        kind: DefinitionKind,
    ) -> Result<super::super::scope::DeclarationId> {
        check_binding_name(name)?;
        self.scopes
            .define_inferred(name, ty, kind)
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
