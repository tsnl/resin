//! Concrete typing and conversion rules shared by HIR construction and LIR verification.
//! This module neither traverses syntax nor owns inference variables.

use std::sync::Arc;

use crate::types::definitions;
use crate::types::{Ty, TypeDef, TypeId, TypeTable};

mod builtin;
mod convert;
mod error;
mod rules;

pub use builtin::BuiltinRule;

pub use convert::{Conv, Converted, ExplicitConversion};
pub use error::{TypeError, TypeErrorKind};
pub use rules::{BuiltinCall, FieldAccess};

/// Type IDs are local to this context's definition table.
#[derive(Debug, Clone, Default)]
pub struct TyperContext {
    definitions: TypeTable,
    pub(crate) string_type: Option<Ty>,
}

impl TyperContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_definitions(definitions: impl Into<TypeTable>) -> Self {
        let definitions = definitions.into();
        let string_type = definitions
            .iter()
            .enumerate()
            .find_map(|(index, definition)| {
                (definition
                    .name()
                    .is_some_and(|name| name.as_ref() == "String"))
                .then_some(Ty::Defined {
                    definition: TypeId::from_index(index),
                })
            });
        Self {
            definitions,
            string_type,
        }
    }

    pub fn definitions(&self) -> &[TypeDef] {
        &self.definitions
    }

    pub fn into_definitions(self) -> Result<TypeTable, TypeError> {
        for index in 0..self.definitions.len() {
            if self.definitions[index].name().is_some() {
                self.definition_body(TypeId::from_index(index))?;
            }
        }
        Ok(self.definitions)
    }

    pub fn create_type(
        &mut self,
        name: impl Into<Arc<str>>,
        body: Ty,
    ) -> Result<TypeId, TypeError> {
        let definition = self.reserve_type(name);
        if let Err(err) = self.define_type(definition, body) {
            self.definitions.pop_nominal();
            return Err(err);
        }
        Ok(definition)
    }

    pub fn reserve_type(&mut self, name: impl Into<Arc<str>>) -> TypeId {
        self.definitions.reserve(name.into())
    }

    pub fn define_type(&mut self, definition: TypeId, body: Ty) -> Result<(), TypeError> {
        if self.definition(definition)?.body().is_some() {
            return Err(TypeError::new(TypeErrorKind::TypeAlreadyDefined {
                definition,
            }));
        }
        definitions::check_references(&self.definitions, &body)?;
        definitions::check_layout(&self.definitions, definition, &body)?;
        self.definitions.define(definition, body);
        Ok(())
    }

    pub fn definition(&self, definition: TypeId) -> Result<&TypeDef, TypeError> {
        Ok(definitions::get(&self.definitions, definition)?)
    }

    fn definition_body(&self, definition: TypeId) -> Result<&Ty, TypeError> {
        Ok(definitions::body(&self.definitions, definition)?)
    }

    pub fn body(&self, ty: &Ty) -> Result<Ty, TypeError> {
        match ty {
            Ty::Defined { definition } => Ok(self.definition_body(*definition)?.clone()),
            other => Ok(other.clone()),
        }
    }
}

impl TyperContext {
    pub fn define_drop(&mut self, ty: TypeId, function: crate::types::FunctionId) {
        self.definitions.set_drop(ty, function);
    }
}

impl TyperContext {
    pub fn string_type(&self) -> Option<&Ty> {
        self.string_type.as_ref()
    }
    pub fn set_string_type(&mut self, ty: Ty) {
        self.string_type = Some(ty);
    }
}

/// Check a representation-preserving explicit conversion.
pub fn ascription(table: &[TypeDef], from: &Ty, to: &Ty) -> Result<Option<Vec<Conv>>, TypeError> {
    Ok(convert::ascription(table, from, to)?)
}

#[cfg(test)]
mod tests;
