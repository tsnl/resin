//! Bottom-up typing rules over an owned nominal type table.

use std::sync::Arc;

use crate::ir::types::definitions;
use crate::ir::{Ty, TypeDef, TypeId, TypeTable};

mod convert;
mod error;
mod rules;

pub use convert::{Conv, Converted};
pub use error::{TypeError, TypeErrorKind};
pub use rules::{BuiltinCall, FieldAccess};

/// Type IDs are local to this context's definition table.
#[derive(Debug, Clone, Default)]
pub struct TyperContext {
    definitions: TypeTable,
}

impl TyperContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_definitions(definitions: impl Into<TypeTable>) -> Self {
        Self {
            definitions: definitions.into(),
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

#[cfg(test)]
mod tests;
