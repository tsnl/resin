//! Bottom-up typing rules over an owned nominal type table.

use std::sync::Arc;

use crate::ir::types::definitions;
use crate::ir::{Ty, TypeDef, TypeId};

mod convert;
mod error;
mod rules;

pub use convert::{Conv, Converted};
pub use error::{TypeError, TypeErrorKind};
pub use rules::{BuiltinCall, FieldAccess};

/// Type IDs are local to this context's definition table.
#[derive(Debug, Clone, Default)]
pub struct TyperContext {
    definitions: Vec<TypeDef>,
}

impl TyperContext {
    pub const fn new() -> Self {
        Self {
            definitions: Vec::new(),
        }
    }

    pub fn from_definitions(definitions: Vec<TypeDef>) -> Self {
        Self { definitions }
    }

    pub fn definitions(&self) -> &[TypeDef] {
        &self.definitions
    }

    pub fn into_definitions(self) -> Result<Vec<TypeDef>, TypeError> {
        for index in 0..self.definitions.len() {
            self.definition_body(TypeId::from_index(index))?;
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
            self.definitions.pop();
            return Err(err);
        }
        Ok(definition)
    }

    pub fn reserve_type(&mut self, name: impl Into<Arc<str>>) -> TypeId {
        let definition = TypeId::from_index(self.definitions.len());
        self.definitions.push(TypeDef {
            name: name.into(),
            body: None,
        });
        definition
    }

    pub fn define_type(&mut self, definition: TypeId, body: Ty) -> Result<(), TypeError> {
        if self.definition(definition)?.body().is_some() {
            return Err(TypeError::new(TypeErrorKind::TypeAlreadyDefined {
                definition,
            }));
        }
        definitions::check_references(&self.definitions, &body)?;
        definitions::check_layout(&self.definitions, definition, &body)?;
        self.definitions[definition.index()].body = Some(body);
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
