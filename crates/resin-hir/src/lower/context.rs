//! Source-only type namespaces layered over the shared concrete type rules.
use super::namespaces::{FunctionDecl, MethodDefinitions, Namespace};
use resin_common::prelude::*;

use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut},
};

#[derive(Debug, Clone, Default)]
pub(crate) struct Context {
    pub(super) typer: TyperContext,
    pub(super) namespaces: BTreeMap<TypeId, Namespace>,
    pub(super) functions: BTreeMap<FunctionId, FunctionDecl>,
    pub(super) method_definitions: Vec<MethodDefinitions>,
}
impl Context {
    pub(crate) fn receiver_definition(&self, ty: &Ty) -> Option<TypeId> {
        let mut ty = ty;
        while let Some(pointee) = ty.deref_target() {
            ty = pointee;
        }
        let Ty::Defined { definition } = ty else {
            return None;
        };
        Some(*definition)
    }

    pub fn new() -> Self {
        Self::default()
    }
    pub fn into_definitions(self) -> Result<TypeTable, TypeError> {
        self.typer.into_definitions()
    }
}
impl Deref for Context {
    type Target = TyperContext;
    fn deref(&self) -> &TyperContext {
        &self.typer
    }
}
impl DerefMut for Context {
    fn deref_mut(&mut self) -> &mut TyperContext {
        &mut self.typer
    }
}
