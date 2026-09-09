use super::scope::{ContextView, DeclarationId, Scopes, Symbol};
use resin_common::prelude::*;

use std::collections::HashMap;

pub(super) struct Environment {
    pub context: ContextView,
    functions: HashMap<DeclarationId, FunctionId>,
}
impl Environment {
    pub fn new() -> Self {
        Self {
            context: Scopes::new().finish(),
            functions: HashMap::new(),
        }
    }
    pub fn bind(&mut self, id: DeclarationId, function: FunctionId) {
        self.functions.insert(id, function);
    }
    pub fn binding(&self, id: DeclarationId) -> Option<FunctionId> {
        self.functions.get(&id).copied()
    }
    pub fn symbol(&self, name: &str) -> Option<Symbol> {
        self.context
            .lookup(name, false)
            .or_else(|| self.context.lookup(name, true))
            .map(|definition| Symbol { definition })
    }
    pub fn lookup_value(&self, name: &str) -> Option<FunctionId> {
        self.binding(self.context.lookup(name, false)?)
    }
}
