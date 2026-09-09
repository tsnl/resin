use crate::types::{LocalId, Ty};
use std::collections::HashMap;
pub(super) type DeclarationId = crate::hir::BindingId;

#[derive(Clone)]
pub(crate) struct ValueBinding {
    pub(super) local: LocalId,
    pub(crate) ty: Ty,
    pub(super) initialization: Initialization,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Initialization {
    Uninitialized,
    Initializing,
    Initialized,
}
/// Map resolved HIR bindings to storage and track their initialization across branches.
#[derive(Clone)]
pub(super) struct Environment {
    values: HashMap<DeclarationId, ValueBinding>,
}
impl Environment {
    pub(super) fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }
    pub(super) fn bind(&mut self, id: DeclarationId, binding: ValueBinding) {
        self.values.insert(id, binding);
    }
    pub(super) fn binding_mut(&mut self, id: DeclarationId) -> Option<&mut ValueBinding> {
        self.values.get_mut(&id)
    }
    pub(super) fn binding(&self, id: DeclarationId) -> Option<&ValueBinding> {
        self.values.get(&id)
    }
    pub(super) fn intersect_initialization(&mut self, other: &Self) {
        for (id, binding) in &mut self.values {
            if let Some(other) = other.values.get(id)
                && binding.initialization != other.initialization
            {
                binding.initialization = Initialization::Uninitialized;
            }
        }
    }
}
