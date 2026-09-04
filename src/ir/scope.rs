//! Lexical scopes used during IR generation.
//!
//! Value and type names stay in separate namespaces, but they share one
//! [`Scope`] frame so push/pop and lookup stay in lockstep.

use std::{collections::HashMap, sync::Arc};

use super::{GlobalId, LocalId, Ty, TypeId};

/// A value name resolved in the current environment.
#[derive(Clone)]
pub struct ValueBinding {
    pub kind: ValueBindingKind,
    pub ty: Option<Ty>,
    pub initializing: bool,
    pub depth: usize,
}

#[derive(Clone, Copy)]
pub enum ValueBindingKind {
    Global(GlobalId),
    Local(LocalId),
}

/// One lexical frame with the language's two namespaces.
#[derive(Clone, Default)]
pub struct Scope {
    values: HashMap<Arc<str>, ValueBinding>,
    types: HashMap<Arc<str>, TypeId>,
}

impl Scope {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn define_value(&mut self, name: Arc<str>, binding: ValueBinding) -> Result<(), Arc<str>> {
        if self.values.contains_key(&name) {
            return Err(name);
        }
        self.values.insert(name, binding);
        Ok(())
    }

    pub fn define_type(&mut self, name: Arc<str>, definition: TypeId) -> Result<(), Arc<str>> {
        if self.types.contains_key(&name) {
            return Err(name);
        }
        self.types.insert(name, definition);
        Ok(())
    }

    pub fn lookup_value(&self, name: &str) -> Option<&ValueBinding> {
        self.values.get(name)
    }

    pub fn lookup_type(&self, name: &str) -> Option<TypeId> {
        self.types.get(name).copied()
    }

    pub fn lookup_value_mut(&mut self, name: &str) -> Option<&mut ValueBinding> {
        self.values.get_mut(name)
    }
}

/// Nested lexical environments. Lookup walks from the innermost frame.
#[derive(Clone)]
pub struct Scopes {
    frames: Vec<Scope>,
}

impl Scopes {
    pub fn new() -> Self {
        Self {
            frames: vec![Scope::new()],
        }
    }

    pub fn push(&mut self) {
        self.frames.push(Scope::new());
    }

    pub fn pop(&mut self) {
        debug_assert!(self.frames.len() > 1, "cannot pop the outermost scope");
        self.frames.pop();
    }

    pub fn define_value(&mut self, name: Arc<str>, binding: ValueBinding) -> Result<(), Arc<str>> {
        self.innermost().define_value(name, binding)
    }

    pub fn define_type(&mut self, name: Arc<str>, definition: TypeId) -> Result<(), Arc<str>> {
        self.innermost().define_type(name, definition)
    }

    pub fn lookup_value(&self, name: &str) -> Option<&ValueBinding> {
        self.frames
            .iter()
            .rev()
            .find_map(|scope| scope.lookup_value(name))
    }

    pub fn lookup_type(&self, name: &str) -> Option<TypeId> {
        self.frames
            .iter()
            .rev()
            .find_map(|scope| scope.lookup_type(name))
    }

    pub fn lookup_value_mut(&mut self, name: &str) -> Option<&mut ValueBinding> {
        self.frames
            .iter_mut()
            .rev()
            .find_map(|scope| scope.lookup_value_mut(name))
    }

    fn innermost(&mut self) -> &mut Scope {
        self.frames.last_mut().expect("scope stack is never empty")
    }
}

impl Default for Scopes {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_and_types_do_not_clash() {
        let mut scope = Scope::new();
        let list = TypeId::from_index(0);
        scope.define_type("List".into(), list).unwrap();
        scope
            .define_value(
                "List".into(),
                ValueBinding {
                    kind: ValueBindingKind::Global(GlobalId::from_index(0)),
                    ty: None,
                    initializing: false,
                    depth: 0,
                },
            )
            .unwrap();
        assert_eq!(scope.lookup_type("List"), Some(list));
        assert!(scope.lookup_value("List").is_some());
    }

    #[test]
    fn inner_frame_shadows_outer_value() {
        let mut scopes = Scopes::new();
        scopes
            .define_value(
                "x".into(),
                ValueBinding {
                    kind: ValueBindingKind::Global(GlobalId::from_index(0)),
                    ty: None,
                    initializing: false,
                    depth: 0,
                },
            )
            .unwrap();
        scopes.push();
        scopes
            .define_value(
                "x".into(),
                ValueBinding {
                    kind: ValueBindingKind::Local(LocalId::from_index(0)),
                    ty: None,
                    initializing: false,
                    depth: 1,
                },
            )
            .unwrap();
        assert!(matches!(
            scopes.lookup_value("x").unwrap().kind,
            ValueBindingKind::Local(_)
        ));
        scopes.pop();
        assert!(matches!(
            scopes.lookup_value("x").unwrap().kind,
            ValueBindingKind::Global(_)
        ));
    }
}
