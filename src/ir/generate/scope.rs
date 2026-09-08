//! Lexical scopes with separate value and type namespaces.

use std::{collections::HashMap, sync::Arc};

use crate::ir::{FunctionId, LocalId, Ty, TypeId};
use crate::{
    analysis::semantic::Trace,
    ast::{Ident, SourceLocation},
    ir::TyperContext,
};

#[derive(Clone)]
pub(super) enum Symbol {
    Value(ValueBinding),
    Type(Ty),
}

#[derive(Clone)]
pub(super) struct ValueBinding {
    pub(super) kind: ValueBindingKind,
    pub(super) shader: bool,
    pub(super) ty: Option<Ty>,
    pub(super) initialization: Initialization,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Initialization {
    Uninitialized,
    Initializing,
    Initialized,
}

#[derive(Clone, Copy)]
pub(super) enum ValueBindingKind {
    Local(LocalId),
    Function(FunctionId),
}

#[derive(Clone, Default)]
struct Scope {
    values: HashMap<Arc<str>, ValueBinding>,
    types: HashMap<Arc<str>, Ty>,
    origins: HashMap<(Arc<str>, bool), SourceLocation>,
}

impl Scope {
    fn new() -> Self {
        Self::default()
    }

    fn define_value(&mut self, name: Arc<str>, binding: ValueBinding) -> Result<(), Arc<str>> {
        if self.values.contains_key(&name) {
            return Err(name);
        }
        self.values.insert(name, binding);
        Ok(())
    }

    fn define_type(&mut self, name: Arc<str>, ty: Ty) -> Result<(), Arc<str>> {
        if self.types.contains_key(&name) {
            return Err(name);
        }
        self.types.insert(name, ty);
        Ok(())
    }

    fn lookup_value(&self, name: &str) -> Option<&ValueBinding> {
        self.values.get(name)
    }

    fn lookup_type(&self, name: &str) -> Option<Ty> {
        self.types.get(name).cloned()
    }

    fn lookup_value_mut(&mut self, name: &str) -> Option<&mut ValueBinding> {
        self.values.get_mut(name)
    }
}

#[derive(Clone)]
pub(super) struct Scopes {
    frames: Vec<Scope>,
    trace: Option<Trace>,
}

impl Scopes {
    pub(super) fn untraced(&self) -> Self {
        Self {
            trace: None,
            ..self.clone()
        }
    }

    pub(super) fn traced(trace: Trace) -> Self {
        Self {
            trace: Some(trace),
            ..Self::new()
        }
    }

    pub(super) fn record_import(&mut self, name: Arc<str>, is_type: bool, origin: SourceLocation) {
        if self.trace.is_some() {
            self.innermost().origins.insert((name, is_type), origin);
        }
    }

    pub(super) fn record_definition(
        &mut self,
        name: &Ident,
        is_type: bool,
        ty: Option<&Ty>,
        typer: &TyperContext,
    ) {
        if let Some(trace) = &self.trace {
            let location = trace.location(name.span);
            if let Some(ty) = ty {
                trace.typed(location.clone(), ty, typer);
            }
            self.innermost()
                .origins
                .insert((name.val.clone(), is_type), location);
        }
    }

    pub(super) fn record_reference(&self, name: &Ident, is_type: bool) {
        if let Some(trace) = &self.trace
            && let Some(origin) = self.origin(&name.val, is_type)
        {
            trace
                .data
                .borrow_mut()
                .references
                .insert(trace.location(name.span), origin.clone());
        }
    }

    pub(super) fn record_fields(&self, name: &Ident, ty: &Ty, typer: &TyperContext) {
        if let Some(trace) = &self.trace {
            trace.record_fields(trace.location(name.span), ty, typer);
        }
    }

    pub(super) fn record_method_definition(&self, receiver: TypeId, name: &Ident) {
        if let Some(trace) = &self.trace {
            trace.data.borrow_mut().method_origins.insert(
                (receiver, name.val.rsplit('.').next().unwrap().to_string()),
                trace.location(name.span),
            );
        }
    }

    pub(super) fn record_method(
        &self,
        name: &Ident,
        ty: &Ty,
        associated: bool,
        typer: &TyperContext,
    ) {
        if let Some(trace) = &self.trace {
            trace.record_method(name, ty, associated, typer);
        }
    }

    pub(super) fn record_binding_type(&self, name: &Arc<str>, ty: &Ty, typer: &TyperContext) {
        if let Some(trace) = &self.trace
            && let Some(origin) = self.origin(name, false)
        {
            trace.typed(origin.clone(), ty, typer);
        }
    }

    fn origin(&self, name: &Arc<str>, is_type: bool) -> Option<&SourceLocation> {
        self.frames
            .iter()
            .rev()
            .find_map(|frame| frame.origins.get(&(name.clone(), is_type)))
    }

    pub(super) fn symbol(&self, name: &str) -> Option<Symbol> {
        self.lookup_value(name)
            .cloned()
            .map(Symbol::Value)
            .or_else(|| self.lookup_type(name).map(Symbol::Type))
    }

    pub(super) fn import(&mut self, name: Arc<str>, symbol: Symbol) {
        let result = match symbol {
            Symbol::Value(binding) => self.innermost().define_value(name, binding),
            Symbol::Type(ty) => self.innermost().define_type(name, ty),
        };
        result.expect("import conflicts were checked");
    }

    pub(super) fn new() -> Self {
        Self {
            frames: vec![Scope::new()],
            trace: None,
        }
    }

    pub(super) fn push(&mut self) {
        self.frames.push(Scope::new());
    }

    pub(super) fn pop(&mut self) {
        debug_assert!(self.frames.len() > 1, "cannot pop the outermost scope");
        self.frames.pop();
    }

    pub(super) fn define_value(
        &mut self,
        name: Arc<str>,
        binding: ValueBinding,
    ) -> Result<(), Arc<str>> {
        self.innermost().define_value(name, binding)
    }

    pub(super) fn define_type(
        &mut self,
        name: Arc<str>,
        definition: TypeId,
    ) -> Result<(), Arc<str>> {
        self.innermost()
            .define_type(name, Ty::Defined { definition })
    }

    pub(super) fn define_foreign_type(&mut self, name: Arc<str>) -> Result<(), Arc<str>> {
        self.innermost()
            .define_type(name.clone(), Ty::Foreign { name })
    }

    pub(super) fn define_alias(&mut self, name: Arc<str>, ty: Ty) -> Result<(), Arc<str>> {
        self.innermost().define_type(name, ty)
    }

    pub(super) fn lookup_value(&self, name: &str) -> Option<&ValueBinding> {
        self.frames
            .iter()
            .rev()
            .find_map(|scope| scope.lookup_value(name))
    }

    pub(super) fn lookup_type(&self, name: &str) -> Option<Ty> {
        self.frames
            .iter()
            .rev()
            .find_map(|scope| scope.lookup_type(name))
    }

    pub(super) fn lookup_value_mut(&mut self, name: &str) -> Option<&mut ValueBinding> {
        self.frames
            .iter_mut()
            .rev()
            .find_map(|scope| scope.lookup_value_mut(name))
    }

    /// A binding is definitely initialized only if both control-flow paths initialize it.
    pub(super) fn intersect_initialization(&mut self, other: &Self) {
        for (frame, other) in self.frames.iter_mut().zip(&other.frames) {
            for (name, binding) in &mut frame.values {
                if let Some(other) = other.values.get(name)
                    && binding.initialization != other.initialization
                {
                    binding.initialization = Initialization::Uninitialized;
                }
            }
        }
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
        scope
            .define_type("List".into(), Ty::Defined { definition: list })
            .unwrap();
        scope
            .define_value(
                "List".into(),
                ValueBinding {
                    shader: false,
                    kind: ValueBindingKind::Function(FunctionId::from_index(0)),
                    ty: None,
                    initialization: Initialization::Initialized,
                },
            )
            .unwrap();
        assert_eq!(
            scope.lookup_type("List"),
            Some(Ty::Defined { definition: list })
        );
        assert!(scope.lookup_value("List").is_some());
    }

    #[test]
    fn inner_frame_shadows_outer_value() {
        let mut scopes = Scopes::new();
        scopes
            .define_value(
                "x".into(),
                ValueBinding {
                    shader: false,
                    kind: ValueBindingKind::Function(FunctionId::from_index(0)),
                    ty: None,
                    initialization: Initialization::Initialized,
                },
            )
            .unwrap();
        scopes.push();
        scopes
            .define_value(
                "x".into(),
                ValueBinding {
                    shader: false,
                    kind: ValueBindingKind::Local(LocalId::from_index(0)),
                    ty: None,
                    initialization: Initialization::Initialized,
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
            ValueBindingKind::Function(_)
        ));
    }
}
