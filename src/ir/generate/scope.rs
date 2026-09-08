//! Constructed lexical contexts and a separate mapping to emitted storage.
use crate::{
    ast::{Ident, SourceLocation, Span},
    ir::generate::semantic::{Definition, DefinitionKind, SemanticData},
    ir::{
        FunctionId, LocalId, Ty, TypeId, TyperContext,
        typecheck::infer::{solver::Solver, types::Type},
    },
};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    path::PathBuf,
    rc::Rc,
    sync::Arc,
};

pub(crate) type DeclarationId = usize;
#[derive(Clone, Copy)]
pub(super) struct Symbol {
    pub(super) definition: DeclarationId,
}
#[derive(Clone)]
pub(in crate::ir) struct ValueBinding {
    pub(super) kind: ValueBindingKind,
    pub(super) shader: bool,
    pub(in crate::ir) ty: Option<Ty>,
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
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Cursor {
    scope: usize,
    prefix: usize,
}
#[derive(Clone, Debug)]
struct Entry {
    name: Arc<str>,
    is_type: bool,
    definition: DeclarationId,
    visible_from: usize,
}
#[derive(Clone, Debug)]
struct Context {
    parent: Option<Cursor>,
    location: SourceLocation,
    entries: Vec<Entry>,
    children: Vec<usize>,
}
#[derive(Clone, Debug, Default)]
pub(crate) struct Contexts {
    scopes: Vec<Context>,
    pub definitions: Vec<Definition>,
    shaders: HashSet<DeclarationId>,
}
impl Contexts {
    fn lookup(&self, mut cursor: Cursor, name: &str, is_type: bool) -> Option<usize> {
        loop {
            let scope = &self.scopes[cursor.scope];
            let mut matches = scope.entries[..cursor.prefix]
                .iter()
                .filter(|entry| entry.name.as_ref() == name && entry.is_type == is_type)
                .map(|entry| entry.definition);
            if let Some(first) = matches.next() {
                return matches.all(|other| other == first).then_some(first);
            }
            cursor = scope.parent?;
        }
    }
    fn cursor_at(&self, path: &std::path::Path, offset: usize) -> Option<Cursor> {
        let mut id = self
            .scopes
            .iter()
            .position(|s| s.parent.is_none() && s.location.path == path)?;
        while let Some(child) = self.scopes[id].children.iter().rev().find(|&&child| {
            let span = self.scopes[child].location.span;
            span.start <= offset && offset <= span.end
        }) {
            id = *child;
        }
        let scope = &self.scopes[id];
        let prefix = scope
            .entries
            .iter()
            .take_while(|entry| entry.visible_from <= offset)
            .count();
        Some(Cursor { scope: id, prefix })
    }
    pub(crate) fn definition(
        &self,
        path: &std::path::Path,
        offset: usize,
        name: &str,
        is_type: bool,
    ) -> Option<&Definition> {
        let cursor = self.cursor_at(path, offset)?;
        self.lookup(cursor, name, is_type)
            .map(|id| &self.definitions[id])
    }
    pub(crate) fn visible(&self, path: &std::path::Path, offset: usize) -> Vec<Definition> {
        let Some(cursor) = self.cursor_at(path, offset) else {
            return vec![];
        };
        let mut names = BTreeMap::new();
        let mut at = Some(cursor);
        while let Some(view) = at {
            let scope = &self.scopes[view.scope];
            for entry in &scope.entries[..view.prefix] {
                if entry.name.is_empty() || self.definitions[entry.definition].member {
                    continue;
                }
                names
                    .entry((entry.name.clone(), entry.is_type))
                    .or_insert_with(|| self.lookup(cursor, &entry.name, entry.is_type));
            }
            at = scope.parent;
        }
        names
            .into_values()
            .flatten()
            .map(|id| self.definitions[id].clone())
            .collect()
    }
}

/// Lookup capability shared by construction, evaluation and emission.
#[derive(Clone)]
pub(in crate::ir) struct ContextView {
    cursor: Cursor,
    path: PathBuf,
    data: Rc<RefCell<SemanticData>>,
}
impl ContextView {
    pub(super) fn capture(&self) -> Cursor {
        self.cursor
    }
    pub(super) fn select(&mut self, cursor: Cursor) -> Cursor {
        std::mem::replace(&mut self.cursor, cursor)
    }
    fn lookup(&self, name: &str, is_type: bool) -> Option<DeclarationId> {
        self.data
            .borrow()
            .contexts
            .lookup(self.cursor, name, is_type)
    }
    pub(in crate::ir) fn resolve_type(&self, name: &Ident) -> Result<Type, super::GenerateError> {
        let id = self
            .lookup(&name.val, true)
            .ok_or_else(|| super::GenerateError {
                span: name.span,
                kind: super::GenerateErrorKind::UnboundType {
                    name: name.val.clone(),
                },
            })?;
        Ok(self.data.borrow().contexts.definitions[id]
            .ty
            .clone()
            .map(Type::from)
            .unwrap_or(Type::Invalid))
    }
}
/// The only capability that can construct contexts. Pending types belong to this module's solver.
pub(in crate::ir) struct Scopes {
    view: ContextView,
    inferred: HashMap<DeclarationId, (Type, bool)>,
    expressions: Vec<(SourceLocation, Type, bool)>,
}
impl Scopes {
    pub(super) fn for_source(path: PathBuf, data: Rc<RefCell<SemanticData>>) -> Self {
        let mut shared = data.borrow_mut();
        let scope = shared.contexts.scopes.len();
        shared.contexts.scopes.push(Context {
            parent: None,
            location: SourceLocation {
                path: path.clone(),
                span: Span {
                    start: 0,
                    end: usize::MAX,
                },
            },
            entries: vec![],
            children: vec![],
        });
        drop(shared);
        Self {
            view: ContextView {
                cursor: Cursor { scope, prefix: 0 },
                path,
                data,
            },
            inferred: HashMap::new(),
            expressions: vec![],
        }
    }
    pub(super) fn new() -> Self {
        Self::for_source(
            "<source>".into(),
            Rc::new(RefCell::new(SemanticData::default())),
        )
    }
    pub(super) fn view(&self) -> &ContextView {
        &self.view
    }
    pub(super) fn capture(&self) -> Cursor {
        self.view.capture()
    }
    pub(super) fn restore(&mut self, cursor: Cursor) {
        self.view.select(cursor);
    }
    pub(super) fn finish(self) -> ContextView {
        self.view
    }
    pub(in crate::ir) fn push_at(&mut self, span: Span) {
        let mut data = self.view.data.borrow_mut();
        let contexts = &mut data.contexts;
        let id = contexts.scopes.len();
        contexts.scopes.push(Context {
            parent: Some(self.view.cursor),
            location: SourceLocation {
                path: self.view.path.clone(),
                span,
            },
            entries: vec![],
            children: vec![],
        });
        contexts.scopes[self.view.cursor.scope].children.push(id);
        self.view.cursor = Cursor {
            scope: id,
            prefix: 0,
        };
    }
    pub(in crate::ir) fn pop(&mut self) {
        self.view.cursor = self.view.data.borrow().contexts.scopes[self.view.cursor.scope]
            .parent
            .expect("cannot pop root context");
    }
    fn declare(
        &mut self,
        name: &Ident,
        kind: DefinitionKind,
    ) -> (DeclarationId, Result<(), Arc<str>>) {
        let mut data = self.view.data.borrow_mut();
        let contexts = &mut data.contexts;
        let scope = &mut contexts.scopes[self.view.cursor.scope];
        let is_type = kind == DefinitionKind::Type;
        let duplicate = (is_type && name.val.as_ref() == "String")
            || scope.entries[..self.view.cursor.prefix]
                .iter()
                .any(|entry| entry.name == name.val && entry.is_type == is_type);
        let id = contexts.definitions.len();
        contexts.definitions.push(Definition {
            name: name.val.to_string(),
            location: SourceLocation {
                path: self.view.path.clone(),
                span: name.span,
            },
            kind,
            label: name.val.to_string(),
            member: kind == DefinitionKind::Function && name.val.contains('.'),
            ty: None,
        });
        let visible_from = if scope.parent.is_none() || kind == DefinitionKind::Parameter {
            scope.location.span.start
        } else {
            name.span.end
        };
        scope.entries.push(Entry {
            name: name.val.clone(),
            is_type,
            definition: id,
            visible_from,
        });
        self.view.cursor.prefix = scope.entries.len();
        (
            id,
            if duplicate {
                Err(name.val.clone())
            } else {
                Ok(())
            },
        )
    }
    pub(in crate::ir) fn define_inferred(
        &mut self,
        name: &Ident,
        ty: Type,
        kind: DefinitionKind,
    ) -> Result<DeclarationId, Arc<str>> {
        let (id, result) = self.declare(name, kind);
        self.inferred
            .insert(id, (ty, kind == DefinitionKind::Function));
        result.map(|()| id)
    }
    pub(in crate::ir) fn set_inferred(&mut self, id: DeclarationId, ty: Type) {
        let function =
            self.view.data.borrow().contexts.definitions[id].kind == DefinitionKind::Function;
        self.inferred.insert(id, (ty, function));
    }
    pub(in crate::ir) fn lookup_inferred(&self, name: &str) -> Option<(Type, bool)> {
        let id = self.view.lookup(name, false)?;
        Some(self.inferred.get(&id).cloned().unwrap_or_else(|| {
            (
                self.view.data.borrow().contexts.definitions[id]
                    .ty
                    .clone()
                    .map(Type::from)
                    .unwrap_or(Type::Invalid),
                false,
            )
        }))
    }
    pub(in crate::ir) fn resolve_type(&self, name: &Ident) -> Result<Type, super::GenerateError> {
        self.view.resolve_type(name)
    }
    pub(in crate::ir) fn mark_shader(&self, name: &Ident) {
        if let Some(id) = self.view.lookup(&name.val, false) {
            self.view.data.borrow_mut().contexts.shaders.insert(id);
        }
    }
    pub(in crate::ir) fn is_shader(&self, name: &str) -> bool {
        self.view
            .lookup(name, false)
            .is_some_and(|id| self.view.data.borrow().contexts.shaders.contains(&id))
    }
    pub(in crate::ir) fn record_inferred(&mut self, span: Span, ty: Type) {
        self.record_members(span, ty, false);
    }
    pub(in crate::ir) fn record_members(&mut self, span: Span, ty: Type, associated: bool) {
        self.expressions.push((
            SourceLocation {
                path: self.view.path.clone(),
                span,
            },
            ty,
            associated,
        ));
    }
    pub(super) fn record_method_definition(&mut self, receiver: TypeId, id: DeclarationId) {
        let mut data = self.view.data.borrow_mut();
        let definition = &mut data.contexts.definitions[id];
        definition.member = true;
        let name = definition.name.rsplit('.').next().unwrap().to_string();
        data.method_origins.entry((receiver, name)).or_insert(id);
    }
    pub(in crate::ir) fn resolve_inferred(&mut self, solver: &Solver, typer: &TyperContext) {
        let mut data = self.view.data.borrow_mut();
        for (id, (ty, _)) in self.inferred.drain() {
            data.contexts.definitions[id].ty = solver.resolve(&ty);
        }
        for (location, ty, associated) in self.expressions.drain(..) {
            if let Some(ty) = solver.resolve(&ty) {
                data.record_members(location, &ty, associated, typer);
            }
        }
        data.typer = typer.clone();
    }
    pub(super) fn import(&mut self, name: Arc<str>, symbol: Symbol) {
        let mut data = self.view.data.borrow_mut();
        let is_type = data.contexts.definitions[symbol.definition].kind == DefinitionKind::Type;
        let scope = &mut data.contexts.scopes[self.view.cursor.scope];
        scope.entries.push(Entry {
            name,
            is_type,
            definition: symbol.definition,
            visible_from: scope.location.span.start,
        });
        self.view.cursor.prefix = scope.entries.len();
    }
    pub(in crate::ir) fn define_invalid_type(&mut self, name: &Ident) {
        let _ = self.declare(name, DefinitionKind::Type);
    }
    pub(in crate::ir) fn define_type(
        &mut self,
        name: &Ident,
        definition: TypeId,
    ) -> Result<DeclarationId, Arc<str>> {
        self.define_alias(name, Ty::Defined { definition })
    }
    pub(in crate::ir) fn define_alias(
        &mut self,
        name: &Ident,
        ty: Ty,
    ) -> Result<DeclarationId, Arc<str>> {
        let (id, result) = self.declare(name, DefinitionKind::Type);
        self.view.data.borrow_mut().contexts.definitions[id].ty = Some(ty);
        result.map(|()| id)
    }
}
/// Emission can select retained contexts and map declarations to storage, but cannot declare names.
#[derive(Clone)]
pub(super) struct Environment {
    pub(super) context: ContextView,
    values: HashMap<DeclarationId, ValueBinding>,
}
impl Environment {
    pub(super) fn new() -> Self {
        Self {
            context: Scopes::new().finish(),
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
    pub(super) fn symbol(&self, name: &str) -> Option<Symbol> {
        self.context
            .lookup(name, false)
            .or_else(|| self.context.lookup(name, true))
            .map(|definition| Symbol { definition })
    }
    pub(super) fn lookup_value(&self, name: &str) -> Option<&ValueBinding> {
        self.values.get(&self.context.lookup(name, false)?)
    }
    pub(super) fn lookup_value_mut(&mut self, name: &str) -> Option<&mut ValueBinding> {
        let id = self.context.lookup(name, false)?;
        self.values.get_mut(&id)
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
