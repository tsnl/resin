//! Retained lexical contexts, with separate inference and storage enrichment.
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

#[derive(Clone)]
pub(super) struct Symbol {
    definition: usize,
    pub(super) binding: Option<ValueBinding>,
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
    definition: usize,
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
    inferred: HashMap<usize, (Type, bool)>,
    shaders: HashSet<usize>,
    expressions: Vec<(SourceLocation, Type, bool)>,
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
        let prefix = if scope.parent.is_none() {
            scope.entries.len()
        } else {
            scope
                .entries
                .iter()
                .take_while(|entry| {
                    let definition = &self.definitions[entry.definition];
                    matches!(
                        definition.kind,
                        DefinitionKind::Parameter | DefinitionKind::Function
                    ) || definition.location.span.end <= offset
                })
                .count()
        };
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
                if self.definitions[entry.definition].member {
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

#[derive(Clone)]
pub(in crate::ir) struct Scopes {
    cursor: Cursor,
    path: std::path::PathBuf,
    data: Rc<RefCell<SemanticData>>,
    values: HashMap<usize, ValueBinding>,
    reuse: bool,
}
impl Scopes {
    pub(in crate::ir) fn planning(&self) -> Self {
        Self {
            reuse: false,
            ..self.clone()
        }
    }
    pub(in crate::ir) fn lowering(&mut self) {
        self.reuse = true;
        self.cursor.prefix = self.data.borrow().contexts.scopes[self.cursor.scope]
            .entries
            .len();
    }
    pub(super) fn for_source(path: PathBuf, shared: Rc<RefCell<SemanticData>>) -> Self {
        let mut data = shared.borrow_mut();
        let id = data.contexts.scopes.len();
        data.contexts.scopes.push(Context {
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
        drop(data);
        Self {
            cursor: Cursor {
                scope: id,
                prefix: 0,
            },
            path,
            data: shared,
            values: HashMap::new(),
            reuse: false,
        }
    }
    pub(super) fn new() -> Self {
        Self::for_source(
            "<source>".into(),
            Rc::new(RefCell::new(SemanticData::default())),
        )
    }
    pub(in crate::ir) fn push_at(&mut self, span: Span) {
        let mut data = self.data.borrow_mut();
        let contexts = &mut data.contexts;
        let found = contexts.scopes[self.cursor.scope]
            .children
            .iter()
            .copied()
            .find(|&id| contexts.scopes[id].location.span == span);
        let id = found.unwrap_or_else(|| {
            let id = contexts.scopes.len();
            contexts.scopes.push(Context {
                parent: Some(self.cursor),
                location: SourceLocation {
                    path: self.path.clone(),
                    span,
                },
                entries: vec![],
                children: vec![],
            });
            contexts.scopes[self.cursor.scope].children.push(id);
            id
        });
        self.cursor = Cursor {
            scope: id,
            prefix: 0,
        };
    }
    pub(in crate::ir) fn pop(&mut self) {
        self.cursor = self.data.borrow().contexts.scopes[self.cursor.scope]
            .parent
            .expect("cannot pop root context");
    }
    fn declare(&mut self, name: Arc<str>, is_type: bool) -> Result<usize, Arc<str>> {
        let mut data = self.data.borrow_mut();
        let contexts = &mut data.contexts;
        let scope = &contexts.scopes[self.cursor.scope];
        let duplicate = (is_type && name.as_ref() == "String")
            || !self.reuse
                && scope.entries[..self.cursor.prefix]
                    .iter()
                    .any(|e| e.name == name && e.is_type == is_type);
        let existing = if self.reuse {
            scope
                .entries
                .iter()
                .position(|e| e.name == name && e.is_type == is_type)
        } else {
            None
        };
        let id = if let Some(index) = existing {
            self.cursor.prefix = self.cursor.prefix.max(index + 1);
            scope.entries[index].definition
        } else {
            let id = contexts.definitions.len();
            contexts.definitions.push(Definition {
                name: name.to_string(),
                location: SourceLocation {
                    path: self.path.clone(),
                    span: Span { start: 0, end: 0 },
                },
                kind: if is_type {
                    DefinitionKind::Type
                } else {
                    DefinitionKind::Variable
                },
                label: name.to_string(),
                member: false,
                ty: None,
            });
            contexts.scopes[self.cursor.scope].entries.push(Entry {
                name: name.clone(),
                is_type,
                definition: id,
            });
            id
        };
        if existing.is_none() {
            self.cursor.prefix = contexts.scopes[self.cursor.scope].entries.len();
        }
        if duplicate { Err(name) } else { Ok(id) }
    }
    fn lookup(&self, name: &str, is_type: bool) -> Option<usize> {
        self.data
            .borrow()
            .contexts
            .lookup(self.cursor, name, is_type)
    }
    pub(in crate::ir) fn define_inferred(
        &mut self,
        name: &Ident,
        ty: Type,
        function: bool,
    ) -> Result<(), Arc<str>> {
        if name.val.is_empty() {
            return Ok(());
        }
        if function && let Some(id) = self.lookup(&name.val, false) {
            let mut data = self.data.borrow_mut();
            let definition = &mut data.contexts.definitions[id];
            if definition.location.path == self.path && definition.location.span == name.span {
                definition.kind = DefinitionKind::Function;
                data.contexts.inferred.insert(id, (ty, true));
                return Ok(());
            }
        }
        let result = self.declare(name.val.clone(), false);
        let mut data = self.data.borrow_mut();
        let id = data.contexts.scopes[self.cursor.scope].entries[self.cursor.prefix - 1].definition;
        let definition = &mut data.contexts.definitions[id];
        definition.location.span = name.span;
        if function {
            definition.kind = DefinitionKind::Function;
            definition.member = name.val.contains('.');
        }
        data.contexts.inferred.insert(id, (ty, function));
        result.map(|_| ())
    }
    pub(in crate::ir) fn mark_shader(&self, name: &Ident) {
        if let Some(id) = self.lookup(&name.val, false) {
            self.data.borrow_mut().contexts.shaders.insert(id);
        }
    }
    pub(in crate::ir) fn is_shader(&self, name: &str) -> bool {
        self.lookup(name, false)
            .is_some_and(|id| self.data.borrow().contexts.shaders.contains(&id))
    }
    pub(in crate::ir) fn define_invalid_type(&mut self, name: &Ident) {
        let _ = self.declare(name.val.clone(), true);
        let mut data = self.data.borrow_mut();
        let id = data.contexts.scopes[self.cursor.scope].entries[self.cursor.prefix - 1].definition;
        data.contexts.definitions[id].location.span = name.span;
    }
    pub(in crate::ir) fn set_inferred(&mut self, name: &Ident, ty: Type) {
        if let Some(id) = self.lookup(&name.val, false) {
            self.data
                .borrow_mut()
                .contexts
                .inferred
                .insert(id, (ty, false));
        }
    }
    pub(in crate::ir) fn lookup_inferred(&self, name: &str) -> Option<(Type, bool)> {
        let id = self.lookup(name, false)?;
        let data = self.data.borrow();
        Some(data.contexts.inferred.get(&id).cloned().unwrap_or_else(|| {
            (
                data.contexts.definitions[id]
                    .ty
                    .clone()
                    .map(Type::from)
                    .unwrap_or(Type::Invalid),
                false,
            )
        }))
    }
    pub(in crate::ir) fn set_definition_kind(&mut self, name: &Ident, kind: DefinitionKind) {
        if let Some(id) = self.lookup(&name.val, kind == DefinitionKind::Type) {
            self.data.borrow_mut().contexts.definitions[id].kind = kind;
        }
    }
    pub(in crate::ir) fn record_inferred(&self, span: Span, ty: Type) {
        self.record_members(span, ty, false);
    }
    pub(in crate::ir) fn record_members(&self, span: Span, ty: Type, associated: bool) {
        self.data.borrow_mut().contexts.expressions.push((
            SourceLocation {
                path: self.path.clone(),
                span,
            },
            ty,
            associated,
        ));
    }
    pub(super) fn record_method_definition(&mut self, receiver: TypeId, name: &Ident) {
        let id = self
            .lookup(&name.val, false)
            .filter(|&id| {
                let data = self.data.borrow();
                let location = &data.contexts.definitions[id].location;
                location.path == self.path && location.span == name.span
            })
            .unwrap_or_else(|| {
                let _ = self.declare(name.val.clone(), false);
                self.data.borrow().contexts.scopes[self.cursor.scope].entries
                    [self.cursor.prefix - 1]
                    .definition
            });
        let mut data = self.data.borrow_mut();
        let definition = &mut data.contexts.definitions[id];
        definition.location.span = name.span;
        definition.kind = DefinitionKind::Function;
        definition.member = true;
        data.method_origins
            .entry((receiver, name.val.rsplit('.').next().unwrap().to_string()))
            .or_insert(id);
    }
    pub(in crate::ir) fn resolve_inferred(&self, solver: &Solver, typer: &TyperContext) {
        let mut data = self.data.borrow_mut();
        let inferred: Vec<_> = data
            .contexts
            .inferred
            .iter()
            .filter(|(id, _)| data.contexts.definitions[**id].location.path == self.path)
            .filter_map(|(&id, (ty, _))| solver.resolve(ty).map(|ty| (id, ty)))
            .collect();
        for (id, ty) in inferred {
            data.contexts.definitions[id].ty = Some(ty);
        }
        let expressions: Vec<_> = data
            .contexts
            .expressions
            .iter()
            .filter(|(location, _, _)| location.path == self.path)
            .filter_map(|(location, ty, associated)| {
                solver
                    .resolve(ty)
                    .map(|ty| (location.clone(), ty, *associated))
            })
            .collect();
        for (location, ty, associated) in expressions {
            data.record_members(location, &ty, associated, typer);
        }
        data.contexts
            .expressions
            .retain(|(location, _, _)| location.path != self.path);
        let finished: Vec<_> = data
            .contexts
            .inferred
            .keys()
            .copied()
            .filter(|id| data.contexts.definitions[*id].location.path == self.path)
            .collect();
        for id in finished {
            data.contexts.inferred.remove(&id);
        }
        data.typer = typer.clone();
    }
    pub(super) fn record_definition(
        &self,
        name: &Ident,
        is_type: bool,
        ty: Option<&Ty>,
        typer: &TyperContext,
    ) {
        if let Some(id) = self.lookup(&name.val, is_type) {
            let mut data = self.data.borrow_mut();
            let definition = &mut data.contexts.definitions[id];
            definition.location.span = name.span;
            definition.ty = ty.cloned().or_else(|| definition.ty.clone());
            data.typer = typer.clone();
        }
    }
    pub(super) fn record_binding_type(&self, name: &Arc<str>, ty: &Ty, typer: &TyperContext) {
        if let Some(id) = self.lookup(name, false) {
            let mut data = self.data.borrow_mut();
            data.contexts.definitions[id].ty = Some(ty.clone());
            data.typer = typer.clone();
        }
    }
    pub(super) fn symbol(&self, name: &str) -> Option<Symbol> {
        let definition = self
            .lookup(name, false)
            .or_else(|| self.lookup(name, true))?;
        Some(Symbol {
            definition,
            binding: self.values.get(&definition).cloned(),
        })
    }
    pub(super) fn import(&mut self, name: Arc<str>, symbol: Symbol) {
        let mut data = self.data.borrow_mut();
        let is_type = data.contexts.definitions[symbol.definition].kind == DefinitionKind::Type;
        let scope = &mut data.contexts.scopes[self.cursor.scope];
        scope.entries.push(Entry {
            name,
            is_type,
            definition: symbol.definition,
        });
        self.cursor.prefix = scope.entries.len();
        if let Some(binding) = symbol.binding {
            self.values.insert(symbol.definition, binding);
        }
    }
    pub(super) fn define_value(
        &mut self,
        name: Arc<str>,
        binding: ValueBinding,
    ) -> Result<(), Arc<str>> {
        let id = self.declare(name, false)?;
        self.values.insert(id, binding);
        Ok(())
    }
    pub(in crate::ir) fn define_type(
        &mut self,
        name: Arc<str>,
        definition: TypeId,
    ) -> Result<(), Arc<str>> {
        self.define_alias(name, Ty::Defined { definition })
    }
    pub(super) fn define_foreign_type(&mut self, name: Arc<str>) -> Result<(), Arc<str>> {
        self.define_alias(name.clone(), Ty::Foreign { name })
    }
    pub(in crate::ir) fn define_alias(&mut self, name: Arc<str>, ty: Ty) -> Result<(), Arc<str>> {
        let id = self.declare(name, true)?;
        self.data.borrow_mut().contexts.definitions[id].ty = Some(ty);
        Ok(())
    }
    pub(in crate::ir) fn lookup_value(&self, name: &str) -> Option<&ValueBinding> {
        self.values.get(&self.lookup(name, false)?)
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
    pub(super) fn lookup_value_mut(&mut self, name: &str) -> Option<&mut ValueBinding> {
        let id = self.lookup(name, false)?;
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
impl Default for Scopes {
    fn default() -> Self {
        Self::new()
    }
}
