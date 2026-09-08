//! Editor-only semantic recovery. `None` is an unknown type, never unit or a
//! wildcard accepted by the compiler. This pass emits no IR and reuses the
//! compiler's scopes, type evaluator, conversions, and expression typing rules.
use super::{
    eval::Evaluator,
    scope::{Initialization, Scopes, Symbol, ValueBinding, ValueBindingKind},
};
use crate::{
    analysis::semantic::{SemanticData, Trace},
    ast::{Ident, Program, SourceLocation, Span, Stmt, StmtKind, Term, TermKind, Type},
    ir::{LocalId, RecordField, Ty, TyperContext},
};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc, sync::Arc};

pub(crate) fn analyze(program: &Program) -> SemanticData {
    let data = Rc::new(RefCell::new(SemanticData::default()));
    let mut typer = TyperContext::new();
    let mut exports: Vec<BTreeMap<Arc<str>, (Symbol, SourceLocation)>> = Vec::new();
    for source in &program.modules {
        let trace = Trace {
            path: source.path.clone(),
            data: data.clone(),
        };
        let mut pass = Recovery {
            scopes: Scopes::traced(trace.clone()),
            typer: &mut typer,
            trace,
        };
        let mut imports = BTreeMap::new();
        for &dependency in &source.imports {
            for (name, (symbol, origin)) in &exports[dependency] {
                let entry = imports
                    .entry(name.clone())
                    .or_insert_with(|| Some((symbol.clone(), origin.clone())));
                if entry
                    .as_ref()
                    .is_some_and(|(_, previous)| previous != origin)
                {
                    *entry = None;
                }
            }
        }
        for (name, (symbol, origin)) in imports
            .into_iter()
            .filter_map(|(name, binding)| binding.map(|b| (name, b)))
        {
            pass.scopes.import(name.clone(), symbol.clone());
            pass.scopes
                .record_import(name, matches!(symbol, Symbol::Type(_)), origin);
        }
        for stmt in source.file.declarations() {
            match &stmt.val {
                StmtKind::ForeignType { name } => {
                    let _ = pass.scopes.define_foreign_type(name.val.clone());
                    pass.scopes.record_definition(name, true, None, pass.typer);
                }
                StmtKind::DefineType { .. } | StmtKind::Struct { .. } => pass.statement(stmt),
                _ => {}
            }
        }
        for stmt in source.file.declarations() {
            if let StmtKind::Function {
                name,
                params,
                result,
                ..
            }
            | StmtKind::ForeignFunction {
                name,
                params,
                result,
                ..
            } = &stmt.val
            {
                let params: Option<Vec<_>> = params.iter().map(|(_, ann)| pass.ty(ann)).collect();
                let result = pass.ty(result);
                let ty = params.zip(result).map(|(p, r)| Ty::Function {
                    param: Box::new(Ty::parameter(&p)),
                    result: Box::new(r),
                });
                pass.bind(name, ty);
                if matches!(&stmt.val, StmtKind::Function { decorators, .. } if decorators.len() == 1 && matches!(decorators[0].val.as_ref(), "compute_shader" | "vertex_shader" | "fragment_shader"))
                    && let Some(binding) = pass.scopes.lookup_value_mut(&name.val)
                {
                    binding.shader = true;
                }
            }
        }
        for stmt in source.file.declarations() {
            if let StmtKind::Function {
                params,
                result,
                body,
                ..
            } = &stmt.val
            {
                pass.scopes.push();
                for (name, ann) in params {
                    let ty = pass.ty(ann);
                    pass.bind(name, ty);
                }
                let expected = pass.ty(result);
                pass.term(body, expected.as_ref());
                pass.scopes.pop();
            }
        }
        let mut exported = BTreeMap::new();
        for name in &source.file.exports {
            if let Some(symbol) = pass.scopes.symbol(&name.val) {
                pass.scopes
                    .record_reference(name, matches!(symbol, Symbol::Type(_)));
                if let Some(origin) = data
                    .borrow()
                    .references
                    .get(&pass.trace.location(name.span))
                    .cloned()
                {
                    exported.insert(name.val.clone(), (symbol, origin));
                }
            }
        }
        exports.push(exported);
    }
    data.borrow().clone()
}

struct Recovery<'a> {
    scopes: Scopes,
    typer: &'a mut TyperContext,
    trace: Trace,
}
impl Recovery<'_> {
    fn properties(&self, base: &Term, ty: Ty) -> Ty {
        if matches!(&base.val, TermKind::Var { name } if self.scopes.lookup_value(&name.val).is_some_and(|binding| binding.shader))
        {
            Ty::shader_properties()
        } else {
            ty
        }
    }

    fn ty(&self, ty: &Type) -> Option<Ty> {
        Evaluator {
            scopes: &self.scopes,
            typer: self.typer,
            checked: None,
        }
        .ty(ty)
        .ok()
    }
    fn bind(&mut self, name: &Ident, ty: Option<Ty>) -> bool {
        if name.val.is_empty() {
            return false;
        }
        let binding = ValueBinding {
            shader: false,
            kind: ValueBindingKind::Local(LocalId::from_index(0)),
            ty: ty.clone(),
            initialization: Initialization::Initialized,
        };
        if self.scopes.define_value(name.val.clone(), binding).is_ok() {
            self.scopes
                .record_definition(name, false, ty.as_ref(), self.typer);
            if ty.is_none() {
                self.trace
                    .data
                    .borrow_mut()
                    .types
                    .insert(self.trace.location(name.span), "?".into());
            }
            true
        } else {
            false
        }
    }
    fn statement(&mut self, stmt: &Stmt) {
        match &stmt.val {
            StmtKind::Define { name, init } => {
                // Shadow immediately, including a broken/self-referential initializer.
                let bound = self.bind(name, None);
                let ty = self.term(init, None);
                if !bound {
                    return;
                }
                if let Some(binding) = self.scopes.lookup_value_mut(&name.val) {
                    binding.ty = ty.clone();
                }
                if let Some(ty) = ty {
                    self.scopes.record_binding_type(&name.val, &ty, self.typer);
                }
            }
            StmtKind::Declare { name, ann } => {
                let ty = self.ty(ann);
                self.bind(name, ty);
            }
            StmtKind::Struct { name, body: init } => {
                if name.val.is_empty() {
                    return;
                }
                let id = self.typer.reserve_type(name.val.clone());
                if self.scopes.define_type(name.val.clone(), id).is_ok() {
                    self.scopes.record_definition(name, true, None, self.typer);
                    if let Some(body) = self.ty(init) {
                        let _ = self.typer.define_type(id, body);
                    }
                }
            }
            StmtKind::Expr { term } => {
                self.term(term, None);
            }
            StmtKind::Defer { body } => {
                self.term(body, None);
            }
            StmtKind::DefineType { name, init } => {
                if let Some(ty) = self.ty(init)
                    && self
                        .scopes
                        .define_alias(name.val.clone(), ty.clone())
                        .is_ok()
                {
                    self.scopes
                        .record_definition(name, true, Some(&ty), self.typer);
                }
            }
            _ => {}
        }
    }
    fn term(&mut self, term: &Term, expected: Option<&Ty>) -> Option<Ty> {
        let found = match &term.val {
            TermKind::Unwrap { value } => self.term(value, None).and_then(|ty| ty.without_none()),
            TermKind::Try { value } => match self.term(value, None) {
                Some(Ty::Result { value, .. }) => Some(*value),
                _ => None,
            },
            TermKind::Match { value, arms } => {
                let input = self.term(value, None);
                let mut output = expected.cloned();
                for arm in arms {
                    self.scopes.push();
                    let payload = match (&input, &arm.variant) {
                        (Some(Ty::Result { value, .. }), crate::ast::MatchVariant::Ok) => {
                            Some(*value.clone())
                        }
                        (Some(Ty::Result { error, .. }), crate::ast::MatchVariant::Err) => {
                            Some(*error.clone())
                        }
                        (_, crate::ast::MatchVariant::Type(ann)) => self.ty(ann),
                        _ => None,
                    };
                    if let Some(name) = &arm.name {
                        self.bind(name, payload);
                    }
                    let ty = self.term(&arm.body, output.as_ref());
                    if output.is_none() {
                        output = ty;
                    }
                    self.scopes.pop();
                }
                output
            }
            TermKind::Hole { children } => {
                for child in children {
                    self.term(child, None);
                }
                None
            }
            TermKind::FieldHole { base } => {
                if let Some(ty) = self.term(base, None) {
                    let ty = self.properties(base, ty);
                    self.trace.record_fields(
                        self.trace.location(Span {
                            start: term.span.end,
                            end: term.span.end,
                        }),
                        &ty,
                        self.typer,
                    );
                }
                None
            }
            TermKind::None => Some(Ty::None),
            TermKind::Unit => Some(Ty::Unit),
            TermKind::Num { value } => Evaluator {
                scopes: &self.scopes,
                typer: self.typer,
                checked: None,
            }
            .number(term.span, value, expected)
            .ok()
            .map(|(_, ty)| ty),
            TermKind::String { value } => Some(Ty::Array {
                element: Box::new(Ty::UInt8),
                length: value.len(),
            }),
            TermKind::Var { name } => {
                self.scopes.record_reference(name, false);
                self.scopes
                    .lookup_value(&name.val)
                    .and_then(|b| b.ty.clone())
            }
            TermKind::Type { ty } => self.ty(ty).map(|_| Ty::Type),
            TermKind::Block { stmts, tail } => {
                self.scopes.push();
                for stmt in stmts {
                    self.statement(stmt);
                }
                let ty = self.term(tail, expected);
                self.scopes.pop();
                ty
            }
            TermKind::Field { base, name } => {
                let ty = self.term(base, None);
                ty.and_then(|ty| {
                    let ty = self.properties(base, ty);
                    self.trace
                        .record_fields(self.trace.location(name.span), &ty, self.typer);
                    self.typer.type_field(&ty, &name.val).ok().map(|f| f.ty)
                })
            }
            TermKind::Address { place } => self.term(place, None).map(|ty| Ty::Pointer {
                pointee: Box::new(ty),
            }),
            TermKind::Deref { pointer } => self
                .term(pointer, None)
                .and_then(|ty| self.typer.type_deref(&ty).ok()),
            TermKind::Record { fields } => {
                let expected_fields =
                    expected
                        .and_then(|t| self.typer.body(t).ok())
                        .and_then(|t| {
                            if let Ty::Record { fields } = t {
                                Some(fields)
                            } else {
                                None
                            }
                        });
                let typed: Vec<_> = fields
                    .iter()
                    .map(|(name, value)| {
                        let expected = expected_fields
                            .as_ref()
                            .and_then(|fs| fs.iter().find(|f| f.name == name.val))
                            .map(|f| &f.ty);
                        self.term(value, expected).map(|ty| RecordField {
                            name: name.val.clone(),
                            ty,
                        })
                    })
                    .collect();
                typed
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .and_then(|fs| self.typer.type_record(&fs).ok())
            }
            TermKind::Array { elems } => {
                let element = expected
                    .and_then(|t| self.typer.body(t).ok())
                    .and_then(|t| {
                        if let Ty::Array { element, .. } = t {
                            Some(*element)
                        } else {
                            None
                        }
                    });
                let typed: Vec<_> = elems
                    .iter()
                    .map(|e| self.term(e, element.as_ref()))
                    .collect();
                typed
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .and_then(|ts| match element {
                        Some(e) => self.typer.type_array_of(&e, &ts).ok(),
                        None => self.typer.type_array(&ts).ok(),
                    })
            }
            TermKind::Builtin { name, args } => {
                let typed: Vec<_> = args.iter().map(|arg| self.term(arg, None)).collect();
                typed
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .and_then(|ts| {
                        self.typer
                            .type_builtin_call(name, &ts)
                            .ok()
                            .map(|b| b.result)
                    })
            }
            TermKind::Call { func, arg } => {
                if let TermKind::Type { ty } = &func.val {
                    let target = self.ty(ty);
                    let shape = target.as_ref().and_then(|t| self.typer.body(t).ok());
                    let value = self.term(arg, shape.as_ref());
                    target
                        .zip(value)
                        .and_then(|(t, v)| self.typer.type_ascription(&t, &v).ok())
                } else {
                    let callee = self.term(func, None);
                    let param = callee.as_ref().and_then(|c| {
                        if let Ty::Function { param, .. } = c {
                            Some(*param.clone())
                        } else {
                            None
                        }
                    });
                    let argument = self.term(arg, param.as_ref());
                    callee
                        .zip(argument)
                        .and_then(|(f, a)| self.typer.type_call(&f, &a).ok())
                }
            }
            TermKind::Assign { place, value } => {
                let place = self.term(place, None);
                let value = self.term(value, place.as_ref());
                place
                    .zip(value)
                    .and_then(|(p, v)| self.typer.type_assign(&p, &v).ok())
            }
            TermKind::If { cond, then, els } => {
                let cond = self.term(cond, Some(&Ty::Bool));
                let then = self.term(then, expected);
                let els = self.term(els, expected);
                cond.zip(then)
                    .zip(els)
                    .and_then(|((c, t), e)| self.typer.type_if(&c, &t, &e).ok())
            }
            TermKind::While { cond, body } => {
                let cond = self.term(cond, Some(&Ty::Bool));
                self.term(body, None);
                cond.map(|_| Ty::Unit)
            }
        };
        found.filter(|ty| expected.is_none_or(|e| self.typer.same(e, ty).is_ok()))
    }
}
