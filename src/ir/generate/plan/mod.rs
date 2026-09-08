//! Walk expressions once, selecting emission operations and adding typing constraints together.
//! Operations are replayable for cleanup, and run after their dependency group's types resolve.
mod context;
mod expressions;
mod groups;
use super::{GenerateError, Generator, scope::Scopes};
use crate::ir::typecheck::SourceModuleId;
use crate::ir::typecheck::infer::{
    Inference,
    solver::Solver,
    types::{Head, Type},
};
use crate::{
    ast::{self, Ident, SourceFile, Span, StmtKind},
    ir::{Ty, TyperContext},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::Arc,
};
pub(super) fn error(span: Span, message: impl Into<Arc<str>>) -> GenerateError {
    GenerateError::inference(span, message)
}
type Result<T> = std::result::Result<T, GenerateError>;
type Emit = Rc<dyn Fn(&mut Generator, &Ty) -> Result<Ty>>;
pub(super) type Statement = Rc<dyn Fn(&mut Generator) -> Result<()>>;
#[derive(Clone)]
pub(super) struct Term {
    pub span: Span,
    pub ty: Type,
    pub form: Form,
    pub emit: Emit,
}
// Only addressability and literal syntax survive planning; there is no second AST dispatcher.
#[derive(Clone)]
pub(super) enum Form {
    Var { name: Ident },
    Field { base: Rc<Term>, name: Ident },
    Deref { pointer: Rc<Term> },
    Num { value: Arc<str> },
    Unit,
    Record,
    Type { ty: Type },
    Other,
}
pub(super) struct MatchArm {
    pub variant: Option<Annotation>,
    pub failure: bool,
    pub name: Option<Ident>,
    pub body: Term,
}
pub(super) struct Annotation {
    pub ty: Type,
    pub span: Span,
}
impl Annotation {
    pub(super) fn resolve(&self, g: &Generator) -> Result<Ty> {
        g.solver.require(&self.ty, self.span)
    }
}
pub(super) struct Signature {
    pub params: Vec<(Ident, Annotation)>,
    pub result: Annotation,
}
impl Planner<'_> {
    fn ann(&mut self, ann: &ast::Type, infer: bool) -> Annotation {
        let scoped = !matches!(ann.val, ast::TypeKind::Unit);
        if scoped {
            self.scopes.push_at(ann.span);
        }
        let checkpoint = self.solver.clone();
        let holes = self.holes.len();
        let result = self.annotation(ann, infer);
        if scoped {
            self.scopes.pop();
        }
        let ty = result.unwrap_or_else(|error| {
            self.errors.push(error);
            self.typing.solver = checkpoint;
            self.holes.truncate(holes);
            Type::Invalid
        });
        if self.output != Type::Invalid {
            self.constrain((
                ann.span,
                crate::ir::typecheck::infer::constraints::Constraint::Depends(ty.clone()),
            ));
        }
        Annotation { ty, span: ann.span }
    }
    fn signature(
        &mut self,
        params: &[(Ident, ast::Type)],
        result: &ast::Type,
        infer: bool,
    ) -> Signature {
        let params = params
            .iter()
            .map(|(name, ann)| (name.clone(), self.ann(ann, false)))
            .collect();
        let result = self.ann(result, infer);
        Signature { params, result }
    }
}

pub(super) struct PlannedFile {
    pub signatures: BTreeMap<Arc<str>, Signature>,
    pub bodies: BTreeMap<Arc<str>, Term>,
    pub solver: Solver,
    pub errors: Vec<GenerateError>,
}
struct Planner<'a> {
    typing: Inference<'a>,
    scopes: Scopes,
    errors: Vec<GenerateError>,
    dependencies: BTreeSet<Arc<str>>,
    holes: Vec<(Span, Type)>,
    expressions: Vec<(Span, Type)>,
    result: Type,
    source_module: SourceModuleId,
}
impl<'a> Planner<'a> {
    fn new(typer: &'a mut TyperContext, scopes: Scopes, source_module: SourceModuleId) -> Self {
        Self {
            typing: Inference::new(typer),
            scopes,
            errors: vec![],
            dependencies: BTreeSet::new(),
            holes: vec![],
            expressions: vec![],
            result: Ty::Unit.into(),
            source_module,
        }
    }
}
impl<'a> Deref for Planner<'a> {
    type Target = Inference<'a>;
    fn deref(&self) -> &Self::Target {
        &self.typing
    }
}
impl DerefMut for Planner<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.typing
    }
}
pub(super) fn file(
    file: &SourceFile,
    typer: &mut TyperContext,
    scopes: &Scopes,
    source_module: SourceModuleId,
) -> PlannedFile {
    let mut planner = Planner::new(typer, scopes.planning(), source_module);
    let mut signatures = BTreeMap::new();
    let mut functions = vec![];
    for stmt in file.declarations() {
        let (name, params, result, body) = match &stmt.val {
            StmtKind::Function {
                name,
                params,
                result,
                body,
                ..
            } => (name, params, result, Some(body)),
            StmtKind::ForeignFunction {
                name,
                params,
                result,
                ..
            } => (name, params, result, None),
            _ => continue,
        };
        let signature = planner.signature(params, result, body.is_some());
        let ty = Type::function(
            Type::parameter(
                signature
                    .params
                    .iter()
                    .map(|(_, ann)| ann.ty.clone())
                    .collect(),
            ),
            signature.result.ty.clone(),
        );
        let declared = crate::ir::typecheck::check_binding_name(name).and_then(|()| {
            planner
                .scopes
                .define_inferred(name, ty, true)
                .map_err(|duplicate| GenerateError {
                    span: name.span,
                    kind: crate::ir::GenerateErrorKind::DuplicateValue { name: duplicate },
                })
        });
        if let Err(error) = declared {
            planner.errors.push(error);
            continue;
        }
        if matches!(&stmt.val, StmtKind::Function { decorators, .. }
            if decorators.len() == 1 && matches!(decorators[0].val.as_ref(),
                "compute_shader" | "vertex_shader" | "fragment_shader"))
        {
            planner.scopes.mark_shader(name);
        }
        if let Some(body) = body {
            functions.push((name, body));
        }
        signatures.insert(name.val.clone(), signature);
    }
    let names: BTreeMap<_, _> = functions
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.val.clone(), i))
        .collect();
    let mut bodies = BTreeMap::new();
    let mut edges = vec![];
    let mut constraints = vec![];
    let mut expressions = vec![];
    for (name, body) in &functions {
        let signature = &signatures[&name.val];
        let errors_before = planner.errors.len();
        planner.scopes.push_at(body.span);
        planner.result = signature.result.ty.clone();
        for (name, ann) in &signature.params {
            if let Err(error) = planner.bind(name, ann.ty.clone()) {
                planner.errors.push(error);
            }
            planner.scopes.set_definition_kind(
                name,
                crate::ir::generate::semantic::DefinitionKind::Parameter,
            );
        }
        let term = planner.term(body, Some(signature.result.ty.clone()));
        planner.scopes.pop();
        if planner.errors.len() != errors_before {
            planner.solver.invalidate(&term.ty);
        }
        // Signature holes belong to the body that proves them, even when a
        // recursive caller has already constrained those holes on a prior attempt.
        planner.constraints.push((
            signature.result.span,
            signature.result.ty.clone(),
            crate::ir::typecheck::infer::constraints::Constraint::Depends(term.ty.clone()),
        ));
        bodies.insert(name.val.clone(), term);
        edges.push(
            std::mem::take(&mut planner.dependencies)
                .iter()
                .filter_map(|name| names.get(name).copied())
                .collect(),
        );
        constraints.push(std::mem::take(&mut planner.constraints));
        expressions.push(std::mem::take(&mut planner.expressions));
    }
    for group in groups::groups(&edges) {
        let mut roots = vec![];
        for &i in &group {
            planner.constraints.append(&mut constraints[i]);
            roots.extend(expressions[i].iter().map(|(_, ty)| ty.clone()));
            roots.push(signatures[&functions[i].0.val].result.ty.clone());
        }
        let errors = planner.solve(&roots);
        planner.errors.extend(errors);
        for &i in &group {
            let result = &signatures[&functions[i].0.val].result;
            for (span, ty) in
                std::iter::once(&(result.span, result.ty.clone())).chain(expressions[i].iter())
            {
                if planner.solver.invalid(ty) {
                    continue;
                }
                if let Err(mut error) = planner.solver.require(ty, *span) {
                    if matches!(planner.solver.head(ty), Type::Node(Head::Array(0), _)) {
                        error = GenerateError::typing(
                            *span,
                            crate::ir::TypeError {
                                kind: crate::ir::TypeErrorKind::EmptyArrayNeedsElementType,
                            },
                        );
                    }
                    planner.errors.push(error);
                    if ty == &result.ty {
                        planner.solver.invalidate(&result.ty);
                    }
                }
            }
        }
    }
    for (span, ty) in &planner.holes {
        if !planner.solver.invalid(ty)
            && let Err(error) = planner.solver.require(ty, *span)
        {
            planner.errors.push(error);
        }
    }
    planner
        .scopes
        .resolve_inferred(&planner.solver, planner.typer);
    PlannedFile {
        signatures,
        bodies,
        solver: planner.typing.solver,
        errors: planner.errors,
    }
}
