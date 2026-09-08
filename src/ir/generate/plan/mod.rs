//! Walk expressions once, selecting emission operations and adding typing constraints together.
//! Operations are replayable for cleanup, and run after their dependency group's types resolve.
mod context;
mod expressions;
mod groups;
use super::semantic::DefinitionKind;
use super::{
    GenerateError, Generator,
    scope::{ContextView, Cursor, DeclarationId, Scopes},
};
use crate::ir::typecheck::SourceModuleId;
use crate::ir::typecheck::infer::{
    Inference, Rule,
    solver::{Solver, VariableId},
    types::{Head, Type},
};
use crate::{
    ast::{self, Ident, SourceFile, Span, StmtKind},
    ir::{Ty, TyperContext},
};
use std::{
    collections::{BTreeMap, BTreeSet},
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
    rule: Rule,
    pub span: Span,
    pub context: Cursor,
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
    pub binding: Option<DeclarationId>,
    pub body: Term,
}
pub(super) struct Annotation {
    holes: Vec<(Span, VariableId)>,
    pub ty: Type,
    pub span: Span,
}
impl Annotation {
    pub(super) fn concrete(ty: Ty, span: Span) -> Self {
        Self {
            ty: ty.into(),
            span,
            holes: vec![],
        }
    }
    pub(super) fn resolve(&self, g: &Generator) -> Result<Ty> {
        g.solver.require(&self.ty, self.span)
    }
}
pub(super) struct Signature {
    pub declaration: Option<DeclarationId>,
    pub parameters: Vec<Option<DeclarationId>>,
    pub params: Vec<(Ident, Annotation)>,
    pub result: Annotation,
}
impl Planner<'_> {
    fn ann(&mut self, ann: &ast::Type, infer: bool) -> Annotation {
        let scoped = !matches!(ann.val, ast::TypeKind::Unit);
        if scoped {
            self.scopes.push_at(ann.span);
        }
        let checkpoint = self.typing.solver.clone();
        let result = super::annotation::Decoder {
            solver: &mut self.typing.solver,
            holes: Vec::new(),
            resolve: &mut |name| {
                if name.val.as_ref() == "String" {
                    return Ok(self
                        .typing
                        .typer
                        .string_type
                        .clone()
                        .expect("builtin String")
                        .into());
                }
                self.scopes.resolve_type(name)
            },
        }
        .decode(ann, infer);
        if scoped {
            self.scopes.pop();
        }
        let decoded = result.unwrap_or_else(|error| {
            self.errors.push(error);
            self.typing.solver = checkpoint;
            super::annotation::Decoded {
                ty: Type::Invalid,
                holes: Vec::new(),
            }
        });
        self.holes.extend_from_slice(&decoded.holes);
        Annotation {
            ty: decoded.ty,
            holes: decoded.holes,
            span: ann.span,
        }
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
        Signature {
            params,
            result,
            declaration: None,
            parameters: vec![],
        }
    }
}

pub(super) struct PlannedFile {
    pub context: ContextView,
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
    holes: Vec<(Span, VariableId)>,
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
pub(super) fn file(
    file: &SourceFile,
    typer: &mut TyperContext,
    scopes: Scopes,
    source_module: SourceModuleId,
    mut methods: BTreeMap<Arc<str>, DeclarationId>,
) -> PlannedFile {
    let mut planner = Planner::new(typer, scopes, source_module);
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
        let mut signature = planner.signature(params, result, body.is_some());
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
        let declared = if let Some(id) = methods.remove(&name.val) {
            planner.scopes.set_inferred(id, ty);
            Ok(id)
        } else {
            crate::ir::typecheck::check_binding_name(name).and_then(|()| {
                planner
                    .scopes
                    .define_inferred(name, ty, DefinitionKind::Function)
                    .map_err(|duplicate| GenerateError {
                        span: name.span,
                        kind: crate::ir::GenerateErrorKind::DuplicateValue { name: duplicate },
                    })
            })
        };
        match declared {
            Ok(id) => signature.declaration = Some(id),
            Err(error) => {
                planner.errors.push(error);
                continue;
            }
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
        let signature = signatures.get_mut(&name.val).unwrap();
        let errors_before = planner.errors.len();
        planner.scopes.push_at(body.span);
        planner.result = signature.result.ty.clone();
        for (name, ann) in &signature.params {
            let binding = planner
                .bind(name, ann.ty.clone(), DefinitionKind::Parameter)
                .map_err(|error| planner.errors.push(error))
                .ok();
            signature.parameters.push(binding);
        }
        let term = planner.term(body, Some(signature.result.ty.clone()));
        planner.scopes.pop();
        if planner.errors.len() != errors_before {
            planner.typing.fail(term.rule);
        }
        planner.typing.infer_from(
            &signature.result.holes,
            signature.result.span,
            term.ty.clone(),
        );
        bodies.insert(name.val.clone(), term);
        edges.push(
            std::mem::take(&mut planner.dependencies)
                .iter()
                .filter_map(|name| names.get(name).copied())
                .collect(),
        );
        constraints.push(std::mem::take(&mut planner.typing.constraints));
        expressions.push(std::mem::take(&mut planner.expressions));
    }
    for group in groups::groups(&edges) {
        let mut roots = vec![];
        for &i in &group {
            planner.typing.constraints.append(&mut constraints[i]);
            roots.extend(expressions[i].iter().map(|(_, ty)| ty.clone()));
            roots.push(signatures[&functions[i].0.val].result.ty.clone());
        }
        let errors = planner.typing.solve(&roots);
        planner.errors.extend(errors);
        for &i in &group {
            let result = &signatures[&functions[i].0.val].result;
            for (span, ty) in
                std::iter::once(&(result.span, result.ty.clone())).chain(expressions[i].iter())
            {
                if planner.typing.solver.invalid(ty) {
                    continue;
                }
                if let Err(mut error) = planner.typing.solver.require(ty, *span) {
                    if matches!(
                        planner.typing.solver.head(ty),
                        Type::Node(Head::Array(0), _)
                    ) {
                        error = GenerateError::typing(
                            *span,
                            crate::ir::TypeError {
                                kind: crate::ir::TypeErrorKind::EmptyArrayNeedsElementType,
                            },
                        );
                    }
                    planner.errors.push(error);
                    if ty == &result.ty {
                        for (_, variable) in &result.holes {
                            planner.typing.solver.invalidate(*variable);
                        }
                    }
                }
            }
        }
    }
    for (span, variable) in &planner.holes {
        let ty = variable.ty();
        if !planner.typing.solver.invalid(&ty)
            && let Err(error) = planner.typing.solver.require(&ty, *span)
        {
            planner.errors.push(error);
        }
    }
    planner
        .scopes
        .resolve_inferred(&planner.typing.solver, planner.typing.typer);
    PlannedFile {
        context: planner.scopes.finish(),
        signatures,
        bodies,
        solver: planner.typing.solver,
        errors: planner.errors,
    }
}
