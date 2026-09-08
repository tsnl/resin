//! Walk expressions once, selecting emission operations and adding typing constraints together.
//! Emission operations run after their dependency group's types resolve.
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
    pub references: Vec<Ident>,
}
impl Annotation {
    pub(super) fn resolve(&self, g: &Generator) -> Result<Ty> {
        for name in &self.references {
            g.scopes.record_reference(name, true);
        }
        g.solver.require(&self.ty, self.span)
    }
}
pub(super) struct Signature {
    pub params: Vec<(Ident, Annotation)>,
    pub result: Annotation,
}
impl Planner<'_> {
    fn ann(&mut self, ann: &ast::Type, infer: bool) -> Result<Annotation> {
        let ty = self.annotation(ann, infer)?;
        Ok(Annotation {
            ty,
            span: ann.span,
            references: std::mem::take(&mut self.references),
        })
    }
    fn signature(
        &mut self,
        params: &[(Ident, ast::Type)],
        result: &ast::Type,
        infer: bool,
    ) -> Result<Signature> {
        let params = params
            .iter()
            .map(|(name, ann)| Ok((name.clone(), self.ann(ann, false)?)))
            .collect::<Result<Vec<_>>>()?;
        let result = self.ann(result, infer)?;
        Ok(Signature { params, result })
    }
}

pub(super) struct PlannedFile {
    pub signatures: BTreeMap<Arc<str>, Signature>,
    pub bodies: BTreeMap<Arc<str>, Term>,
    pub solver: Solver,
}
struct Planner<'a> {
    typing: Inference<'a>,
    scopes: Scopes,
    locals: Vec<BTreeMap<Arc<str>, Type>>,
    functions: BTreeMap<Arc<str>, Type>,
    dependencies: BTreeSet<Arc<str>>,
    holes: Vec<(Span, Type)>,
    references: Vec<Ident>,
    expressions: Vec<(Span, Type)>,
    result: Type,
    source_module: SourceModuleId,
}
impl<'a> Planner<'a> {
    fn new(typer: &'a mut TyperContext, scopes: Scopes, source_module: SourceModuleId) -> Self {
        Self {
            typing: Inference::new(typer),
            scopes,
            locals: vec![],
            functions: BTreeMap::new(),
            dependencies: BTreeSet::new(),
            holes: vec![],
            references: vec![],
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
) -> Result<PlannedFile> {
    let mut planner = Planner::new(typer, scopes.untraced(), source_module);
    let mut signatures = BTreeMap::new();
    let mut functions = vec![];
    for stmt in file.declarations() {
        let (name, params, result, infer) = match &stmt.val {
            StmtKind::Function {
                name,
                params,
                result,
                body,
                ..
            } => {
                functions.push((name, body));
                (name, params, result, true)
            }
            StmtKind::ForeignFunction {
                name,
                params,
                result,
                ..
            } => (name, params, result, false),
            _ => continue,
        };
        crate::ir::typecheck::check_binding_name(name)?;
        let signature = planner.signature(params, result, infer)?;
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
        if planner.functions.insert(name.val.clone(), ty).is_some() {
            return Err(GenerateError {
                span: name.span,
                kind: crate::ir::GenerateErrorKind::DuplicateValue {
                    name: name.val.clone(),
                },
            });
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
        planner.push();
        planner.result = signature.result.ty.clone();
        for (name, ann) in &signature.params {
            planner.bind(name, ann.ty.clone())?;
        }
        let term = planner.term(body, Some(signature.result.ty.clone()))?;
        planner.pop();
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
        planner.solve(&roots)?;
        for &i in &group {
            let result = &signatures[&functions[i].0.val].result;
            planner.solver.require(&result.ty, result.span)?;
            for (span, ty) in &expressions[i] {
                if let Err(error) = planner.solver.require(ty, *span) {
                    // Empty arrays need an element annotation even when discarded.
                    if matches!(planner.solver.head(ty), Type::Node(Head::Array(0), _)) {
                        return Err(GenerateError::typing(
                            *span,
                            crate::ir::TypeError {
                                kind: crate::ir::TypeErrorKind::EmptyArrayNeedsElementType,
                            },
                        ));
                    }
                    return Err(error);
                }
            }
        }
    }
    for (span, ty) in &planner.holes {
        planner.solver.require(ty, *span)?;
    }
    Ok(PlannedFile {
        signatures,
        bodies,
        solver: planner.typing.solver,
    })
}
