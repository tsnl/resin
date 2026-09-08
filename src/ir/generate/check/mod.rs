//! First expression pass: resolve declarations and types, producing a typed tree.
//! No IR instructions or deferred code-generation operations are created here.
mod context;
mod expressions;
mod groups;
mod resolve;
#[cfg(test)]
mod tests;
use super::semantic::DefinitionKind;
use super::typed;
use super::{
    GenerateError,
    scope::{ContextView, DeclarationId, Scopes},
};
use crate::ir::typecheck::SourceModuleId;
use crate::ir::typecheck::infer::{
    Inference,
    solver::VariableId,
    types::{Head, Type},
};
use crate::{
    ast::{self, Ident, SourceFile, Span, StmtKind},
    ir::{Ty, TyperContext},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
type Result<T> = std::result::Result<T, GenerateError>;
type Term = typed::Term<Type>;
type Statement = typed::Statement<Type>;
type MatchArm = typed::MatchArm<Type>;
pub(super) struct Annotation {
    holes: Vec<(Span, VariableId)>,
    pub ty: Type,
    pub span: Span,
}
impl Annotation {
    fn into_tree(self) -> typed::Annotation<Type> {
        typed::Annotation {
            ty: self.ty,
            span: self.span,
        }
    }
}
pub(super) struct Signature {
    pub declaration: Option<DeclarationId>,
    pub parameters: Vec<Option<DeclarationId>>,
    pub params: Vec<(Ident, Annotation)>,
    pub result: Annotation,
}
impl Checker<'_> {
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

pub(super) struct CheckedFile {
    pub context: ContextView,
    pub signatures: BTreeMap<Arc<str>, typed::Signature>,
    pub bodies: BTreeMap<Arc<str>, typed::Term>,
    pub errors: Vec<GenerateError>,
}
struct Checker<'a> {
    typing: Inference<'a>,
    scopes: Scopes,
    errors: Vec<GenerateError>,
    dependencies: BTreeSet<Arc<str>>,
    holes: Vec<(Span, VariableId)>,
    expressions: Vec<(Span, Type)>,
    result: Type,
    source_module: SourceModuleId,
}
impl<'a> Checker<'a> {
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
) -> CheckedFile {
    let mut checker = Checker::new(typer, scopes, source_module);
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
        let mut signature = checker.signature(params, result, body.is_some());
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
            checker.scopes.set_inferred(id, ty);
            Ok(id)
        } else {
            crate::ir::typecheck::check_binding_name(name).and_then(|()| {
                checker
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
                checker.errors.push(error);
                continue;
            }
        }
        if matches!(&stmt.val, StmtKind::Function { decorators, .. }
            if decorators.len() == 1 && matches!(decorators[0].val.as_ref(),
                "compute_shader" | "vertex_shader" | "fragment_shader"))
        {
            checker.scopes.mark_shader(name);
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
        let errors_before = checker.errors.len();
        checker.scopes.push_at(body.span);
        checker.result = signature.result.ty.clone();
        for (name, ann) in &signature.params {
            let binding = checker
                .bind(name, ann.ty.clone(), DefinitionKind::Parameter)
                .map_err(|error| checker.errors.push(error))
                .ok();
            signature.parameters.push(binding);
        }
        let (rule, term) = checker.term(body, Some(signature.result.ty.clone()));
        checker.scopes.pop();
        if checker.errors.len() != errors_before {
            checker.typing.fail(rule);
        }
        checker.typing.infer_from(
            &signature.result.holes,
            signature.result.span,
            term.ty.clone(),
        );
        bodies.insert(name.val.clone(), term);
        edges.push(
            std::mem::take(&mut checker.dependencies)
                .iter()
                .filter_map(|name| names.get(name).copied())
                .collect(),
        );
        constraints.push(std::mem::take(&mut checker.typing.constraints));
        expressions.push(std::mem::take(&mut checker.expressions));
    }
    for group in groups::groups(&edges) {
        let mut roots = vec![];
        for &i in &group {
            checker.typing.constraints.append(&mut constraints[i]);
            roots.extend(expressions[i].iter().map(|(_, ty)| ty.clone()));
            roots.push(signatures[&functions[i].0.val].result.ty.clone());
        }
        let errors = checker.typing.solve(&roots);
        checker.errors.extend(errors);
        for &i in &group {
            let result = &signatures[&functions[i].0.val].result;
            for (span, ty) in
                std::iter::once(&(result.span, result.ty.clone())).chain(expressions[i].iter())
            {
                if checker.typing.solver.invalid(ty) {
                    continue;
                }
                if let Err(mut error) = checker.typing.solver.require(ty, *span) {
                    if matches!(
                        checker.typing.solver.head(ty),
                        Type::Node(Head::Array(0), _)
                    ) {
                        error = GenerateError::typing(
                            *span,
                            crate::ir::TypeError {
                                kind: crate::ir::TypeErrorKind::EmptyArrayNeedsElementType,
                            },
                        );
                    }
                    checker.errors.push(error);
                    if ty == &result.ty {
                        for (_, variable) in &result.holes {
                            checker.typing.solver.invalidate(*variable);
                        }
                    }
                }
            }
        }
    }
    for (span, variable) in &checker.holes {
        let ty = variable.ty();
        if !checker.typing.solver.invalid(&ty)
            && let Err(error) = checker.typing.solver.require(&ty, *span)
        {
            checker.errors.push(error);
        }
    }
    checker
        .scopes
        .resolve_inferred(&checker.typing.solver, checker.typing.typer);
    let mut checked_signatures = BTreeMap::new();
    for (name, signature) in signatures {
        let resolved = (|| {
            Ok(typed::Signature {
                declaration: signature.declaration,
                parameters: signature.parameters,
                params: signature
                    .params
                    .into_iter()
                    .map(|(name, ann)| Ok((name, ann.into_tree().resolve(&checker.typing.solver)?)))
                    .collect::<Result<_>>()?,
                result: signature
                    .result
                    .into_tree()
                    .resolve(&checker.typing.solver)?,
            })
        })();
        match resolved {
            Ok(signature) => {
                checked_signatures.insert(name, signature);
            }
            Err(error) => {
                if !checker.errors.contains(&error) {
                    checker.errors.push(error);
                }
            }
        }
    }
    let mut checked_bodies = BTreeMap::new();
    for (name, body) in bodies {
        if checker.typing.solver.invalid(&body.ty) {
            continue;
        }
        match body.resolve(&checker.typing.solver) {
            Ok(body) => {
                checked_bodies.insert(name, body);
            }
            Err(error) => {
                if !checker.errors.contains(&error) {
                    checker.errors.push(error);
                }
            }
        }
    }
    CheckedFile {
        context: checker.scopes.finish(),
        signatures: checked_signatures,
        bodies: checked_bodies,
        errors: checker.errors,
    }
}
