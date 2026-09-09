//! Declare all signatures, check bodies, solve dependency groups, then resolve the tree.
use super::*;
use crate::lower::infer::{Equation, solver::Solver, types::Head};
use resin_ast::{SourceFile, StmtKind};

type Signatures = BTreeMap<Arc<str>, Signature>;
type SourceBodies<'a> = Vec<(&'a Ident, &'a resin_ast::Term)>;

struct Body {
    name: Arc<str>,
    term: Term,
    dependencies: BTreeSet<Arc<str>>,
    constraints: Vec<Equation>,
    expressions: Vec<(Span, Type)>,
}

pub(in crate::lower) fn file(
    file: &SourceFile,
    typer: &mut Context,
    scopes: Scopes,
    source_module: SourceModuleId,
    methods: BTreeMap<Arc<str>, DeclarationId>,
) -> CheckedFile {
    let mut checker = Checker::new(typer, scopes, source_module);
    let (mut signatures, sources) = checker.declarations(file, methods);
    let mut bodies = checker.bodies(&mut signatures, sources);
    checker.solve_functions(&signatures, &mut bodies);
    checker.require_holes();
    checker.finish(signatures, bodies)
}

impl Checker<'_> {
    fn declarations<'s>(
        &mut self,
        file: &'s SourceFile,
        mut methods: BTreeMap<Arc<str>, DeclarationId>,
    ) -> (Signatures, SourceBodies<'s>) {
        let mut signatures = BTreeMap::new();
        let mut bodies = vec![];
        for stmt in file.declarations() {
            if let Some((name, body, signature)) = self.declaration(&stmt.val, &mut methods) {
                if let Some(body) = body {
                    bodies.push((name, body));
                }
                signatures.insert(name.val.clone(), signature);
            }
        }
        (signatures, bodies)
    }

    fn declaration<'s>(
        &mut self,
        stmt: &'s StmtKind,
        methods: &mut BTreeMap<Arc<str>, DeclarationId>,
    ) -> Option<(&'s Ident, Option<&'s resin_ast::Term>, Signature)> {
        let (name, params, result, body) = match stmt {
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
            _ => return None,
        };
        let mut signature = self.signature(params, result, body.is_some());
        match self.declare(name, signature.ty(), methods.remove(&name.val)) {
            Ok(id) => signature.declaration = Some(id),
            Err(error) => {
                self.errors.push(error);
                return None;
            }
        }
        if is_shader(stmt) {
            self.scopes.mark_shader(name);
        }
        Some((name, body, signature))
    }

    fn declare(
        &mut self,
        name: &Ident,
        ty: Type,
        method: Option<DeclarationId>,
    ) -> Result<DeclarationId> {
        if let Some(id) = method {
            self.scopes.set_inferred(id, ty);
            return Ok(id);
        }
        crate::lower::infer::check_binding_name(name)?;
        self.scopes
            .define_inferred(name, ty, DefinitionKind::Function)
            .map_err(|name_| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateValue { name: name_ },
            })
    }

    fn bodies(&mut self, signatures: &mut Signatures, sources: SourceBodies<'_>) -> Vec<Body> {
        sources
            .into_iter()
            .map(|(name, body)| {
                let term = self.function_body(body, signatures.get_mut(&name.val).unwrap());
                Body {
                    name: name.val.clone(),
                    term,
                    dependencies: std::mem::take(&mut self.dependencies),
                    constraints: std::mem::take(&mut self.typing.constraints),
                    expressions: std::mem::take(&mut self.expressions),
                }
            })
            .collect()
    }

    fn function_body(&mut self, source: &resin_ast::Term, signature: &mut Signature) -> Term {
        let errors_before = self.errors.len();
        self.scopes.push_at(source.span);
        self.result = signature.result.ty.clone();
        self.bind_parameters(signature);
        let (rule, term) = self.term(source, Some(signature.result.ty.clone()));
        self.scopes.pop();
        if self.errors.len() != errors_before {
            self.typing.fail(rule);
        }
        self.typing.infer_from(
            &signature.result.holes,
            signature.result.span,
            term.ty.clone(),
        );
        term
    }

    fn bind_parameters(&mut self, signature: &mut Signature) {
        for (name, ann) in &signature.params {
            let binding = self
                .bind(name, ann.ty.clone(), DefinitionKind::Parameter)
                .map_err(|error| self.errors.push(error))
                .ok();
            signature.parameters.push(binding);
        }
    }

    fn solve_functions(&mut self, signatures: &Signatures, bodies: &mut [Body]) {
        for group in groups::groups(&dependencies(bodies)) {
            let roots = self.group_roots(signatures, bodies, &group);
            self.errors.extend(self.typing.solve(&roots));
            for index in group {
                self.require_function(
                    &signatures[&bodies[index].name].result,
                    &bodies[index].expressions,
                );
            }
        }
    }

    fn group_roots(
        &mut self,
        signatures: &Signatures,
        bodies: &mut [Body],
        group: &[usize],
    ) -> Vec<Type> {
        let mut roots = vec![];
        for &index in group {
            let body = &mut bodies[index];
            self.typing.constraints.append(&mut body.constraints);
            roots.extend(body.expressions.iter().map(|(_, ty)| ty.clone()));
            roots.push(signatures[&body.name].result.ty.clone());
        }
        roots
    }

    fn require_function(&mut self, result: &Annotation, expressions: &[(Span, Type)]) {
        for (span, ty) in std::iter::once(&(result.span, result.ty.clone())).chain(expressions) {
            if let Err(error) = self.require(ty, *span) {
                self.errors.push(error);
                if ty == &result.ty {
                    for (_, variable) in &result.holes {
                        self.typing.solver.invalidate(*variable);
                    }
                }
            }
        }
    }

    fn require(&self, ty: &Type, span: Span) -> Result<()> {
        if self.typing.solver.invalid(ty) {
            return Ok(());
        }
        self.typing
            .solver
            .require(ty, span)
            .map(|_| ())
            .map_err(|error| {
                if matches!(self.typing.solver.head(ty), Type::Node(Head::Array(0), _)) {
                    GenerateError::typing(
                        span,
                        TypeError {
                            kind: TypeErrorKind::EmptyArrayNeedsElementType,
                        },
                    )
                } else {
                    error
                }
            })
    }

    fn require_holes(&mut self) {
        for (span, variable) in &self.holes {
            let ty = variable.ty();
            if !self.typing.solver.invalid(&ty)
                && let Err(error) = self.typing.solver.require(&ty, *span)
            {
                self.errors.push(error);
            }
        }
    }

    fn finish(mut self, signatures: Signatures, bodies: Vec<Body>) -> CheckedFile {
        self.scopes
            .resolve_inferred(&self.typing.solver, self.typing.typer);
        let signatures = self.resolve_signatures(signatures);
        let bodies = self.resolve_bodies(bodies);
        CheckedFile {
            context: self.scopes.finish(),
            signatures,
            bodies,
            errors: self.errors,
        }
    }

    fn resolve_signatures(
        &mut self,
        signatures: Signatures,
    ) -> BTreeMap<Arc<str>, typed::Signature> {
        signatures
            .into_iter()
            .filter_map(|(name, signature)| {
                let resolved = signature.resolve(&self.typing.solver);
                self.record(resolved).map(|signature| (name, signature))
            })
            .collect()
    }

    fn resolve_bodies(&mut self, bodies: Vec<Body>) -> BTreeMap<Arc<str>, typed::Term> {
        bodies
            .into_iter()
            .filter_map(|body| {
                if self.typing.solver.invalid(&body.term.ty) {
                    return None;
                }
                let resolved = body.term.resolve(&self.typing.solver);
                self.record(resolved).map(|term| (body.name, term))
            })
            .collect()
    }

    fn record<T>(&mut self, result: Result<T>) -> Option<T> {
        result
            .map_err(|error| {
                if !self.errors.contains(&error) {
                    self.errors.push(error);
                }
            })
            .ok()
    }
}

impl Signature {
    fn ty(&self) -> Type {
        Type::function(
            Type::parameter(self.params.iter().map(|(_, ann)| ann.ty.clone()).collect()),
            self.result.ty.clone(),
        )
    }

    fn resolve(self, solver: &Solver) -> Result<typed::Signature> {
        Ok(typed::Signature {
            declaration: self.declaration,
            parameters: self.parameters,
            params: self
                .params
                .into_iter()
                .map(|(name, ann)| Ok((name, ann.into_tree().resolve(solver)?)))
                .collect::<Result<_>>()?,
            result: self.result.into_tree().resolve(solver)?,
        })
    }
}

fn dependencies(bodies: &[Body]) -> Vec<Vec<usize>> {
    let names: BTreeMap<_, _> = bodies
        .iter()
        .enumerate()
        .map(|(i, body)| (&body.name, i))
        .collect();
    bodies
        .iter()
        .map(|body| {
            body.dependencies
                .iter()
                .filter_map(|name| names.get(name).copied())
                .collect()
        })
        .collect()
}

fn is_shader(stmt: &StmtKind) -> bool {
    matches!(stmt, StmtKind::Function { decorators, .. } if decorators.len() == 1
        && matches!(decorators[0].val.as_ref(), "compute_shader" | "vertex_shader" | "fragment_shader"))
}
