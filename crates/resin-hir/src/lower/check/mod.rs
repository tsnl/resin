//! Infer source expressions and complete public HIR within one construction boundary.
//! Scopes, recovery, and the solver stay here; only completed bodies reach assembly.
use crate::DefinitionKind;
use crate::lower::{
    context::Context,
    infer::{
        Constraint, Equation, Head, Inference, Pattern, Rule, Solver, Type, VariableId,
        check_binding_name,
    },
    scope::{ContextView, DeclarationId, Scopes},
    typed::{self, StatementKind, TermKind},
};
use crate::{GenerateError, GenerateErrorKind};
use resin_ast::{SourceFile, StmtKind};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

mod constants;

//
// Checking state and signatures
//

type Result<T> = std::result::Result<T, GenerateError>;
type Term = typed::Term;
type Statement = typed::Statement;
type MatchArm = typed::MatchArm;
#[derive(Clone)]
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
#[derive(Clone)]
pub(super) struct Signature {
    pub type_params: Vec<crate::TypeParameter>,
    pub declaration: Option<DeclarationId>,
    pub parameters: Vec<Option<DeclarationId>>,
    pub params: Vec<(resin_ast::BindingPattern, Annotation)>,
    pub result: Annotation,
}
impl Checker<'_> {
    fn ann(&mut self, ann: &resin_ast::Type, infer: bool) -> Annotation {
        let scoped = !matches!(ann.val, resin_ast::TypeKind::Unit);
        if scoped {
            self.scopes.push_at(ann.span);
        }
        let checkpoint = self.typing.solver.clone();
        let result = super::eval::Decoder {
            solver: &mut self.typing.solver,
            holes: Vec::new(),
            scopes: self.scopes.view(),
        }
        .decode(ann, infer);
        if scoped {
            self.scopes.pop();
        }
        let decoded = result.unwrap_or_else(|error| {
            self.errors.push(error);
            self.typing.solver = checkpoint;
            super::eval::Decoded {
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
        params: &[(resin_ast::BindingPattern, resin_ast::Type)],
        result: &resin_ast::Type,
        infer: bool,
    ) -> Signature {
        let params = params
            .iter()
            .map(|(name, ann)| (name.clone(), self.ann(ann, false)))
            .collect();
        let result = self.ann(result, infer);
        Signature {
            type_params: vec![],
            params,
            result,
            declaration: None,
            parameters: vec![],
        }
    }
}

/// Completed signatures and HIR bodies, with independent editor facts and diagnostics.
/// Failed definitions do not discard healthy bodies; no inference state survives here.
pub(super) struct CheckedFile {
    pub context: ContextView,
    pub declarations: Vec<typed::Declaration>,
    pub signatures: BTreeMap<DeclarationId, typed::Signature>,
    pub bodies: BTreeMap<DeclarationId, crate::Term>,
    pub errors: Vec<GenerateError>,
}
struct Checker<'a> {
    loop_depth: usize,
    function_ids: std::collections::HashMap<DeclarationId, FunctionId>,
    typing: Inference<'a>,
    scopes: Scopes,
    errors: Vec<GenerateError>,
    dependencies: BTreeSet<DeclarationId>,
    holes: Vec<(Span, VariableId)>,
    expressions: Vec<(Span, Type)>,
    result: Type,
    iota: Option<usize>,
    // Eager self-reference is invalid independently of operator inference.
    // Layout operands are checked for types only and do not access storage.
    initializers: Vec<DeclarationId>,
    unevaluated: bool,
}
impl<'a> Checker<'a> {
    fn new(typer: &'a mut Context, scopes: Scopes) -> Self {
        Self {
            function_ids: Default::default(),
            typing: Inference::new(typer),
            scopes,
            errors: vec![],
            dependencies: BTreeSet::new(),
            holes: vec![],
            expressions: vec![],
            loop_depth: 0,
            result: Ty::Unit.into(),
            iota: None,
            initializers: vec![],
            unevaluated: false,
        }
    }
}

//
// Local name lookup and bindings
//

impl Checker<'_> {
    pub fn bind(&mut self, name: &Ident, ty: Type, kind: DefinitionKind) -> Result<DeclarationId> {
        check_binding_name(name)?;
        self.scopes
            .define_inferred(name, ty, kind)
            .map_err(|duplicate| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateValue { name: duplicate },
            })
    }

    pub fn value(
        &mut self,
        name: &Ident,
        explicit: Option<Vec<Type>>,
    ) -> Result<(DeclarationId, Type, Vec<Type>)> {
        for argument in explicit.iter().flatten() {
            super::eval::reference_type(&self.typing.solver, argument, false, name.span)?;
        }
        let (declaration, ty, function) =
            self.scopes
                .lookup_inferred(&name.val)
                .ok_or_else(|| GenerateError {
                    span: name.span,
                    kind: GenerateErrorKind::UnboundValue {
                        name: name.val.clone(),
                    },
                })?;
        if !self.unevaluated && self.initializers.contains(&declaration) {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::EagerRecursion {
                    name: name.val.clone(),
                },
            });
        }
        if !function {
            if explicit.is_some() {
                return Err(GenerateError::inference(
                    name.span,
                    "only a function declaration accepts type arguments",
                ));
            }
            return Ok((declaration, ty, vec![]));
        }
        self.dependencies.insert(declaration);
        if self
            .function_ids
            .get(&declaration)
            .and_then(|id| self.typing.typer.functions.get(id))
            .is_some_and(|function| {
                matches!(
                    function.body,
                    super::context::FunctionBody::GpuPipelineFactory { .. }
                        | super::context::FunctionBody::GpuPipelineRecord { .. }
                )
            })
        {
            return Err(GenerateError::inference(
                name.span,
                "GPU pipeline bridges require direct calls; native bridge references cannot escape",
            ));
        }
        let parameters = self.scopes.parameters(declaration);
        let (ty, arguments) = self
            .typing
            .solver
            .apply(ty, &parameters, explicit, name.span)?;
        Ok((declaration, ty, arguments))
    }
}

//
// Declarations, bodies, and dependency solving
//

type Signatures = BTreeMap<DeclarationId, Signature>;
type SourceBodies<'a> = Vec<(DeclarationId, &'a resin_ast::Term)>;

struct Body {
    declaration: DeclarationId,
    term: Term,
    dependencies: BTreeSet<DeclarationId>,
    constraints: Vec<Equation>,
    expressions: Vec<(Span, Type)>,
}

pub(in crate::lower) fn file(
    file: &SourceFile,
    generator: &mut super::Generator,
    scopes: Scopes,
    methods: BTreeMap<Arc<str>, DeclarationId>,
) -> CheckedFile {
    let mut checker = Checker::new(&mut generator.typer, scopes);
    checker.module_constants(file);
    let (declarations, mut signatures, sources) = checker.declarations(file, methods);
    for declaration in &declarations {
        generator
            .function_bindings
            .entry(declaration.id)
            .or_insert_with(|| {
                let id = FunctionId::from_index(generator.functions.len());
                generator.functions.push(None);
                id
            });
    }
    checker.function_ids = generator.function_bindings.clone();
    checker.struct_hooks(file, &declarations, &signatures);
    checker.declare_gpu_contracts(
        &declarations,
        &signatures,
        &mut generator.functions,
        &mut generator.function_bindings,
        &generator.source,
    );
    checker.declare_gpu_bridges(&declarations, &signatures);
    let mut bodies = checker.bodies(&mut signatures, sources);
    checker.solve_functions(&signatures, &mut bodies);
    checker.require_holes();
    let signatures = checker.resolve_signatures(signatures);
    checker.complete_method_signatures(&declarations, &signatures);
    checker.scopes.resolve_inferred(
        &checker.typing.solver,
        checker.typing.typer,
        &checker.typing.methods,
        &checker.function_ids,
    );
    let Checker {
        mut typing,
        scopes,
        errors,
        ..
    } = checker;
    let solver = std::mem::take(&mut typing.solver);
    let methods = std::mem::take(&mut typing.methods);
    drop(typing);
    let mut checked = CheckedFile {
        context: scopes.finish(),
        declarations,
        signatures,
        bodies: BTreeMap::new(),
        errors,
    };
    // Reserve every concrete signature before completing any body, including recursion.
    let errors = generator.declare_checked_functions(&checked);
    checked.errors.extend(errors);
    for body in bodies {
        if solver.invalid(&body.term.actual) {
            continue;
        }
        let Some(signature) = checked.signatures.get(&body.declaration) else {
            continue;
        };
        let completed = super::elaborate::function(
            &body.term,
            signature
                .parameters
                .iter()
                .zip(&signature.params)
                .filter_map(|(id, (pattern, _))| id.map(|id| (id, pattern.mutable))),
            &solver,
            &methods,
            &generator.typer,
            &generator.function_bindings,
            &generator.module.shaders,
        );
        match completed {
            Ok(completed) => {
                for shader in completed.shaders {
                    generator.module.shaders.get_mut(&shader).unwrap().embedded = true;
                }
                checked.bodies.insert(body.declaration, completed.body);
            }
            Err(error) if !checked.errors.contains(&error) => checked.errors.push(error),
            Err(_) => {}
        }
    }
    checked
}

impl Checker<'_> {
    fn struct_hooks(
        &mut self,
        file: &SourceFile,
        declarations: &[typed::Declaration],
        signatures: &Signatures,
    ) {
        for declaration in declarations
            .iter()
            .filter(|declaration| matches!(declaration.name.val.as_ref(), "drop" | "repr_bytes"))
        {
            let hook = declaration.name.val.as_ref();
            let result = (|| {
                let signature = signatures[&declaration.id]
                    .clone()
                    .resolve(&self.typing.solver)?;
                let invalid = || {
                    GenerateError::inference(
                        declaration.name.span,
                        if hook == "drop" {
                            "drop must have signature fn drop<T>(value: RefMut<Owner<T>>) with only the owner's type parameters"
                        } else {
                            "repr_bytes must take Ref<Owner<T>> and return (Ptr<u8>, u64), with only the owner's type parameters"
                        },
                    )
                };
                let [(_, parameter)] = signature.params.as_slice() else {
                    return Err(invalid());
                };
                let (pointee, result) = match (hook, &parameter.ty) {
                    (
                        "drop",
                        crate::Type::Reference {
                            referent,
                            mutable: true,
                        },
                    ) => (referent, crate::Type::Unit),
                    (
                        "repr_bytes",
                        crate::Type::Reference {
                            referent,
                            mutable: false,
                        },
                    ) => (referent, super::types::ty(&Ty::byte_span(false))),
                    _ => return Err(invalid()),
                };
                let crate::Type::Defined {
                    definition,
                    arguments,
                } = pointee.as_ref()
                else {
                    return Err(invalid());
                };
                let owner = self
                    .typing
                    .typer
                    .nominal_schemes
                    .get(definition)
                    .ok_or_else(invalid)?;
                if signature.result.ty != result
                    || arguments.len() != owner.type_params.len()
                    || arguments
                        != &signature
                            .type_params
                            .iter()
                            .map(|parameter| crate::Type::Parameter {
                                parameter: parameter.id,
                            })
                            .collect::<Vec<_>>()
                {
                    return Err(invalid());
                }
                if !file.stmts.iter().any(|statement| matches!(&statement.val, StmtKind::Struct { name, .. } if name.val == owner.name)) {
                    return Err(GenerateError::inference(declaration.name.span, format!("a {hook} hook must be defined in the same module as its struct")));
                }
                let function = self.function_ids[&declaration.id];
                if hook == "drop" {
                    if self
                        .typing
                        .typer
                        .definition(*definition)
                        .ok()
                        .is_some_and(|definition| definition.drop_hook().is_some())
                    {
                        return Err(GenerateError::inference(
                            declaration.name.span,
                            "duplicate drop hook for this struct",
                        ));
                    }
                    self.typing.typer.define_drop(*definition, function);
                } else {
                    let owner = self
                        .typing
                        .typer
                        .nominal_schemes
                        .get_mut(definition)
                        .expect("checked nominal owner");
                    if owner.text_view.is_some() {
                        return Err(GenerateError::inference(
                            declaration.name.span,
                            "duplicate repr_bytes hook for this struct",
                        ));
                    }
                    owner.text_view = Some(function);
                }
                Ok(())
            })();
            self.record(result);
        }
    }

    fn declare_gpu_bridges(
        &mut self,
        declarations: &[typed::Declaration],
        signatures: &Signatures,
    ) {
        for declaration in declarations {
            let typed::DeclarationKind::Function { decorators } = &declaration.kind else {
                continue;
            };
            for decorator in decorators
                .iter()
                .filter(|decorator| super::gpu::is_bridge(&decorator.val))
            {
                let result = (|| {
                    let signature = signatures[&declaration.id]
                        .clone()
                        .resolve(&self.typing.solver)?;
                    let function = self.function_ids[&declaration.id];
                    let params = signature
                        .params
                        .iter()
                        .map(|(_, annotation)| annotation.ty.clone())
                        .collect();
                    if !signature.type_params.is_empty()
                        || !self.typing.typer.register_function(
                            function,
                            params,
                            &signature.result.ty,
                        )
                    {
                        return Err(GenerateError::inference(
                            decorator.span,
                            "GPU bridges require a fixed, explicit signature",
                        ));
                    }
                    self.typing.typer.register_bridge(function, decorator)
                })();
                self.record(result);
            }
        }
    }

    fn declare_gpu_contracts(
        &mut self,
        declarations: &[typed::Declaration],
        signatures: &Signatures,
        functions: &mut Vec<Option<crate::Function>>,
        bindings: &mut std::collections::HashMap<DeclarationId, FunctionId>,
        source: &Source,
    ) {
        // Contracts have fully explicit signatures. Install their type relations
        // before any body can request a factory or recording operation. Sequence
        // contracts depend on pointer contracts, regardless of source order.
        for operation in [
            "gpu_pointer_projection",
            "gpu_span_projection",
            "gpu_compute_pipeline_type",
            "gpu_graphics_pipeline_type",
            "gpu_ray_tracing_pipeline_type",
        ] {
            for declaration in declarations {
                if !matches!(&declaration.kind, typed::DeclarationKind::Intrinsic { operation: name } if name.as_ref() == operation)
                {
                    continue;
                }
                let result = signatures[&declaration.id]
                    .clone()
                    .resolve(&self.typing.solver);
                let Some(signature) = self.record(result) else {
                    continue;
                };
                if self.record(super::check_parameters(&signature)).is_none() {
                    continue;
                }
                let id = *bindings.entry(declaration.id).or_insert_with(|| {
                    let id = FunctionId::from_index(functions.len());
                    functions.push(None);
                    id
                });
                let mut function = crate::Function {
                    location: Some(SourceLocation {
                        source: source.clone(),
                        span: declaration.name.span,
                    }),
                    name: declaration.name.val.clone(),
                    signature: super::elaborate_signature(&signature),
                    body: None,
                    foreign_header: None,
                };
                let result =
                    super::gpu_projections::define(self.typing.typer, &mut function, id, operation);
                self.record(result);
                functions[id.index()] = Some(function);
            }
        }
    }

    fn declarations<'s>(
        &mut self,
        file: &'s SourceFile,
        mut methods: BTreeMap<Arc<str>, DeclarationId>,
    ) -> (Vec<typed::Declaration>, Signatures, SourceBodies<'s>) {
        let mut declarations = vec![];
        let mut signatures = BTreeMap::new();
        let mut bodies = vec![];
        for stmt in file.declarations() {
            if let Some((declaration, body, signature)) = self.declaration(&stmt.val, &mut methods)
            {
                if let Some(body) = body {
                    bodies.push((declaration.id, body));
                }
                signatures.insert(declaration.id, signature);
                declarations.push(declaration);
            }
        }
        (declarations, signatures, bodies)
    }

    fn declaration<'s>(
        &mut self,
        stmt: &'s StmtKind,
        methods: &mut BTreeMap<Arc<str>, DeclarationId>,
    ) -> Option<(typed::Declaration, Option<&'s resin_ast::Term>, Signature)> {
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
            }
            | StmtKind::IntrinsicFunction {
                name,
                params,
                result,
                ..
            } => (name, params, result, None),
            _ => return None,
        };
        let method = methods.get(&name.val).copied();
        let owner = method.and_then(|id| self.typing.typer.method_owners.get(&id).copied());
        let owner_parameters = owner
            .map(|owner| {
                self.typing.typer.nominal_schemes[&owner]
                    .type_params
                    .clone()
            })
            .unwrap_or_default();
        let type_scope = match stmt {
            StmtKind::Function { type_params, .. }
            | StmtKind::IntrinsicFunction { type_params, .. } => {
                type_params.first().map(|parameter| Span {
                    start: parameter.span.start,
                    end: body.map_or(result.span.end, |body| body.span.start),
                })
            }
            _ => None,
        }
        .or_else(|| {
            (!owner_parameters.is_empty()).then_some(Span {
                start: name.span.start,
                end: body.map_or(result.span.end, |body| body.span.start),
            })
        });
        if let Some(span) = type_scope {
            self.scopes.push_at(span);
        }
        let mut binders = owner_parameters.clone();
        for parameter in &owner_parameters {
            self.scopes.import(
                parameter.name.val.clone(),
                super::scope::Symbol {
                    definition: parameter.id.index(),
                },
            );
        }
        if let StmtKind::Function { type_params, .. }
        | StmtKind::IntrinsicFunction { type_params, .. } = stmt
        {
            for parameter in type_params {
                match self.scopes.define_type_parameter(parameter) {
                    Ok(parameter) => binders.push(parameter),
                    Err(error) => self.errors.push(error),
                }
            }
        }
        let mut signature = self.signature(params, result, body.is_some());
        signature.type_params = binders;
        if type_scope.is_some() {
            self.scopes.pop();
        }
        match self.declare(name, signature.ty(), methods.remove(&name.val)) {
            Ok(id) => signature.declaration = Some(id),
            Err(error) => {
                self.errors.push(error);
                return None;
            }
        }
        let id = signature.declaration.unwrap();
        self.scopes.set_parameters(id, &signature.type_params);
        if let Some(owner) = owner
            && matches!(stmt, StmtKind::Function { decorators, .. } if decorators.is_empty())
        {
            self.typing.typer.define_source_method(
                owner,
                name.val.rsplit('.').next().unwrap(),
                super::context::SourceMethod {
                    declaration: id,
                    type_params: signature.type_params[owner_parameters.len()..].to_vec(),
                    owner_params: owner_parameters,
                    params: signature
                        .params
                        .iter()
                        .map(|(_, annotation)| annotation.ty.clone())
                        .collect(),
                    result: signature.result.ty.clone(),
                },
            );
        }
        let kind = match stmt {
            StmtKind::Function { decorators, .. } => typed::DeclarationKind::Function {
                decorators: decorators.clone(),
            },
            StmtKind::IntrinsicFunction { operation, .. } => typed::DeclarationKind::Intrinsic {
                operation: operation.clone(),
            },
            StmtKind::ForeignFunction { header, .. } => typed::DeclarationKind::Foreign {
                header: header.clone(),
            },
            _ => unreachable!("function declaration"),
        };
        let declaration = typed::Declaration {
            id: signature.declaration.unwrap(),
            name: name.clone(),
            kind,
        };
        Some((declaration, body, signature))
    }

    fn complete_method_signatures(
        &mut self,
        declarations: &[typed::Declaration],
        signatures: &BTreeMap<DeclarationId, typed::Signature>,
    ) {
        let local = declarations
            .iter()
            .map(|declaration| declaration.id)
            .collect::<BTreeSet<_>>();
        for method in self.typing.typer.source_methods.values_mut() {
            let Some(signature) = signatures.get(&method.declaration) else {
                if local.contains(&method.declaration) {
                    method.params.fill(Type::Invalid);
                    method.result = Type::Invalid;
                }
                continue;
            };
            method.params = signature
                .params
                .iter()
                .map(|(_, annotation)| Type::from_hir(&annotation.ty))
                .collect();
            method.result = Type::from_hir(&signature.result.ty);
        }
    }

    fn method_dependencies(&mut self, name: &crate::MethodName) {
        // Receiver ownership may depend on an earlier call's inferred result.
        // Include same-name candidates until solving selects the nominal owner.
        self.dependencies.extend(
            self.typing
                .typer
                .source_methods
                .iter()
                .filter(|((_, candidate), _)| candidate == name)
                .map(|(_, method)| method.declaration),
        );
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
            .map(|(declaration, body)| {
                let term = self.function_body(body, signatures.get_mut(&declaration).unwrap());
                Body {
                    declaration,
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
        for parameter in &signature.type_params {
            self.scopes.import(
                parameter.name.val.clone(),
                super::scope::Symbol {
                    definition: parameter.id.index(),
                },
            );
        }
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
            term.actual.clone(),
        );
        term
    }

    fn bind_parameters(&mut self, signature: &mut Signature) {
        for (name, ann) in &signature.params {
            let binding = self
                .bind(&name.name, ann.ty.clone(), DefinitionKind::Parameter)
                .map_err(|error| self.errors.push(error))
                .ok();
            signature.parameters.push(binding);
        }
    }

    fn solve_functions(&mut self, signatures: &Signatures, bodies: &mut [Body]) {
        for group in groups(&dependencies(bodies)) {
            let roots = self.group_roots(signatures, bodies, &group);
            self.errors.extend(self.typing.solve(&roots));
            for index in group {
                self.require_function(
                    &signatures[&bodies[index].declaration].result,
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
            roots.push(signatures[&body.declaration].result.ty.clone());
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
            .require_complete(ty, span)
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
                && let Err(error) = self.typing.solver.require_complete(&ty, *span)
            {
                self.errors.push(error);
            }
        }
    }

    fn resolve_signatures(
        &mut self,
        signatures: Signatures,
    ) -> BTreeMap<DeclarationId, typed::Signature> {
        signatures
            .into_iter()
            .filter_map(|(declaration, signature)| {
                let resolved = signature.resolve(&self.typing.solver);
                self.record(resolved)
                    .map(|signature| (declaration, signature))
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
            self.params.iter().map(|(_, ann)| ann.ty.clone()).collect(),
            self.result.ty.clone(),
        )
    }

    fn resolve(self, solver: &Solver) -> Result<typed::Signature> {
        Ok(typed::Signature {
            type_params: self.type_params,
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
    let declarations: BTreeMap<_, _> = bodies
        .iter()
        .enumerate()
        .map(|(i, body)| (&body.declaration, i))
        .collect();
    bodies
        .iter()
        .map(|body| {
            body.dependencies
                .iter()
                .filter_map(|declaration| declarations.get(declaration).copied())
                .collect()
        })
        .collect()
}

//
// Expression constraints and typed tree construction
//

impl Checker<'_> {
    pub fn term(&mut self, term: &resin_ast::Term, expected: Option<Type>) -> (Rule, Term) {
        let reference_result = matches!(
            term.val,
            resin_ast::TermKind::Var { .. } | resin_ast::TermKind::MethodCall { .. }
        ) || matches!(&term.val, resin_ast::TermKind::Call { func, .. } if !matches!(func.val, resin_ast::TermKind::Type { .. }));
        let (rule, out) = if reference_result {
            self.typing.reference_expression()
        } else {
            self.typing.expression()
        };
        let mut checked = Expression {
            checker: self,
            rule,
        }
        .check(term, expected.clone(), out.clone());
        let consumed = expected.unwrap_or_else(|| {
            if reference_result {
                Type::value(out)
            } else {
                out
            }
        });
        checked.ty = consumed;
        (rule, checked)
    }
}

/// One expression owns all of its equations and automatically depends on its children.
struct Expression<'p, 'a> {
    checker: &'p mut Checker<'a>,
    rule: Rule,
}

impl Expression<'_, '_> {
    fn overload_candidates(&mut self, name: &Ident) -> Vec<super::infer::OverloadCandidate> {
        self.checker
            .scopes
            .lookup_overloads(&name.val)
            .into_iter()
            .filter(|(_, _, function)| *function)
            .filter_map(|(declaration, signature, _)| {
                let function = self.checker.function_ids.get(&declaration).copied()?;
                self.checker.dependencies.insert(declaration);
                Some(super::infer::OverloadCandidate {
                    function,
                    declaration,
                    signature,
                    parameters: self.checker.scopes.parameters(declaration),
                })
            })
            .collect()
    }

    fn overload_reference(
        &mut self,
        name: &Ident,
        explicit: Option<Vec<Type>>,
        out: Type,
    ) -> Option<TermKind> {
        let candidates = self.overload_candidates(name);
        if candidates.len() < 2 {
            return None;
        }
        self.checker
            .scopes
            .record_call(name, Ty::Unit.into(), vec![], true, self.rule);
        self.constrain((
            name.span,
            Constraint::Overload {
                lookup: super::infer::Overload {
                    name: name.val.clone(),
                    candidates,
                    primitive: None,
                    expected: None,
                    type_args: explicit,
                    args: None,
                    out,
                },
            },
        ));
        Some(TermKind::MethodReference {
            rule: self.rule,
            name: name.clone(),
        })
    }

    fn overload_call(
        &mut self,
        func: &resin_ast::Term,
        args: &[resin_ast::Term],
        out: Type,
        expected: Option<Type>,
    ) -> Option<TermKind> {
        let (name, explicit) = match &func.val {
            resin_ast::TermKind::Var { name } => (name, None),
            resin_ast::TermKind::TypeApply { function, args } => {
                let resin_ast::TermKind::Var { name } = &function.val else {
                    return None;
                };
                (
                    name,
                    Some(
                        args.iter()
                            .map(|arg| self.annotation(arg, true).ty)
                            .collect(),
                    ),
                )
            }
            _ => return None,
        };
        if self
            .checker
            .scopes
            .lookup_overloads(&name.val)
            .iter()
            .any(|(_, _, function)| !function)
        {
            return None;
        }
        let candidates = self.overload_candidates(name);
        let primitive = super::context::is_primitive_operation(&name.val).then(|| name.val.clone());
        if candidates.is_empty() && primitive.is_none() {
            return None;
        }
        let args = args
            .iter()
            .map(|arg| self.child(arg, None))
            .collect::<Vec<_>>();
        self.checker.scopes.record_call(
            name,
            args.first()
                .map_or_else(|| Ty::Unit.into(), |arg| arg.ty.clone()),
            args.iter().map(|arg| arg.ty.clone()).collect(),
            true,
            self.rule,
        );
        self.constrain((
            name.span,
            Constraint::Overload {
                lookup: super::infer::Overload {
                    name: name.val.clone(),
                    candidates,
                    primitive,
                    expected,
                    type_args: explicit,
                    args: Some(args.iter().map(|arg| arg.ty.clone()).collect()),
                    out,
                },
            },
        ));
        Some(TermKind::Builtin {
            rule: self.rule,
            name: name.val.clone(),
            name_span: name.span,
            args,
        })
    }

    fn method_reference(
        &mut self,
        receiver_type: typed::Annotation<Type>,
        name: &Ident,
        type_args: Option<Vec<Type>>,
        out: Type,
    ) -> TermKind {
        self.checker.method_dependencies(&name.val.clone().into());
        self.checker
            .scopes
            .record_members(name.span, receiver_type.ty.clone(), true);
        self.checker
            .scopes
            .record_call(name, receiver_type.ty.clone(), vec![], true, self.rule);
        self.constrain((
            name.span,
            Constraint::MethodReference {
                receiver: receiver_type.ty.clone(),
                name: name.val.clone(),
                type_args,
                out,
            },
        ));
        TermKind::MethodReference {
            rule: self.rule,
            name: name.clone(),
        }
    }

    fn child(&mut self, term: &resin_ast::Term, expected: Option<Type>) -> Term {
        let (_, child) = self.checker.term(term, expected);
        self.checker
            .typing
            .depends(self.rule, term.span, child.actual.clone());
        child
    }

    fn annotation(&mut self, ann: &resin_ast::Type, infer: bool) -> Annotation {
        let annotation = self.checker.ann(ann, infer);
        self.checker
            .typing
            .depends(self.rule, ann.span, annotation.ty.clone());
        annotation
    }

    fn constrain(&mut self, constraint: (Span, Constraint)) {
        self.checker.typing.constrain(self.rule, constraint);
    }

    fn check(&mut self, term: &resin_ast::Term, expected: Option<Type>, out: Type) -> Term {
        let context = self.checker.scopes.capture();
        let checked = self.term_inner(term, expected, out.clone());
        let checked = match checked {
            Ok(checked) => checked,
            Err(error) => {
                self.checker.scopes.restore(context);
                self.checker.errors.push(error.clone());
                self.checker.typing.fail(self.rule);
                Term {
                    span: term.span,
                    ty: out.clone(),
                    actual: out.clone(),
                    kind: TermKind::Error(error),
                }
            }
        };
        self.checker.expressions.push((term.span, out.clone()));
        self.checker.scopes.record_inferred(term.span, out);
        checked
    }

    fn builtin(
        &mut self,
        name: &Arc<str>,
        name_span: &Span,
        args: &[resin_ast::Term],
        span: Span,
        out: &Type,
        expected: &Option<Type>,
    ) -> TermKind {
        self.checker
            .method_dependencies(&crate::MethodName::Operator {
                symbol: name.clone(),
                arity: args.len(),
            });
        let args = args
            .iter()
            .map(|arg| {
                let expected = if matches!(name.as_ref(), "&&" | "||" | "assert") {
                    Some(Ty::Bool.into())
                } else {
                    expression_exits(arg).then(|| self.checker.typing.solver.fresh())
                };
                self.child(arg, expected)
            })
            .collect::<Vec<_>>();
        self.checker.scopes.record_call(
            &Ident::new(name.clone(), *name_span),
            args[0].ty.clone(),
            args.iter().map(|arg| arg.ty.clone()).collect(),
            true,
            self.rule,
        );
        let dunder = crate::OPERATOR_METHODS
            .iter()
            .find(|(_, symbol, arity)| *symbol == name.as_ref() && *arity == args.len())
            .map(|(method, _, _)| *method);
        let constraint = if let Some(dunder) = dunder {
            let candidates = self.overload_candidates(&Ident::new(dunder.into(), *name_span));
            Constraint::Overload {
                lookup: super::infer::Overload {
                    name: name.clone(),
                    candidates,
                    primitive: Some(name.clone()),
                    expected: expected.clone(),
                    type_args: None,
                    args: Some(args.iter().map(|arg| arg.ty.clone()).collect()),
                    out: out.clone(),
                },
            }
        } else {
            Constraint::Builtin(
                name.clone(),
                args.iter().map(|arg| arg.ty.clone()).collect(),
                out.clone(),
            )
        };
        self.constrain((span, constraint));
        TermKind::Builtin {
            rule: self.rule,
            name_span: *name_span,
            name: name.clone(),
            args,
        }
    }

    fn call(
        &mut self,
        func: &resin_ast::Term,
        args: &[resin_ast::Term],
        span: Span,
        out: &Type,
        expected: &Option<Type>,
    ) -> Result<(TermKind, Option<Type>)> {
        let mut equate = None;
        let kind = if let resin_ast::TermKind::Var { name } = &func.val
            && name.val.as_ref() == "absurd"
        {
            let arg = single_argument(args, span)?;
            let arg = self.child(arg, Some(Ty::union([]).into()));
            TermKind::Absurd { arg: Box::new(arg) }
        } else if let resin_ast::TermKind::Var { name } = &func.val
            && matches!(name.val.as_ref(), "size_of" | "align_of")
        {
            let arg = single_argument(args, span)?;
            // Check the operand for typing only. Never execute its effects or read its locals.
            let ann = if let resin_ast::TermKind::Type { ty } = &arg.val {
                self.annotation(ty, false)
            } else {
                let unevaluated = std::mem::replace(&mut self.checker.unevaluated, true);
                let term = self.child(arg, None);
                self.checker.unevaluated = unevaluated;
                Annotation {
                    holes: Vec::new(),
                    ty: term.ty,
                    span: term.span,
                }
            };
            equate = Some(Ty::UInt64.into());
            self.constrain((span, Constraint::Layout(ann.ty.clone())));
            TermKind::Layout {
                ty: ann.into_tree(),
                size: name.val.as_ref() == "size_of",
            }
        } else if let resin_ast::TermKind::Type { ty } = &func.val
            && matches!(&ty.val, resin_ast::TypeKind::App { head, .. } if head.val.as_ref() == "Err")
        {
            let ann = self.annotation(ty, true);
            super::eval::reference_type(&self.checker.typing.solver, &ann.ty, false, span)?;
            let Type::Node(super::infer::Head::Error, parts) =
                self.checker.typing.solver.head(&ann.ty)
            else {
                unreachable!("Err annotation");
            };
            let arg = self.child(single_argument(args, span)?, Some(parts[0].clone()));
            equate = Some(ann.ty.clone());
            TermKind::Ascribe {
                ty: ann.into_tree(),
                arg: Box::new(arg),
            }
        } else if let resin_ast::TermKind::Type { ty } = &func.val {
            let unit = resin_ast::Term {
                span,
                val: resin_ast::TermKind::Unit,
            };
            let arg = if args.is_empty() {
                &unit
            } else {
                single_argument(args, span)?
            };
            let ann = self.annotation(ty, true);
            super::eval::reference_type(&self.checker.typing.solver, &ann.ty, false, span)?;
            let arg = {
                let literal = matches!(arg.val, resin_ast::TermKind::Num { .. })
                    || matches!(&arg.val, resin_ast::TermKind::Builtin { name, args, .. } if matches!(name.as_ref(), "+" | "-") && matches!(args.as_slice(), [resin_ast::Term { val: resin_ast::TermKind::Num { .. }, .. }]));
                let arg = self.child(arg, None);
                self.constrain((
                    span,
                    Constraint::Ascribe(arg.ty.clone(), ann.ty.clone(), literal),
                ));
                arg
            };
            equate = Some(ann.ty.clone());
            TermKind::Ascribe {
                ty: ann.into_tree(),
                arg: Box::new(arg),
            }
        } else if let Some(call) = self.overload_call(func, args, out.clone(), expected.clone()) {
            call
        } else {
            let func = self.child(func, None);
            let args = args
                .iter()
                .map(|arg| self.child(arg, None))
                .collect::<Vec<_>>();
            self.constrain((
                span,
                Constraint::Call(
                    func.ty.clone(),
                    args.iter().map(|arg| arg.ty.clone()).collect(),
                    out.clone(),
                ),
            ));
            TermKind::Call {
                func: Box::new(func),
                args,
            }
        };
        Ok((kind, equate))
    }

    fn type_application(
        &mut self,
        func: &resin_ast::Term,
        type_args: &[resin_ast::Type],
        span: Span,
        out: Type,
    ) -> Result<(TermKind, Option<Type>)> {
        let mut equate = None;
        let arguments: Vec<_> = type_args
            .iter()
            .map(|ann| self.annotation(ann, true).ty)
            .collect();
        for argument in &arguments {
            super::eval::reference_type(&self.checker.typing.solver, argument, false, span)?;
        }
        let kind = match &func.val {
            resin_ast::TermKind::Var { name } => {
                if let Some(reference) =
                    self.overload_reference(name, Some(arguments.clone()), out.clone())
                {
                    reference
                } else {
                    let (declaration, ty, type_args) = self.checker.value(name, Some(arguments))?;
                    equate = Some(ty);
                    TermKind::Var {
                        declaration,
                        name: name.clone(),
                        type_args,
                    }
                }
            }
            resin_ast::TermKind::Field { base, name } => {
                let base = self.child(base, None);
                let TermKind::Type { ty } = base.kind else {
                    return Err(GenerateError::inference(
                        span,
                        "method references require a type receiver",
                    ));
                };
                self.method_reference(ty, name, Some(arguments), out.clone())
            }
            _ => {
                return Err(GenerateError::inference(
                    span,
                    "type arguments require a function declaration",
                ));
            }
        };
        Ok((kind, equate))
    }

    fn match_expression(
        &mut self,
        value: &resin_ast::Term,
        arms: &[resin_ast::MatchArm],
        out: &Type,
    ) -> TermKind {
        let input = self.child(value, None);
        let mut checked = vec![];
        for arm in arms {
            self.checker.scopes.push_at(arm.body.span);
            let payload = self.checker.typing.solver.fresh();
            let (variant, pattern) = match &arm.variant {
                resin_ast::MatchVariant::Wildcard => (None, Pattern::Error),
                resin_ast::MatchVariant::Error => (None, Pattern::Error),
                resin_ast::MatchVariant::Type(ty) => {
                    let ann = self.annotation(ty, false);
                    let ty = ann.ty.clone();
                    (Some(ann.into_tree()), Pattern::Type(ty))
                }
            };
            if !matches!(arm.variant, resin_ast::MatchVariant::Wildcard) {
                self.constrain((
                    arm.body.span,
                    Constraint::Variant(input.ty.clone(), pattern, payload.clone()),
                ));
            }
            let binding = arm.pattern.as_ref().and_then(|pattern| {
                self.checker
                    .bind(&pattern.name, payload, DefinitionKind::Variable)
                    .map_err(|error| self.checker.errors.push(error))
                    .ok()
            });
            let body = self.child(&arm.body, Some(out.clone()));
            checked.push(MatchArm {
                error: matches!(arm.variant, resin_ast::MatchVariant::Error),
                wildcard: matches!(arm.variant, resin_ast::MatchVariant::Wildcard),
                mutable: arm.pattern.as_ref().is_some_and(|pattern| pattern.mutable),
                binding,
                variant,
                body,
            });
            self.checker.scopes.pop();
        }
        TermKind::Match {
            value: Box::new(input),
            arms: checked,
        }
    }

    fn receiver_call(
        &mut self,
        receiver: &resin_ast::Term,
        name: &Ident,
        type_args: &[resin_ast::Type],
        args: &[resin_ast::Term],
        span: Span,
        context: (Option<Type>, Type),
    ) -> Result<Term> {
        let (expected, out) = context;
        let mut func = resin_ast::Term {
            span: name.span,
            val: resin_ast::TermKind::Var { name: name.clone() },
        };
        if !type_args.is_empty() {
            func = resin_ast::Term {
                span: name.span,
                val: resin_ast::TermKind::TypeApply {
                    function: Box::new(func),
                    args: type_args.to_vec(),
                },
            };
        }
        let args = std::iter::once(receiver.clone())
            .chain(args.iter().cloned())
            .collect();
        let call = resin_ast::Term {
            span,
            val: resin_ast::TermKind::Call {
                func: Box::new(func),
                args,
            },
        };
        self.term_inner(&call, expected, out)
    }

    fn term_inner(
        &mut self,
        term: &resin_ast::Term,
        expected: Option<Type>,
        out: Type,
    ) -> Result<Term> {
        if let resin_ast::TermKind::MethodCall {
            receiver,
            name,
            type_args,
            args,
        } = &term.val
        {
            return self.receiver_call(receiver, name, type_args, args, term.span, (expected, out));
        }
        let propagate = matches!(
            term.val,
            resin_ast::TermKind::If { .. }
                | resin_ast::TermKind::Match { .. }
                | resin_ast::TermKind::Block { .. }
        ) || matches!(&term.val, resin_ast::TermKind::Call { func, .. } if matches!(&func.val, resin_ast::TermKind::Var { name } if matches!(name.val.as_ref(), "ok" | "err" | "absurd")));
        let contextual = propagate && expected.is_some();
        if propagate && let Some(expected) = &expected {
            self.constrain((term.span, Constraint::Equal(expected.clone(), out.clone())));
        }
        let span = term.span;
        let mut equate = None;
        let kind = match &term.val {
            resin_ast::TermKind::SizeOf { ty } => {
                let ann = self.annotation(ty, false);
                super::eval::reference_type(&self.checker.typing.solver, &ann.ty, false, span)?;
                self.constrain((span, Constraint::SizeOf(ann.ty.clone())));
                equate = Some(Ty::UInt64.into());
                TermKind::SizeOf {
                    ty: ann.into_tree(),
                }
            }
            resin_ast::TermKind::Hole { children } => {
                for child in children {
                    self.child(child, None);
                }
                return Err(GenerateError {
                    span,
                    kind: GenerateErrorKind::IncompleteSyntax,
                });
            }
            resin_ast::TermKind::FieldHole { base } => {
                let base = self.child(base, None);
                let (ty, associated) = match &base.kind {
                    TermKind::Type { ty } => (ty.ty.clone(), true),
                    _ => (base.ty.clone(), false),
                };
                self.checker.scopes.record_members(
                    Span {
                        start: span.end,
                        end: span.end,
                    },
                    ty,
                    associated,
                );
                return Err(GenerateError {
                    span,
                    kind: GenerateErrorKind::IncompleteSyntax,
                });
            }
            resin_ast::TermKind::Unit => {
                equate = Some(Ty::Unit.into());
                TermKind::Unit
            }
            resin_ast::TermKind::Bool { value } => {
                equate = Some(Ty::Bool.into());
                TermKind::Bool { value: *value }
            }
            resin_ast::TermKind::None => {
                equate = Some(Ty::None.into());
                TermKind::None
            }
            resin_ast::TermKind::Unwrap { value } => {
                let input = self.child(value, None);
                self.constrain((span, Constraint::ExcludeNone(input.ty.clone(), out.clone())));
                TermKind::Unwrap {
                    value: Box::new(input),
                }
            }
            resin_ast::TermKind::Num { value } => {
                equate = Some(self.checker.typing.solver.number(value));
                TermKind::Num {
                    value: value.clone(),
                }
            }
            resin_ast::TermKind::String { value } => {
                equate = Some(Ty::Str.into());
                TermKind::String {
                    value: value.clone(),
                }
            }
            resin_ast::TermKind::Var { name } => {
                if name.val.as_ref() == "iota" {
                    let index = self.checker.iota.ok_or_else(|| {
                        GenerateError::inference(
                            span,
                            "iota is only available in a const initializer",
                        )
                    })?;
                    equate = Some(self.checker.typing.solver.number(&index.to_string()));
                    TermKind::Num {
                        value: index.to_string().into(),
                    }
                } else if let Some(value) = self.checker.scopes.constant(&name.val) {
                    equate = Some(Type::from_hir(&value.ty));
                    TermKind::Constant { value }
                } else if let Some(reference) = self.overload_reference(name, None, out.clone()) {
                    reference
                } else {
                    let (declaration, ty, type_args) = self.checker.value(name, None)?;
                    equate = Some(ty);
                    TermKind::Var {
                        declaration,
                        name: name.clone(),
                        type_args,
                    }
                }
            }
            resin_ast::TermKind::TypeApply { function, args } => {
                let (kind, result) = self.type_application(function, args, span, out.clone())?;
                equate = result;
                kind
            }
            resin_ast::TermKind::Type { ty } => {
                let ann = self.annotation(ty, true);
                equate = Some(Ty::Type.into());
                TermKind::Type {
                    ty: ann.into_tree(),
                }
            }
            resin_ast::TermKind::Break | resin_ast::TermKind::Continue => {
                if self.checker.loop_depth == 0 {
                    return Err(GenerateError::inference(
                        span,
                        "break and continue require a loop body",
                    ));
                }
                equate = Some(Ty::union([]).into());
                if matches!(term.val, resin_ast::TermKind::Break) {
                    TermKind::Break
                } else {
                    TermKind::Continue
                }
            }
            resin_ast::TermKind::Return { value } => {
                let value = self.child(value, Some(self.checker.result.clone()));
                equate = Some(Ty::union([]).into());
                TermKind::Return {
                    value: Box::new(value),
                }
            }
            resin_ast::TermKind::Try { value } => {
                let input = self.child(value, None);
                let result = self.checker.result.clone();
                self.constrain((span, Constraint::Try(input.ty.clone(), out.clone(), result)));
                TermKind::Try {
                    value: Box::new(input),
                }
            }
            resin_ast::TermKind::Match { value, arms } => self.match_expression(value, arms, &out),
            resin_ast::TermKind::If { cond, then, els } => {
                let cond = self.child(cond, None);
                self.constrain((cond.span, Constraint::Boolean(cond.ty.clone())));
                let then = self.child(then, Some(out.clone()));
                let els = self.child(els, Some(out.clone()));
                TermKind::If {
                    cond: Box::new(cond),
                    then: Box::new(then),
                    els: Box::new(els),
                }
            }
            resin_ast::TermKind::While { cond, body } => {
                let depth = std::mem::replace(&mut self.checker.loop_depth, 0);
                let cond = self.child(cond, None);
                self.checker.loop_depth = depth + 1;
                self.constrain((cond.span, Constraint::Boolean(cond.ty.clone())));
                let body = self.child(body, None);
                self.checker.loop_depth = depth;
                equate = Some(Ty::Unit.into());
                TermKind::While {
                    cond: Box::new(cond),
                    body: Box::new(body),
                }
            }
            resin_ast::TermKind::Block { stmts, tail } => {
                self.checker.scopes.push_at(term.span);
                let exits = stmts.iter().any(statement_exits);
                let stmts = stmts
                    .iter()
                    .map(|stmt| self.statement(stmt))
                    .collect::<Vec<_>>();
                let tail = self.child(tail, (!exits).then(|| out.clone()));
                if exits {
                    equate = Some(Ty::union([]).into());
                }
                self.checker.scopes.pop();
                TermKind::Block {
                    stmts,
                    tail: Box::new(tail),
                }
            }
            resin_ast::TermKind::Record { fields } => {
                let fields = fields
                    .iter()
                    .map(|(name, term)| (name.clone(), self.child(term, None)))
                    .collect::<Vec<_>>();
                self.constrain((
                    span,
                    Constraint::Record(
                        fields
                            .iter()
                            .map(|(name, term)| (name.val.clone(), term.ty.clone()))
                            .collect(),
                        out.clone(),
                    ),
                ));
                TermKind::Record { fields }
            }
            resin_ast::TermKind::Array { elems } => {
                let element = self.checker.typing.solver.fresh();
                let elems = elems
                    .iter()
                    .map(|elem| self.child(elem, Some(element.clone())))
                    .collect::<Vec<_>>();
                equate = Some(Type::Node(Head::Array(elems.len()), vec![element]));
                TermKind::Array { elems }
            }
            resin_ast::TermKind::Builtin {
                name,
                name_span,
                args,
            } => self.builtin(name, name_span, args, span, &out, &expected),
            resin_ast::TermKind::MethodCall { .. } => unreachable!("receiver call was desugared"),
            resin_ast::TermKind::Call { func, args } => {
                let (kind, result) = self.call(func, args, span, &out, &expected)?;
                equate = result;
                kind
            }
            resin_ast::TermKind::Assign { place, value } => {
                let place = self.child(place, None);
                let value = self.child(value, Some(place.ty.clone()));
                equate = Some(Type::from(Ty::Unit));
                TermKind::Assign {
                    place: Box::new(place),
                    value: Box::new(value),
                }
            }
            resin_ast::TermKind::Address { place } => {
                let place = self.child(place, None);
                let mut bases = vec![];
                let mut base = &place;
                loop {
                    base = match &base.kind {
                        TermKind::Field { base, .. } => {
                            bases.push(super::infer::AddressBase::Field {
                                receiver: base.ty.clone(),
                            });
                            base
                        }
                        TermKind::Deref { pointer } => {
                            bases.push(super::infer::AddressBase::Deref {
                                pointer: pointer.ty.clone(),
                            });
                            pointer
                        }
                        _ => break,
                    };
                }
                self.constrain((
                    span,
                    Constraint::Address {
                        pointee: place.ty.clone(),
                        bases,
                        result: out.clone(),
                    },
                ));
                TermKind::Address {
                    place: Box::new(place),
                }
            }
            resin_ast::TermKind::Deref { pointer } => {
                let pointer = self.child(pointer, None);
                self.constrain((span, Constraint::Deref(pointer.ty.clone(), out.clone())));
                TermKind::Deref {
                    pointer: Box::new(pointer),
                }
            }
            resin_ast::TermKind::Field { base, name } => {
                let base = self.child(base, None);
                if let TermKind::Type { ty } = &base.kind {
                    self.method_reference(ty.clone(), name, None, out.clone())
                } else {
                    self.checker
                        .scopes
                        .record_members(name.span, base.ty.clone(), false);
                    self.constrain((
                        span,
                        Constraint::Field(base.ty.clone(), name.val.clone(), out.clone()),
                    ));
                    TermKind::Field {
                        base: Box::new(base),
                        name: name.clone(),
                    }
                }
            }
        };
        if !contextual
            && matches!(
                term.val,
                resin_ast::TermKind::If { .. } | resin_ast::TermKind::Match { .. }
            )
            && expression_exits(term)
        {
            equate = Some(Ty::union([]).into());
        }
        if let Some(ty) = equate {
            if contextual {
                self.constrain((span, Constraint::Coerce(ty, out.clone())));
            } else {
                self.constrain((span, Constraint::Equal(ty, out.clone())));
            }
        }
        if !propagate && let Some(expected) = expected {
            self.constrain((span, Constraint::Coerce(out.clone(), expected)));
        }
        Ok(Term {
            span,
            actual: out.clone(),
            ty: out,
            kind,
        })
    }

    fn statement(&mut self, stmt: &resin_ast::Stmt) -> Statement {
        let result = self.statement_inner(stmt);
        let kind = match result {
            Ok(statement) => statement,
            Err(error) => {
                match &stmt.val {
                    StmtKind::Declare { pattern, .. } => {
                        let name = &pattern.name;
                        let _ = self
                            .checker
                            .bind(name, Type::Invalid, DefinitionKind::Variable);
                    }
                    StmtKind::DefineType { name, .. } => {
                        self.checker.scopes.define_invalid_type(name);
                    }
                    _ => {}
                }
                self.checker.errors.push(error.clone());
                StatementKind::Error(error)
            }
        };
        Statement { kind }
    }

    fn statement_inner(&mut self, stmt: &resin_ast::Stmt) -> Result<StatementKind> {
        let span = stmt.span;
        Ok(match &stmt.val {
            StmtKind::Const { specs } => {
                self.checker.local_constants(specs)?;
                StatementKind::CompileTimeDefinition
            }
            StmtKind::Define { pattern, ann, init } => {
                let name = &pattern.name;
                let ty = if let Some(ann) = ann {
                    self.annotation(ann, true).ty
                } else {
                    self.checker.typing.solver.fresh()
                };
                let binding = self
                    .checker
                    .bind(name, ty.clone(), DefinitionKind::Variable)
                    .map_err(|error| self.checker.errors.push(error))
                    .ok();
                if let Some(binding) = binding {
                    self.checker.initializers.push(binding);
                }
                let init = self.child(init, Some(ty));
                if binding.is_some() {
                    self.checker.initializers.pop();
                }
                if let Some(binding) = binding {
                    let binding_type = if ann.is_some() {
                        init.ty.clone()
                    } else {
                        Type::value(init.actual.clone())
                    };
                    self.checker.scopes.set_inferred(binding, binding_type);
                }
                StatementKind::Define {
                    mutable: pattern.mutable,
                    binding,
                    name: name.clone(),
                    init,
                }
            }
            StmtKind::Declare { pattern, ann } => {
                let name = &pattern.name;
                let ann = self.annotation(ann, true);
                let binding = self
                    .checker
                    .bind(name, ann.ty.clone(), DefinitionKind::Variable)?;
                StatementKind::Declare {
                    mutable: pattern.mutable,
                    binding,
                    name: name.clone(),
                    ty: ann.into_tree(),
                }
            }
            StmtKind::Struct {
                name,
                body,
                methods,
                type_params,
            } => {
                if let Some(method) = methods.first() {
                    return Err(GenerateError::inference(
                        method.span,
                        "local structs cannot define methods",
                    ));
                }
                self.checker
                    .scopes
                    .nominal(name, type_params, body, self.checker.typing.typer)?;
                StatementKind::CompileTimeDefinition
            }
            StmtKind::DefineType {
                name,
                init,
                type_params,
            } => {
                self.checker.scopes.alias(name, type_params, init)?;
                StatementKind::CompileTimeDefinition
            }
            StmtKind::Expr { term } => {
                let term = self.child(term, None);
                StatementKind::Expr { term }
            }
            _ => {
                return Err(GenerateError::inference(
                    span,
                    "unexpected declaration in function body",
                ));
            }
        })
    }
}

//
// Signature annotation resolution
//

impl typed::Annotation<Type> {
    fn resolve(self, solver: &Solver) -> Result<typed::Annotation<crate::Type>> {
        Ok(typed::Annotation {
            ty: solver.require_complete(&self.ty, self.span)?,
            span: self.span,
        })
    }
}

//
// Dependency groups
//

/// Strongly connected groups in dependency-first order.
fn groups(edges: &[Vec<usize>]) -> Vec<Vec<usize>> {
    struct Walk<'a> {
        edges: &'a [Vec<usize>],
        index: Vec<Option<usize>>,
        low: Vec<usize>,
        active: Vec<bool>,
        stack: Vec<usize>,
        next: usize,
        groups: Vec<Vec<usize>>,
    }
    impl Walk<'_> {
        fn visit(&mut self, node: usize) {
            self.index[node] = Some(self.next);
            self.low[node] = self.next;
            self.next += 1;
            self.stack.push(node);
            self.active[node] = true;
            for &dependency in &self.edges[node] {
                if self.index[dependency].is_none() {
                    self.visit(dependency);
                    self.low[node] = self.low[node].min(self.low[dependency]);
                } else if self.active[dependency] {
                    self.low[node] = self.low[node].min(self.index[dependency].unwrap());
                }
            }
            if self.low[node] == self.index[node].unwrap() {
                let mut group = Vec::new();
                loop {
                    let member = self.stack.pop().unwrap();
                    self.active[member] = false;
                    group.push(member);
                    if member == node {
                        break;
                    }
                }
                self.groups.push(group);
            }
        }
    }
    let n = edges.len();
    let mut walk = Walk {
        edges,
        index: vec![None; n],
        low: vec![0; n],
        active: vec![false; n],
        stack: vec![],
        next: 0,
        groups: vec![],
    };
    for node in 0..n {
        if walk.index[node].is_none() {
            walk.visit(node);
        }
    }
    walk.groups
}

#[cfg(test)]
mod tests;

fn single_argument(args: &[resin_ast::Term], span: Span) -> Result<&resin_ast::Term> {
    match args {
        [arg] => Ok(arg),
        _ => Err(GenerateError::inference(
            span,
            format!("expected 1 argument, found {}", args.len()),
        )),
    }
}

// Syntactic exits do not constrain a block's unreachable tail to its result type.
fn statement_exits(statement: &resin_ast::Stmt) -> bool {
    match &statement.val {
        StmtKind::Expr { term } => expression_exits(term),
        StmtKind::Define { init, .. } => expression_exits(init),
        _ => false,
    }
}

fn expression_exits(term: &resin_ast::Term) -> bool {
    use resin_ast::TermKind::*;
    match &term.val {
        Return { .. } | Break | Continue => true,
        Block { stmts, tail } => stmts.iter().any(statement_exits) || expression_exits(tail),
        If { cond, then, els } => {
            expression_exits(cond) || (expression_exits(then) && expression_exits(els))
        }
        Match { value, arms } => {
            expression_exits(value) || arms.iter().all(|arm| expression_exits(&arm.body))
        }
        While { cond, .. } => expression_exits(cond),
        Call { func, args } => expression_exits(func) || args.iter().any(expression_exits),
        Builtin { name, args, .. } if matches!(name.as_ref(), "&&" | "||") => {
            expression_exits(&args[0])
        }
        Builtin { args, .. } | Array { elems: args } => args.iter().any(expression_exits),
        Record { fields } => fields.iter().any(|(_, value)| expression_exits(value)),
        MethodCall { receiver, args, .. } => {
            expression_exits(receiver) || args.iter().any(expression_exits)
        }
        Assign { place, value } => expression_exits(place) || expression_exits(value),
        Unwrap { value } | Try { value } => expression_exits(value),
        Address { place } => expression_exits(place),
        Deref { pointer } => expression_exits(pointer),
        Field { base, .. } => expression_exits(base),
        _ => false,
    }
}
