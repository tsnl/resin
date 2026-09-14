//! AST → HIR: resolve declarations, infer dependency groups, and complete function bodies.
use crate::{
    Analysis, Annotation, CheckedProgram, DefinitionKind, Function, Module, Parameter, Signature,
};
use crate::{GenerateError, GenerateErrorKind};
use resin_ast::{Program, SourceFile, StmtKind};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    rc::Rc,
    sync::Arc,
};

use self::{
    check::CheckedFile,
    context::{Context, SourceModuleId, SourceOrigin},
    eval::Evaluator,
    scope::{ContextView, DeclarationId, Scopes, Symbol},
};

mod check;
pub(crate) mod context;
mod elaborate;
pub(crate) mod eval;
mod gpu;
mod gpu_projections;
pub(crate) mod infer;
mod primitives;
pub(crate) mod scope;
mod typed;
mod types;

/// Check a standalone source file. Imports require `generate_program`.
pub fn generate(file: &SourceFile) -> Result<Module, GenerateError> {
    if let Some(import) = file.imports.first() {
        return Err(GenerateError {
            span: import.span,
            kind: GenerateErrorKind::UnresolvedImport {
                path: import.val.clone(),
            },
        });
    }
    let mut generator = Generator::new();
    generator.generate_file(file, Scopes::new());
    if !generator.errors.is_empty() {
        return Err(generator.errors.remove(0));
    }
    generator.module.entries = generator.exported_functions(file)?;
    Ok(generator.finish())
}

struct Generator {
    module: Module,
    functions: Vec<Option<Function>>,
    source: Source,
    typer: Context,
    scopes: ContextView,
    function_bindings: HashMap<DeclarationId, FunctionId>,
    errors: Vec<GenerateError>,
}
impl Generator {
    fn new() -> Self {
        Self {
            module: Module::default(),
            functions: vec![],
            source: Source::new("<source>", ""),
            typer: Context::with_builtins(),
            scopes: Scopes::new().finish(),
            function_bindings: HashMap::new(),
            errors: vec![],
        }
    }
    fn generate_file(&mut self, file: &SourceFile, mut scopes: Scopes) {
        let prepared = scopes.prepare(file, &mut self.typer);
        self.errors.extend(prepared.errors);
        let methods = self.declare_methods(prepared.methods, &mut scopes);
        let checked = check::file(file, self, scopes, methods);
        self.errors.extend(checked.errors.iter().cloned());
        self.scopes = checked.context.clone();
        self.define_functions(checked);
    }

    fn finish(mut self) -> Module {
        self.module.types = self.typer.into_definitions();
        self.module.functions = self
            .functions
            .into_iter()
            .map(|function| function.expect("completed declaration"))
            .collect();
        self.module
    }

    fn reserve_function(&mut self, declaration: DeclarationId) -> FunctionId {
        if let Some(&function) = self.function_bindings.get(&declaration) {
            return function;
        }
        let function = FunctionId::from_index(self.functions.len());
        self.functions.push(None);
        self.function_bindings.insert(declaration, function);
        function
    }

    fn function(&self, function: FunctionId) -> &Function {
        self.functions[function.index()]
            .as_ref()
            .expect("completed signature")
    }

    fn function_mut(&mut self, function: FunctionId) -> &mut Function {
        self.functions[function.index()]
            .as_mut()
            .expect("completed signature")
    }
}

struct Export {
    origin: SourceOrigin,
    symbol: Symbol,
}

type Exports = BTreeMap<Arc<str>, Export>;

pub fn generate_program(program: &Program) -> Result<crate::Module, SourceError> {
    let mut compilation = analyze_program(program);
    if !compilation.diagnostics.is_empty() {
        Err(compilation.diagnostics.remove(0))
    } else {
        Ok(compilation.module.expect("successful compilation"))
    }
}

/// Check modules in import order while retaining independent editor facts after errors.
pub fn analyze_program(program: &Program) -> CheckedProgram {
    let diagnostics = invalid_dependencies(program);
    if !diagnostics.is_empty() {
        return CheckedProgram {
            module: None,
            diagnostics,
            semantics: Default::default(),
        };
    }
    let mut builder = ProgramBuilder::new(program);
    for index in 0..program.modules.len() {
        builder.module(index);
    }
    builder.entries();
    builder.finish()
}

fn invalid_dependencies(program: &Program) -> Vec<SourceError> {
    let mut errors = Vec::new();
    for (index, source) in program.modules.iter().enumerate() {
        for &(span, dependency) in &source.imports {
            if dependency >= index {
                errors.push(source.error(span, "import dependency must precede its consumer"));
            }
        }
    }
    errors
}

struct ProgramBuilder<'a> {
    program: &'a Program,
    generator: Generator,
    data: Rc<RefCell<Analysis>>,
    exports: Vec<Exports>,
    diagnostics: Vec<SourceError>,
}

impl<'a> ProgramBuilder<'a> {
    fn new(program: &'a Program) -> Self {
        Self {
            program,
            generator: Generator::new(),
            data: Default::default(),
            exports: vec![],
            diagnostics: vec![],
        }
    }

    fn module(&mut self, index: usize) {
        let source = &self.program.modules[index];
        self.begin_module(index);
        let mut scopes = Scopes::for_source(source.source.clone(), self.data.clone());
        let mut names = BTreeMap::new();
        self.imports(index, &mut scopes, &mut names);
        self.declarations(index, &mut names);
        self.generator.generate_file(&source.file, scopes);
        self.diagnostics.extend(
            std::mem::take(&mut self.generator.errors)
                .into_iter()
                .map(|e| source.error(e.span, e)),
        );
        self.export_module(index, &names);
    }

    fn begin_module(&mut self, index: usize) {
        let source = &self.program.modules[index];
        self.generator.source = source.source.clone();
    }

    fn imports(
        &mut self,
        index: usize,
        scopes: &mut Scopes,
        names: &mut BTreeMap<Arc<str>, SourceOrigin>,
    ) {
        let source = &self.program.modules[index];
        for &(span, dependency) in &source.imports {
            self.data.borrow_mut().imports.insert(
                SourceLocation {
                    source: source.source.clone(),
                    span,
                },
                self.program.modules[dependency].source.clone(),
            );
            self.import_exports(index, dependency, span, scopes, names);
        }
    }

    fn import_exports(
        &mut self,
        index: usize,
        dependency: usize,
        span: Span,
        scopes: &mut Scopes,
        names: &mut BTreeMap<Arc<str>, SourceOrigin>,
    ) {
        for (name, export) in &self.exports[dependency] {
            match bind(self.program, index, names, name, export.origin, span) {
                Ok(false) => continue,
                Err(error) => self.diagnostics.push(error),
                Ok(true) => {}
            }
            scopes.import(name.clone(), export.symbol);
        }
    }

    fn declarations(&mut self, index: usize, names: &mut BTreeMap<Arc<str>, SourceOrigin>) {
        for stmt in self.program.modules[index].file.declarations() {
            let Some(name) = declaration_name(&stmt.val) else {
                continue;
            };
            let origin = SourceOrigin {
                module: SourceModuleId::from_index(index),
                span: name.span,
            };
            if let Err(error) = bind(self.program, index, names, &name.val, origin, name.span) {
                self.diagnostics.push(error);
            }
        }
    }

    fn export_module(&mut self, index: usize, names: &BTreeMap<Arc<str>, SourceOrigin>) {
        let source = &self.program.modules[index];
        let (symbols, errors) = self.generator.exports(&source.file);
        self.diagnostics
            .extend(errors.into_iter().map(|e| source.error(e.span, e)));
        self.exports.push(
            symbols
                .into_iter()
                .filter_map(|(name, symbol)| {
                    names
                        .get(&name)
                        .copied()
                        .map(|origin| (name, Export { origin, symbol }))
                })
                .collect(),
        );
    }

    fn entries(&mut self) {
        let Some(source) = self.program.modules.last() else {
            return;
        };
        match self.generator.exported_functions(&source.file) {
            Ok(entries) => self.generator.module.entries = entries,
            Err(error) => self.diagnostics.push(source.error(error.span, error)),
        }
    }

    fn finish(self) -> CheckedProgram {
        let module = if self.diagnostics.is_empty() {
            Some(self.generator.finish())
        } else {
            None
        };
        CheckedProgram {
            module,
            diagnostics: self.diagnostics,
            semantics: self.data.borrow().clone(),
        }
    }
}

fn declaration_name(stmt: &StmtKind) -> Option<&Ident> {
    match stmt {
        StmtKind::ForeignType { name }
        | StmtKind::ForeignFunction { name, .. }
        | StmtKind::IntrinsicFunction { name, .. }
        | StmtKind::Function { name, .. }
        | StmtKind::Define { name, .. }
        | StmtKind::DefineType { name, .. }
        | StmtKind::Struct { name, .. }
        | StmtKind::Declare { name, .. } => Some(name),
        _ => None,
    }
}

fn bind(
    program: &Program,
    module: usize,
    names: &mut BTreeMap<Arc<str>, SourceOrigin>,
    name: &Arc<str>,
    origin: SourceOrigin,
    span: Span,
) -> Result<bool, SourceError> {
    if let Some(previous) = names.get(name) {
        if *previous == origin {
            return Ok(false);
        }
        let location =
            |origin: &SourceOrigin| program.modules[origin.module.index()].location(origin.span);
        let mut error = program.modules[module].error(
            span,
            format!(
                "conflicting binding `{name}`\n  first defined at {}\n  also defined at {}",
                location(previous),
                location(&origin),
            ),
        );
        for origin in [previous, &origin] {
            error.related.push(SourceNote {
                location: SourceLocation {
                    source: program.modules[origin.module.index()].source.clone(),
                    span: origin.span,
                },
                message: "defined here".into(),
            });
        }
        return Err(error);
    }
    names.insert(name.clone(), origin);
    Ok(true)
}

impl Generator {
    fn exported_functions(
        &self,
        file: &SourceFile,
    ) -> Result<BTreeMap<Arc<str>, FunctionId>, GenerateError> {
        let (symbols, errors) = self.exports(file);
        if let Some(error) = errors.into_iter().next() {
            return Err(error);
        }
        Ok(symbols
            .into_iter()
            .filter_map(|(name, symbol)| {
                Some((
                    name,
                    self.function_bindings.get(&symbol.definition).copied()?,
                ))
            })
            .collect())
    }
    fn exports(&self, file: &SourceFile) -> (BTreeMap<Arc<str>, Symbol>, Vec<GenerateError>) {
        let mut exports = BTreeMap::new();
        let mut errors = Vec::new();
        for name in &file.exports {
            if exports.contains_key(&name.val) {
                errors.push(GenerateError {
                    span: name.span,
                    kind: GenerateErrorKind::DuplicateExport {
                        name: name.val.clone(),
                    },
                });
            } else if let Some(symbol) = self.symbol(&name.val) {
                exports.insert(name.val.clone(), symbol);
            } else {
                errors.push(GenerateError {
                    span: name.span,
                    kind: GenerateErrorKind::UnknownExport {
                        name: name.val.clone(),
                    },
                });
            }
        }
        (exports, errors)
    }
}

struct Method<'a> {
    owner: TypeId,
    statement: &'a resin_ast::Stmt,
}

struct PreparedTypes<'a> {
    methods: Vec<Method<'a>>,
    errors: Vec<GenerateError>,
}

impl Scopes {
    fn prepare<'a>(&mut self, file: &'a SourceFile, typer: &mut Context) -> PreparedTypes<'a> {
        let mut errors = Vec::new();
        let mut methods = Vec::new();
        for stmt in &file.stmts {
            let result = match &stmt.val {
                StmtKind::Define { .. } | StmtKind::Declare { .. } | StmtKind::Expr { .. } => {
                    Err(GenerateError {
                        span: stmt.span,
                        kind: GenerateErrorKind::InvalidModuleItem,
                    })
                }
                StmtKind::ForeignType { name } => self
                    .define_alias(
                        name,
                        Ty::Foreign {
                            name: name.val.clone(),
                        },
                    )
                    .map(|_| ())
                    .map_err(|name| GenerateError {
                        span: stmt.span,
                        kind: GenerateErrorKind::DuplicateType { name },
                    }),
                _ => Ok(()),
            };
            if let Err(error) = result {
                errors.push(error);
            }
        }
        for stmt in &file.stmts {
            let result = match &stmt.val {
                StmtKind::DefineType {
                    name,
                    init,
                    type_params,
                } => self.alias(name, type_params, init),
                StmtKind::Struct {
                    name,
                    body,
                    methods: owned,
                    type_params,
                } => (|| {
                    let id = self.nominal(name, type_params, body, typer)?;
                    methods.extend(owned.iter().map(|statement| Method {
                        owner: id,
                        statement,
                    }));
                    Ok(())
                })(),
                _ => Ok(()),
            };
            if let Err(error) = result {
                errors.push(error);
            }
        }
        PreparedTypes { methods, errors }
    }
    fn nominal(
        &mut self,
        name: &Ident,
        parameters: &[Ident],
        body: &resin_ast::Type,
        typer: &mut Context,
    ) -> Result<TypeId, GenerateError> {
        let definition = typer.declare_type(name.val.clone());
        let declaration =
            self.define_type(name, definition)
                .map_err(|duplicate| GenerateError {
                    span: name.span,
                    kind: GenerateErrorKind::DuplicateType { name: duplicate },
                })?;
        let captures = self.type_parameters();
        self.push_at(Span {
            start: name.span.end,
            end: body.span.end,
        });
        let result = (|| {
            let parameters = parameters
                .iter()
                .map(|name| self.define_type_parameter(name))
                .collect::<Result<Vec<_>, _>>()?;
            self.define_nominal_scheme(declaration, definition, &parameters, &captures);
            let parameters = captures.into_iter().chain(parameters).collect();
            if let resin_ast::TypeKind::Record { fields } = &body.val {
                self.record_field_definitions(definition, fields);
            }
            let body = Evaluator {
                scopes: self.view(),
            }
            .scheme(body)?;
            typer
                .define_nominal(definition, parameters, body)
                .map_err(|error| GenerateError::typing(name.span, error))?;
            Ok(definition)
        })();
        self.pop();
        result
    }

    fn alias(
        &mut self,
        name: &Ident,
        parameters: &[Ident],
        body: &resin_ast::Type,
    ) -> Result<(), GenerateError> {
        let declaration = self.begin_alias(name)?;
        self.push_at(Span {
            start: parameters
                .first()
                .map_or(body.span.start, |parameter| parameter.span.start),
            end: body.span.end,
        });
        let result = (|| {
            let parameters = parameters
                .iter()
                .map(|name| self.define_type_parameter(name))
                .collect::<Result<Vec<_>, _>>()?;
            self.set_parameters(declaration, &parameters);
            Evaluator {
                scopes: self.view(),
            }
            .scheme(body)
        })();
        self.pop();
        self.finish_alias(declaration, result.as_ref().ok().cloned());
        result.map(|_| ())
    }
}

type Declarations = BTreeMap<Arc<str>, DeclarationId>;

impl Generator {
    fn declare_methods(&mut self, methods: Vec<Method<'_>>, scopes: &mut Scopes) -> Declarations {
        let mut declarations = BTreeMap::new();
        // All module types and aliases are available before method signatures.
        for method in methods {
            if let Err(error) = self.declare_method(method, scopes, &mut declarations) {
                self.errors.push(error);
            }
        }
        declarations
    }

    fn declare_method(
        &mut self,
        method: Method<'_>,
        scopes: &mut Scopes,
        declarations: &mut Declarations,
    ) -> Result<(), GenerateError> {
        let StmtKind::Function {
            type_params,
            name,
            params,
            result,
            decorators,
            ..
        } = &method.statement.val
        else {
            return Ok(());
        };
        let declaration = reserve_method(name, scopes, declarations)?;
        let definition = method.owner;
        self.typer.method_owners.insert(declaration, definition);
        scopes.record_method_definition(definition, declaration);
        if decorators
            .iter()
            .any(|decorator| !gpu::is_bridge(&decorator.val))
        {
            return Err(GenerateError::inference(
                name.span,
                "methods cannot be shader entries",
            ));
        }
        // Ordinary methods are reserved here and their schemes are constructed
        // once, with other function declarations. Native GPU bridges still need
        // concrete metadata before checking any calls in this file.
        if decorators.is_empty() {
            let function = self.reserve_function(declaration);
            if name.val.rsplit('.').next() == Some("drop") {
                // Ownership affects every signature's operations, including GPU
                // projection. Reserve its identity before inference; validate the
                // completed hook signature with the other method declarations.
                self.typer.define_drop(definition, function);
            }
            return Ok(());
        }
        require_fixed_bridge_signature(type_params)?;
        if !self.typer.nominal_schemes[&definition]
            .type_params
            .is_empty()
        {
            return Err(GenerateError::inference(
                name.span,
                "GPU bridges require a fixed owner type",
            ));
        }
        let evaluator = Evaluator {
            scopes: scopes.view(),
        };
        let signature = method_signature(&evaluator, declaration, params, result)?;
        let function = self.declare_function(name, &signature)?;
        self.register_method(definition, name, function)?;
        if decorators.len() > 1 {
            return Err(GenerateError::inference(
                name.span,
                "a method can have only one bridge decorator",
            ));
        }
        if let Some(decorator) = decorators.first() {
            if decorator.val.as_ref() == "gpu_allocator" {
                self.register_gpu_allocator(definition, function, name)?;
            } else {
                self.register_gpu_bridge(definition, function, decorator)?;
            }
        }
        Ok(())
    }

    fn register_gpu_allocator(
        &mut self,
        owner: TypeId,
        function: FunctionId,
        name: &Ident,
    ) -> Result<(), GenerateError> {
        let declaration = self.typer.declared_function(function);
        let params = &declaration.params;
        let receiver_matches = params
            .first()
            .is_some_and(|receiver| self.typer.receiver_definition(receiver) == Some(owner));
        let result_matches = matches!(&declaration.result, Ty::Result { value, .. } if **value == Ty::GpuView);
        if !receiver_matches
            || params.get(1..) != Some(&[Ty::UInt64, Ty::UInt64, Ty::Int32][..])
            || !result_matches
        {
            return Err(GenerateError::inference(
                name.span,
                "@gpu_allocator requires (self, bytes: ulong, alignment: ulong, memory: int) -> Result<GpuView, E>",
            ));
        }
        if self.typer.gpu_allocators.insert(owner, function).is_some() {
            return Err(GenerateError::inference(
                name.span,
                "a type can declare only one GPU allocator",
            ));
        }
        Ok(())
    }

    fn register_method(
        &mut self,
        owner: TypeId,
        name: &Ident,
        function: FunctionId,
    ) -> Result<(), GenerateError> {
        let short = name.val.rsplit('.').next().unwrap();
        if !self.typer.define_method(owner, short.into(), function) {
            return Err(GenerateError::inference(name.span, "duplicate method"));
        }
        if short == "drop" {
            self.register_drop(owner, name, function)?;
        }
        Ok(())
    }

    fn register_drop(
        &mut self,
        owner: TypeId,
        name: &Ident,
        function: FunctionId,
    ) -> Result<(), GenerateError> {
        let signature = &self.function(function).signature;
        let owner_parameters = &self.typer.nominal_schemes[&owner].type_params;
        let pointer = crate::Type::Pointer {
            pointee: Box::new(crate::Type::Defined {
                definition: owner,
                arguments: owner_parameters
                    .iter()
                    .map(|parameter| crate::Type::Parameter {
                        parameter: parameter.id,
                    })
                    .collect(),
            }),
        };
        if !signature
            .type_params
            .iter()
            .map(|parameter| parameter.id)
            .eq(owner_parameters.iter().map(|parameter| parameter.id))
            || signature.params.len() != 1
            || signature.params[0].annotation.ty != pointer
            || signature.result.ty != crate::Type::Unit
        {
            return Err(GenerateError::inference(
                name.span,
                "drop must have signature drop(receiver: Ptr<T>) -> ()",
            ));
        }
        self.typer.define_drop(owner, function);
        Ok(())
    }
}

fn reserve_method(
    name: &Ident,
    scopes: &mut Scopes,
    declarations: &mut Declarations,
) -> Result<DeclarationId, GenerateError> {
    if declarations.contains_key(&name.val) {
        return Err(duplicate(name));
    }
    let declaration = scopes
        .define_inferred(name, infer::Type::Invalid, DefinitionKind::Function)
        .map_err(|_| duplicate(name))?;
    declarations.insert(name.val.clone(), declaration);
    Ok(declaration)
}

fn duplicate(name: &Ident) -> GenerateError {
    GenerateError {
        span: name.span,
        kind: GenerateErrorKind::DuplicateValue {
            name: name.val.clone(),
        },
    }
}

fn method_signature(
    evaluator: &Evaluator<'_>,
    declaration: DeclarationId,
    params: &[(Ident, resin_ast::Type)],
    result: &resin_ast::Type,
) -> Result<typed::Signature, GenerateError> {
    Ok(typed::Signature {
        type_params: vec![],
        declaration: Some(declaration),
        parameters: vec![],
        params: params
            .iter()
            .map(|(name, ty)| Ok((name.clone(), method_annotation(evaluator, ty)?)))
            .collect::<Result<_, GenerateError>>()?,
        result: method_annotation(evaluator, result)?,
    })
}

fn method_annotation(
    evaluator: &Evaluator<'_>,
    source: &resin_ast::Type,
) -> Result<typed::Annotation<crate::Type>, GenerateError> {
    Ok(typed::Annotation {
        ty: types::ty(&evaluator.ty(source)?),
        span: source.span,
    })
}

fn elaborate_signature(source: &typed::Signature) -> Signature {
    Signature {
        type_params: source.type_params.clone(),
        params: source
            .params
            .iter()
            .enumerate()
            .map(|(index, (name, a))| Parameter {
                binding: source.parameters.get(index).copied().flatten(),
                name: name.clone(),
                annotation: elaborate_annotation(a),
            })
            .collect(),
        result: elaborate_annotation(&source.result),
    }
}
fn elaborate_annotation(source: &typed::Annotation<crate::Type>) -> Annotation {
    Annotation {
        ty: source.ty.clone(),
        span: source.span,
    }
}

impl Generator {
    fn declare_function(
        &mut self,
        name: &Ident,
        source: &typed::Signature,
    ) -> Result<FunctionId, GenerateError> {
        check_parameters(source)?;
        let id = self.reserve_function(source.declaration.expect("checked function"));
        // Native bridges and the remaining compiler-provided method declarations
        // consume concrete signatures. Ordinary source calls use schemes in scopes.
        let solver = infer::Solver::default();
        let params = source
            .params
            .iter()
            .map(|(_, a)| solver.resolve(&infer::Type::from_hir(&a.ty)))
            .collect::<Option<Vec<_>>>();
        if let (Some(params), Some(result)) = (
            params,
            solver.resolve(&infer::Type::from_hir(&source.result.ty)),
        ) {
            self.typer.register_function(id, params, result);
        }
        self.functions[id.index()] = Some(Function {
            location: Some(SourceLocation {
                source: self.source.clone(),
                span: name.span,
            }),
            name: name.val.clone(),
            signature: elaborate_signature(source),
            body: None,
            foreign_header: None,
        });
        Ok(id)
    }
    fn declare_foreign(
        &mut self,
        header: &str,
        name: &Ident,
        signature: &typed::Signature,
    ) -> Result<FunctionId, GenerateError> {
        let id = self.declare_function(name, signature)?;
        let shape = |ty: &crate::Type| {
            if matches!(ty, crate::Type::Pointer { .. }) {
                Some(Ty::Pointer {
                    pointee: Box::new(Ty::Unit),
                })
            } else {
                infer::Solver::default().resolve(&infer::Type::from_hir(ty))
            }
        };
        let invalid = || GenerateError {
            span: name.span,
            kind: GenerateErrorKind::InvalidForeignSignature,
        };
        // Native ABI classification needs pointer width, not the pointee layout.
        // Preserve actual source annotations for LIR's concrete foreign signature.
        let params = signature
            .params
            .iter()
            .map(|(_, annotation)| shape(&annotation.ty))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(invalid)?;
        let result = shape(&signature.result.ty).ok_or_else(invalid)?;
        let foreign = Foreign {
            header: header.into(),
            params,
        };
        if !foreign.valid(&result) {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::InvalidForeignSignature,
            });
        }
        self.function_mut(id).foreign_header = Some(foreign.header);
        Ok(id)
    }
}

fn check_parameters(signature: &typed::Signature) -> Result<(), GenerateError> {
    let mut names = std::collections::HashSet::new();
    for (name, _) in &signature.params {
        crate::lower::infer::check_binding_name(name)?;
        if !names.insert(&name.val) {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateValue {
                    name: name.val.clone(),
                },
            });
        }
    }
    Ok(())
}

impl Generator {
    fn declare_checked_functions(&mut self, checked: &CheckedFile) -> Vec<GenerateError> {
        let mut errors = Vec::new();
        for declaration in &checked.declarations {
            let Some(signature) = checked.signatures.get(&declaration.id) else {
                continue;
            };
            if let Err(error) = self.declare_checked(declaration, signature) {
                errors.push(error);
            }
        }
        errors
    }

    fn declare_checked(
        &mut self,
        declaration: &typed::Declaration,
        signature: &typed::Signature,
    ) -> Result<(), GenerateError> {
        let name = &declaration.name;
        match &declaration.kind {
            typed::DeclarationKind::Function { decorators } => {
                let id = self.declare_source_function(name, signature)?;
                for decorator in decorators {
                    if (!gpu::is_bridge(&decorator.val)) || !name.val.contains('.') {
                        self.declare_shader(id, decorator)?;
                    }
                }
            }
            typed::DeclarationKind::Intrinsic { operation } => {
                let id = self.declare_source_function(name, signature)?;
                if !gpu_projections::define(
                    &mut self.typer,
                    self.functions[id.index()].as_mut().unwrap(),
                    id,
                    operation,
                )? {
                    primitives::define(self.function_mut(id), operation)?;
                }
            }
            typed::DeclarationKind::Foreign { header } => {
                self.declare_foreign(header, name, signature)?;
            }
        }
        Ok(())
    }

    fn declare_source_function(
        &mut self,
        name: &Ident,
        signature: &typed::Signature,
    ) -> Result<FunctionId, GenerateError> {
        let declaration = signature.declaration.expect("checked declaration");
        if let Some(&function) = self.function_bindings.get(&declaration)
            && self.functions[function.index()].is_some()
        {
            return Ok(function);
        }
        let function = self.declare_function(name, signature)?;
        if let Some(&owner) = self.typer.method_owners.get(&declaration) {
            self.register_method(owner, name, function)?;
        }
        Ok(function)
    }

    fn declare_shader(&mut self, id: FunctionId, decorator: &Ident) -> Result<(), GenerateError> {
        if !self.function(id).signature.type_params.is_empty() {
            return Err(shader_error(
                decorator,
                "shader entries require a fixed signature without template parameters",
            ));
        }
        let stage = shader_stage(decorator)?;
        if self.module.shaders.contains_key(&id) {
            return Err(shader_error(
                decorator,
                "a function can have only one shader decorator",
            ));
        }
        let signature = &self.function(id).signature;
        let parameters = signature
            .params
            .iter()
            .map(|parameter| self.stage_shape(&parameter.annotation.ty, decorator))
            .collect::<Result<Vec<_>, _>>()?;
        let result = self.stage_shape(&signature.result.ty, decorator)?;
        resin_types::shader::validate(&self.typer, &parameters, &result, false, stage)
            .map_err(|message| shader_error(decorator, &message))?;
        self.module.shaders.insert(
            id,
            resin_types::shader::ShaderEntry {
                stage: stage.into(),
                embedded: false,
            },
        );
        Ok(())
    }

    fn stage_shape(&self, ty: &crate::Type, decorator: &Ident) -> Result<Ty, GenerateError> {
        let mut active = std::collections::BTreeSet::new();
        let mut remaining = 65536;
        self.stage_shape_inner(
            &infer::Type::from_hir(ty),
            decorator,
            &mut active,
            0,
            &mut remaining,
        )
    }

    fn stage_shape_inner(
        &self,
        ty: &infer::Type,
        decorator: &Ident,
        active: &mut std::collections::BTreeSet<crate::Type>,
        depth: usize,
        remaining: &mut usize,
    ) -> Result<Ty, GenerateError> {
        use infer::{Head, Type};
        if depth >= 256 {
            return Err(shader_error(
                decorator,
                "shader interface expansion exceeds the HIR depth limit of 256",
            ));
        }
        if *remaining == 0 {
            return Err(shader_error(
                decorator,
                "shader interface expansion exceeds the HIR size limit of 65536",
            ));
        }
        *remaining -= 1;
        let solver = infer::Solver::default();
        let ty = solver.shape_hint(ty);
        match &ty {
            // Stage signatures constrain the root to be a pointer. Its pointee
            // layout is checked when LIR materializes the actual signature.
            Type::Node(Head::Pointer, _) => Ok(Ty::Pointer {
                pointee: Box::new(Ty::Unit),
            }),
            Type::Node(
                Head::Nominal { definition } | Head::Atom(Ty::Defined { definition }),
                _,
            ) => {
                let application = solver.require_bounded(&ty, decorator.span)?;
                if !active.insert(application.clone()) {
                    return Err(shader_error(decorator, "recursive shader interface layout"));
                }
                let body = self
                    .typer
                    .nominal_body(&ty, &solver)
                    .or_else(|| {
                        self.typer
                            .body(&Ty::Defined {
                                definition: *definition,
                            })
                            .ok()
                            .map(infer::Type::from)
                    })
                    .ok_or_else(|| shader_error(decorator, "incomplete shader interface type"))?;
                let result = self.stage_shape_inner(&body, decorator, active, depth + 1, remaining);
                active.remove(&application);
                result
            }
            Type::Node(Head::Record(names), fields) => Ok(Ty::Record {
                fields: names
                    .iter()
                    .zip(fields)
                    .map(|(name, ty)| {
                        Ok(RecordField {
                            name: name.clone(),
                            ty: self.stage_shape_inner(
                                ty,
                                decorator,
                                active,
                                depth + 1,
                                remaining,
                            )?,
                        })
                    })
                    .collect::<Result<_, GenerateError>>()?,
            }),
            _ => solver
                .resolve(&ty)
                .ok_or_else(|| shader_error(decorator, "shader entries require a fixed signature")),
        }
    }

    fn define_functions(&mut self, checked: CheckedFile) {
        for (declaration, body) in checked.bodies {
            let Some(signature) = checked.signatures.get(&declaration) else {
                continue;
            };
            let Some(&id) = self.function_bindings.get(&declaration) else {
                continue;
            };
            let function = self.function_mut(id);
            function.signature = elaborate_signature(signature);
            function.body = Some(body);
        }
    }
}

fn shader_stage(decorator: &Ident) -> Result<&'static str, GenerateError> {
    match decorator.val.as_ref() {
        "compute_shader" => Ok("compute"),
        "vertex_shader" => Ok("vertex"),
        "fragment_shader" => Ok("fragment"),
        _ => Err(shader_error(decorator, "unknown decorator")),
    }
}

fn shader_error(decorator: &Ident, message: &str) -> GenerateError {
    GenerateError {
        span: decorator.span,
        kind: GenerateErrorKind::InvalidShader {
            message: message.into(),
        },
    }
}

impl Generator {
    fn symbol(&self, name: &str) -> Option<Symbol> {
        self.scopes
            .lookup(name, false)
            .or_else(|| self.scopes.lookup(name, true))
            .map(|definition| Symbol { definition })
    }
}

fn require_fixed_bridge_signature(params: &[Ident]) -> Result<(), GenerateError> {
    match params.first() {
        None => Ok(()),
        Some(parameter) => Err(GenerateError::inference(
            parameter.span,
            "GPU bridge declarations require a fixed signature without template parameters",
        )),
    }
}
