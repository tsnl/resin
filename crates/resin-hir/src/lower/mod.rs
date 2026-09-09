//! AST → HIR: declare modules and functions, check bodies, then elaborate a resolved tree.
use crate::{
    Analysis, Annotation, CheckedProgram, DefinitionKind, Function, Module, Parameter, Signature,
};
use crate::{GenerateError, GenerateErrorKind};
use resin_ast::{Program, SourceFile, StmtKind};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap},
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
pub(crate) mod infer;
pub(crate) mod scope;
mod typed;

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
    generator.finish()
}

struct Generator {
    module: Module,
    source: Source,
    source_module: SourceModuleId,
    typer: Context,
    scopes: ContextView,
    function_bindings: HashMap<DeclarationId, FunctionId>,
    errors: Vec<GenerateError>,
}
impl Generator {
    fn new() -> Self {
        Self {
            module: Module::default(),
            source: Source::new("<source>", ""),
            source_module: SourceModuleId::from_index(0),
            typer: Context::with_builtins(),
            scopes: Scopes::new().finish(),
            function_bindings: HashMap::new(),
            errors: vec![],
        }
    }
    fn generate_file(&mut self, file: &SourceFile, mut scopes: Scopes) {
        self.errors
            .extend(scopes.prepare(file, &mut self.typer, self.source_module));
        let methods = self.declare_methods(file, &mut scopes);
        let checked = check::file(file, &mut self.typer, scopes, self.source_module, methods);
        self.errors.extend(checked.errors.iter().cloned());
        self.scopes = checked.context.clone();
        self.declare_checked_functions(file, &checked);
        self.elaborate_functions(file, &checked);
    }

    fn finish(mut self) -> Result<Module, GenerateError> {
        self.module.types = self
            .typer
            .into_definitions()
            .map_err(|error| GenerateError::typing(Span { start: 0, end: 0 }, error))?;
        Ok(self.module)
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
        self.generator.source_module = SourceModuleId::from_index(index);
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

    fn finish(mut self) -> CheckedProgram {
        let module = if self.diagnostics.is_empty() {
            self.generator
                .finish()
                .map_err(|error| {
                    self.diagnostics.push(program_error(self.program, error));
                })
                .ok()
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
        | StmtKind::Function { name, .. }
        | StmtKind::Define { name, .. }
        | StmtKind::DefineType { name, .. }
        | StmtKind::Struct { name, .. }
        | StmtKind::Declare { name, .. } => Some(name),
        _ => None,
    }
}

fn program_error(program: &Program, error: GenerateError) -> SourceError {
    if let Some(source) = program.modules.last() {
        source.error(error.span, error)
    } else {
        SourceError::new(
            Source::new("<source>", ""),
            Some(error.span),
            error.to_string(),
        )
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

impl Scopes {
    fn prepare(
        &mut self,
        file: &SourceFile,
        typer: &mut Context,
        source_module: crate::lower::context::SourceModuleId,
    ) -> Vec<GenerateError> {
        let mut errors = Vec::new();
        for stmt in file.declarations() {
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
        for stmt in file.declarations() {
            let result = match &stmt.val {
                StmtKind::DefineType { name, init } => match self.annotation(init, typer) {
                    Ok(ty) => {
                        self.define_alias(name, ty)
                            .map(|_| ())
                            .map_err(|name| GenerateError {
                                span: init.span,
                                kind: GenerateErrorKind::DuplicateType { name },
                            })
                    }
                    Err(error) => {
                        self.define_invalid_type(name);
                        Err(error)
                    }
                },
                StmtKind::Struct { name, body } => (|| {
                    let id = typer.declare_type(
                        name.val.clone(),
                        crate::lower::context::SourceOrigin {
                            module: source_module,
                            span: name.span,
                        },
                    );
                    self.define_type(name, id).map_err(|name| GenerateError {
                        span: stmt.span,
                        kind: GenerateErrorKind::DuplicateType { name },
                    })?;
                    let ty = self.annotation(body, typer)?;
                    typer
                        .define_type(id, ty)
                        .map_err(|error| GenerateError::typing(body.span, error))
                })(),
                _ => Ok(()),
            };
            if let Err(error) = result {
                errors.push(error);
            }
        }
        errors
    }
    fn annotation(&mut self, ann: &resin_ast::Type, typer: &Context) -> Result<Ty, GenerateError> {
        self.push_at(ann.span);
        let result = Evaluator {
            scopes: self.view(),
            typer,
        }
        .ty(ann);
        self.pop();
        result
    }
}

type Declarations = BTreeMap<Arc<str>, DeclarationId>;

impl Generator {
    fn declare_methods(&mut self, file: &SourceFile, scopes: &mut Scopes) -> Declarations {
        let mut declarations = BTreeMap::new();
        for stmt in file.declarations() {
            if let Err(error) = self.declare_method(&stmt.val, scopes, &mut declarations) {
                self.errors.push(error);
            }
        }
        declarations
    }

    fn declare_method(
        &mut self,
        stmt: &StmtKind,
        scopes: &mut Scopes,
        declarations: &mut Declarations,
    ) -> Result<(), GenerateError> {
        let StmtKind::Function {
            receiver: Some(receiver),
            name,
            params,
            result,
            decorators,
            ..
        } = stmt
        else {
            return Ok(());
        };
        let declaration = reserve_method(name, scopes, declarations)?;
        let definition = self.method_owner(receiver, scopes)?;
        scopes.record_method_definition(definition, declaration);
        if !decorators.is_empty() {
            return Err(GenerateError::inference(
                name.span,
                "methods cannot be shader entries",
            ));
        }
        let evaluator = Evaluator {
            scopes: scopes.view(),
            typer: &self.typer,
        };
        let signature = method_signature(&evaluator, declaration, params, result)?;
        let function = self.declare_function(name, &signature)?;
        self.register_method(definition, name, function)
    }

    fn method_owner(&self, receiver: &Ident, scopes: &Scopes) -> Result<TypeId, GenerateError> {
        let evaluator = Evaluator {
            scopes: scopes.view(),
            typer: &self.typer,
        };
        let Ty::Defined { definition } = evaluator.type_name(receiver)? else {
            return Err(GenerateError::inference(
                receiver.span,
                "impl requires a nominal struct type",
            ));
        };
        if self
            .typer
            .type_origin(definition)
            .map(|origin| origin.module)
            != Some(self.source_module)
        {
            return Err(GenerateError::inference(
                receiver.span,
                "impl requires a type defined in this module",
            ));
        }
        Ok(definition)
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
        let declaration = self.typer.declared_function(function);
        let pointer = Ty::Pointer {
            pointee: Box::new(Ty::Defined { definition: owner }),
        };
        if declaration.params != [pointer] || declaration.result != Ty::Unit {
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
) -> Result<typed::Annotation, GenerateError> {
    Ok(typed::Annotation {
        ty: evaluator.ty(source)?,
        span: source.span,
    })
}

fn elaborate_signature(source: &typed::Signature) -> Signature {
    Signature {
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
fn elaborate_annotation(source: &typed::Annotation) -> Annotation {
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
        let id = FunctionId::from_index(self.module.functions.len());
        let params = source.params.iter().map(|(_, a)| a.ty.clone()).collect();
        self.typer
            .register_function(id, params, source.result.ty.clone());
        self.module.functions.push(Function {
            location: Some(SourceLocation {
                source: self.source.clone(),
                span: name.span,
            }),
            name: name.val.clone(),
            signature: elaborate_signature(source),
            body: None,
            foreign: None,
        });
        self.function_bindings
            .insert(source.declaration.expect("checked function"), id);
        Ok(id)
    }
    fn declare_foreign(
        &mut self,
        header: &str,
        name: &Ident,
        signature: &typed::Signature,
    ) -> Result<FunctionId, GenerateError> {
        let id = self.declare_function(name, signature)?;
        let params = self.typer.declared_function(id).params.clone();
        let foreign = Foreign {
            header: header.into(),
            params,
        };
        if !foreign.valid(&signature.result.ty) {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::InvalidForeignSignature,
            });
        }
        self.module.functions[id.index()].foreign = Some(foreign);
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
    fn declare_checked_functions(&mut self, file: &SourceFile, checked: &CheckedFile) {
        let mut declared = BTreeSet::new();
        for stmt in file.declarations() {
            let Some(name) = function_name(&stmt.val) else {
                continue;
            };
            let Some(signature) = checked.signatures.get(&name.val) else {
                continue;
            };
            if declared.insert(&name.val)
                && let Err(error) = self.declare_checked(&stmt.val, name, signature)
            {
                self.errors.push(error);
            }
        }
    }

    fn declare_checked(
        &mut self,
        stmt: &StmtKind,
        name: &Ident,
        signature: &typed::Signature,
    ) -> Result<(), GenerateError> {
        match stmt {
            StmtKind::Function { decorators, .. } => {
                if let Some(id) = self.function_identity(name, signature)? {
                    for decorator in decorators {
                        self.declare_shader(id, decorator)?;
                    }
                }
            }
            StmtKind::ForeignFunction { header, .. } => {
                self.declare_foreign(header, name, signature)?;
            }
            _ => unreachable!("function declaration"),
        }
        Ok(())
    }

    fn function_identity(
        &mut self,
        name: &Ident,
        signature: &typed::Signature,
    ) -> Result<Option<FunctionId>, GenerateError> {
        if name.val.contains('.') {
            return Ok(signature
                .declaration
                .and_then(|id| self.function_bindings.get(&id).copied()));
        }
        self.declare_function(name, signature).map(Some)
    }

    fn declare_shader(&mut self, id: FunctionId, decorator: &Ident) -> Result<(), GenerateError> {
        let stage = shader_stage(decorator)?;
        if self.module.shaders.contains_key(&id) {
            return Err(shader_error(
                decorator,
                "a function can have only one shader decorator",
            ));
        }
        let signature = &self.module.functions[id.index()].signature;
        resin_types::shader::validate(
            &self.typer,
            &signature.parameter_type(),
            &signature.result.ty,
            false,
            stage,
        )
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

    fn elaborate_functions(&mut self, file: &SourceFile, checked: &CheckedFile) {
        for stmt in file.declarations() {
            let StmtKind::Function { name, .. } = &stmt.val else {
                continue;
            };
            let Some(body) = checked.bodies.get(&name.val) else {
                continue;
            };
            let Some(signature) = checked.signatures.get(&name.val) else {
                continue;
            };
            let Some(id) = self.lookup_function(&name.val) else {
                continue;
            };
            self.elaborate_function(id, signature, body);
        }
    }

    fn elaborate_function(
        &mut self,
        id: FunctionId,
        signature: &typed::Signature,
        body: &typed::Term,
    ) {
        match self.elaborate(body) {
            Ok(body) => {
                self.module.functions[id.index()].signature = elaborate_signature(signature);
                self.module.functions[id.index()].body = Some(body);
            }
            Err(error) if !self.errors.contains(&error) => self.errors.push(error),
            Err(_) => {}
        }
    }
}

fn function_name(stmt: &StmtKind) -> Option<&Ident> {
    match stmt {
        StmtKind::Function { name, .. } | StmtKind::ForeignFunction { name, .. } => Some(name),
        _ => None,
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

    fn lookup_function(&self, name: &str) -> Option<FunctionId> {
        self.function_bindings
            .get(&self.scopes.lookup(name, false)?)
            .copied()
    }
}
