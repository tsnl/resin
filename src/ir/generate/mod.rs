//! AST → typed stack IR.

use std::sync::Arc;

use crate::ast::{SourceFile, Span, StmtKind};
use crate::ir::{BlockId, Instr, LocalId, Module, Terminator, Ty, TyperContext};

mod annotation;
mod bindings;
mod builder;
mod builtin_methods;
mod builtins;
mod cleanup;
mod error;
pub(super) mod eval;
mod flow;
mod functions;
mod methods;
mod modules;
mod places;
mod plan;
use plan::Term;
pub(crate) mod scope;
pub(crate) mod semantic;
mod sums;
mod terms;

pub use error::{GenerateError, GenerateErrorKind};
pub(crate) use modules::analyze_program;
pub use modules::generate_program;

use crate::ir::typecheck::SourceModuleId;
use builder::FunctionBuilder;
use eval::Evaluator;
use scope::{Cursor, Environment, Scopes, ValueBindingKind};

/// Lower a source file to a verified IR module.
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
    if let Some(error) = std::mem::take(&mut generator.errors).into_iter().next() {
        return Err(error);
    }
    generator.module.entries = generator.exported_functions(file)?;
    generator
        .finish()
        .map(crate::ir::verify::VerifiedModule::into_module)
}

struct Generator {
    module: Module,
    source_path: std::path::PathBuf,
    source_module: SourceModuleId,
    source_span: Span,
    function_id: Option<crate::ir::FunctionId>,
    typer: TyperContext,
    function: Option<FunctionBuilder>,
    environment: Environment,
    solver: crate::ir::typecheck::infer::solver::Solver,
    owned: Vec<Vec<LocalId>>,
    errors: Vec<GenerateError>,
}

impl Generator {
    fn new() -> Self {
        Self {
            module: Module::default(),
            source_path: "<source>".into(),
            source_module: SourceModuleId::from_index(0),
            source_span: Span { start: 0, end: 0 },
            function_id: None,
            typer: builtins::typer(),
            function: None,
            environment: Environment::new(),
            solver: Default::default(),
            owned: vec![],
            errors: vec![],
        }
    }

    fn generate_file(&mut self, file: &SourceFile, mut scopes: Scopes) {
        self.errors
            .extend(scopes.prepare(file, &mut self.typer, self.source_module));
        let methods = self.declare_methods(file, &mut scopes);
        let planned = plan::file(file, &mut self.typer, scopes, self.source_module, methods);
        self.solver = planned.solver;
        self.errors.extend(planned.errors);
        self.environment.context = planned.context;
        let mut declared = std::collections::BTreeSet::new();
        for stmt in file.declarations() {
            let result = (|| {
                match &stmt.val {
                    StmtKind::Function {
                        name, decorators, ..
                    } => {
                        let Some(signature) = planned.signatures.get(&name.val) else {
                            return Ok(());
                        };
                        if !declared.insert(name.val.clone()) {
                            return Ok(());
                        }
                        let id = if name.val.contains('.') {
                            let Some(binding) = signature
                                .declaration
                                .and_then(|id| self.environment.binding(id))
                            else {
                                return Ok(());
                            };
                            let ValueBindingKind::Function(id) = binding.kind else {
                                return Ok(());
                            };
                            id
                        } else {
                            self.declare_function(name, signature)?
                        };
                        for decorator in decorators {
                            let stage = match decorator.val.as_ref() {
                                "compute_shader" => "compute",
                                "vertex_shader" => "vertex",
                                "fragment_shader" => "fragment",
                                _ => {
                                    return Err(GenerateError {
                                        span: decorator.span,
                                        kind: GenerateErrorKind::InvalidShader {
                                            message: "unknown decorator".into(),
                                        },
                                    });
                                }
                            };
                            if self.module.shaders.contains_key(&id) {
                                return Err(GenerateError {
                                    span: decorator.span,
                                    kind: GenerateErrorKind::InvalidShader {
                                        message: "a function can have only one shader decorator"
                                            .into(),
                                    },
                                });
                            }
                            crate::ir::shader::validate(
                                &self.typer,
                                &self.module.functions[id.index()],
                                stage,
                            )
                            .map_err(|message| GenerateError {
                                span: decorator.span,
                                kind: GenerateErrorKind::InvalidShader {
                                    message: message.into(),
                                },
                            })?;
                            self.environment.lookup_value_mut(&name.val).unwrap().shader = true;
                            self.module.shaders.insert(
                                id,
                                crate::ir::shader::ShaderEntry {
                                    stage: stage.into(),
                                    embedded: false,
                                },
                            );
                        }
                    }
                    StmtKind::ForeignFunction { header, name, .. } => {
                        if let Some(signature) = planned.signatures.get(&name.val)
                            && declared.insert(name.val.clone())
                        {
                            self.declare_foreign(header, name, signature)?;
                        }
                    }
                    _ => {}
                }
                Ok(())
            })();
            if let Err(error) = result {
                self.errors.push(error);
            }
        }
        for stmt in file.declarations() {
            if let StmtKind::Function { name, .. } = &stmt.val
                && let Some(body) = planned.bodies.get(&name.val)
                && self.environment.lookup_value(&name.val).is_some()
                && !self.solver.invalid(&body.ty)
            {
                let environment = self.environment.clone();
                let result = self.gen_function(name, &planned.signatures[&name.val], body);
                self.environment = environment;
                if let Err(error) = result {
                    if !self.errors.contains(&error) {
                        self.errors.push(error);
                    }
                    self.function = None;
                    self.function_id = None;
                    self.owned.clear();
                }
            }
        }
    }

    fn finish(mut self) -> Result<crate::ir::verify::VerifiedModule, GenerateError> {
        self.module.types = self.typer.into_definitions().map_err(|err| GenerateError {
            span: Span { start: 0, end: 0 },
            kind: GenerateErrorKind::Type(err.kind),
        })?;
        crate::ir::verify::VerifiedModule::new(self.module).map_err(|err| GenerateError {
            span: Span { start: 0, end: 0 },
            kind: GenerateErrorKind::InvalidIr(err),
        })
    }

    // `to` requests an emitted value conversion, never a typing context.
    fn gen_term(&mut self, term: &Term, to: Option<&Ty>) -> Result<Ty, GenerateError> {
        let before = std::mem::replace(&mut self.source_span, term.span);
        let context = self.environment.context.select(term.context);
        let result = (|| {
            let checked = self.solver.require(&term.ty, term.span)?;
            let found = (term.emit)(self, &checked)?;
            let found = self.coerce(term.span, found, &checked)?;
            if let Some(to) = to {
                self.coerce(term.span, found, to)
            } else {
                Ok(found)
            }
        })();
        self.source_span = before;
        self.environment.context.select(context);
        result
    }

    fn with_context<T>(
        &mut self,
        context: Cursor,
        emit: impl FnOnce(&mut Self) -> Result<T, GenerateError>,
    ) -> Result<T, GenerateError> {
        let before = self.environment.context.select(context);
        let result = emit(self);
        self.environment.context.select(before);
        result
    }

    fn alloc_local(&mut self, ty: Ty, name: Option<Arc<str>>) -> LocalId {
        let local = self.function().local(ty, name);
        if let Some(owned) = self.owned.last_mut() {
            owned.push(local);
        }
        local
    }

    fn save_top(&mut self, ty: &Ty) -> LocalId {
        let local = self.alloc_local(ty.clone(), None);
        self.emit(Instr::SetLocal { local });
        local
    }

    fn load_local(&mut self, local: LocalId) {
        self.emit(Instr::LocalAddress { local });
        self.emit(Instr::Load);
    }

    fn record_origin(&mut self) {
        if let Some(function) = self.function_id {
            let (block, instruction) = self.function().position();
            self.module.origins.instructions.insert(
                (function, block, instruction),
                crate::ast::SourceLocation {
                    path: self.source_path.clone(),
                    span: self.source_span,
                },
            );
        }
    }

    fn emit(&mut self, instr: Instr) {
        self.record_origin();
        self.function().emit(instr);
    }

    fn terminate(&mut self, terminator: Terminator) {
        self.record_origin();
        self.function().terminate(terminator);
    }

    fn new_block(&mut self, hint: &str) -> BlockId {
        self.function().new_block(hint)
    }

    fn switch(&mut self, block: BlockId) {
        self.function().switch(block);
    }

    fn function(&mut self) -> &mut FunctionBuilder {
        self.function.as_mut().expect("inside a function body")
    }

    fn evaluator(&self) -> Evaluator<'_> {
        Evaluator {
            scopes: &self.environment.context,
            typer: &self.typer,
        }
    }
}
