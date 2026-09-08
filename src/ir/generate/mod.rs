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
use scope::{Scopes, ValueBindingKind};

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
    generator.generate_file(file);
    if let Some(error) = std::mem::take(&mut generator.errors).into_iter().next() {
        return Err(error);
    }
    generator.module.entries = generator.exported_functions(file)?;
    generator.finish().map(|(module, _)| module)
}

struct Generator {
    module: Module,
    source_path: std::path::PathBuf,
    source_module: SourceModuleId,
    source_span: Span,
    function_id: Option<crate::ir::FunctionId>,
    typer: TyperContext,
    function: Option<FunctionBuilder>,
    scopes: Scopes,
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
            scopes: Scopes::new(),
            solver: Default::default(),
            owned: vec![],
            errors: vec![],
        }
    }

    fn generate_file(&mut self, file: &SourceFile) {
        for stmt in file.declarations() {
            let result = match &stmt.val {
                StmtKind::Define { .. } | StmtKind::Declare { .. } | StmtKind::Expr { .. } => {
                    Err(GenerateError {
                        span: stmt.span,
                        kind: GenerateErrorKind::InvalidModuleItem,
                    })
                }
                StmtKind::ForeignType { name } => {
                    let result =
                        self.scopes
                            .define_foreign_type(name.val.clone())
                            .map_err(|name| GenerateError {
                                span: stmt.span,
                                kind: GenerateErrorKind::DuplicateType { name },
                            });
                    self.scopes.record_definition(
                        name,
                        true,
                        Some(&Ty::Foreign {
                            name: name.val.clone(),
                        }),
                        &self.typer,
                    );
                    result
                }
                _ => Ok(()),
            };
            if let Err(error) = result {
                self.errors.push(error);
            }
        }
        for stmt in file.declarations() {
            let result = match &stmt.val {
                StmtKind::DefineType { name, init } => self.gen_define_type(name, init),
                StmtKind::Struct { name, body } => self.gen_struct(name, body),
                _ => Ok(()),
            };
            if let Err(error) = result {
                self.errors.push(error);
            }
        }
        self.declare_methods(file);
        let planned = plan::file(file, &mut self.typer, &self.scopes, self.source_module);
        self.solver = planned.solver;
        self.errors.extend(planned.errors);
        self.scopes.lowering();
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
                            let Some(binding) = self.scopes.lookup_value(&name.val) else {
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
                            self.scopes.lookup_value_mut(&name.val).unwrap().shader = true;
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
                && self.scopes.lookup_value(&name.val).is_some()
                && !self.solver.invalid(&body.ty)
            {
                let scopes = self.scopes.clone();
                if let Err(error) = self.gen_function(name, &planned.signatures[&name.val], body) {
                    if !self.errors.contains(&error) {
                        self.errors.push(error);
                    }
                    self.scopes = scopes;
                    self.function = None;
                    self.function_id = None;
                    self.owned.clear();
                }
            }
        }
    }

    fn finish(mut self) -> Result<(Module, crate::ir::verify::ModuleTypes), GenerateError> {
        self.module.types = self.typer.into_definitions().map_err(|err| GenerateError {
            span: Span { start: 0, end: 0 },
            kind: GenerateErrorKind::Type(err.kind),
        })?;
        let verification =
            crate::ir::verify::analyze(&self.module).map_err(|err| GenerateError {
                span: Span { start: 0, end: 0 },
                kind: GenerateErrorKind::InvalidIr(err),
            })?;
        self.module.types = verification.types.clone();
        Ok((self.module, verification))
    }

    // `to` requests an emitted value conversion, never a typing context.
    fn gen_term(&mut self, term: &Term, to: Option<&Ty>) -> Result<Ty, GenerateError> {
        let before = std::mem::replace(&mut self.source_span, term.span);
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
            scopes: &self.scopes,
            typer: &self.typer,
        }
    }
}
