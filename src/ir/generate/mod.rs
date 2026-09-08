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
mod recover;
pub(crate) use recover::analyze as analyze_recovering;
pub(super) mod scope;
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
    generator.generate_file(file)?;
    generator.module.entries = generator.exported_functions(file)?;
    generator.finish()
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
        }
    }

    fn generate_file(&mut self, file: &SourceFile) -> Result<(), GenerateError> {
        for stmt in file.declarations() {
            if matches!(
                stmt.val,
                StmtKind::Define { .. } | StmtKind::Declare { .. } | StmtKind::Expr { .. }
            ) {
                return Err(GenerateError {
                    span: stmt.span,
                    kind: GenerateErrorKind::InvalidModuleItem,
                });
            }
        }
        for stmt in file.declarations() {
            if let StmtKind::ForeignType { name } = &stmt.val {
                self.scopes
                    .define_foreign_type(name.val.clone())
                    .map_err(|name| GenerateError {
                        span: stmt.span,
                        kind: GenerateErrorKind::DuplicateType { name },
                    })?;
                self.scopes.record_definition(
                    name,
                    true,
                    Some(&Ty::Foreign {
                        name: name.val.clone(),
                    }),
                    &self.typer,
                );
            }
        }
        for stmt in file.declarations() {
            match &stmt.val {
                StmtKind::DefineType { name, init } => self.gen_define_type(name, init)?,
                StmtKind::Struct { name, body } => self.gen_struct(name, body)?,
                _ => {}
            }
        }
        self.declare_methods(file)?;
        let planned = plan::file(file, &mut self.typer, &self.scopes, self.source_module)?;
        self.solver = planned.solver;
        for stmt in file.declarations() {
            if let StmtKind::Function {
                name, decorators, ..
            } = &stmt.val
            {
                let id = if !name.val.contains('.') {
                    self.declare_function(name, &planned.signatures[&name.val])?
                } else {
                    let ValueBindingKind::Function(id) = self.resolve_value(name)?.kind else {
                        unreachable!()
                    };
                    id
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
                                message: "a function can have only one shader decorator".into(),
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
            if let StmtKind::ForeignFunction { header, name, .. } = &stmt.val {
                self.declare_foreign(header, name, &planned.signatures[&name.val])?;
            }
        }
        for stmt in file.declarations() {
            if let StmtKind::Function { name, .. } = &stmt.val {
                self.gen_function(
                    name,
                    &planned.signatures[&name.val],
                    &planned.bodies[&name.val],
                )?;
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Module, GenerateError> {
        self.module.types = self.typer.into_definitions().map_err(|err| GenerateError {
            span: Span { start: 0, end: 0 },
            kind: GenerateErrorKind::Type(err.kind),
        })?;
        let analysis = crate::ir::verify::analyze(&self.module).map_err(|err| GenerateError {
            span: Span { start: 0, end: 0 },
            kind: GenerateErrorKind::InvalidIr(err),
        })?;
        self.module.types = analysis.types;
        Ok(self.module)
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
