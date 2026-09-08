//! AST → typed stack IR.

use std::sync::Arc;

use crate::ast::{SourceFile, Span, Stmt, StmtKind, Term, TermKind};
use crate::ir::typer::SourceModuleId;
use crate::ir::{BlockId, Instr, LocalId, Module, Terminator, Ty, TyperContext, Value};

mod bindings;
mod builder;
mod check;
mod cleanup;
mod error;
mod eval;
mod flow;
mod functions;
mod methods;
mod modules;
mod places;
mod recover;
pub(crate) use recover::analyze as analyze_recovering;
mod scope;
mod sums;
mod terms;

pub use error::{GenerateError, GenerateErrorKind};
pub(crate) use modules::analyze_program;
pub use modules::generate_program;

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
    checked: check::Checked,
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
            typer: TyperContext::new(),
            function: None,
            scopes: Scopes::new(),
            checked: check::Checked::default(),
            owned: vec![],
        }
    }

    fn generate_file(&mut self, file: &SourceFile) -> Result<(), GenerateError> {
        self.checked = check::Checked::default();
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
        self.checked = check::file(file, &mut self.typer, &self.scopes, self.source_module)?;
        for stmt in file.declarations() {
            if let StmtKind::Function {
                name,
                params,
                result,
                decorators,
                ..
            } = &stmt.val
            {
                let id = if !name.val.contains('.') {
                    self.declare_function(name, params, result)?
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
            if let StmtKind::ForeignFunction {
                header,
                name,
                params,
                result,
            } = &stmt.val
            {
                self.declare_foreign(header, name, params, result)?;
            }
        }
        for stmt in file.declarations() {
            if let StmtKind::Function {
                name, params, body, ..
            } = &stmt.val
            {
                self.gen_function(name, params, body)?;
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

    fn gen_stmt(&mut self, stmt: &Stmt) -> Result<(), GenerateError> {
        match &stmt.val {
            StmtKind::Impl { .. }
            | StmtKind::Function { .. }
            | StmtKind::ForeignFunction { .. }
            | StmtKind::ForeignType { .. } => {
                unreachable!("functions and foreign types are module items")
            }
            StmtKind::Define { name, init } => self.gen_define(name, init),
            StmtKind::DefineType { name, init } => self.gen_define_type(name, init),
            StmtKind::Struct { name, body } => self.gen_struct(name, body),
            StmtKind::Declare { name, ann } => self.gen_declare(name, ann),
            StmtKind::Expr { term } => {
                self.gen_term(term, None)?;
                self.emit(Instr::Discard);
                Ok(())
            }
        }
    }

    // `to` requests an emitted value conversion, never a typing context.
    fn gen_term(&mut self, term: &Term, to: Option<&Ty>) -> Result<Ty, GenerateError> {
        let before = std::mem::replace(&mut self.source_span, term.span);
        let result = (|| {
            let checked = self.checked.expressions[&std::ptr::from_ref(term)].clone();
            let found = self.gen_term_inner(term, &checked)?;
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

    fn gen_term_inner(&mut self, term: &Term, expected: &Ty) -> Result<Ty, GenerateError> {
        match &term.val {
            TermKind::Hole { .. } | TermKind::FieldHole { .. } => Err(GenerateError {
                span: term.span,
                kind: GenerateErrorKind::IncompleteSyntax,
            }),
            TermKind::Var { name } => self.gen_var(name),
            TermKind::Num { value } => {
                let (pushed, ty) = self.evaluator().number(term.span, value, Some(expected))?;
                self.emit(Instr::Push { value: pushed });
                Ok(ty)
            }
            TermKind::Unit => {
                self.emit(Instr::Push { value: Value::Unit });
                Ok(Ty::Unit)
            }
            TermKind::None => {
                self.emit(Instr::Push { value: Value::None });
                Ok(Ty::None)
            }
            TermKind::String { value } => {
                let bytes = value.as_bytes();
                self.emit(Instr::Push {
                    value: Value::Array {
                        value: crate::ir::ArrayValue {
                            element_ty: Ty::UInt8,
                            elements: bytes.iter().map(|&value| Value::UInt8 { value }).collect(),
                        },
                    },
                });
                Ok(Ty::Array {
                    element: Box::new(Ty::UInt8),
                    length: bytes.len(),
                })
            }
            TermKind::Type { ty } => {
                let value = self.evaluator().ty(ty)?;
                self.emit(Instr::Push {
                    value: Value::Type { ty: value },
                });
                Ok(Ty::Type)
            }
            TermKind::If { cond, then, els } => self.gen_if(cond, then, els, expected),
            TermKind::Unwrap { value } => {
                let input = self.gen_term(value, None)?;
                self.emit(Instr::ExcludeNone);
                Ok(input.without_none().expect("checked None exclusion"))
            }
            TermKind::Try { value } => self.gen_try(term.span, value),
            TermKind::Match { value, arms } => self.gen_match(term.span, value, arms, expected),
            TermKind::While { cond, body } => self.gen_while(cond, body),
            TermKind::Array { elems } => self.gen_array(elems, expected),
            TermKind::Record { fields } => self.gen_record(fields, expected),
            TermKind::Block { stmts, tail } => self.gen_block(stmts, tail, expected),
            TermKind::MethodCall {
                receiver,
                name,
                arg,
            } => self.gen_method_call(receiver, name, arg),
            TermKind::Call { func, arg } => self.gen_call(term.span, func, arg, expected),
            TermKind::Builtin { name, args } => self.gen_builtin(term.span, name, args, expected),
            TermKind::Assign { place, value } => self.gen_assign(place, value),
            TermKind::Address { place } => self.gen_place(place),
            TermKind::Deref { pointer } => {
                let checked =
                    self.checked.expressions[&std::ptr::from_ref(pointer.as_ref())].clone();
                let pointer_ty = if matches!(checked, Ty::Arc { .. }) {
                    self.hold_arc_address(pointer)?
                } else {
                    self.gen_term(pointer, None)?
                };
                let ty = self
                    .typer
                    .type_deref(&pointer_ty)
                    .map_err(|err| GenerateError::typing(term.span, err))?;
                self.emit(Instr::Load);
                Ok(ty)
            }
            TermKind::Field { base, name } => self.gen_field_value(term.span, base, name),
        }
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
            checked: Some(&self.checked.holes),
        }
    }
}
