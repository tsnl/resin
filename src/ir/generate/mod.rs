//! AST → typed stack IR.

use std::sync::Arc;

use crate::ast::{SourceFile, Span, Stmt, StmtKind, Term, TermKind};
use crate::ir::{
    BlockId, Function, Global, GlobalId, Instr, Local, LocalId, Module, Terminator, Ty,
    TyperContext, Value, verify,
};

mod bindings;
mod builder;
mod error;
mod eval;
mod flow;
mod functions;
mod places;
mod scope;
mod terms;

pub use error::{GenerateError, GenerateErrorKind};

use builder::FunctionBuilder;
use eval::Evaluator;
use functions::FunctionState;
use scope::Scopes;

/// Lower a source file to a verified IR module.
/// `functions[0]` initializes globals and evaluates top-level expressions.
pub fn generate(file: &SourceFile) -> Result<Module, GenerateError> {
    Generator::new().generate_file(file)
}

struct Generator {
    module: Module,
    typer: TyperContext,
    functions: Vec<FunctionState>,
    scopes: Scopes,
}

impl Generator {
    fn new() -> Self {
        let mut module = Module::default();
        module.functions.push(Function {
            name: Some("init".into()),
            nonlocals: Vec::new(),
            param: LocalId::from_index(0),
            result: Ty::Unit,
            locals: vec![Local {
                name: None,
                ty: Ty::Unit,
            }],
            entry: BlockId::from_index(0),
            blocks: Vec::new(),
        });
        Self {
            module,
            typer: TyperContext::new(),
            functions: vec![FunctionState::new(Some("init".into()))],
            scopes: Scopes::new(),
        }
    }

    fn generate_file(mut self, file: &SourceFile) -> Result<Module, GenerateError> {
        for stmt in &file.stmts {
            self.gen_stmt(stmt)?;
        }
        self.emit(Instr::Push { value: Value::Unit });
        self.terminate(Terminator::Return);
        let init = self
            .functions
            .pop()
            .expect("module initializer")
            .builder
            .finish();
        self.module.functions[0] = init;
        self.module.types = self.typer.into_definitions().map_err(|err| GenerateError {
            span: Span { start: 0, end: 0 },
            kind: GenerateErrorKind::Type(err.kind),
        })?;
        verify(&self.module).map_err(|err| GenerateError {
            span: Span { start: 0, end: 0 },
            kind: GenerateErrorKind::InvalidIr(err),
        })?;
        Ok(self.module)
    }

    fn gen_stmt(&mut self, stmt: &Stmt) -> Result<(), GenerateError> {
        match &stmt.val {
            StmtKind::Define { name, init } => self.gen_define(name, init),
            StmtKind::DefineType { name, init } => self.gen_define_type(name, init),
            StmtKind::Declare { name, ann } => self.gen_declare(name, ann),
            StmtKind::Expr { term } => {
                self.gen_term(term, None)?;
                self.emit(Instr::Discard);
                Ok(())
            }
        }
    }

    fn gen_term(&mut self, term: &Term, expected: Option<&Ty>) -> Result<Ty, GenerateError> {
        let found = self.gen_term_inner(term, expected)?;
        if let Some(expected) = expected {
            self.typer
                .same(expected, &found)
                .map_err(|err| GenerateError::typing(term.span, err))?;
        }
        Ok(found)
    }

    fn gen_term_inner(&mut self, term: &Term, expected: Option<&Ty>) -> Result<Ty, GenerateError> {
        match &term.val {
            TermKind::Var { name } => self.gen_var(name),
            TermKind::Num { value } => {
                let (pushed, ty) = self.evaluator().number(term.span, value, expected)?;
                self.emit(Instr::Push { value: pushed });
                Ok(ty)
            }
            TermKind::Unit => {
                self.emit(Instr::Push { value: Value::Unit });
                Ok(Ty::Unit)
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
            TermKind::Lambda { params, body } => self.gen_lambda(params, body, expected, None),
            TermKind::If { cond, then, els } => self.gen_if(cond, then, els, expected),
            TermKind::Array { elems } => self.gen_array(term.span, elems, expected),
            TermKind::Record { fields } => self.gen_record(term.span, fields, expected),
            TermKind::Block { stmts, tail } => self.gen_block(stmts, tail, expected),
            TermKind::Call { func, arg } => self.gen_call(term.span, func, arg),
            TermKind::Builtin { name, args } => self.gen_builtin(term.span, name, args, expected),
            TermKind::Assign { place, value } => self.gen_assign(place, value),
            TermKind::Deref { pointer } => {
                let pointer_ty = self.gen_term(pointer, None)?;
                let converted = self
                    .typer
                    .as_pointer(&pointer_ty)
                    .map_err(|err| GenerateError::typing(term.span, err))?;
                self.emit_value_conv(&converted.steps);
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
        self.function().local(ty, name)
    }

    fn alloc_global(&mut self, ty: Ty, name: Arc<str>) -> GlobalId {
        let id = GlobalId::from_index(self.module.globals.len());
        self.module.globals.push(Global { name, ty });
        id
    }

    fn emit(&mut self, instr: Instr) {
        self.function().emit(instr);
    }

    fn terminate(&mut self, terminator: Terminator) {
        self.function().terminate(terminator);
    }

    fn new_block(&mut self, hint: &str) -> BlockId {
        self.function().new_block(hint)
    }

    fn switch(&mut self, block: BlockId) {
        self.function().switch(block);
    }

    fn current_depth(&self) -> usize {
        self.functions.len() - 1
    }

    fn function(&mut self) -> &mut FunctionBuilder {
        &mut self.functions.last_mut().expect("function").builder
    }

    fn evaluator(&self) -> Evaluator<'_> {
        Evaluator {
            scopes: &self.scopes,
            typer: &self.typer,
        }
    }
}
