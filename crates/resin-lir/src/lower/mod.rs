//! HIR → LIR: choose storage and make evaluation, cleanup, and control flow explicit.
pub use crate::diagnostic::{GenerateError, GenerateErrorKind};
use crate::hir::{self, Term};
use crate::source::Span;
use crate::types::TyperContext;
use crate::{BlockId, FunctionId, Instr, LocalId, Module, Terminator, Ty};
use builder::FunctionBuilder;
use scope::Environment;
use std::sync::Arc;

mod arguments;
mod bindings;
mod builder;
mod cleanup;
mod expressions;
mod flow;
mod functions;
mod places;
mod scope;
mod sums;
mod terms;

use crate::Error;

pub fn generate(source: &hir::Module) -> Result<Module, Error> {
    analyze(source).map_err(|mut errors| errors.remove(0))
}

pub fn analyze(source: &hir::Module) -> Result<Module, Vec<Error>> {
    let mut generator = Generator::new(source);
    let mut errors = vec![];
    for (index, function) in source.functions.iter().enumerate() {
        let id = FunctionId::from_index(index);
        if let Err(error) = generator.gen_function(id, function) {
            errors.push(Error {
                function: id,
                error,
            });
        }
    }
    if errors.is_empty() {
        Ok(generator.module)
    } else {
        Err(errors)
    }
}

struct Generator {
    module: Module,
    source_path: std::path::PathBuf,
    source_span: Span,
    function_id: Option<FunctionId>,
    typer: TyperContext,
    function: Option<FunctionBuilder>,
    environment: Environment,
    owned: Vec<Vec<LocalId>>,
}
impl Generator {
    fn new(source: &hir::Module) -> Self {
        let module = Module {
            types: source.types.clone(),
            entries: source.entries.clone(),
            shaders: source.shaders.clone(),
            origins: crate::SourceMap {
                sources: source.origins.sources.clone(),
                functions: source.origins.functions.clone(),
                instructions: Default::default(),
            },
            functions: source.functions.iter().map(functions::prototype).collect(),
        };
        Self {
            module,
            source_path: "<source>".into(),
            source_span: Span { start: 0, end: 0 },
            function_id: None,
            typer: TyperContext::from_definitions(source.types.clone()),
            function: None,
            environment: Environment::new(),
            owned: vec![],
        }
    }
    fn gen_term(&mut self, term: &Term, to: Option<&Ty>) -> Result<Ty, GenerateError> {
        let before = std::mem::replace(&mut self.source_span, term.span);
        let result = self
            .lower_term(term)
            .and_then(|ty| self.coerce(term.span, ty, &term.ty))
            .and_then(|ty| match to {
                Some(to) => self.coerce(term.span, ty, to),
                None => Ok(ty),
            });
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
                crate::source::SourceLocation {
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
}
