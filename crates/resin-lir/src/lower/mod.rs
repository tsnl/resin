//! HIR → LIR: choose storage and make evaluation, cleanup, and control flow explicit.
use crate::{BlockId, Instr, Module, Terminator};
use builder::FunctionBuilder;
use resin_hir::{BindingId, Term};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::{collections::HashMap, sync::Arc};

mod arguments;
mod bindings;
mod builder;
mod expressions;
mod flow;
mod functions;
mod places;
mod sums;
mod terms;

use crate::{Error, ErrorKind};

pub fn generate(source: &resin_hir::Module) -> Result<Module, Error> {
    analyze(source).map_err(|mut errors| errors.remove(0))
}

pub fn analyze(source: &resin_hir::Module) -> Result<Module, Vec<Error>> {
    let mut generator = Generator::new(source);
    let mut errors = vec![];
    for (index, function) in source.functions.iter().enumerate() {
        let id = FunctionId::from_index(index);
        if let Err(error) = generator.gen_function(id, function) {
            errors.push(Error {
                function: id,
                span: error.span,
                kind: error.kind,
            });
        }
    }
    if errors.is_empty() {
        Ok(generator.module)
    } else {
        Err(errors)
    }
}

//
// Resolved bindings and initialization across control-flow paths
//

#[derive(Clone)]
struct ValueBinding {
    local: LocalId,
    ty: Ty,
    initialization: Initialization,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Initialization {
    Uninitialized,
    Initializing,
    Initialized,
}

struct Generator {
    module: Module,
    source: Option<Source>,
    source_span: Span,
    function_id: Option<FunctionId>,
    typer: TyperContext,
    function: Option<FunctionBuilder>,
    bindings: HashMap<BindingId, ValueBinding>,
    owned: Vec<Vec<LocalId>>,
}
impl Generator {
    fn new(source: &resin_hir::Module) -> Self {
        let module = Module {
            types: source.types.clone(),
            entries: source.entries.clone(),
            shaders: source.shaders.clone(),
            origins: crate::SourceMap {
                functions: source
                    .functions
                    .iter()
                    .enumerate()
                    .filter_map(|(index, function)| {
                        function
                            .location
                            .clone()
                            .map(|location| (FunctionId::from_index(index), location))
                    })
                    .collect(),
                instructions: Default::default(),
            },
            functions: source.functions.iter().map(functions::prototype).collect(),
        };
        Self {
            module,
            source: None,
            source_span: Span { start: 0, end: 0 },
            function_id: None,
            typer: TyperContext::from_definitions(source.types.clone()),
            function: None,
            bindings: HashMap::new(),
            owned: vec![],
        }
    }
    fn gen_term(&mut self, term: &Term, to: Option<&Ty>) -> Result<Ty, LowerError> {
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
        let (Some(function), Some(source)) = (self.function_id, self.source.clone()) else {
            return;
        };
        let (block, instruction) = self.function().position();
        self.module.origins.instructions.insert(
            (function, block, instruction),
            SourceLocation {
                source,
                span: self.source_span,
            },
        );
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

//
// Preserve the result while destroying owned locals in reverse scope order
//

impl Generator {
    fn cleanup(&mut self, first_scope: usize, result: &Ty) {
        let owned = self.locals_to_drop(first_scope);
        if owned.is_empty() {
            return;
        }
        let saved = self.save_top(result);
        for local in owned {
            self.emit(Instr::DropLocal { local });
        }
        if result.needs_drop(self.typer.definitions()) {
            self.emit(Instr::TakeLocal { local: saved });
        } else {
            self.load_local(saved);
        }
    }

    fn locals_to_drop(&self, first_scope: usize) -> Vec<LocalId> {
        let function = self.function.as_ref().expect("inside a function body");
        self.owned[first_scope..]
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().rev().copied())
            .filter(|id| {
                function
                    .local_type(*id)
                    .needs_drop(self.typer.definitions())
            })
            .collect()
    }
}

//
// A binding is initialized after a join only when incoming paths agree
//

impl Generator {
    fn intersect_initialization(&mut self, other: &HashMap<BindingId, ValueBinding>) {
        for (id, binding) in &mut self.bindings {
            if let Some(other) = other.get(id)
                && binding.initialization != other.initialization
            {
                binding.initialization = Initialization::Uninitialized;
            }
        }
    }
}

// Failures acquire their function identity when the function lowering returns.
struct LowerError {
    span: Span,
    kind: ErrorKind,
}

impl LowerError {
    fn invalid_hir(span: Span, message: impl Into<Arc<str>>) -> Self {
        Self {
            span,
            kind: ErrorKind::InvalidHir {
                message: message.into(),
            },
        }
    }

    fn typing(span: Span, error: TypeError) -> Self {
        Self {
            span,
            kind: ErrorKind::Type { kind: error.kind },
        }
    }
}
