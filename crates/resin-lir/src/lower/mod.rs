//! HIR → LIR: choose storage and make evaluation, cleanup, and control flow explicit.
use crate::lower::concrete::Term;
use crate::{BlockId, Instr, Module, Terminator};
use builder::FunctionBuilder;
use resin_hir::BindingId;
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

mod arguments;
mod bindings;
mod builder;
mod concrete;
mod expressions;
mod flow;
mod functions;
mod places;
mod specialize;
mod sums;
mod terms;

use crate::{Error, ErrorKind};

pub fn generate(source: &resin_hir::Module) -> Result<Module, Error> {
    analyze(source).map_err(|mut errors| errors.remove(0))
}

pub fn analyze(source: &resin_hir::Module) -> Result<Module, Vec<Error>> {
    let definitions = specialize::definitions(&source.types).map_err(|error| {
        vec![Error {
            source: None,
            span: Span { start: 0, end: 0 },
            kind: ErrorKind::Type { kind: error.kind },
        }]
    })?;
    let typer = TyperContext::from_definitions(definitions);
    let mut functions = Vec::with_capacity(source.functions.len());
    let mut errors = vec![];
    for function in &source.functions {
        match functions::lower(&specialize::function(function), &typer) {
            Ok(function) => functions.push(function),
            Err(error) => errors.push(error),
        }
    }
    if errors.is_empty() {
        Ok(assemble(source, typer, functions))
    } else {
        Err(errors)
    }
}

/// A completed function uses local instruction positions; assembly supplies its ID.
struct LoweredFunction {
    function: crate::Function,
    location: Option<SourceLocation>,
    origins: BTreeMap<(BlockId, usize), SourceLocation>,
}

fn assemble(
    source: &resin_hir::Module,
    typer: TyperContext,
    functions: Vec<LoweredFunction>,
) -> Module {
    let mut module = Module {
        types: typer
            .into_definitions()
            .expect("completed nominal definitions"),
        entries: source.entries.clone(),
        shaders: source.shaders.clone(),
        ..Default::default()
    };
    for (index, lowered) in functions.into_iter().enumerate() {
        let id = FunctionId::from_index(index);
        module.functions.push(lowered.function);
        if let Some(location) = lowered.location {
            module.origins.functions.insert(id, location);
        }
        module.origins.instructions.extend(
            lowered
                .origins
                .into_iter()
                .map(|((block, instruction), location)| ((id, block, instruction), location)),
        );
    }
    module
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

struct FunctionLowering<'types> {
    source: Option<Source>,
    source_span: Span,
    origins: BTreeMap<(BlockId, usize), SourceLocation>,
    typer: &'types TyperContext,
    function: FunctionBuilder,
    bindings: HashMap<BindingId, ValueBinding>,
    owned: Vec<Vec<LocalId>>,
}
impl FunctionLowering<'_> {
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
        let local = self.function.local(ty, name);
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
        let Some(source) = self.source.clone() else {
            return;
        };
        let (block, instruction) = self.function.position();
        self.origins.insert(
            (block, instruction),
            SourceLocation {
                source,
                span: self.source_span,
            },
        );
    }

    fn emit(&mut self, instr: Instr) {
        self.record_origin();
        self.function.emit(instr);
    }

    fn terminate(&mut self, terminator: Terminator) {
        self.record_origin();
        self.function.terminate(terminator);
    }

    fn new_block(&mut self, hint: &str, height: usize) -> BlockId {
        self.function.new_block(hint, height)
    }

    fn switch(&mut self, block: BlockId) {
        self.function.switch(block);
    }
}

//
// Preserve the result while destroying owned locals in reverse scope order
//

impl FunctionLowering<'_> {
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
        let function = &self.function;
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

impl FunctionLowering<'_> {
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

// Failures acquire their source when the function lowering returns.
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
