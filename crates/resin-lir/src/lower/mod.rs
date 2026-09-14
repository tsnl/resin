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
mod instances;
mod places;
mod specialize;
mod substitute;
mod sums;
mod terms;

use crate::{Error, ErrorKind};

pub fn analyze(
    source: &resin_hir::Module,
    options: &crate::LoweringOptions,
) -> Result<Module, Vec<Error>> {
    let mut instances = instances::Instances::new(source, options);
    instances.reserve_roots().map_err(|error| vec![error])?;
    instances
        .reserve_types()
        .map_err(|error| vec![instances.lower_error(error, None, None)])?;
    let functions = instances.lower()?;
    instances.assemble(functions)
}

pub fn instantiate(
    source: &resin_hir::Module,
    entries: &[crate::Entry],
    options: &crate::LoweringOptions,
) -> Result<Module, Vec<Error>> {
    let mut instances = instances::Instances::new(source, options);
    instances
        .reserve_entries(entries)
        .map_err(|error| vec![error])?;
    let functions = instances.lower()?;
    instances.assemble(functions)
}

/// A completed function uses local instruction positions; assembly supplies its ID.
struct LoweredFunction {
    function: crate::Function,
    location: Option<SourceLocation>,
    origins: BTreeMap<(BlockId, usize), SourceLocation>,
}

//
// Resolved bindings and their concrete storage
//

#[derive(Clone)]
struct ValueBinding {
    local: LocalId,
    ty: Ty,
}

#[derive(Default)]
struct Scope {
    locals: Vec<LocalId>,
    operand_base: usize,
}

struct FunctionLowering<'types> {
    source: Option<Source>,
    source_span: Span,
    origins: BTreeMap<(BlockId, usize), SourceLocation>,
    typer: &'types TyperContext,
    function: FunctionBuilder,
    bindings: HashMap<BindingId, ValueBinding>,
    owned: Vec<Scope>,
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
            owned.locals.push(local);
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

    fn new_block(&mut self, hint: &str, inherited: usize, produced: usize) -> BlockId {
        self.function.new_block(hint, inherited, produced)
    }

    fn switch(&mut self, block: BlockId) {
        self.function.switch(block);
    }

    fn enter_scope(&mut self) {
        self.owned.push(Scope {
            locals: vec![],
            operand_base: self.function.stack_len(),
        });
    }
}

//
// Preserve the result while destroying locals and pending operands in lifetime order
//

impl FunctionLowering<'_> {
    fn cleanup(&mut self, first_scope: usize, result: &Ty) {
        let base = self.owned[first_scope].operand_base;
        let owned = self.locals_to_drop(first_scope);
        if owned.is_empty() && self.function.stack_len() == base + 1 {
            return;
        }
        let saved = self.save_top(result);
        // Pending expression values and locals share one destruction order.
        // Preserve operands inherited from an enclosing scope on normal exit;
        // cleanup from the function scope also abandons unfinished expressions.
        for local in owned {
            while self.function.stack_len() > base && self.function.stack_is_newer_than(local) {
                self.emit(Instr::Discard);
            }
            self.emit(Instr::DropLocal { local });
        }
        while self.function.stack_len() > base {
            self.emit(Instr::Discard);
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
            .flat_map(|scope| scope.locals.iter().rev().copied())
            .filter(|id| {
                function
                    .local_type(*id)
                    .needs_drop(self.typer.definitions())
            })
            .collect()
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
