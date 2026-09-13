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
    let definitions = definitions(source, &mut instances).map_err(|error| vec![error])?;
    let typer = TyperContext::from_definitions(definitions);
    let functions = instances.lower(&typer)?;
    Ok(assemble(source, &instances, typer, functions))
}

fn definitions(
    source: &resin_hir::Module,
    instances: &mut instances::Instances<'_>,
) -> Result<TypeTable, Error> {
    let substitution = substitute::Substitution::default();
    let mut definitions = Vec::with_capacity(source.types.len());
    for definition in &source.types {
        let body = substitution
            .ty(&definition.body)
            .map_err(|error| instances.lower_error(error, None, None))?;
        let drop = definition
            .drop
            .map(|id| instances.request(id, vec![], None, None))
            .transpose()?;
        definitions.push(TypeDef::Nominal {
            name: definition.name.clone(),
            body: Some(body),
            drop,
        });
    }
    let definitions = TypeTable::from(definitions);
    for (index, definition) in definitions.iter().enumerate() {
        let body = definition.body().expect("completed nominal body");
        resin_types::check_references(&definitions, body)
            .and_then(|()| resin_types::check_layout(&definitions, TypeId::from_index(index), body))
            .map_err(|error| {
                instances.lower_error(
                    LowerError::typing(Span { start: 0, end: 0 }, error),
                    None,
                    None,
                )
            })?;
    }
    Ok(definitions)
}

/// A completed function uses local instruction positions; assembly supplies its ID.
struct LoweredFunction {
    function: crate::Function,
    location: Option<SourceLocation>,
    origins: BTreeMap<(BlockId, usize), SourceLocation>,
}

fn assemble(
    source: &resin_hir::Module,
    instances: &instances::Instances<'_>,
    typer: TyperContext,
    functions: Vec<LoweredFunction>,
) -> Module {
    let mut module = Module {
        types: typer
            .into_definitions()
            .expect("completed nominal definitions"),
        entries: source
            .entries
            .iter()
            .filter_map(|(name, id)| instances.ordinary(*id).map(|id| (name.clone(), id)))
            .collect(),
        shaders: source
            .shaders
            .iter()
            .filter_map(|(id, shader)| instances.ordinary(*id).map(|id| (id, shader.clone())))
            .collect(),
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
// Resolved bindings and their concrete storage
//

#[derive(Clone)]
struct ValueBinding {
    local: LocalId,
    ty: Ty,
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
