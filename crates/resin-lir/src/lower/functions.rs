//! Allocate storage for parameters and turn one structured body into blocks.
use super::ValueBinding;
use super::builder::FunctionBuilder;
use super::{FunctionLowering, LowerError, LoweredFunction};
use crate::lower::concrete::{Function, Signature};
use crate::{BlockId, Local, Terminator};
use resin_source::prelude::*;
use resin_types::prelude::*;

pub(super) fn lower(
    source: &Function,
    typer: &TyperContext,
    profile: crate::Profile,
) -> Result<LoweredFunction, crate::Error> {
    if let Some(foreign) = &source.foreign {
        return Ok(lower_foreign(source, foreign, profile));
    }
    let mut lowering = FunctionLowering::new(source, typer, profile);
    lowering.lower_body(source).map_err(|error| crate::Error {
        source: lowering.source.clone(),
        span: error.span,
        kind: error.kind,
        applications: vec![],
    })?;
    Ok(LoweredFunction {
        function: lowering.function.finish(),
        location: source.location.clone(),
        origins: lowering.origins,
    })
}

fn lower_foreign(
    source: &Function,
    foreign: &crate::Foreign,
    profile: crate::Profile,
) -> LoweredFunction {
    LoweredFunction {
        function: crate::Function {
            name: Some(source.name.clone()),
            profile,
            foreign: Some(foreign.clone()),
            result: source.signature.result.clone(),
            parameter_count: source.signature.params.len(),
            locals: source
                .signature
                .params
                .iter()
                .map(|parameter| Local {
                    name: Some(parameter.name.val.clone()),
                    ty: parameter.ty.clone(),
                })
                .collect(),
            entry: BlockId::from_index(0),
            blocks: vec![],
        },
        location: source.location.clone(),
        origins: Default::default(),
    }
}

impl<'types> FunctionLowering<'types> {
    fn new(source: &Function, typer: &'types TyperContext, profile: crate::Profile) -> Self {
        let mut function = FunctionBuilder::new(Some(source.name.clone()), profile);
        function.result(source.signature.result.clone());
        Self {
            source: source
                .location
                .as_ref()
                .map(|location| location.source.clone()),
            source_span: source
                .location
                .as_ref()
                .map_or(Span { start: 0, end: 0 }, |location| location.span),
            origins: Default::default(),
            typer,
            function,
            bindings: Default::default(),
            owned: vec![super::Scope::default()],
            loop_scopes: vec![],
            parallel_depth: 0,
        }
    }

    fn lower_body(&mut self, source: &Function) -> Result<(), LowerError> {
        if let Some(body) = &source.body {
            self.bind_params(&source.signature);
            self.gen_term(body, Some(&source.signature.result))?;
            self.cleanup(0, &source.signature.result);
            self.terminate(Terminator::Return);
        } else {
            self.bind_params(&source.signature);
        }
        Ok(())
    }

    fn bind_params(&mut self, signature: &Signature) {
        for parameter in &signature.params {
            let local = self
                .function
                .parameter(Some(parameter.name.val.clone()), parameter.ty.clone());
            self.owned[0].locals.push(local);
            if let Some(binding) = parameter.binding {
                self.bindings.insert(
                    binding,
                    ValueBinding {
                        local,
                        ty: parameter.ty.clone(),
                    },
                );
            }
        }
    }
}
