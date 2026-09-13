//! Allocate storage for parameters and turn one structured body into blocks.
use super::ValueBinding;
use super::builder::FunctionBuilder;
use super::{FunctionLowering, LowerError, LoweredFunction};
use crate::lower::concrete::{Function, Parameter, Signature};
use crate::{BlockId, Instr, Local, Terminator};
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

fn lower_foreign(source: &Function, foreign: &Foreign, profile: crate::Profile) -> LoweredFunction {
    LoweredFunction {
        function: crate::Function {
            name: Some(source.name.clone()),
            profile,
            foreign: Some(foreign.clone()),
            result: source.signature.result.clone(),
            locals: vec![Local {
                name: None,
                ty: source.signature.parameter_type(),
            }],
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
            owned: vec![vec![LocalId::from_index(0)]],
        }
    }

    fn lower_body(&mut self, source: &Function) -> Result<(), LowerError> {
        if let Some(body) = &source.body {
            self.bind_params(&source.signature);
            self.gen_term(body, Some(&source.signature.result))?;
            self.cleanup(0, &source.signature.result);
            self.terminate(Terminator::Return);
        } else {
            self.function
                .parameter(None, source.signature.parameter_type());
        }
        Ok(())
    }

    fn bind_params(&mut self, signature: &Signature) {
        let params = &signature.params;
        let ty = signature.parameter_type();
        let name = (params.len() == 1).then(|| params[0].name.val.clone());
        self.function.parameter(name, ty.clone());
        for (index, parameter) in params.iter().enumerate() {
            self.bind_parameter(parameter, index, params.len() == 1);
        }
        if params.len() > 1 && ty.needs_drop(self.typer.definitions()) {
            self.emit(Instr::ForgetLocal {
                local: LocalId::from_index(0),
            });
        }
    }

    fn bind_parameter(&mut self, parameter: &Parameter, index: usize, single: bool) {
        let ty = &parameter.ty;
        let local = if single {
            LocalId::from_index(0)
        } else {
            let local = self.alloc_local(ty.clone(), Some(parameter.name.val.clone()));
            self.unpack_parameter(local, index, ty);
            local
        };
        self.bindings.insert(
            parameter.binding.expect("checked body parameter"),
            ValueBinding {
                local,
                ty: ty.clone(),
            },
        );
    }

    fn unpack_parameter(&mut self, local: LocalId, index: usize, ty: &Ty) {
        if ty.needs_drop(self.typer.definitions()) {
            self.parameter_field(index);
            self.emit(Instr::TransferLoad);
            self.emit(Instr::SetLocal { local });
        } else {
            self.emit(Instr::LocalAddress { local });
            self.parameter_field(index);
            self.emit(Instr::Load);
            self.emit(Instr::Store);
            self.emit(Instr::Discard);
        }
    }

    fn parameter_field(&mut self, index: usize) {
        self.emit(Instr::LocalAddress {
            local: LocalId::from_index(0),
        });
        self.emit(Instr::AccessStatic { index });
    }
}
