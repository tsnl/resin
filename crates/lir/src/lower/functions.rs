//! Allocate storage for parameters and turn one structured body into blocks.
use super::builder::FunctionBuilder;
use super::scope::{Environment, Initialization, ValueBinding};
use super::{GenerateError, Generator};
use crate::hir::{Function, Parameter, Signature};
use crate::{FunctionId, Instr, LocalId, Terminator, Ty};

impl Generator {
    pub(super) fn gen_function(
        &mut self,
        id: FunctionId,
        source: &Function,
    ) -> Result<(), GenerateError> {
        self.begin_function(id, source);
        if let Some(body) = &source.body {
            self.bind_params(&source.signature);
            self.gen_term(body, Some(&source.signature.result.ty))?;
            self.cleanup(0, &source.signature.result.ty);
            self.terminate(Terminator::Return);
        } else {
            self.function()
                .parameter(None, source.signature.parameter_type());
        }
        self.finish_function(id, source);
        Ok(())
    }

    fn begin_function(&mut self, id: FunctionId, source: &Function) {
        self.function_id = Some(id);
        self.environment = Environment::new();
        self.owned.clear();
        self.owned.push(vec![LocalId::from_index(0)]);
        self.set_function_origin(id);
        let mut builder = FunctionBuilder::new(Some(source.name.clone()));
        builder.result(source.signature.result.ty.clone());
        self.function = Some(builder);
    }

    fn set_function_origin(&mut self, id: FunctionId) {
        if let Some(origin) = self.module.origins.functions.get(&id) {
            self.source_path = origin.path.clone();
            self.source_span = origin.span;
        }
    }

    fn finish_function(&mut self, id: FunctionId, source: &Function) {
        self.owned.pop();
        let mut function = self.function.take().unwrap().finish();
        set_foreign(&mut function, source);
        self.module.functions[id.index()] = function;
        self.function_id = None;
    }

    fn bind_params(&mut self, signature: &Signature) {
        let params = &signature.params;
        let ty = signature.parameter_type();
        let name = (params.len() == 1).then(|| params[0].name.val.clone());
        self.function().parameter(name, ty.clone());
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
        let ty = &parameter.annotation.ty;
        let local = if single {
            LocalId::from_index(0)
        } else {
            let local = self.alloc_local(ty.clone(), Some(parameter.name.val.clone()));
            self.unpack_parameter(local, index, ty);
            local
        };
        self.environment.bind(
            parameter.binding.expect("checked body parameter"),
            ValueBinding {
                local,
                ty: ty.clone(),
                initialization: Initialization::Initialized,
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

pub(super) fn prototype(source: &Function) -> crate::Function {
    let mut builder = FunctionBuilder::new(Some(source.name.clone()));
    builder.parameter(None, source.signature.parameter_type());
    builder.result(source.signature.result.ty.clone());
    let mut function = builder.finish();
    set_foreign(&mut function, source);
    function
}

fn set_foreign(function: &mut crate::Function, source: &Function) {
    function.foreign = source.foreign.clone();
    if function.foreign.is_some() {
        function.blocks.clear();
    }
}
