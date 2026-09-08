use super::plan::{Signature, Term};
use crate::ast::Ident;
use crate::ir::{
    Foreign, FunctionId, Instr, LocalId, Terminator, Ty, typecheck::check_binding_name,
};

use super::builder::FunctionBuilder;
use super::scope::{Initialization, ValueBinding, ValueBindingKind};
use super::{GenerateError, GenerateErrorKind, Generator};

impl Generator {
    pub(super) fn declare_foreign(
        &mut self,
        header: &str,
        name: &Ident,
        signature: &Signature,
    ) -> Result<FunctionId, GenerateError> {
        let id = self.declare_function(name, signature)?;
        let params = self.typer.declared_function(id).params.clone();
        let function = &mut self.module.functions[id.index()];
        let foreign = Foreign {
            header: header.into(),
            params,
        };
        if !foreign.valid(&function.result) {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::InvalidForeignSignature,
            });
        }
        function.foreign = Some(foreign);
        function.blocks.clear();
        Ok(id)
    }

    pub(super) fn declare_function(
        &mut self,
        name: &Ident,
        signature: &Signature,
    ) -> Result<FunctionId, GenerateError> {
        let params = &signature.params;
        let mut names = std::collections::HashSet::new();
        for (name, _) in params {
            check_binding_name(name)?;
            if !names.insert(&name.val) {
                return Err(GenerateError {
                    span: name.span,
                    kind: GenerateErrorKind::DuplicateValue {
                        name: name.val.clone(),
                    },
                });
            }
        }
        let params = params
            .iter()
            .map(|(_, ann)| ann.resolve(self))
            .collect::<Result<Vec<_>, _>>()?;
        let param = Ty::parameter(&params);
        let result = signature.result.resolve(self)?;
        let id = FunctionId::from_index(self.module.functions.len());
        self.typer.register_function(id, params, result.clone());
        self.module.origins.functions.insert(
            id,
            crate::ast::SourceLocation {
                path: self.source_path.clone(),
                span: name.span,
            },
        );
        let mut builder = FunctionBuilder::new(Some(name.val.clone()));
        builder.parameter(None, param.clone());
        builder.result(result.clone());
        self.module.functions.push(builder.finish());
        self.bind_value(
            name,
            ValueBinding {
                shader: false,
                kind: ValueBindingKind::Function(id),
                ty: Some(Ty::Function {
                    param: Box::new(param),
                    result: Box::new(result),
                }),
                initialization: Initialization::Initialized,
            },
        )?;
        Ok(id)
    }

    pub(super) fn gen_function(
        &mut self,
        name: &Ident,
        signature: &Signature,
        body: &Term,
    ) -> Result<(), GenerateError> {
        let binding = self.resolve_value(name)?;
        let ValueBindingKind::Function(id) = binding.kind else {
            unreachable!()
        };
        self.function_id = Some(id);
        self.source_span = name.span;
        let result = self.module.functions[id.index()].result.clone();
        self.function = Some(FunctionBuilder::new(Some(name.val.clone())));
        self.function().result(result.clone());
        self.scopes.push();
        self.owned.push(vec![LocalId::from_index(0)]);
        self.bind_params(signature)?;
        self.gen_term(body, Some(&result))?;
        self.cleanup(0, &result);
        self.owned.pop();
        self.terminate(Terminator::Return);
        self.module.functions[id.index()] = self.function.take().unwrap().finish();
        self.function_id = None;
        self.scopes.pop();
        Ok(())
    }

    fn bind_params(&mut self, signature: &Signature) -> Result<Ty, GenerateError> {
        let params = &signature.params;
        let mut param_tys = Vec::with_capacity(params.len());
        for (_, ann) in params {
            param_tys.push(ann.resolve(self)?);
        }
        let param_ty = Ty::parameter(&param_tys);
        let parameter = LocalId::from_index(0);
        self.function().parameter(
            if params.len() == 1 {
                Some(params[0].0.val.clone())
            } else {
                None
            },
            param_ty.clone(),
        );
        for (index, ((name, _), ty)) in params.iter().zip(&param_tys).enumerate() {
            let local = if params.len() == 1 {
                parameter
            } else {
                let local = self.alloc_local(ty.clone(), Some(name.val.clone()));
                if ty.needs_drop(self.typer.definitions()) {
                    self.emit(Instr::LocalAddress { local: parameter });
                    self.emit(Instr::AccessStatic { index });
                    self.emit(Instr::TransferLoad);
                    self.emit(Instr::SetLocal { local });
                } else {
                    self.emit(Instr::LocalAddress { local });
                    self.emit(Instr::LocalAddress { local: parameter });
                    self.emit(Instr::AccessStatic { index });
                    self.emit(Instr::Load);
                    self.emit(Instr::Store);
                    self.emit(Instr::Discard);
                }
                local
            };
            self.bind_value(
                name,
                ValueBinding {
                    shader: false,
                    kind: ValueBindingKind::Local(local),
                    ty: Some(ty.clone()),
                    initialization: Initialization::Initialized,
                },
            )?;
        }
        if params.len() > 1 && param_ty.needs_drop(self.typer.definitions()) {
            self.emit(Instr::ForgetLocal { local: parameter });
        }
        Ok(param_ty)
    }
}
