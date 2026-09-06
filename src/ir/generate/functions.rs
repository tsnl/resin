use crate::ast::{Ident, Term, Type};
use crate::ir::{Foreign, FunctionId, Instr, LocalId, Terminator, Ty};

use super::builder::FunctionBuilder;
use super::scope::{Initialization, ValueBinding, ValueBindingKind};
use super::{GenerateError, GenerateErrorKind, Generator};

impl Generator {
    pub(super) fn declare_foreign(
        &mut self,
        header: &str,
        name: &Ident,
        params: &[(Ident, Type)],
        result: &Type,
    ) -> Result<(), GenerateError> {
        let params = self.declare_function(name, params, result)?;
        let function = self.module.functions.last_mut().unwrap();
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
        Ok(())
    }

    pub(super) fn declare_function(
        &mut self,
        name: &Ident,
        params: &[(Ident, Type)],
        result: &Type,
    ) -> Result<Vec<Ty>, GenerateError> {
        let mut names = std::collections::HashSet::new();
        for (name, _) in params {
            Self::check_binding_name(name)?;
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
            .map(|(_, ann)| self.evaluator().ty(ann))
            .collect::<Result<Vec<_>, _>>()?;
        let param = Ty::parameter(&params);
        let result = self.evaluator().ty(result)?;
        let id = FunctionId::from_index(self.module.functions.len());
        let mut builder = FunctionBuilder::new(Some(name.val.clone()));
        builder.parameter(None, param.clone());
        builder.result(result.clone());
        self.module.functions.push(builder.finish());
        self.bind_value(
            name,
            ValueBinding {
                kind: ValueBindingKind::Function(id),
                ty: Some(Ty::Function {
                    param: Box::new(param),
                    result: Box::new(result),
                }),
                initialization: Initialization::Initialized,
                depth: 0,
            },
        )?;
        Ok(params)
    }

    pub(super) fn gen_function(
        &mut self,
        name: &Ident,
        params: &[(Ident, Type)],
        body: &Term,
    ) -> Result<(), GenerateError> {
        let binding = self.resolve_value(name)?;
        let ValueBindingKind::Function(id) = binding.kind else {
            unreachable!()
        };
        let result = self.module.functions[id.index()].result.clone();
        let enclosing = self.scopes.clone();
        self.functions
            .push(FunctionBuilder::new(Some(name.val.clone())));
        self.scopes.push();
        self.bind_params(params)?;
        self.gen_term(body, Some(&result))?;
        self.function().result(result);
        self.terminate(Terminator::Return);
        self.module.functions[id.index()] = self.functions.pop().unwrap().finish();
        self.scopes = enclosing;
        Ok(())
    }

    fn bind_params(&mut self, params: &[(Ident, Type)]) -> Result<Ty, GenerateError> {
        let mut param_tys = Vec::with_capacity(params.len());
        for (_, ann) in params {
            param_tys.push(self.evaluator().ty(ann)?);
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
                self.emit(Instr::LocalAddress { local });
                self.emit(Instr::LocalAddress { local: parameter });
                self.emit(Instr::AccessStatic { index });
                self.emit(Instr::Load);
                self.emit(Instr::Store);
                self.emit(Instr::Discard);
                local
            };
            self.bind_value(
                name,
                ValueBinding {
                    kind: ValueBindingKind::Local(local),
                    ty: Some(ty.clone()),
                    initialization: Initialization::Initialized,
                    depth: self.current_depth(),
                },
            )?;
        }
        Ok(param_ty)
    }
}
