use super::{GenerateError, Generator, check::error};
use crate::{
    ast::{Ident, SourceFile, StmtKind, Term, TermKind},
    ir::{Instr, Ty, typer::ReceiverConversion},
};

impl Generator {
    pub(super) fn declare_methods(&mut self, file: &SourceFile) -> Result<(), GenerateError> {
        for stmt in file.declarations() {
            let StmtKind::Function {
                receiver: Some(receiver),
                name,
                params,
                result,
                decorators,
                ..
            } = &stmt.val
            else {
                continue;
            };
            let Some(Ty::Defined { definition }) = self.scopes.lookup_type(&receiver.val) else {
                return Err(error(receiver.span, "impl requires a nominal struct type"));
            };
            if self.typer.type_origin(definition) != Some(self.source_module) {
                return Err(error(
                    receiver.span,
                    "impl requires a type defined in this module",
                ));
            }
            if !decorators.is_empty() {
                return Err(error(name.span, "methods cannot be shader entries"));
            }
            let function = self.declare_function(name, params, result)?;
            let short = name.val.rsplit('.').next().unwrap();
            if !self.typer.define_method(definition, short.into(), function) {
                return Err(error(name.span, "duplicate method"));
            }
            self.scopes.record_method_definition(definition, name);
        }
        Ok(())
    }

    pub(super) fn gen_method_call(
        &mut self,
        base: &Term,
        name: &Ident,
        arg: &Term,
    ) -> Result<Ty, GenerateError> {
        let (receiver, associated) = if let TermKind::Type { ty } = &base.val {
            (self.evaluator().ty(ty)?, true)
        } else {
            (
                self.checked.expressions[&std::ptr::from_ref(base)].clone(),
                false,
            )
        };
        let function = self
            .typer
            .method(&receiver, &name.val)
            .cloned()
            .ok_or_else(|| error(name.span, "unknown method"))?;
        self.scopes
            .record_method(name, &receiver, associated, &self.typer);
        self.emit(Instr::Function {
            function: function.function,
        });
        if associated {
            self.gen_term(arg, Some(&Ty::parameter(&function.params)))?;
        } else {
            let (first, remaining) = function
                .params
                .split_first()
                .expect("checked receiver parameter");
            self.gen_receiver(base, &receiver, first)?;
            self.gen_method_arguments(arg, remaining)?;
            if !remaining.is_empty() {
                self.emit(Instr::MakeRecord {
                    fields: (0..function.params.len())
                        .map(|i| format!("_{i}").into())
                        .collect(),
                });
            }
        }
        self.emit(Instr::Call);
        Ok(function.result)
    }

    fn gen_receiver(&mut self, base: &Term, from: &Ty, to: &Ty) -> Result<(), GenerateError> {
        match ReceiverConversion::between(from, to).expect("checked receiver conversion") {
            ReceiverConversion::Value => {
                self.gen_term(base, Some(to))?;
            }
            ReceiverConversion::Address => {
                self.check_place_initialized(base)?;
                self.gen_place(base)?;
            }
            ReceiverConversion::Load => {
                self.gen_term(base, None)?;
                self.emit(Instr::Load);
            }
        }
        Ok(())
    }

    fn gen_method_arguments(&mut self, arg: &Term, params: &[Ty]) -> Result<(), GenerateError> {
        self.gen_term(arg, Some(&Ty::parameter(params)))?;
        match params.len() {
            0 => self.emit(Instr::Discard),
            1 => {}
            count => {
                let saved = self.save_top(&Ty::parameter(params));
                for index in 0..count {
                    self.emit(Instr::LocalAddress { local: saved });
                    self.emit(Instr::AccessStatic { index });
                    self.emit(Instr::Load);
                }
            }
        }
        Ok(())
    }
}
