use super::{
    GenerateError, Generator,
    plan::{Annotation, Signature, Term, error},
};
use crate::{
    ast::{Ident, SourceFile, StmtKind},
    ir::{
        Instr, Ty,
        typecheck::{FunctionBody, ReceiverConversion},
    },
};

impl Generator {
    pub(super) fn declare_methods(&mut self, file: &SourceFile) {
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
            let receiver_type = self
                .scopes
                .resolve_type(receiver)
                .and_then(|ty| self.solver.require(&ty, receiver.span));
            let definition = match receiver_type {
                Ok(Ty::Defined { definition }) => definition,
                Ok(_) => {
                    self.errors
                        .push(error(receiver.span, "impl requires a nominal struct type"));
                    continue;
                }
                Err(error) => {
                    self.errors.push(error);
                    continue;
                }
            };
            if self
                .typer
                .type_origin(definition)
                .map(|origin| origin.module)
                != Some(self.source_module)
            {
                self.errors.push(error(
                    receiver.span,
                    "impl requires a type defined in this module",
                ));
                continue;
            }
            let checked = (|| {
                if !decorators.is_empty() {
                    return Err(error(name.span, "methods cannot be shader entries"));
                }
                let signature = Signature {
                    params: params
                        .iter()
                        .map(|(name, ann)| {
                            Ok((
                                name.clone(),
                                Annotation {
                                    ty: self.evaluator().ty(ann)?.into(),
                                    span: ann.span,
                                },
                            ))
                        })
                        .collect::<Result<_, GenerateError>>()?,
                    result: Annotation {
                        ty: self.evaluator().ty(result)?.into(),
                        span: result.span,
                    },
                };
                let function = self.declare_function(name, &signature)?;
                let short = name.val.rsplit('.').next().unwrap();
                if !self.typer.define_method(definition, short.into(), function) {
                    return Err(error(name.span, "duplicate method"));
                }
                if short == "drop" {
                    let declaration = self.typer.declared_function(function);
                    if declaration.params
                        != [Ty::Pointer {
                            pointee: Box::new(Ty::Defined { definition }),
                        }]
                        || declaration.result != Ty::Unit
                    {
                        return Err(error(
                            name.span,
                            "drop must have signature drop(receiver: Ptr<T>) -> ()",
                        ));
                    }
                    self.typer.define_drop(definition, function);
                }
                Ok(())
            })();
            self.scopes.record_method_definition(definition, name);
            if let Err(error) = checked {
                self.errors.push(error);
            }
        }
    }

    pub(super) fn hold_arc_address(&mut self, term: &Term) -> Result<Ty, GenerateError> {
        let ty = self.gen_term(term, None)?;
        let Ty::Arc { pointee } = &ty else {
            return Err(error(term.span, "expected Arc<T>"));
        };
        let pointee = *pointee.clone();
        let owner = self.save_top(&ty);
        self.load_local(owner);
        self.emit(Instr::ArcData);
        Ok(Ty::Pointer {
            pointee: Box::new(pointee),
        })
    }

    pub(super) fn gen_method_call(
        &mut self,
        receiver: Option<&Term>,
        receiver_type: &Ty,
        name: &Ident,
        arg: &Term,
    ) -> Result<Ty, GenerateError> {
        let associated = receiver.is_none();
        let function = self
            .typer
            .method(receiver_type, &name.val)
            .ok_or_else(|| error(name.span, "unknown method"))?;
        if let FunctionBody::Defined(function) = &function.body {
            self.emit(Instr::Function {
                function: *function,
            });
        }
        if associated {
            if matches!(function.body, FunctionBody::Generated(_)) {
                self.gen_method_arguments(arg, &function.params)?;
            } else {
                self.gen_term(arg, Some(&Ty::parameter(&function.params)))?;
            }
        } else {
            let (first, remaining) = function
                .params
                .split_first()
                .expect("checked receiver parameter");
            self.gen_receiver(receiver.unwrap(), receiver_type, first)?;
            self.gen_method_arguments(arg, remaining)?;
            if !remaining.is_empty() && matches!(function.body, FunctionBody::Defined(_)) {
                self.emit(Instr::MakeRecord {
                    fields: (0..function.params.len())
                        .map(|i| format!("_{i}").into())
                        .collect(),
                });
            }
        }
        match function.body {
            FunctionBody::Defined(_) => self.emit(Instr::Call),
            FunctionBody::Generated(instructions) => {
                for instruction in instructions {
                    self.emit(instruction);
                }
            }
        }
        Ok(function.result)
    }

    fn gen_receiver(&mut self, receiver: &Term, from: &Ty, to: &Ty) -> Result<(), GenerateError> {
        match ReceiverConversion::between(from, to).expect("checked receiver conversion") {
            conversion @ (ReceiverConversion::ArcAddress | ReceiverConversion::ArcLoad) => {
                self.hold_arc_address(receiver)?;
                if matches!(conversion, ReceiverConversion::ArcLoad) {
                    self.emit(Instr::Load);
                }
            }
            ReceiverConversion::Value => {
                self.gen_term(receiver, Some(to))?;
            }
            ReceiverConversion::Address => {
                self.check_place_initialized(receiver)?;
                match self.gen_operand(receiver)? {
                    super::places::Operand::Place(_) => {}
                    super::places::Operand::Value(ty) => {
                        let local = self.save_top(&ty);
                        self.emit(Instr::LocalAddress { local });
                    }
                }
            }
            ReceiverConversion::Load => {
                self.gen_term(receiver, None)?;
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
                    self.emit(Instr::TransferLoad);
                }
                self.emit(Instr::ForgetLocal { local: saved });
            }
        }
        Ok(())
    }
}
