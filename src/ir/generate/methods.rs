use super::{GenerateError, Generator, check::error};
use crate::{
    ast::{Ident, SourceFile, Span, StmtKind, Term, TermKind},
    ir::{FunctionId, Instr, Ty, types::Method},
};

impl Generator {
    pub(super) fn declare_methods(&mut self, file: &SourceFile) -> Result<(), GenerateError> {
        for stmt in file.declarations() {
            let StmtKind::Function {
                owner: Some(owner),
                name,
                params,
                result,
                decorators,
                ..
            } = &stmt.val
            else {
                continue;
            };
            if !file
                .declarations()
                .any(|s| matches!(&s.val, StmtKind::Struct { name, .. } if name.val == owner.val))
            {
                return Err(error(
                    owner.span,
                    "impl requires a struct defined in this module",
                ));
            }
            if !decorators.is_empty() {
                return Err(error(name.span, "methods cannot be shader entries"));
            }
            let Ty::Defined { definition } = self.scopes.lookup_type(&owner.val).unwrap() else {
                unreachable!()
            };
            let short = name.val.rsplit('.').next().unwrap();
            if self
                .typer
                .definition(definition)
                .unwrap()
                .methods
                .contains_key(short)
            {
                return Err(error(name.span, "duplicate method"));
            }
            let function = FunctionId::from_index(self.module.functions.len());
            let typed = self.declare_function(name, params, result)?;
            let result = self.module.functions[function.index()].result.clone();
            let receiver = params
                .first()
                .is_some_and(|(n, _)| n.val.as_ref() == "self");
            let owner_ty = Ty::Defined { definition };
            if receiver
                && !matches!(&typed[0], t if t == &owner_ty || t == &Ty::Pointer { pointee: Box::new(owner_ty.clone()) })
            {
                return Err(error(
                    name.span,
                    "self must have type T or Ptr<T> for the impl type",
                ));
            }
            self.typer.define_method(
                definition,
                short.into(),
                Method {
                    function,
                    params: typed,
                    result,
                    receiver,
                },
            );
        }
        Ok(())
    }

    pub(super) fn gen_method_call(
        &mut self,
        span: Span,
        base: &Term,
        name: &Ident,
        arg: &Term,
    ) -> Result<Option<Ty>, GenerateError> {
        let (base_ty, _associated) = if let TermKind::Type { ty } = &base.val {
            (self.evaluator().ty(ty)?, true)
        } else {
            (
                self.checked.expressions[&std::ptr::from_ref(base)].clone(),
                false,
            )
        };
        let Some(method) = self.typer.method(&base_ty, &name.val).cloned() else {
            return Ok(None);
        };
        if method.receiver {
            let receiver = &method.params[0];
            let owner = match receiver {
                Ty::Pointer { pointee } => pointee.as_ref(),
                ty => ty,
            };
            let pointer = Ty::Pointer {
                pointee: Box::new(owner.clone()),
            };
            let compatible = receiver == &base_ty
                || receiver == &pointer && (&base_ty == owner)
                || receiver == owner && (base_ty == pointer);
            if !compatible {
                return Err(error(
                    span,
                    "method receiver requires T or Ptr<T> matching its declared self type",
                ));
            }
        }
        self.emit(Instr::Function {
            function: method.function,
        });
        if method.receiver {
            let expected = &method.params[0];
            if expected == &base_ty {
                self.gen_term(base, Some(expected))?;
            } else if matches!(expected, Ty::Pointer { .. }) {
                self.check_place_initialized(base)?;
                self.gen_place(base)?;
            } else {
                self.gen_term(base, None)?;
                self.emit(Instr::Load);
            }
            let remaining = &method.params[1..];
            if remaining.is_empty() {
                self.gen_term(arg, Some(&Ty::Unit))?;
                self.emit(Instr::Discard);
            } else if remaining.len() == 1 {
                self.gen_term(arg, Some(&remaining[0]))?;
            } else {
                let arg_ty = Ty::parameter(remaining);
                self.gen_term(arg, Some(&arg_ty))?;
                let saved = self.save_top(&arg_ty);
                for i in 0..remaining.len() {
                    self.emit(Instr::LocalAddress { local: saved });
                    self.emit(Instr::AccessStatic { index: i });
                    self.emit(Instr::Load);
                }
            }
            if method.params.len() > 1 {
                self.emit(Instr::MakeRecord {
                    fields: (0..method.params.len())
                        .map(|i| format!("_{i}").into())
                        .collect(),
                });
            }
        } else {
            self.gen_term(arg, Some(&Ty::parameter(&method.params)))?;
        }
        self.emit(Instr::Call);
        Ok(Some(method.result))
    }
}
