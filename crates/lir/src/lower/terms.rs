use std::{collections::HashMap, sync::Arc};

use crate::hir::Term;
use crate::source::{Ident, Span};
use crate::types::Conv;
use crate::{Instr, Ty};

use super::{GenerateError, Generator};

impl Generator {
    pub(super) fn gen_array(&mut self, elems: &[Term], ty: &Ty) -> Result<Ty, GenerateError> {
        let Ty::Array { element, length } = ty else {
            unreachable!("checked array")
        };
        for elem in elems {
            self.gen_term(elem, Some(element))?;
        }
        self.emit(Instr::MakeArray {
            elements: *length,
            element: *element.clone(),
        });
        Ok(ty.clone())
    }

    pub(super) fn gen_record(
        &mut self,
        fields: &[(Ident, Term)],
        ty: &Ty,
    ) -> Result<Ty, GenerateError> {
        let Ty::Record {
            fields: expected_fields,
        } = ty
        else {
            unreachable!("checked record")
        };
        // Evaluate in source order; layout order must not reorder effects.
        let mut values = HashMap::with_capacity(fields.len());
        for (name, value) in fields {
            let field = expected_fields
                .iter()
                .find(|field| field.name == name.val)
                .unwrap();
            let local = self.alloc_local(field.ty.clone(), None);
            if field.ty.needs_drop(self.typer.definitions()) {
                self.gen_term(value, Some(&field.ty))?;
                self.emit(Instr::SetLocal { local });
            } else {
                self.emit(Instr::LocalAddress { local });
                self.gen_term(value, Some(&field.ty))?;
                self.emit(Instr::Store);
                self.emit(Instr::Discard);
            }
            values.insert(name.val.clone(), local);
        }
        let names = expected_fields
            .iter()
            .map(|field| {
                let local = values[&field.name];
                if field.ty.needs_drop(self.typer.definitions()) {
                    self.emit(Instr::TakeLocal { local });
                } else {
                    self.load_local(local);
                }
                field.name.clone()
            })
            .collect();
        self.emit(Instr::MakeRecord { fields: names });
        Ok(ty.clone())
    }

    pub(super) fn gen_call(&mut self, func: &Term, arg: &Term) -> Result<Ty, GenerateError> {
        let callee_ty = self.gen_term(func, None)?;
        let Ty::Function { param, .. } = &callee_ty else {
            unreachable!("inferred callable type")
        };
        self.gen_term(arg, Some(param))?;
        self.emit(Instr::Call);
        let Ty::Function { result, .. } = callee_ty else {
            unreachable!("checked call")
        };
        Ok(*result)
    }

    pub(super) fn gen_conversion(
        &mut self,
        term: &Term,
        arg: &Term,
        conversion: &crate::types::check::ExplicitConversion,
    ) -> Result<Ty, GenerateError> {
        use crate::types::check::ExplicitConversion::*;
        let from = self.gen_term(arg, None)?;
        let ty = term.ty.clone();
        match conversion {
            Widen => return self.coerce(term.span, from, &ty),
            NumericCast => self.emit(Instr::NumericCast { ty: ty.clone() }),
            PointerCast => self.emit(Instr::PointerCast { ty: ty.clone() }),
            Ascribe(steps) if !steps.is_empty() => self.emit(Instr::Ascribe { ty: ty.clone() }),
            Ascribe(_) => {}
        }
        Ok(ty)
    }

    pub(super) fn gen_builtin(
        &mut self,
        _span: Span,
        name: &str,
        args: &[Term],
        expected: &Ty,
    ) -> Result<Ty, GenerateError> {
        let mut arg_tys = Vec::with_capacity(args.len());
        for arg in args {
            arg_tys.push(self.gen_term(arg, None)?);
        }
        self.emit(Instr::CallBuiltin {
            name: Arc::from(name),
            params: arg_tys,
            result: expected.clone(),
        });
        Ok(expected.clone())
    }

    pub(super) fn emit_value_conv(&mut self, steps: &[Conv]) {
        for step in steps {
            match step {
                Conv::Unwrap { definition } => {
                    let ty = self
                        .typer
                        .body(&Ty::Defined {
                            definition: *definition,
                        })
                        .expect("validated type definition");
                    self.emit(Instr::Ascribe { ty });
                }
                Conv::Wrap { definition } => {
                    self.emit(Instr::Ascribe {
                        ty: Ty::Defined {
                            definition: *definition,
                        },
                    });
                }
                Conv::Deref => self.emit(Instr::Load),
                Conv::MakeSpan | Conv::SpanRecord => {
                    unreachable!("span conversions require a target type")
                }
            }
        }
    }
}
