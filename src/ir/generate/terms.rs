use std::{collections::HashMap, sync::Arc};

use crate::ast::{Ident, Span, Term, TermKind, Type};
use crate::ir::{Conv, Instr, Ty, TypeError, TypeErrorKind};

use super::{GenerateError, Generator};

impl Generator {
    pub(super) fn gen_shared_payload(
        &mut self,
        span: Span,
        ty: &Ty,
        init: &Term,
    ) -> Result<(), GenerateError> {
        let body = self
            .typer
            .body(ty)
            .map_err(|e| GenerateError::typing(span, e))?;
        if matches!(init.val, TermKind::Unit)
            && matches!(&body, Ty::Record { fields } if fields.is_empty())
        {
            self.emit(Instr::MakeRecord { fields: vec![] });
        } else {
            self.gen_term(init, Some(&body))?;
        }
        if &body != ty {
            self.emit(Instr::Ascribe { ty: ty.clone() });
        }
        Ok(())
    }

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

    pub(super) fn gen_call(
        &mut self,
        span: Span,
        func: &Term,
        arg: &Term,
        expected: &Ty,
    ) -> Result<Ty, GenerateError> {
        if let TermKind::Type { ty } = &func.val {
            return self.gen_ascription(span, ty, arg);
        }
        if let TermKind::Var { name } = &func.val {
            match name.val.as_ref() {
                "replace" => {
                    let pair = self.gen_term(arg, None)?;
                    let saved = self.save_top(&pair);
                    for index in 0..2 {
                        self.emit(Instr::LocalAddress { local: saved });
                        self.emit(Instr::AccessStatic { index });
                        self.emit(Instr::TransferLoad);
                    }
                    self.emit(Instr::ForgetLocal { local: saved });
                    self.emit(Instr::Replace);
                    return Ok(expected.clone());
                }
                "absurd" => {
                    self.gen_term(arg, Some(&Ty::union([])))?;
                    self.emit(Instr::Eliminate {
                        result: expected.clone(),
                    });
                    return Ok(expected.clone());
                }
                "size_of" | "align_of" => {
                    let ty = if let TermKind::Type { ty } = &arg.val {
                        self.evaluator().ty(ty)?
                    } else {
                        self.checked.expressions[&std::ptr::from_ref(arg)].clone()
                    };
                    let layout = crate::ir::layout::layout(self.typer.definitions(), &ty)
                        .map_err(|e| super::check::error(span, e.to_string()))?;
                    self.emit(Instr::Push {
                        value: crate::ir::Value::UInt64 {
                            value: if name.val.as_ref() == "size_of" {
                                layout.size
                            } else {
                                layout.align
                            } as u64,
                        },
                    });
                    return Ok(Ty::UInt64);
                }
                "ok" | "err" => {
                    return self.gen_result(span, name.val.as_ref() == "err", arg, expected);
                }
                "print" => {
                    return self.gen_builtin(span, "print", std::slice::from_ref(arg), &Ty::Unit);
                }
                _ => {}
            }
        }

        use super::places::Operand;
        self.check_place_initialized(func)?;
        let operand = self.gen_operand(func)?;
        let (callee_ty, address) = match operand {
            Operand::Value(ty) => (ty, false),
            Operand::Place(ty) => (ty, true),
        };
        // Read the callee exactly once. Array places retain their storage address.
        let shape = self
            .typer
            .body(&callee_ty)
            .map_err(|e| GenerateError::typing(func.span, e))?;
        if address && !matches!(shape, Ty::Array { .. }) {
            self.emit(Instr::Load);
        }
        if let Ty::Span { element } | Ty::Array { element, .. } = &shape {
            if !address && matches!(shape, Ty::Array { .. }) {
                let local = self.alloc_local(callee_ty.clone(), None);
                self.emit(Instr::SetLocal { local });
                self.emit(Instr::LocalAddress { local });
            }
            let index_ty = self.gen_term(arg, None)?;
            if !index_ty.is_integer() {
                return Err(GenerateError::typing(
                    arg.span,
                    TypeError {
                        kind: TypeErrorKind::ExpectedInteger { found: index_ty },
                    },
                ));
            }
            self.emit(Instr::AccessDynamic);
            return Ok(Ty::Pointer {
                pointee: element.clone(),
            });
        }

        let Ty::Function { param, .. } = &callee_ty else {
            return Err(GenerateError::typing(
                func.span,
                TypeError {
                    kind: TypeErrorKind::ExpectedFunction { found: callee_ty },
                },
            ));
        };
        self.gen_term(arg, Some(param))?;
        self.emit(Instr::Call);
        let Ty::Function { result, .. } = callee_ty else {
            unreachable!("checked call")
        };
        Ok(*result)
    }

    fn gen_ascription(&mut self, span: Span, ty: &Type, arg: &Term) -> Result<Ty, GenerateError> {
        let ascribed = self.evaluator().ty(ty)?;
        if let Ty::Arc { pointee } = &ascribed {
            if matches!(arg.val, TermKind::Record { .. } | TermKind::Unit) {
                self.gen_shared_payload(span, pointee, arg)?;
            } else {
                self.gen_term(arg, Some(pointee))?;
            }
            self.emit(Instr::ArcNew);
            return Ok(ascribed);
        }
        if let Ty::Weak { pointee } = &ascribed
            && matches!(arg.val, TermKind::Unit)
        {
            self.emit(Instr::WeakEmpty {
                pointee: *pointee.clone(),
            });
            return Ok(ascribed);
        }
        let context = self
            .typer
            .body(&ascribed)
            .map_err(|err| GenerateError::typing(span, err))?;
        let context = context.span_record().unwrap_or(context);
        let found = if matches!(&context, Ty::Record { fields } if fields.is_empty())
            && matches!(arg.val, TermKind::Unit)
        {
            self.emit(Instr::MakeRecord { fields: vec![] });
            context
        } else {
            self.gen_term(arg, None)?
        };
        if found != ascribed && found.widens_to(&ascribed) {
            return self.coerce(span, found, &ascribed);
        }
        if found != ascribed && found.is_numeric() && ascribed.is_numeric() {
            self.emit(Instr::NumericCast {
                ty: ascribed.clone(),
            });
            return Ok(ascribed);
        }
        if found.pointer_cast(&ascribed) {
            self.emit(Instr::PointerCast {
                ty: ascribed.clone(),
            });
            return Ok(ascribed);
        }
        self.apply_ascription(span, &ascribed, found)
    }

    pub(super) fn gen_builtin(
        &mut self,
        span: Span,
        name: &str,
        args: &[Term],
        expected: &Ty,
    ) -> Result<Ty, GenerateError> {
        if name == "&&" || name == "||" {
            return self.gen_short_circuit(span, name, args);
        }
        if matches!(name, "+" | "-")
            && let [
                Term {
                    val: TermKind::Num { value },
                    ..
                },
            ] = args
        {
            let text = if name == "-" {
                format!("-{value}")
            } else {
                value.to_string()
            };
            let (value, ty) = self.evaluator().number(span, &text, Some(expected))?;
            self.emit(Instr::Push { value });
            return Ok(ty);
        }
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

    pub(super) fn apply_ascription(
        &mut self,
        span: Span,
        expected: &Ty,
        found: Ty,
    ) -> Result<Ty, GenerateError> {
        let steps = self
            .typer
            .ascribe(&found, expected)
            .map_err(|err| GenerateError::typing(span, err))?;
        if steps.iter().any(|step| matches!(step, Conv::Unwrap { definition } if self.typer.definition(*definition).unwrap().drop_hook().is_some())) {
            return Err(super::check::error(span, "cannot unwrap a type with drop; access its fields through a pointer or use replace"));
        }
        if steps
            .iter()
            .any(|s| matches!(s, Conv::MakeSpan | Conv::SpanRecord))
        {
            self.emit(Instr::Ascribe {
                ty: expected.clone(),
            });
        } else {
            self.emit_value_conv(&steps);
        }
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
