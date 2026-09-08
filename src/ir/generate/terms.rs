use std::{collections::HashMap, sync::Arc};

use super::typed::{Term, TermKind};
use crate::ast::{Ident, Span};
use crate::ir::{Conv, Instr, Ty};

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
        if matches!(init.kind, TermKind::Unit)
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

    pub(super) fn gen_call(&mut self, func: &Term, arg: &Term) -> Result<Ty, GenerateError> {
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
            self.gen_term(arg, None)?;
            self.emit(Instr::AccessDynamic);
            return Ok(Ty::Pointer {
                pointee: element.clone(),
            });
        }

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

    pub(super) fn gen_ascription(
        &mut self,
        span: Span,
        ty: &Ty,
        arg: &Term,
    ) -> Result<Ty, GenerateError> {
        let ascribed = ty.clone();
        if let Ty::Arc { pointee } = &ascribed {
            if matches!(arg.kind, TermKind::Record { .. } | TermKind::Unit) {
                self.gen_shared_payload(span, pointee, arg)?;
            } else {
                self.gen_term(arg, Some(pointee))?;
            }
            self.emit(Instr::ArcNew);
            return Ok(ascribed);
        }
        if let Ty::Weak { pointee } = &ascribed
            && matches!(arg.kind, TermKind::Unit)
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
            && matches!(arg.kind, TermKind::Unit)
        {
            self.emit(Instr::MakeRecord { fields: vec![] });
            context
        } else {
            self.gen_term(arg, None)?
        };
        use crate::ir::typecheck::ExplicitConversion;
        match self
            .typer
            .explicit_conversion(&found, &ascribed)
            .map_err(|err| GenerateError::typing(span, err))?
        {
            ExplicitConversion::Widen => return self.coerce(span, found, &ascribed),
            ExplicitConversion::NumericCast => self.emit(Instr::NumericCast {
                ty: ascribed.clone(),
            }),
            ExplicitConversion::PointerCast => self.emit(Instr::PointerCast {
                ty: ascribed.clone(),
            }),
            ExplicitConversion::Ascribe(steps) => {
                if steps.iter().any(|step| matches!(step, Conv::Unwrap { definition } if self.typer.definition(*definition).unwrap().drop_hook().is_some())) {
                    return Err(GenerateError::inference(span, "cannot unwrap a type with drop; access its fields through a pointer or use Ptr.replace"));
                }
                if !steps.is_empty() {
                    self.emit(Instr::Ascribe {
                        ty: ascribed.clone(),
                    });
                }
            }
        }
        Ok(ascribed)
    }

    pub(super) fn gen_builtin(
        &mut self,
        span: Span,
        name: &str,
        args: &[Term],
        expected: &Ty,
    ) -> Result<Ty, GenerateError> {
        if name == "&&" || name == "||" {
            return self.gen_short_circuit(name, args);
        }
        if matches!(name, "+" | "-")
            && let [
                Term {
                    kind: TermKind::Num { value },
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
