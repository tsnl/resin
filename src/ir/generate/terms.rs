use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::ast::{Ident, Span, Term, TermKind, Type};
use crate::ir::{Conv, Instr, RecordField, Ty, TypeError, TypeErrorKind};

use super::{GenerateError, GenerateErrorKind, Generator};

impl Generator {
    pub(super) fn gen_array(
        &mut self,
        span: Span,
        elems: &[Term],
        expected: Option<&Ty>,
    ) -> Result<Ty, GenerateError> {
        let expected_element = expected.and_then(|ty| self.array_element(ty));
        let mut elem_tys = Vec::with_capacity(elems.len());
        for elem in elems {
            elem_tys.push(self.gen_term(elem, expected_element.as_ref())?);
        }
        let ty = if let Some(element) = expected_element {
            self.typer
                .type_array_of(&element, &elem_tys)
                .map_err(|err| GenerateError::typing(span, err))?
        } else {
            self.typer
                .type_array(&elem_tys)
                .map_err(|err| GenerateError::typing(span, err))?
        };
        let Ty::Array { element, length } = &ty else {
            unreachable!("array typing produces an array");
        };
        self.emit(Instr::MakeArray {
            elements: *length,
            element: element.as_ref().clone(),
        });
        Ok(ty)
    }

    pub(super) fn gen_record(
        &mut self,
        span: Span,
        fields: &[(Ident, Term)],
        expected: Option<&Ty>,
    ) -> Result<Ty, GenerateError> {
        if let Some(expected_fields) = expected.and_then(|ty| self.record_fields(ty)) {
            check_fields(span, fields, &expected_fields)?;
            // Evaluate in source order; layout order must not reorder effects.
            let mut values = HashMap::with_capacity(fields.len());
            for (name, value) in fields {
                let field = expected_fields
                    .iter()
                    .find(|field| field.name == name.val)
                    .unwrap();
                let local = self.alloc_local(field.ty.clone(), None);
                self.emit(Instr::LocalAddress { local });
                self.gen_term(value, Some(&field.ty))?;
                self.emit(Instr::Store);
                self.emit(Instr::Discard);
                values.insert(name.val.clone(), local);
            }
            let names = expected_fields
                .iter()
                .map(|field| {
                    self.emit(Instr::LocalAddress {
                        local: values[&field.name],
                    });
                    self.emit(Instr::Load);
                    field.name.clone()
                })
                .collect();
            self.emit(Instr::MakeRecord { fields: names });
            return Ok(Ty::Record {
                fields: expected_fields,
            });
        }

        let mut names = Vec::with_capacity(fields.len());
        let mut typed = Vec::with_capacity(fields.len());
        for (name, value) in fields {
            let ty = self.gen_term(value, None)?;
            names.push(name.val.clone());
            typed.push(RecordField {
                name: name.val.clone(),
                ty,
            });
        }
        let ty = self
            .typer
            .type_record(&typed)
            .map_err(|err| GenerateError::typing(span, err))?;
        self.emit(Instr::MakeRecord { fields: names });
        Ok(ty)
    }

    pub(super) fn gen_call(
        &mut self,
        span: Span,
        func: &Term,
        arg: &Term,
    ) -> Result<Ty, GenerateError> {
        if let TermKind::Type { ty } = &func.val {
            return self.gen_ascription(span, ty, arg);
        }
        if let TermKind::Var { name } = &func.val
            && name.val.as_ref() == "shader"
            && self.scopes.lookup_value("shader").is_none()
        {
            return self.gen_shader(span, arg);
        }
        if let TermKind::Var { name } = &func.val
            && name.val.as_ref() == "print"
            && self.scopes.lookup_value("print").is_none()
        {
            return self.gen_builtin(span, "print", std::slice::from_ref(arg), None);
        }

        let callee_ty = self.gen_term(func, None)?;
        let converted = self
            .typer
            .as_function(&callee_ty)
            .map_err(|err| GenerateError::typing(func.span, err))?;
        self.emit_value_conv(&converted.steps);
        let Ty::Function { param, .. } = &converted.ty else {
            return Err(GenerateError::typing(
                func.span,
                TypeError {
                    kind: TypeErrorKind::ExpectedFunction { found: callee_ty },
                },
            ));
        };
        let arg_ty = self.gen_term(arg, Some(param))?;
        let result = self
            .typer
            .type_call(&callee_ty, &arg_ty)
            .map_err(|err| GenerateError::typing(span, err))?;
        self.emit(Instr::Call);
        Ok(result)
    }

    fn gen_ascription(&mut self, span: Span, ty: &Type, arg: &Term) -> Result<Ty, GenerateError> {
        let ascribed = self.evaluator().ty(ty)?;
        let context = self
            .typer
            .body(&ascribed)
            .map_err(|err| GenerateError::typing(span, err))?;
        let found = self.gen_term_inner(arg, Some(&context))?;
        if found.pointer_cast(&ascribed) {
            self.emit(Instr::PointerCast {
                ty: ascribed.clone(),
            });
            return Ok(ascribed);
        }
        self.apply_ascription(span, &ascribed, found)
    }

    fn gen_shader(&mut self, span: Span, arg: &Term) -> Result<Ty, GenerateError> {
        let invalid = || GenerateError {
            span,
            kind: GenerateErrorKind::InvalidShader {
                message: "expected shader(named_function, \"compute\" | \"vertex\" | \"fragment\")"
                    .into(),
            },
        };
        let TermKind::Record { fields } = &arg.val else {
            return Err(invalid());
        };
        let [(first, function), (second, stage)] = fields.as_slice() else {
            return Err(invalid());
        };
        let (TermKind::Var { name }, TermKind::String { value: stage }) =
            (&function.val, &stage.val)
        else {
            return Err(invalid());
        };
        if first.val.as_ref() != "_0"
            || second.val.as_ref() != "_1"
            || !matches!(stage.as_ref(), "compute" | "vertex" | "fragment")
        {
            return Err(invalid());
        }
        let binding = self.resolve_value(name)?;
        let super::scope::ValueBindingKind::Function(function) = binding.kind else {
            return Err(invalid());
        };
        if self.module.functions[function.index()].foreign.is_some() {
            return Err(invalid());
        }
        self.emit(Instr::Shader {
            function,
            stage: stage.clone(),
        });
        Ok(Ty::shader())
    }

    pub(super) fn gen_builtin(
        &mut self,
        span: Span,
        name: &str,
        args: &[Term],
        expected: Option<&Ty>,
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
            let (value, ty) = self.evaluator().number(span, &text, expected)?;
            self.emit(Instr::Push { value });
            return Ok(ty);
        }
        let numeric_context = expected.filter(|ty| ty.is_numeric()).filter(|_| {
            matches!(
                name,
                "+" | "-" | "*" | "/" | "%" | "~" | "<<" | ">>" | "&" | "|" | "^"
            )
        });
        let mut arg_tys = Vec::with_capacity(args.len());
        for arg in args {
            arg_tys.push(self.gen_term(arg, numeric_context)?);
        }
        if name == "!" && arg_tys.len() == 1 {
            let converted = self
                .typer
                .as_bool(&arg_tys[0])
                .map_err(|err| GenerateError::typing(span, err))?;
            self.emit_value_conv(&converted.steps);
            arg_tys[0] = converted.ty;
        }
        let call = self
            .typer
            .type_builtin_call(name, &arg_tys)
            .map_err(|err| GenerateError::typing(span, err))?;
        self.emit(Instr::CallBuiltin {
            name: Arc::from(name),
            params: call.params,
            result: call.result.clone(),
        });
        Ok(call.result)
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
        self.emit_value_conv(&steps);
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
            }
        }
    }

    fn array_element(&self, ty: &Ty) -> Option<Ty> {
        match self.typer.body(ty).ok()? {
            Ty::Array { element, .. } => Some(*element),
            _ => None,
        }
    }

    fn record_fields(&self, ty: &Ty) -> Option<Vec<RecordField>> {
        match self.typer.body(ty).ok()? {
            Ty::Record { fields } => Some(fields),
            _ => None,
        }
    }
}

fn check_fields(
    span: Span,
    fields: &[(Ident, Term)],
    expected: &[RecordField],
) -> Result<(), GenerateError> {
    let mut names = HashSet::with_capacity(fields.len());
    for (name, _) in fields {
        if !names.insert(name.val.clone()) {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::Type(TypeErrorKind::DuplicateField {
                    name: name.val.clone(),
                }),
            });
        }
    }
    for field in expected {
        if !names.contains(&field.name) {
            return Err(GenerateError {
                span,
                kind: GenerateErrorKind::MissingField {
                    name: field.name.clone(),
                },
            });
        }
    }
    for (name, _) in fields {
        if !expected.iter().any(|field| field.name == name.val) {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::ExtraField {
                    name: name.val.clone(),
                },
            });
        }
    }
    Ok(())
}
