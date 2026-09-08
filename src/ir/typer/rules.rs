use std::{collections::HashSet, sync::Arc};

use crate::ir::{RecordField, Ty};

use super::{Conv, TypeError, TypeErrorKind, TyperContext};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldAccess {
    pub ty: Ty,
    pub index: usize,
    pub steps: Vec<Conv>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinCall {
    pub params: Vec<Ty>,
    pub result: Ty,
}

impl TyperContext {
    pub const fn type_unit(&self) -> Ty {
        Ty::Unit
    }

    pub fn type_num(&self, value: &str) -> Ty {
        if let (_, Some(ty)) = crate::ir::literal::split(value) {
            return ty;
        }
        let value = value.strip_prefix('-').unwrap_or(value);
        let is_hex = value.starts_with("0x") || value.starts_with("0X");
        if value.contains('.') || (!is_hex && value.contains(['e', 'E'])) {
            Ty::Float64
        } else {
            Ty::Int32
        }
    }

    pub const fn type_type(&self, _value: &Ty) -> Ty {
        Ty::Type
    }

    pub fn type_var(&self, binding: &Ty) -> Ty {
        binding.clone()
    }

    pub fn type_block(&self, tail: &Ty) -> Ty {
        tail.clone()
    }

    pub fn type_assign(&self, place: &Ty, value: &Ty) -> Result<Ty, TypeError> {
        self.same(place, value)?;
        Ok(value.clone())
    }

    pub fn type_array(&self, elements: &[Ty]) -> Result<Ty, TypeError> {
        let Some(element) = elements.first() else {
            return Err(TypeError::new(TypeErrorKind::EmptyArrayNeedsElementType));
        };
        self.type_array_of(element, elements)
    }

    pub fn type_array_of(&self, element: &Ty, elements: &[Ty]) -> Result<Ty, TypeError> {
        for found in elements {
            self.same(element, found)?;
        }
        Ok(Ty::Array {
            element: Box::new(element.clone()),
            length: elements.len(),
        })
    }

    pub fn type_record(&self, fields: &[RecordField]) -> Result<Ty, TypeError> {
        let mut names = HashSet::with_capacity(fields.len());
        for field in fields {
            if !names.insert(field.name.clone()) {
                return Err(TypeError::new(TypeErrorKind::DuplicateField {
                    name: field.name.clone(),
                }));
            }
        }
        Ok(Ty::Record {
            fields: fields.to_vec(),
        })
    }

    pub fn type_function(&self, param: &Ty, body: &Ty) -> Ty {
        Ty::Function {
            param: Box::new(param.clone()),
            result: Box::new(body.clone()),
        }
    }

    pub fn type_if(&self, condition: &Ty, then_ty: &Ty, else_ty: &Ty) -> Result<Ty, TypeError> {
        self.as_bool(condition)?;
        self.same(then_ty, else_ty)?;
        Ok(then_ty.clone())
    }

    pub fn type_deref(&self, pointer: &Ty) -> Result<Ty, TypeError> {
        let Some(pointee) = pointer.deref_target() else {
            return Err(TypeError::new(TypeErrorKind::ExpectedPointer {
                found: pointer.clone(),
            }));
        };
        Ok(pointee.clone())
    }

    pub fn type_field(&self, base: &Ty, name: &str) -> Result<FieldAccess, TypeError> {
        let converted = self.as_record(base)?;
        let shape = converted.ty.span_record().unwrap_or(converted.ty);
        let Ty::Record { fields } = shape else {
            return Err(TypeError::new(TypeErrorKind::ExpectedRecord {
                found: base.clone(),
            }));
        };
        for (index, field) in fields.into_iter().enumerate() {
            if field.name.as_ref() == name {
                return Ok(FieldAccess {
                    ty: field.ty,
                    index,
                    steps: converted.steps,
                });
            }
        }
        Err(TypeError::new(TypeErrorKind::UnknownField {
            name: Arc::from(name),
        }))
    }

    pub fn type_call(&self, callee: &Ty, arg: &Ty) -> Result<Ty, TypeError> {
        if let Ty::Span { element } | Ty::Array { element, .. } = self.body(callee)? {
            if !arg.is_integer() {
                return Err(TypeError::new(TypeErrorKind::ExpectedInteger {
                    found: arg.clone(),
                }));
            }
            return Ok(Ty::Pointer { pointee: element });
        }
        let Ty::Function { param, result } = callee else {
            return Err(TypeError::new(TypeErrorKind::ExpectedFunction {
                found: callee.clone(),
            }));
        };
        self.same(param, arg)?;
        Ok(*result.clone())
    }

    pub fn type_ascription(&self, expected: &Ty, value: &Ty) -> Result<Ty, TypeError> {
        self.ascribe(value, expected)?;
        Ok(expected.clone())
    }

    pub fn type_builtin_call(&self, name: &str, args: &[Ty]) -> Result<BuiltinCall, TypeError> {
        if matches!(
            name,
            "+" | "-" | "~" | "*" | "/" | "%" | "<<" | ">>" | "&" | "|" | "^"
        ) && args.iter().any(|ty| matches!(ty, Ty::Pointer { .. }))
        {
            return Err(TypeError::new(TypeErrorKind::PointerArithmetic));
        }
        let result = match (name, args) {
            ("print", [arg]) => self.type_print(arg)?,
            ("+" | "-" | "~", [arg]) => arg.clone(),
            ("!", [arg]) => {
                self.as_bool(arg)?;
                Ty::Bool
            }
            ("+" | "-" | "*" | "/" | "%" | "<<" | ">>" | "&" | "|" | "^", [left, right]) => {
                self.same(left, right)?;
                left.clone()
            }
            ("==" | "!=" | "<" | "<=" | ">" | ">=", [left, right]) => {
                self.same(left, right)?;
                Ty::Bool
            }
            ("&&" | "||", [left, right]) => {
                self.as_bool(left)?;
                self.as_bool(right)?;
                Ty::Bool
            }
            (
                "+" | "-" | "~" | "!" | "*" | "/" | "%" | "<<" | ">>" | "&" | "|" | "^" | "=="
                | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||" | "print",
                _,
            ) => {
                return Err(TypeError::new(TypeErrorKind::InvalidBuiltinArgumentCount {
                    name: Arc::from(name),
                    found: args.len(),
                }));
            }
            _ => {
                return Err(TypeError::new(TypeErrorKind::UnknownBuiltin {
                    name: Arc::from(name),
                }));
            }
        };
        Ok(BuiltinCall {
            params: args.to_vec(),
            result,
        })
    }

    fn type_print(&self, arg: &Ty) -> Result<Ty, TypeError> {
        let invalid =
            || TypeError::new(TypeErrorKind::InvalidPrintArguments { found: arg.clone() });
        let Ty::Record { fields } = arg else {
            return Err(invalid());
        };
        let [format, values] = fields.as_slice() else {
            return Err(invalid());
        };
        if format.name.as_ref() != "_0"
            || values.name.as_ref() != "_1"
            || !matches!(&format.ty, Ty::Array { element, .. } if **element == Ty::UInt8)
        {
            return Err(invalid());
        }
        let fields = match &values.ty {
            Ty::Unit => &[][..],
            Ty::Record { fields } => fields,
            _ => return Err(invalid()),
        };
        for (i, field) in fields.iter().enumerate() {
            if field.name.as_ref() != format!("_{i}") {
                return Err(invalid());
            }
            let ty = &field.ty;
            if !(ty.is_numeric()
                || matches!(ty, Ty::Bool | Ty::Unit | Ty::Pointer { .. })
                || matches!(ty, Ty::Array { element, .. } if **element == Ty::UInt8))
            {
                return Err(TypeError::new(TypeErrorKind::UnprintableType {
                    found: ty.clone(),
                }));
            }
        }
        Ok(Ty::Unit)
    }
}
