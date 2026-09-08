use std::sync::Arc;

use crate::types::Ty;

use super::{BuiltinRule, Conv, TypeError, TypeErrorKind, TyperContext};

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
    pub fn type_num(&self, value: &str) -> Ty {
        crate::types::literal::split(value)
            .1
            .unwrap_or_else(|| crate::types::literal::unsuffixed_type(value))
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

    pub fn type_builtin_call(&self, name: &str, args: &[Ty]) -> Result<BuiltinCall, TypeError> {
        let rule = BuiltinRule::lookup(name, args.len())?;
        let result = match rule {
            BuiltinRule::Print if self.is_string(&args[0]) => Ty::Unit,
            BuiltinRule::Print => {
                return Err(TypeError::new(TypeErrorKind::InvalidPrintArguments {
                    found: args[0].clone(),
                }));
            }
            BuiltinRule::Format => self.type_format(&args[0])?,
            BuiltinRule::StringFromStr => {
                self.same(&Ty::byte_span(), &args[0])?;
                self.string_type.clone().ok_or_else(|| {
                    TypeError::new(TypeErrorKind::UnknownBuiltin { name: name.into() })
                })?
            }
            BuiltinRule::Boolean => {
                for arg in args {
                    self.as_bool(arg)?;
                }
                Ty::Bool
            }
            BuiltinRule::Arithmetic | BuiltinRule::Comparison => {
                if rule == BuiltinRule::Arithmetic
                    && args.iter().any(|ty| matches!(ty, Ty::Pointer { .. }))
                {
                    return Err(TypeError::new(TypeErrorKind::PointerArithmetic));
                }
                for arg in &args[1..] {
                    self.same(&args[0], arg)?;
                }
                if rule == BuiltinRule::Comparison {
                    Ty::Bool
                } else {
                    args[0].clone()
                }
            }
        };
        Ok(BuiltinCall {
            params: args.to_vec(),
            result,
        })
    }

    pub(crate) fn is_string(&self, ty: &Ty) -> bool {
        ty == &Ty::byte_span() || self.string_type.as_ref() == Some(ty)
    }

    fn type_format(&self, arg: &Ty) -> Result<Ty, TypeError> {
        let invalid =
            || TypeError::new(TypeErrorKind::InvalidFormatArguments { found: arg.clone() });
        let Ty::Record { fields } = arg else {
            return Err(invalid());
        };
        let [format, values] = fields.as_slice() else {
            return Err(invalid());
        };
        if format.name.as_ref() != "_0"
            || values.name.as_ref() != "_1"
            || !self.is_string(&format.ty)
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
                || self.is_string(ty))
            {
                return Err(TypeError::new(TypeErrorKind::UnformattableType {
                    found: ty.clone(),
                }));
            }
        }
        self.string_type.clone().ok_or_else(invalid)
    }
}
