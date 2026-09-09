use crate::lower::context::Context;
use std::fmt;

use crate::ast::{Ident, Span, Type};
use crate::types::{Ty, Value};

use super::scope::ContextView;
use super::{GenerateError, GenerateErrorKind};

pub(crate) struct Evaluator<'a> {
    pub(crate) scopes: &'a ContextView,
    pub(crate) typer: &'a Context,
}

impl Evaluator<'_> {
    pub(super) fn type_name(&self, name: &Ident) -> Result<Ty, GenerateError> {
        let ty = self.scopes.resolve_type(name)?;
        crate::lower::infer::solver::Solver::default().require(&ty, name.span)
    }

    pub(super) fn number(
        &self,
        span: Span,
        text: &str,
        expected: Option<&Ty>,
    ) -> Result<(Value, Ty), GenerateError> {
        let ty = self.numeric_type(span, text, expected)?;
        let (text, _) = crate::types::literal::split(text);
        let value = parse_number(text, &ty).map_err(|message| GenerateError {
            span,
            kind: GenerateErrorKind::InvalidLiteral {
                message: message.into(),
            },
        })?;
        Ok((value, ty))
    }

    fn numeric_type(
        &self,
        span: Span,
        text: &str,
        expected: Option<&Ty>,
    ) -> Result<Ty, GenerateError> {
        if let (_, Some(ty)) = crate::types::literal::split(text) {
            return Ok(ty);
        }
        if let Some(expected) = expected {
            let shape = self
                .typer
                .body(expected)
                .map_err(|err| GenerateError::typing(span, err))?;
            if shape.is_numeric() {
                return Ok(shape);
            }
        }
        Ok(self.typer.type_num(text))
    }

    pub(crate) fn ty(&self, ty: &Type) -> Result<Ty, GenerateError> {
        let mut solver = crate::lower::infer::solver::Solver::default();
        let inferred = super::annotation::Decoder {
            solver: &mut solver,
            holes: Vec::new(),
            resolve: &mut |name| {
                if name.val.as_ref() == "String" {
                    return Ok(self
                        .typer
                        .string_type()
                        .cloned()
                        .expect("builtin String")
                        .into());
                }
                self.scopes.resolve_type(name)
            },
        }
        .decode(ty, false)?;
        solver.require(&inferred.ty, ty.span)
    }
}

fn parse_number(text: &str, ty: &Ty) -> Result<Value, String> {
    let compact: String = text.chars().filter(|c| *c != '_').collect();
    match ty {
        Ty::Float32 => {
            if is_hex_literal(&compact) {
                return Err("hexadecimal float literals are not supported".into());
            }
            let value: f32 = compact
                .parse()
                .map_err(|err| format!("invalid float literal: {err}"))?;
            if !value.is_finite() {
                return Err("float literal out of range for float32".into());
            }
            Ok(Value::Float32 { value })
        }
        Ty::Float64 => Ok(Value::Float64 {
            value: parse_float(&compact)?,
        }),
        Ty::Int8 => Ok(Value::Int8 {
            value: parse_signed(&compact)?,
        }),
        Ty::Int16 => Ok(Value::Int16 {
            value: parse_signed(&compact)?,
        }),
        Ty::Int32 => Ok(Value::Int32 {
            value: parse_signed(&compact)?,
        }),
        Ty::Int64 => Ok(Value::Int64 {
            value: parse_signed(&compact)?,
        }),
        Ty::UInt8 => Ok(Value::UInt8 {
            value: parse_unsigned(&compact)?,
        }),
        Ty::UInt16 => Ok(Value::UInt16 {
            value: parse_unsigned(&compact)?,
        }),
        Ty::UInt32 => Ok(Value::UInt32 {
            value: parse_unsigned(&compact)?,
        }),
        Ty::UInt64 => Ok(Value::UInt64 {
            value: parse_unsigned(&compact)?,
        }),
        other => Err(format!("cannot use numeric literal as {other:?}")),
    }
}

fn parse_float(text: &str) -> Result<f64, String> {
    if is_hex_literal(text) {
        return Err("hexadecimal float literals are not supported".into());
    }
    let value: f64 = text
        .parse()
        .map_err(|err| format!("invalid float literal: {err}"))?;
    if !value.is_finite() {
        return Err("float literal out of range for float64".into());
    }
    Ok(value)
}

fn parse_signed<T: TryFrom<i128>>(text: &str) -> Result<T, String>
where
    T::Error: fmt::Display,
{
    let (negative, magnitude) = text.strip_prefix('-').map_or((false, text), |s| (true, s));
    let value = if let Some(hex) = magnitude
        .strip_prefix("0x")
        .or_else(|| magnitude.strip_prefix("0X"))
    {
        i128::from_str_radix(hex, 16).map_err(|err| format!("invalid hex literal: {err}"))?
    } else {
        magnitude
            .parse::<i128>()
            .map_err(|err| format!("invalid integer literal: {err}"))?
    };
    let value = if negative { -value } else { value };
    T::try_from(value).map_err(|err| format!("integer literal out of range: {err}"))
}

fn parse_unsigned<T: TryFrom<u128>>(text: &str) -> Result<T, String>
where
    T::Error: fmt::Display,
{
    let value = if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u128::from_str_radix(hex, 16).map_err(|err| format!("invalid hex literal: {err}"))?
    } else {
        text.parse()
            .map_err(|err| format!("invalid integer literal: {err}"))?
    };
    T::try_from(value).map_err(|err| format!("integer literal out of range: {err}"))
}

fn is_hex_literal(value: &str) -> bool {
    let value = value.strip_prefix('-').unwrap_or(value);
    value.len() >= 2
        && value.as_bytes()[0] == b'0'
        && (value.as_bytes()[1] == b'x' || value.as_bytes()[1] == b'X')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Ident, TypeKind};
    use crate::lower::scope::Scopes;

    #[test]
    fn evaluation_only_needs_scopes_and_types() {
        let typer = Context::new();
        let mut scopes = Scopes::new();
        scopes
            .define_alias(
                &Ident::new("Byte".into(), Span { start: 0, end: 4 }),
                Ty::Int8,
            )
            .unwrap();
        let evaluator = Evaluator {
            scopes: scopes.view(),
            typer: &typer,
        };
        let span = Span { start: 4, end: 8 };
        let named = Type::new(
            TypeKind::Atom {
                name: Ident::new("Byte".into(), span),
            },
            span,
        );
        let ty = evaluator.ty(&named).unwrap();
        assert_eq!(ty, Ty::Int8);
        assert_eq!(
            evaluator.number(span, "-128", Some(&ty)).unwrap(),
            (Value::Int8 { value: -128 }, Ty::Int8),
        );
        let error = evaluator.number(span, "128", Some(&ty)).unwrap_err();
        assert_eq!(error.span, span);
        assert!(matches!(
            error.kind,
            GenerateErrorKind::InvalidLiteral { .. }
        ));
    }

    #[test]
    fn numeric_values_keep_the_selected_type() {
        let scopes = Scopes::new();
        let typer = Context::new();
        let evaluator = Evaluator {
            scopes: scopes.view(),
            typer: &typer,
        };
        let span = Span { start: 0, end: 0 };
        for (text, expected, value, ty) in [
            ("0x1_e", None, Value::Int64 { value: 30 }, Ty::Int64),
            ("1e3", None, Value::Float64 { value: 1000.0 }, Ty::Float64),
            (
                "1.5",
                Some(Ty::Float32),
                Value::Float32 { value: 1.5 },
                Ty::Float32,
            ),
            (
                "255",
                Some(Ty::UInt8),
                Value::UInt8 { value: 255 },
                Ty::UInt8,
            ),
        ] {
            assert_eq!(
                evaluator.number(span, text, expected.as_ref()).unwrap(),
                (value, ty),
            );
        }
    }
}
