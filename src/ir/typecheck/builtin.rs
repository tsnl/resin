use super::{TypeError, TypeErrorKind};

/// Operand relationships shared by concrete checking and inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinRule {
    Arithmetic,
    Comparison,
    Boolean,
    Print,
    Format,
    StringFromStr,
}

impl BuiltinRule {
    pub(crate) fn lookup(name: &str, arity: usize) -> Result<Self, TypeError> {
        let (rule, valid_arity) = match name {
            "+" | "-" => (Self::Arithmetic, matches!(arity, 1 | 2)),
            "~" => (Self::Arithmetic, arity == 1),
            "*" | "/" | "%" | "<<" | ">>" | "&" | "|" | "^" => (Self::Arithmetic, arity == 2),
            "==" | "!=" | "<" | "<=" | ">" | ">=" => (Self::Comparison, arity == 2),
            "!" => (Self::Boolean, arity == 1),
            "&&" | "||" => (Self::Boolean, arity == 2),
            "print" => (Self::Print, arity == 1),
            "fmt" => (Self::Format, arity == 1),
            "string_from_str" => (Self::StringFromStr, arity == 1),
            _ => {
                return Err(TypeError::new(TypeErrorKind::UnknownBuiltin {
                    name: name.into(),
                }));
            }
        };
        if !valid_arity {
            return Err(TypeError::new(TypeErrorKind::InvalidBuiltinArgumentCount {
                name: name.into(),
                found: arity,
            }));
        }
        Ok(rule)
    }
}
