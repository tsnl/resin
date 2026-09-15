//! Checked scalar arithmetic for compile-time constants.
use crate::{Ty, Value};

fn integer(value: &Value) -> Option<i128> {
    Some(match value {
        Value::Int8 { value } => i128::from(*value),
        Value::Int16 { value } => i128::from(*value),
        Value::Int32 { value } => i128::from(*value),
        Value::Int64 { value } => i128::from(*value),
        Value::UInt8 { value } => i128::from(*value),
        Value::UInt16 { value } => i128::from(*value),
        Value::UInt32 { value } => i128::from(*value),
        Value::UInt64 { value } => i128::from(*value),
        _ => return None,
    })
}

fn integer_type(value: &Value) -> Option<Ty> {
    Some(match value {
        Value::Int8 { .. } => Ty::Int8,
        Value::Int16 { .. } => Ty::Int16,
        Value::Int32 { .. } => Ty::Int32,
        Value::Int64 { .. } => Ty::Int64,
        Value::UInt8 { .. } => Ty::UInt8,
        Value::UInt16 { .. } => Ty::UInt16,
        Value::UInt32 { .. } => Ty::UInt32,
        Value::UInt64 { .. } => Ty::UInt64,
        _ => return None,
    })
}

fn width(ty: &Ty) -> u32 {
    match ty {
        Ty::Int8 | Ty::UInt8 => 8,
        Ty::Int16 | Ty::UInt16 => 16,
        Ty::Int32 | Ty::UInt32 => 32,
        Ty::Int64 | Ty::UInt64 => 64,
        _ => unreachable!("integer type"),
    }
}

fn integer_value(value: i128, ty: &Ty) -> Result<Value, String> {
    crate::literal::parse(&value.to_string(), ty)
        .map_err(|_| format!("constant value {value} overflows {ty:?}"))
}

fn comparison<T: PartialOrd>(name: &str, left: T, right: T) -> Option<Value> {
    Some(Value::Bool {
        value: match name {
            "==" => left == right,
            "!=" => left != right,
            "<" => left < right,
            "<=" => left <= right,
            ">" => left > right,
            ">=" => left >= right,
            _ => return None,
        },
    })
}

pub(super) fn operation(name: &str, operands: &[Value]) -> Result<Value, String> {
    let Some(left) = operands.first() else {
        return Err("constant operator requires operands".into());
    };
    if let Some(ty) = integer_type(left) {
        return integer_operation(name, operands, &ty);
    }
    match operands {
        [Value::Float32 { value: left }] => float32(name, *left, None),
        [
            Value::Float32 { value: left },
            Value::Float32 { value: right },
        ] => float32(name, *left, Some(*right)),
        [Value::Float64 { value: left }] => float64(name, *left, None),
        [
            Value::Float64 { value: left },
            Value::Float64 { value: right },
        ] => float64(name, *left, Some(*right)),
        [Value::Bool { value }] if name == "!" => Ok(Value::Bool { value: !value }),
        [Value::Bool { value: left }, Value::Bool { value: right }] => match name {
            "&&" => Ok(Value::Bool {
                value: *left && *right,
            }),
            "||" => Ok(Value::Bool {
                value: *left || *right,
            }),
            _ => comparison(name, left, right).ok_or_else(unsupported),
        },
        _ => Err(unsupported()),
    }
}

fn unsupported() -> String {
    "unsupported constant operator or operand types".into()
}

fn integer_operation(name: &str, operands: &[Value], ty: &Ty) -> Result<Value, String> {
    if operands
        .iter()
        .any(|value| integer_type(value).as_ref() != Some(ty))
    {
        return Err("constant operands must have the same type".into());
    }
    let left = integer(&operands[0]).unwrap();
    if operands.len() == 1 {
        let value = match name {
            "+" => left,
            "-" => -left,
            "~" if matches!(ty, Ty::UInt8 | Ty::UInt16 | Ty::UInt32 | Ty::UInt64) => {
                left ^ ((1_i128 << width(ty)) - 1)
            }
            "~" => !left,
            _ => return Err(unsupported()),
        };
        return integer_value(value, ty);
    }
    if operands.len() != 2 {
        return Err(unsupported());
    }
    let right = integer(&operands[1]).unwrap();
    if let Some(value) = comparison(name, left, right) {
        return Ok(value);
    }
    if matches!(name, "/" | "%") && right == 0 {
        return Err("division by zero in constant expression".into());
    }
    if matches!(name, "<<" | ">>") && !(0..i128::from(width(ty))).contains(&right) {
        return Err(format!(
            "constant shift count must be between 0 and {}",
            width(ty) - 1
        ));
    }
    let value = match name {
        "+" => left.checked_add(right),
        "-" => left.checked_sub(right),
        "*" => left.checked_mul(right),
        "/" => left.checked_div(right),
        "%" => left.checked_rem(right),
        "<<" => left.checked_mul(1_i128 << right),
        ">>" => Some(left >> right),
        "&" => Some(left & right),
        "|" => Some(left | right),
        "^" => Some(left ^ right),
        _ => return Err(unsupported()),
    }
    .ok_or("integer overflow in constant expression")?;
    integer_value(value, ty)
}

// Keep each operation at its declared precision, including intermediate rounding.
fn float32(name: &str, left: f32, right: Option<f32>) -> Result<Value, String> {
    let value = match (name, right) {
        ("+", None) => left,
        ("-", None) => -left,
        ("+", Some(right)) => left + right,
        ("-", Some(right)) => left - right,
        ("*", Some(right)) => left * right,
        ("/" | "%", Some(0.0)) => return Err("division by zero in constant expression".into()),
        ("/", Some(right)) => left / right,
        ("%", Some(right)) => left % right,
        (_, Some(right)) => return comparison(name, left, right).ok_or_else(unsupported),
        _ => return Err(unsupported()),
    };
    if !value.is_finite() {
        return Err("non-finite float32 constant".into());
    }
    Ok(Value::Float32 { value })
}

fn float64(name: &str, left: f64, right: Option<f64>) -> Result<Value, String> {
    let value = match (name, right) {
        ("+", None) => left,
        ("-", None) => -left,
        ("+", Some(right)) => left + right,
        ("-", Some(right)) => left - right,
        ("*", Some(right)) => left * right,
        ("/" | "%", Some(0.0)) => return Err("division by zero in constant expression".into()),
        ("/", Some(right)) => left / right,
        ("%", Some(right)) => left % right,
        (_, Some(right)) => return comparison(name, left, right).ok_or_else(unsupported),
        _ => return Err(unsupported()),
    };
    if !value.is_finite() {
        return Err("non-finite float64 constant".into());
    }
    Ok(Value::Float64 { value })
}

pub(super) fn convert(value: &Value, to: &Ty) -> Result<Value, String> {
    if *to == Ty::Float32
        && let Some(value) = integer(value)
    {
        return Ok(Value::Float32 {
            value: value as f32,
        });
    }
    if to.is_integer() {
        let value = if let Some(value) = integer(value) {
            value
        } else {
            let value = match value {
                Value::Float32 { value } => f64::from(*value),
                Value::Float64 { value } => *value,
                _ => return Err("constant conversion requires numeric operands".into()),
            };
            if !value.is_finite() || !(-2_f64.powi(64)..2_f64.powi(64)).contains(&value) {
                return Err("constant conversion overflows integer type".into());
            }
            value.trunc() as i128
        };
        return integer_value(value, to);
    }
    let value = match value {
        Value::Float32 { value } => f64::from(*value),
        Value::Float64 { value } => *value,
        value => integer(value).ok_or("constant conversion requires numeric operands")? as f64,
    };
    match to {
        Ty::Float32 if (value as f32).is_finite() => Ok(Value::Float32 {
            value: value as f32,
        }),
        Ty::Float64 if value.is_finite() => Ok(Value::Float64 { value }),
        _ => Err("constant conversion overflows or requires a numeric type".into()),
    }
}
