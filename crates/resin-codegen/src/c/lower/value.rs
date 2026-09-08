use super::types::Types;
use resin_types::prelude::*;

pub(super) fn literal(types: &Types<'_>, ty: &Ty, value: &Value) -> String {
    match value {
        Value::Str { value } => format!(
            "({}){{ .f0 = {}, .f1 = {} }}",
            types.name(ty),
            types.literal(value),
            value.len()
        ),
        Value::None | Value::Unit => "0".into(),
        Value::Type { ty } => types.id(ty).to_string(),
        Value::Bool { value } => value.to_string(),
        Value::Int8 { value } => signed(i64::from(*value)),
        Value::Int16 { value } => signed(i64::from(*value)),
        Value::Int32 { value } => signed(i64::from(*value)),
        Value::Int64 { value } => signed(*value),
        Value::UInt8 { value } => format!("UINT64_C({value})"),
        Value::UInt16 { value } => format!("UINT64_C({value})"),
        Value::UInt32 { value } => format!("UINT64_C({value})"),
        Value::UInt64 { value } => format!("UINT64_C({value})"),
        Value::Float32 { value } => format!("(float)({})", float(f64::from(*value))),
        Value::Float64 { value } => float(*value),
        Value::Array { value } => {
            let values = value
                .elements
                .iter()
                .map(|v| literal(types, &value.element_ty, v))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "({}){{ .items = {{ {} }} }}",
                types.name(ty),
                if values.is_empty() { "0" } else { &values }
            )
        }
        Value::Record { value } => {
            let Ty::Record { fields } = ty else {
                unreachable!()
            };
            let values = value
                .fields
                .iter()
                .zip(fields)
                .map(|(field, ty)| literal(types, &ty.ty, &field.value))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "({}){{ {} }}",
                types.name(ty),
                if values.is_empty() { "0" } else { &values }
            )
        }
        _ => unreachable!("verifier rejects non-literal immediates"),
    }
}

fn signed(value: i64) -> String {
    if value == i64::MIN {
        "(-INT64_C(9223372036854775807) - INT64_C(1))".into()
    } else if value < 0 {
        format!("-INT64_C({})", -value)
    } else {
        format!("INT64_C({value})")
    }
}

fn float(value: f64) -> String {
    if value.is_nan() {
        "NAN".into()
    } else if value == f64::INFINITY {
        "INFINITY".into()
    } else if value == f64::NEG_INFINITY {
        "(-INFINITY)".into()
    } else {
        format!("{value:e}")
    }
}
