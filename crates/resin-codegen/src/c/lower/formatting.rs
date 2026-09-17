use super::{Slot, types::Types};
use crate::Error;
use resin_types::prelude::*;

pub(super) fn format(types: &Types<'_>, args: &[Slot], result: &Ty) -> Result<String, Error> {
    let invalid = || Error::invalid("format_bytes expects a byte pointer, length and tuple".into());
    let [data, length, arguments] = args else {
        return Err(invalid());
    };
    if result != &Ty::StrongOwner {
        return Err(invalid());
    }
    let fields = match &arguments.ty {
        Ty::Unit => &[][..],
        Ty::Record { fields } => fields.as_slice(),
        _ => return Err(invalid()),
    };
    let mut values = Vec::new();
    for (i, field) in fields.iter().enumerate() {
        if field.name.as_ref() != format!("_{i}") {
            return Err(invalid());
        }
        values.push(value(
            types,
            &field.ty,
            format!("({}).f{i}", arguments.expr),
        )?);
    }
    let count = values.len();
    let values = if count == 0 {
        "NULL".into()
    } else {
        format!("(ResinPrintArg[]){{ {} }}", values.join(", "))
    };
    Ok(format!(
        "resin_format({}, {}, {values}, {count})",
        data.expr, length.expr
    ))
}

pub(super) fn from_bytes(_types: &Types<'_>, args: &[Slot], result: &Ty) -> Result<String, Error> {
    let [data, length] = args else {
        return Err(Error::invalid(
            "byte copy expects pointer and length".into(),
        ));
    };
    if result != &Ty::StrongOwner {
        return Err(Error::invalid("byte copy produces an owner".into()));
    }
    Ok(format!(
        "resin_string_from_str({}, {})",
        data.expr, length.expr
    ))
}

// Verification admits only str or the explicit structural byte transport record.
fn bytes(ty: &Ty, expr: &str) -> Result<(String, String), Error> {
    match ty {
        Ty::Str | Ty::Record { .. } => Ok((format!("({expr}).f0"), format!("({expr}).f1"))),
        _ => Err(Error::invalid(
            "expected str or a structural byte view".into(),
        )),
    }
}

fn value(types: &Types<'_>, ty: &Ty, expr: String) -> Result<String, Error> {
    let (kind, member, value) = match ty {
        Ty::Unit => ("UNIT", "unsigned_value", "0".into()),
        Ty::Bool => ("BOOL", "unsigned_value", expr),
        Ty::Int8 | Ty::Int16 | Ty::Int32 | Ty::Int64 => {
            ("SIGNED", "signed_value", format!("(int64_t)({expr})"))
        }
        Ty::UInt8 | Ty::UInt16 | Ty::UInt32 | Ty::UInt64 => {
            ("UNSIGNED", "unsigned_value", format!("(uint64_t)({expr})"))
        }
        Ty::Float32 => ("FLOAT32", "float_value", format!("(double)({expr})")),
        Ty::Float64 => ("FLOAT64", "float_value", expr),
        Ty::Pointer { .. } => (
            "POINTER",
            "unsigned_value",
            format!("(uint64_t)(uintptr_t)({expr})"),
        ),
        ty if *ty == Ty::Str || *ty == Ty::byte_span() => {
            let (data, length) = bytes(ty, &expr)?;
            (
                "BYTES",
                "bytes",
                format!("{{ .data = {data}, .length = {length} }}"),
            )
        }
        _ => {
            let descriptor = super::representation::descriptor(types, ty);
            (
                "REPR",
                "repr",
                format!("{{ .type = &{descriptor}, .data = &({expr}) }}"),
            )
        }
    };
    Ok(format!(
        "{{ .kind = RESIN_PRINT_{kind}, .value = {{ .{member} = {value} }} }}"
    ))
}
