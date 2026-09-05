use crate::{backend::Error, ir::Ty};

use super::{Slot, types::Types};

pub(super) fn emit(types: &Types<'_>, args: &[Slot], result: &Ty) -> Result<String, Error> {
    let invalid =
        || Error("print expects a byte-string format and a tuple of values; returns ()".into());
    let [arg] = args else { return Err(invalid()) };
    let Ty::Record { fields } = &arg.ty else {
        return Err(invalid());
    };
    let [format, values] = fields.as_slice() else {
        return Err(invalid());
    };
    if result != &Ty::Unit || format.name.as_ref() != "_0" || values.name.as_ref() != "_1" {
        return Err(invalid());
    }
    let Ty::Array { element, length } = &format.ty else {
        return Err(invalid());
    };
    if element.as_ref() != &Ty::UInt8 {
        return Err(invalid());
    }
    let fields = match &values.ty {
        Ty::Unit => &[][..],
        Ty::Record { fields } => fields.as_slice(),
        _ => return Err(invalid()),
    };
    let mut values = Vec::new();
    for (i, field) in fields.iter().enumerate() {
        if field.name.as_ref() != format!("_{i}") {
            return Err(invalid());
        }
        values.push(value(types, &field.ty, format!("({}).f1.f{i}", arg.expr))?);
    }
    let count = values.len();
    let values = if count == 0 {
        "NULL".into()
    } else {
        format!("(ResinPrintArg[]){{ {} }}", values.join(", "))
    };
    Ok(format!(
        "(resin_print(({}).f0.items, {length}, {values}, {count}), 0)",
        arg.expr
    ))
}

fn value(types: &Types<'_>, ty: &Ty, expr: String) -> Result<String, Error> {
    let expr = types.unwrap(ty, expr);
    let (kind, member, value) = match types.shape(ty) {
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
        Ty::Array { element, length } if element.as_ref() == &Ty::UInt8 => (
            "BYTES",
            "bytes",
            format!("{{ .data = ({expr}).items, .length = {length} }}"),
        ),
        _ => return Err(Error(format!("cannot print {ty:?}"))),
    };
    Ok(format!(
        "{{ .kind = RESIN_PRINT_{kind}, .value = {{ .{member} = {value} }} }}"
    ))
}
