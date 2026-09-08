use crate::{backend::Error, ir::Ty};

use super::{Slot, types::Types};

pub(super) fn emit(types: &Types<'_>, args: &[Slot], result: &Ty) -> Result<String, Error> {
    let [arg] = args else {
        return Err(Error("print expects one string".into()));
    };
    if result != &Ty::Unit {
        return Err(Error("print returns ()".into()));
    }
    let (data, length) = bytes(types, &arg.ty, &arg.expr)?;
    Ok(format!("(resin_print({data}, {length}), 0)"))
}

pub(super) fn format(types: &Types<'_>, args: &[Slot], result: &Ty) -> Result<String, Error> {
    let invalid = || Error("fmt expects a string and a tuple of arguments".into());
    let [arg] = args else {
        return Err(invalid());
    };
    let Ty::Record { fields } = &arg.ty else {
        return Err(invalid());
    };
    let [format, arguments] = fields.as_slice() else {
        return Err(invalid());
    };
    if format.name.as_ref() != "_0" || arguments.name.as_ref() != "_1" {
        return Err(invalid());
    }
    let (data, length) = bytes(types, &format.ty, &format!("({}).f0", arg.expr))?;
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
        values.push(value(types, &field.ty, format!("({}).f1.f{i}", arg.expr))?);
    }
    let count = values.len();
    let values = if count == 0 {
        "NULL".into()
    } else {
        format!("(ResinPrintArg[]){{ {} }}", values.join(", "))
    };
    string(
        types,
        result,
        format!("resin_format({data}, {length}, {values}, {count})"),
    )
}

pub(super) fn from_str(types: &Types<'_>, args: &[Slot], result: &Ty) -> Result<String, Error> {
    let [arg] = args else {
        return Err(Error("String.from_str expects one byte span".into()));
    };
    if arg.ty != Ty::byte_span() {
        return Err(Error("String.from_str expects Span<ubyte>".into()));
    }
    let (data, length) = bytes(types, &arg.ty, &arg.expr)?;
    string(
        types,
        result,
        format!("resin_string_from_str({data}, {length})"),
    )
}

fn string(types: &Types<'_>, result: &Ty, allocation: String) -> Result<String, Error> {
    let body = types.shape(result);
    let Ty::Record { fields } = body else {
        return Err(Error("expected String result".into()));
    };
    if fields.len() != 1 || fields[0].ty != Ty::formatted_bytes() {
        return Err(Error("invalid String representation".into()));
    }
    Ok(types.wrap(
        result,
        format!("({}){{ .f0 = {allocation} }}", types.name(body)),
    ))
}

fn bytes(types: &Types<'_>, ty: &Ty, expr: &str) -> Result<(String, String), Error> {
    let expr = types.unwrap(ty, expr.into());
    match types.shape(ty) {
        Ty::Span { element } if **element == Ty::UInt8 => {
            Ok((format!("({expr}).f0"), format!("({expr}).f1")))
        }
        Ty::Record { fields } if fields.len() == 1 && fields[0].ty == Ty::formatted_bytes() => {
            let Ty::Arc { pointee } = &fields[0].ty else {
                unreachable!()
            };
            let span = format!("(({} *)resin_arc_data(({expr}).f0))", types.name(pointee));
            Ok((format!("{span}->f0"), format!("{span}->f1")))
        }
        _ => Err(Error("expected Span<ubyte> or String".into())),
    }
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
        ty @ (Ty::Span { .. } | Ty::Record { .. }) => {
            let (data, length) = bytes(types, ty, &expr)?;
            (
                "BYTES",
                "bytes",
                format!("{{ .data = {data}, .length = {length} }}"),
            )
        }
        _ => return Err(Error(format!("cannot format {ty:?}"))),
    };
    Ok(format!(
        "{{ .kind = RESIN_PRINT_{kind}, .value = {{ .{member} = {value} }} }}"
    ))
}
