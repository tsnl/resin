//! Human-readable names for resolved types, without source namespaces.
use super::{Ty, TypeDef};

pub fn format_type(ty: &Ty, definitions: &[TypeDef]) -> String {
    match ty {
        Ty::Union { variants } => {
            if variants.is_empty() {
                return "Never".into();
            }
            variants
                .iter()
                .map(|member| format_type(member, definitions))
                .collect::<Vec<_>>()
                .join(" | ")
        }
        Ty::Result { value, error } => format!(
            "Result<{}, {}>",
            format_type(value, definitions),
            format_type(error, definitions)
        ),
        Ty::Type => "type".into(),
        Ty::Unit => "()".into(),
        Ty::None => "None".into(),
        Ty::Bool => "bool".into(),
        Ty::Int8 => "sbyte".into(),
        Ty::Int16 => "short".into(),
        Ty::Int32 => "int".into(),
        Ty::Int64 => "long".into(),
        Ty::UInt8 => "ubyte".into(),
        Ty::UInt16 => "ushort".into(),
        Ty::UInt32 => "uint".into(),
        Ty::UInt64 => "ulong".into(),
        Ty::Float32 => "float32".into(),
        Ty::Float64 => "float64".into(),
        Ty::Foreign { name } => name.to_string(),
        Ty::Defined { definition } => definitions
            .get(definition.index())
            .and_then(|d| d.name())
            .map(ToString::to_string)
            .unwrap_or_else(|| "?".into()),
        Ty::Pointer { pointee } => format!("Ptr<{}>", format_type(pointee, definitions)),
        Ty::Arc { pointee } => format!("Arc<{}>", format_type(pointee, definitions)),
        Ty::Weak { pointee } => format!("Weak<{}>", format_type(pointee, definitions)),
        Ty::Span { element } => format!("Span<{}>", format_type(element, definitions)),
        Ty::Array { element, length } => {
            format!("[{}; {length}]", format_type(element, definitions))
        }
        Ty::Record { fields } => format!(
            "{{ {} }}",
            fields
                .iter()
                .map(|f| format!("{}: {}", f.name, format_type(&f.ty, definitions)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Ty::Function { param, result } => format!(
            "({}) -> {}",
            format_type(param, definitions),
            format_type(result, definitions)
        ),
    }
}
