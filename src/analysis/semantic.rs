//! Optional observations of the compiler's own binding and typing decisions.

use crate::{
    ast::{SourceLocation, Span},
    ir::{Ty, TyperContext},
};
use std::{cell::RefCell, collections::BTreeMap, path::PathBuf, rc::Rc};

#[derive(Debug, Clone, Default)]
pub(crate) struct SemanticData {
    pub types: BTreeMap<SourceLocation, String>,
    pub references: BTreeMap<SourceLocation, SourceLocation>,
}

#[derive(Clone)]
pub(crate) struct Trace {
    pub path: PathBuf,
    pub data: Rc<RefCell<SemanticData>>,
}

impl Trace {
    pub fn location(&self, span: Span) -> SourceLocation {
        SourceLocation {
            path: self.path.clone(),
            span,
        }
    }

    pub fn typed(&self, location: SourceLocation, ty: &Ty, typer: &TyperContext) {
        self.data
            .borrow_mut()
            .types
            .insert(location, format_type(ty, typer));
    }
}

pub(crate) fn format_type(ty: &Ty, typer: &TyperContext) -> String {
    match ty {
        Ty::Type => "type".into(),
        Ty::Unit => "()".into(),
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
        Ty::Defined { definition } => typer
            .definitions()
            .get(definition.index())
            .map(|d| d.name.to_string())
            .unwrap_or_else(|| "?".into()),
        Ty::Pointer { pointee } => format!("Ptr ({})", format_type(pointee, typer)),
        Ty::Span { element } => format!("Span ({})", format_type(element, typer)),
        Ty::Array { element, length } => format!("[{}; {length}]", format_type(element, typer)),
        Ty::Record { fields } => format!(
            "{{ {} }}",
            fields
                .iter()
                .map(|f| format!("{}: {}", f.name, format_type(&f.ty, typer)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Ty::Function { param, result } => format!(
            "({}) -> {}",
            format_type(param, typer),
            format_type(result, typer)
        ),
    }
}
