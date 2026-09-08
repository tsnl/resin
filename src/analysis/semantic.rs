//! Optional observations of the compiler's own binding and typing decisions.

use crate::{
    ast::{SourceLocation, Span},
    ir::{Ty, TypeId, TyperContext},
};
use std::{cell::RefCell, collections::BTreeMap, path::PathBuf, rc::Rc};

#[derive(Debug, Clone, Default)]
pub(crate) struct SemanticData {
    pub types: BTreeMap<SourceLocation, String>,
    pub references: BTreeMap<SourceLocation, SourceLocation>,
    pub fields: BTreeMap<SourceLocation, Vec<Member>>,
    pub method_origins: BTreeMap<(TypeId, String), SourceLocation>,
}

#[derive(Debug, Clone)]
pub(crate) struct Member {
    pub name: String,
    pub ty: String,
    pub kind: super::DefinitionKind,
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
        self.record_fields(location.clone(), ty, typer);
        self.data
            .borrow_mut()
            .types
            .insert(location, format_type(ty, typer));
    }

    pub fn record_fields(&self, location: SourceLocation, ty: &Ty, typer: &TyperContext) {
        self.record_members(location, ty, false, typer);
    }

    pub fn record_members(
        &self,
        location: SourceLocation,
        ty: &Ty,
        associated: bool,
        typer: &TyperContext,
    ) {
        let mut members = Vec::new();
        if let Some(result) = typer.index_method(ty, "at", associated) {
            members.push(Member {
                name: "at".into(),
                ty: format!("(integer) -> {}", format_type(&result, typer)),
                kind: super::DefinitionKind::Function,
            });
        }
        if !associated {
            for name in ["get", "downgrade", "upgrade"] {
                if let Some((_, result)) = crate::ir::typer::shared_method(ty, name) {
                    members.push(Member {
                        name: name.into(),
                        ty: format_type(
                            &Ty::Function {
                                param: Box::new(Ty::Unit),
                                result: Box::new(result),
                            },
                            typer,
                        ),
                        kind: super::DefinitionKind::Function,
                    });
                }
            }
        }
        if !associated
            && let Ok(converted) = typer.as_record(ty)
            && let Ty::Record { fields } = converted.ty.span_record().unwrap_or(converted.ty)
        {
            members.extend(fields.into_iter().map(|field| Member {
                name: field.name.to_string(),
                ty: format_type(&field.ty, typer),
                kind: super::DefinitionKind::Field,
            }));
        }
        if let Some(receiver) = typer.receiver_definition(ty) {
            for (name, method) in typer.methods(receiver) {
                if name.as_ref() == "drop" {
                    continue;
                }
                let Some(params) = method.arguments(ty, associated) else {
                    continue;
                };
                let ty = Ty::Function {
                    param: Box::new(Ty::parameter(params)),
                    result: Box::new(method.result.clone()),
                };
                members.retain(|member| member.name != name.as_ref());
                members.push(Member {
                    name: name.to_string(),
                    ty: format_type(&ty, typer),
                    kind: super::DefinitionKind::Function,
                });
            }
        }
        self.data.borrow_mut().fields.insert(location, members);
    }

    pub fn record_method(
        &self,
        name: &crate::ast::Ident,
        ty: &Ty,
        associated: bool,
        typer: &TyperContext,
    ) {
        self.record_members(self.location(name.span), ty, associated, typer);
        if let Some(receiver) = typer.receiver_definition(ty) {
            let mut data = self.data.borrow_mut();
            if let Some(origin) = data
                .method_origins
                .get(&(receiver, name.val.to_string()))
                .cloned()
            {
                data.references.insert(self.location(name.span), origin);
            }
        }
    }
}

pub(crate) fn format_type(ty: &Ty, typer: &TyperContext) -> String {
    match ty {
        Ty::Union { variants } => {
            if variants.is_empty() {
                return "Never".into();
            }
            variants
                .iter()
                .map(|member| format_type(member, typer))
                .collect::<Vec<_>>()
                .join(" | ")
        }
        Ty::Result { value, error } => format!(
            "Result<{}, {}>",
            format_type(value, typer),
            format_type(error, typer)
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
        Ty::Defined { definition } => typer
            .definitions()
            .get(definition.index())
            .and_then(|d| d.name())
            .map(ToString::to_string)
            .unwrap_or_else(|| "?".into()),
        Ty::Pointer { pointee } => format!("Ptr<{}>", format_type(pointee, typer)),
        Ty::Arc { pointee } => format!("Arc<{}>", format_type(pointee, typer)),
        Ty::Weak { pointee } => format!("Weak<{}>", format_type(pointee, typer)),
        Ty::Span { element } => format!("Span<{}>", format_type(element, typer)),
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
