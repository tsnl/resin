//! Retained compiler contexts and their resolved typing facts.

use super::scope::DeclarationId;
use crate::{
    ast::SourceLocation,
    ir::{Ty, TypeId, TyperContext},
};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefinitionKind {
    Function,
    Variable,
    Parameter,
    Type,
    Keyword,
    Field,
}
#[derive(Debug, Clone)]
pub struct Definition {
    pub name: String,
    pub location: SourceLocation,
    pub kind: DefinitionKind,
    pub label: String,
    pub(crate) ty: Option<Ty>,
    pub(crate) member: bool,
}
#[derive(Debug, Clone, Default)]
pub(crate) struct SemanticData {
    pub imports: BTreeMap<SourceLocation, PathBuf>,
    pub contexts: crate::ir::generate::scope::Contexts,
    pub fields: BTreeMap<SourceLocation, Vec<Member>>,
    pub method_origins: BTreeMap<(TypeId, String), DeclarationId>,
    pub typer: TyperContext,
}
#[derive(Debug, Clone)]
pub(crate) struct Member {
    pub name: String,
    pub ty: String,
    pub kind: DefinitionKind,
    pub origin: Option<DeclarationId>,
}
impl SemanticData {
    pub(crate) fn record_members(
        &mut self,
        location: SourceLocation,
        ty: &Ty,
        associated: bool,
        typer: &TyperContext,
    ) {
        let mut members = Vec::new();
        if !associated
            && let Ok(converted) = typer.as_record(ty)
            && let Ty::Record { fields } = converted.ty.span_record().unwrap_or(converted.ty)
        {
            members.extend(fields.into_iter().map(|field| Member {
                name: field.name.to_string(),
                ty: format_type(&field.ty, typer),
                kind: DefinitionKind::Field,
                origin: None,
            }));
        }
        for (name, method) in typer.methods(ty) {
            let Some(params) = method.arguments(ty, associated) else {
                continue;
            };
            let signature = Ty::Function {
                param: Box::new(Ty::parameter(params)),
                result: Box::new(method.result.clone()),
            };
            let origin = if matches!(method.body, crate::ir::typecheck::FunctionBody::Defined(_)) {
                typer.receiver_definition(ty).and_then(|receiver| {
                    self.method_origins
                        .get(&(receiver, name.to_string()))
                        .copied()
                })
            } else {
                None
            };
            members.retain(|member| member.name != name.as_ref());
            members.push(Member {
                name: name.to_string(),
                ty: format_type(&signature, typer),
                kind: DefinitionKind::Function,
                origin,
            });
        }
        self.fields.insert(location, members);
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
