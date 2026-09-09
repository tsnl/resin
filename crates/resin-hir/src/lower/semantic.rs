//! Retained compiler contexts and their resolved typing facts.

use super::scope::DeclarationId;
use crate::lower::context::Context;
use resin_common::prelude::*;

use std::{collections::BTreeMap, path::PathBuf};

pub(crate) use crate::DefinitionKind;
#[derive(Debug, Clone)]
pub(crate) struct Definition {
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
    pub contexts: crate::lower::scope::Contexts,
    pub fields: BTreeMap<SourceLocation, Vec<Member>>,
    pub method_origins: BTreeMap<(TypeId, String), DeclarationId>,
    pub typer: Context,
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
        typer: &Context,
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
            let origin = if matches!(
                method.body,
                crate::lower::namespaces::FunctionBody::Defined(_)
            ) {
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
pub(crate) fn format_type(ty: &Ty, typer: &Context) -> String {
    resin_common::types::print::format_type(ty, typer.definitions())
}
