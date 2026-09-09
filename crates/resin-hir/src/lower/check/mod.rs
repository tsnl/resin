//! First expression pass: resolve declarations and types, producing a typed tree.
//! No IR instructions or deferred code-generation operations are created here.
use crate::lower::context::Context;
mod context;
mod file;
pub(super) use file::file;
mod expressions;
mod groups;
mod resolve;
#[cfg(test)]
mod tests;
use super::semantic::DefinitionKind;
use super::typed;
use super::{
    GenerateError,
    scope::{ContextView, DeclarationId, Scopes},
};
use crate::lower::infer::{
    Inference,
    solver::VariableId,
    types::{Head, Type},
};
use crate::lower::namespaces::SourceModuleId;
use resin_ast::{Ident, Span};
use resin_common::types::Ty;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
type Result<T> = std::result::Result<T, GenerateError>;
type Term = typed::Term<Type>;
type Statement = typed::Statement<Type>;
type MatchArm = typed::MatchArm<Type>;
pub(super) struct Annotation {
    holes: Vec<(Span, VariableId)>,
    pub ty: Type,
    pub span: Span,
}
impl Annotation {
    fn into_tree(self) -> typed::Annotation<Type> {
        typed::Annotation {
            ty: self.ty,
            span: self.span,
        }
    }
}
pub(super) struct Signature {
    pub declaration: Option<DeclarationId>,
    pub parameters: Vec<Option<DeclarationId>>,
    pub params: Vec<(Ident, Annotation)>,
    pub result: Annotation,
}
impl Checker<'_> {
    fn ann(&mut self, ann: &resin_ast::Type, infer: bool) -> Annotation {
        let scoped = !matches!(ann.val, resin_ast::TypeKind::Unit);
        if scoped {
            self.scopes.push_at(ann.span);
        }
        let checkpoint = self.typing.solver.clone();
        let result = super::annotation::Decoder {
            solver: &mut self.typing.solver,
            holes: Vec::new(),
            resolve: &mut |name| {
                if name.val.as_ref() == "String" {
                    return Ok(self
                        .typing
                        .typer
                        .string_type()
                        .cloned()
                        .expect("builtin String")
                        .into());
                }
                self.scopes.resolve_type(name)
            },
        }
        .decode(ann, infer);
        if scoped {
            self.scopes.pop();
        }
        let decoded = result.unwrap_or_else(|error| {
            self.errors.push(error);
            self.typing.solver = checkpoint;
            super::annotation::Decoded {
                ty: Type::Invalid,
                holes: Vec::new(),
            }
        });
        self.holes.extend_from_slice(&decoded.holes);
        Annotation {
            ty: decoded.ty,
            holes: decoded.holes,
            span: ann.span,
        }
    }

    fn signature(
        &mut self,
        params: &[(Ident, resin_ast::Type)],
        result: &resin_ast::Type,
        infer: bool,
    ) -> Signature {
        let params = params
            .iter()
            .map(|(name, ann)| (name.clone(), self.ann(ann, false)))
            .collect();
        let result = self.ann(result, infer);
        Signature {
            params,
            result,
            declaration: None,
            parameters: vec![],
        }
    }
}

pub(super) struct CheckedFile {
    pub context: ContextView,
    pub signatures: BTreeMap<Arc<str>, typed::Signature>,
    pub bodies: BTreeMap<Arc<str>, typed::Term>,
    pub errors: Vec<GenerateError>,
}
struct Checker<'a> {
    typing: Inference<'a>,
    scopes: Scopes,
    errors: Vec<GenerateError>,
    dependencies: BTreeSet<Arc<str>>,
    holes: Vec<(Span, VariableId)>,
    expressions: Vec<(Span, Type)>,
    result: Type,
    source_module: SourceModuleId,
}
impl<'a> Checker<'a> {
    fn new(typer: &'a mut Context, scopes: Scopes, source_module: SourceModuleId) -> Self {
        Self {
            typing: Inference::new(typer),
            scopes,
            errors: vec![],
            dependencies: BTreeSet::new(),
            holes: vec![],
            expressions: vec![],
            result: Ty::Unit.into(),
            source_module,
        }
    }
}
