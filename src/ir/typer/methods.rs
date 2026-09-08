//! Source-level function namespaces. None of this metadata is part of IR.
use super::TyperContext;
use crate::{
    ast::Span,
    ir::{FunctionId, Ty, TypeId},
    util::define_id,
};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::Arc,
};

define_id! { pub(crate) struct SourceModuleId(usize); }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SourceOrigin {
    pub module: SourceModuleId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub(crate) struct FunctionDecl {
    pub function: FunctionId,
    pub params: Vec<Ty>,
    pub result: Ty,
}

#[derive(Debug, Clone)]
pub(super) struct Namespace {
    origin: SourceOrigin,
    functions: BTreeMap<Arc<str>, FunctionId>,
}

#[derive(Clone, Copy)]
pub(crate) enum ReceiverConversion {
    Value,
    Address,
    Load,
    ArcAddress,
    ArcLoad,
}

impl ReceiverConversion {
    pub fn between(from: &Ty, to: &Ty) -> Option<Self> {
        if from == to {
            Some(Self::Value)
        } else if matches!(to, Ty::Pointer { pointee } if pointee.as_ref() == from) {
            Some(Self::Address)
        } else if matches!(from, Ty::Pointer { pointee } if pointee.as_ref() == to) {
            Some(Self::Load)
        } else if matches!(from, Ty::Arc { pointee } if to == &Ty::Pointer { pointee: pointee.clone() })
        {
            Some(Self::ArcAddress)
        } else if matches!(from, Ty::Arc { pointee } if pointee.as_ref() == to) {
            Some(Self::ArcLoad)
        } else {
            None
        }
    }
}

impl FunctionDecl {
    pub fn ty(&self) -> Ty {
        Ty::Function {
            param: Box::new(Ty::parameter(&self.params)),
            result: Box::new(self.result.clone()),
        }
    }
    pub fn arguments(&self, receiver: &Ty, associated: bool) -> Option<&[Ty]> {
        if associated {
            return Some(&self.params);
        }
        let (first, rest) = self.params.split_first()?;
        ReceiverConversion::between(receiver, first)?;
        Some(rest)
    }
}

impl TyperContext {
    pub(crate) fn define_drop(&mut self, ty: TypeId, function: FunctionId) {
        self.definitions.set_drop(ty, function);
    }

    /// Indexing uses the ordinary integer-index and pointer-result
    /// rules. It is a builtin method so a field receiver needs no parentheses.
    pub(crate) fn index_method(&self, receiver: &Ty, name: &str, associated: bool) -> Option<Ty> {
        if associated || name != "at" {
            return None;
        }
        match self.body(receiver).ok()? {
            Ty::Array { element, .. } | Ty::Span { element } => {
                Some(Ty::Pointer { pointee: element })
            }
            _ => None,
        }
    }

    pub(crate) fn declare_type(&mut self, name: Arc<str>, origin: SourceOrigin) -> TypeId {
        let ty = self.definitions.reserve(name);
        self.namespaces.insert(
            ty,
            Namespace {
                origin,
                functions: BTreeMap::new(),
            },
        );
        ty
    }
    pub(crate) fn type_origin(&self, ty: TypeId) -> Option<SourceOrigin> {
        self.namespaces.get(&ty).map(|scope| scope.origin)
    }
    pub(crate) fn register_function(&mut self, function: FunctionId, params: Vec<Ty>, result: Ty) {
        self.functions.insert(
            function,
            FunctionDecl {
                function,
                params,
                result,
            },
        );
    }
    pub(crate) fn declared_function(&self, function: FunctionId) -> &FunctionDecl {
        &self.functions[&function]
    }
    pub(crate) fn define_method(
        &mut self,
        ty: TypeId,
        name: Arc<str>,
        function: FunctionId,
    ) -> bool {
        match self
            .namespaces
            .get_mut(&ty)
            .expect("nominal type has an origin")
            .functions
            .entry(name)
        {
            Entry::Vacant(entry) => {
                entry.insert(function);
                true
            }
            Entry::Occupied(_) => false,
        }
    }
    pub(crate) fn method(&self, ty: &Ty, name: &str) -> Option<&FunctionDecl> {
        let namespace = self.namespaces.get(&self.receiver_definition(ty)?)?;
        self.functions.get(namespace.functions.get(name)?)
    }
    pub(crate) fn methods(&self, ty: TypeId) -> impl Iterator<Item = (&Arc<str>, &FunctionDecl)> {
        self.namespaces
            .get(&ty)
            .into_iter()
            .flat_map(|scope| scope.functions.iter())
            .map(|(name, id)| (name, &self.functions[id]))
    }
}

/// Builtin operations on shared handles; these have no user method declaration.
pub(crate) fn shared_method(ty: &Ty, name: &str) -> Option<(crate::ir::Instr, Ty)> {
    use crate::ir::Instr;
    match (ty, name) {
        (Ty::Arc { pointee }, "get") => Some((
            Instr::ArcData,
            Ty::Pointer {
                pointee: pointee.clone(),
            },
        )),
        (Ty::Arc { pointee }, "downgrade") => Some((
            Instr::Downgrade,
            Ty::Weak {
                pointee: pointee.clone(),
            },
        )),
        (Ty::Weak { pointee }, "upgrade") => Some((
            Instr::Upgrade,
            Ty::union_of([
                Ty::Arc {
                    pointee: pointee.clone(),
                },
                Ty::None,
            ]),
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_declarations_have_an_origin_before_the_body_is_defined() {
        let mut typer = TyperContext::new();
        let origin = SourceOrigin {
            module: SourceModuleId::from_index(3),
            span: Span { start: 17, end: 21 },
        };
        let id = typer.declare_type("Node".into(), origin);
        assert_eq!(typer.type_origin(id), Some(origin));
        assert_eq!(typer.definitions()[id.index()].body(), None);
        typer
            .define_type(id, Ty::Record { fields: vec![] })
            .unwrap();
        assert_eq!(typer.type_origin(id), Some(origin));
    }
}
