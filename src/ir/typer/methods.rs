//! Source-level function namespaces. None of this metadata is part of IR.
use super::TyperContext;
use crate::{
    ir::{FunctionId, Ty, TypeId},
    util::define_id,
};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::Arc,
};

define_id! { pub(crate) struct SourceModuleId(usize); }

#[derive(Debug, Clone)]
pub(crate) struct FunctionDecl {
    pub function: FunctionId,
    pub params: Vec<Ty>,
    pub result: Ty,
}

#[derive(Debug, Clone)]
pub(super) struct Namespace {
    module: SourceModuleId,
    functions: BTreeMap<Arc<str>, FunctionId>,
}

#[derive(Clone, Copy)]
pub(crate) enum ReceiverConversion {
    Value,
    Address,
    Load,
}

impl ReceiverConversion {
    pub fn between(from: &Ty, to: &Ty) -> Option<Self> {
        if from == to {
            Some(Self::Value)
        } else if matches!(to, Ty::Pointer { pointee } if pointee.as_ref() == from) {
            Some(Self::Address)
        } else if matches!(from, Ty::Pointer { pointee } if pointee.as_ref() == to) {
            Some(Self::Load)
        } else {
            None
        }
    }
}

impl FunctionDecl {
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
    pub(crate) fn record_type_origin(&mut self, ty: TypeId, module: SourceModuleId) {
        self.namespaces.entry(ty).or_insert_with(|| Namespace {
            module,
            functions: BTreeMap::new(),
        });
    }
    pub(crate) fn type_origin(&self, ty: TypeId) -> Option<SourceModuleId> {
        self.namespaces.get(&ty).map(|scope| scope.module)
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
