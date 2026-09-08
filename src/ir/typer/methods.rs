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
    pub body: FunctionBody,
    pub params: Vec<Ty>,
    pub result: Ty,
}

/// Ordinary signatures can have a source body or a compiler-provided definition.
#[derive(Debug, Clone)]
pub(crate) enum FunctionBody {
    Defined(FunctionId),
    /// IR body with one stack input per parameter and one result. Expanded at
    /// the call site, preserving addresses that cannot cross shader calls.
    Generated(Vec<crate::ir::Instr>),
}

pub(crate) type MethodDefinitions = fn(&Ty) -> Vec<(Arc<str>, FunctionDecl)>;

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
                body: FunctionBody::Defined(function),
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
    pub(crate) fn register_method_definitions(&mut self, definitions: MethodDefinitions) {
        self.method_definitions.push(definitions);
    }

    pub(crate) fn method(&self, ty: &Ty, name: &str) -> Option<FunctionDecl> {
        self.methods(ty)
            .into_iter()
            .find_map(|(n, method)| (n.as_ref() == name).then_some(method))
    }

    pub(crate) fn methods(&self, ty: &Ty) -> Vec<(Arc<str>, FunctionDecl)> {
        let mut methods = BTreeMap::new();
        if let Some(namespace) = self
            .receiver_definition(ty)
            .and_then(|id| self.namespaces.get(&id))
        {
            for (name, id) in &namespace.functions {
                methods.insert(name.clone(), self.functions[id].clone());
            }
        }
        for definitions in &self.method_definitions {
            if let Ty::Pointer { pointee } = ty {
                methods.extend(definitions(pointee));
            }
            methods.extend(definitions(ty));
        }
        methods.into_iter().collect()
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
