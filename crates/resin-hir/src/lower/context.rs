//! Source type namespaces, nominal origins, and builtin method signatures.
use crate::ReceiverConversion;
use resin_common::define_id;
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::{
    collections::{BTreeMap, btree_map::Entry},
    ops::{Deref, DerefMut},
    sync::Arc,
};

#[derive(Debug, Clone, Default)]
pub(crate) struct Context {
    pub(super) typer: TyperContext,
    pub(super) namespaces: BTreeMap<TypeId, Namespace>,
    pub(super) functions: BTreeMap<FunctionId, FunctionDecl>,
    pub(super) method_definitions: Vec<MethodDefinitions>,
}
impl Context {
    pub(crate) fn receiver_definition(&self, ty: &Ty) -> Option<TypeId> {
        let mut ty = ty;
        while let Some(pointee) = ty.deref_target() {
            ty = pointee;
        }
        let Ty::Defined { definition } = ty else {
            return None;
        };
        Some(*definition)
    }

    pub fn new() -> Self {
        Self::default()
    }
    pub fn into_definitions(self) -> Result<TypeTable, TypeError> {
        self.typer.into_definitions()
    }
}
impl Deref for Context {
    type Target = TyperContext;
    fn deref(&self) -> &TyperContext {
        &self.typer
    }
}
impl DerefMut for Context {
    fn deref_mut(&mut self) -> &mut TyperContext {
        &mut self.typer
    }
}

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
    /// Primitive operation elaborated at its call site, preserving addresses
    /// that cannot cross shader calls.
    Intrinsic(Intrinsic),
}

pub(crate) type MethodDefinitions = fn(&Ty, &Context) -> Vec<(Arc<str>, FunctionDecl)>;

#[derive(Debug, Clone)]
pub(super) struct Namespace {
    origin: SourceOrigin,
    functions: BTreeMap<Arc<str>, FunctionId>,
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

impl Context {
    pub(crate) fn declare_type(&mut self, name: Arc<str>, origin: SourceOrigin) -> TypeId {
        let ty = self.typer.reserve_type(name);
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
                methods.extend(definitions(pointee, self));
            }
            methods.extend(definitions(ty, self));
        }
        methods.into_iter().collect()
    }
}

impl ReceiverConversion {
    pub(crate) fn between(from: &Ty, to: &Ty) -> Option<Self> {
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

impl Context {
    pub(super) fn with_builtins() -> Self {
        let mut typer = Context::new();
        let definition = typer
            .create_type(
                "String",
                Ty::Record {
                    fields: vec![RecordField {
                        name: "bytes".into(),
                        ty: Ty::formatted_bytes(),
                    }],
                },
            )
            .expect("builtin String layout");
        typer.set_string_type(Ty::Defined { definition });
        typer.register_method_definitions(builtin_methods);
        typer
    }
}

pub(super) fn shader_properties() -> Ty {
    Ty::Record {
        fields: vec![RecordField {
            name: "spirv".into(),
            ty: Ty::shader(),
        }],
    }
}

fn pointer(ty: Ty) -> Ty {
    Ty::Pointer {
        pointee: Box::new(ty),
    }
}

fn builtin_methods(receiver: &Ty, typer: &Context) -> Vec<(Arc<str>, FunctionDecl)> {
    if typer.string_type() == Some(receiver) {
        return vec![method(
            "from_str",
            vec![Ty::byte_span()],
            receiver.clone(),
            Intrinsic::StringFromStr,
        )];
    }
    match receiver {
        Ty::Pointer { pointee } => vec![method(
            "replace",
            vec![receiver.clone(), *pointee.clone()],
            *pointee.clone(),
            Intrinsic::Replace,
        )],
        Ty::Array { element, .. } => vec![method(
            "at",
            vec![pointer(receiver.clone()), Ty::UInt64],
            pointer(*element.clone()),
            Intrinsic::Index,
        )],
        Ty::Span { element } => vec![method(
            "at",
            vec![receiver.clone(), Ty::UInt64],
            pointer(*element.clone()),
            Intrinsic::Index,
        )],
        Ty::Arc { pointee } => vec![
            method(
                "get",
                vec![pointer(receiver.clone())],
                pointer(*pointee.clone()),
                Intrinsic::ArcGet,
            ),
            method(
                "downgrade",
                vec![receiver.clone()],
                Ty::Weak {
                    pointee: pointee.clone(),
                },
                Intrinsic::Downgrade,
            ),
        ],
        Ty::Weak { pointee } => vec![method(
            "upgrade",
            vec![receiver.clone()],
            Ty::union_of([
                Ty::Arc {
                    pointee: pointee.clone(),
                },
                Ty::None,
            ]),
            Intrinsic::Upgrade,
        )],
        _ => vec![],
    }
}

fn method(
    name: &str,
    params: Vec<Ty>,
    result: Ty,
    intrinsic: Intrinsic,
) -> (Arc<str>, FunctionDecl) {
    (
        name.into(),
        FunctionDecl {
            body: FunctionBody::Intrinsic(intrinsic),
            params,
            result,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_declarations_have_an_origin_before_the_body_is_defined() {
        let mut typer = Context::new();
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
