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
    pub(super) gpu_allocators: BTreeMap<TypeId, FunctionId>,
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
    GpuNew {
        allocator: FunctionId,
    },
    GpuAllocate {
        allocator: FunctionId,
    },
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

    pub(crate) fn gpu_allocator(&self, receiver: &Ty) -> Option<FunctionId> {
        let definition = self.receiver_definition(receiver)?;
        let function = *self.gpu_allocators.get(&definition)?;
        (self.functions[&function].params[0] == *receiver).then_some(function)
    }

    pub(crate) fn gpu_error(&self, allocator: FunctionId) -> Ty {
        let Ty::Result { error, .. } = &self.functions[&allocator].result else {
            unreachable!("validated GPU allocator")
        };
        *error.clone()
    }

    /// Instantiate compiler allocation methods from the registered allocator's receiver.
    pub(crate) fn method_call(
        &self,
        ty: &Ty,
        name: &str,
        argument: &Ty,
        associated: bool,
    ) -> Option<FunctionDecl> {
        if !associated
            && name == "new"
            && let Some(allocator) = self.gpu_allocator(ty)
        {
            return Some(FunctionDecl {
                body: FunctionBody::GpuNew { allocator },
                params: vec![ty.clone(), argument.clone()],
                result: gpu_result(
                    Ty::GpuPointer {
                        pointee: Box::new(argument.clone()),
                    },
                    self.gpu_error(allocator),
                ),
            });
        }
        if associated
            && name == "allocate_native"
            && *ty
                == (Ty::GpuPointer {
                    pointee: Box::new(Ty::UInt8),
                })
        {
            let Ty::Record { fields } = argument else {
                return None;
            };
            let [handle, owner, ..] = fields.as_slice() else {
                return None;
            };
            if !matches!(handle.ty, Ty::Pointer { .. }) || !matches!(owner.ty, Ty::Arc { .. }) {
                return None;
            }
            return Some(FunctionDecl {
                body: FunctionBody::Intrinsic(Intrinsic::GpuAllocateNative),
                params: vec![
                    handle.ty.clone(),
                    owner.ty.clone(),
                    Ty::UInt64,
                    Ty::UInt64,
                    Ty::Int32,
                ],
                result: Ty::Record {
                    fields: vec![
                        RecordField {
                            name: "value".into(),
                            ty: Ty::union_of([ty.clone(), Ty::None]),
                        },
                        RecordField {
                            name: "status".into(),
                            ty: Ty::Int32,
                        },
                    ],
                },
            });
        }
        if associated && let Ty::Record { fields } = argument {
            let first = &fields.first()?.ty;
            let body = match (ty, name) {
                (Ty::GpuPointer { .. }, "new") => FunctionBody::GpuNew {
                    allocator: self.gpu_allocator(first)?,
                },
                (Ty::GpuSpan { .. }, "allocate") => FunctionBody::GpuAllocate {
                    allocator: self.gpu_allocator(first)?,
                },
                _ => return self.method(ty, name),
            };
            let allocator = self.gpu_allocator(first)?;
            let value = match ty {
                Ty::GpuPointer { pointee } => *pointee.clone(),
                _ => Ty::UInt64,
            };
            return Some(FunctionDecl {
                body,
                params: vec![first.clone(), value],
                result: gpu_result(ty.clone(), self.gpu_error(allocator)),
            });
        }
        self.method(ty, name)
    }

    /// Generic signatures for editor queries before a call provides its arguments.
    /// These labels describe inference parameters, not concrete language types.
    pub(crate) fn generic_method_label(
        &self,
        ty: &Ty,
        associated: bool,
    ) -> Option<(&'static str, String)> {
        let label = |ty: &Ty| resin_types::format_type(ty, self.definitions());
        match (ty, associated) {
            (Ty::GpuPointer { pointee }, true) => Some((
                "new",
                format!(
                    "(gpu: _, value: {}) -> Result<{}, _>",
                    label(pointee),
                    label(ty)
                ),
            )),
            (Ty::GpuSpan { .. }, true) => Some((
                "allocate",
                format!("(gpu: _, count: ulong) -> Result<{}, _>", label(ty)),
            )),
            (_, false) => {
                let allocator = self.gpu_allocator(ty)?;
                Some((
                    "new",
                    format!(
                        "(value: T) -> Result<GpuPtr<T>, {}>",
                        label(&self.gpu_error(allocator))
                    ),
                ))
            }
            _ => None,
        }
    }

    pub(crate) fn projection_method_label(&self, shader: &Ty) -> Option<String> {
        let Ty::Function { param, .. } = shader else {
            return None;
        };
        let Ty::Record { fields } = &**param else {
            return None;
        };
        let Ty::Pointer { pointee } = &fields.get(1)?.ty else {
            return None;
        };
        let argument = pointee.gpu_projection(self.definitions())?;
        Some(format!(
            "(gpu: _, arguments: {}) -> Result<GpuArguments, _>",
            resin_types::format_type(&argument, self.definitions())
        ))
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
        } else if matches!(to, Ty::Pointer { pointee } | Ty::GpuPointer { pointee } if pointee.as_ref() == from)
        {
            Some(Self::Address)
        } else if matches!(from, Ty::Pointer { pointee } | Ty::GpuPointer { pointee } if pointee.as_ref() == to)
        {
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
        return string_constructors(receiver);
    }
    match receiver {
        Ty::Pointer { pointee } => vec![method(
            "replace",
            vec![receiver.clone(), *pointee.clone()],
            *pointee.clone(),
            Intrinsic::Replace,
        )],
        Ty::GpuPointer { pointee } => {
            let mut methods = gpu_methods(receiver, pointee);
            methods.push(method(
                "replace",
                vec![receiver.clone(), *pointee.clone()],
                *pointee.clone(),
                Intrinsic::Replace,
            ));
            methods
        }
        Ty::GpuSpan { element } => {
            let mut methods = gpu_methods(receiver, element);
            methods.push(method(
                "copy_to",
                vec![
                    receiver.clone(),
                    Ty::Span {
                        element: element.clone(),
                    },
                ],
                Ty::Unit,
                Intrinsic::GpuCopyTo,
            ));
            if **element == Ty::UInt8 {
                methods.push(method(
                    "copy_image_native",
                    vec![receiver.clone(), pointer(Ty::UInt8), pointer(Ty::UInt8)],
                    Ty::Int32,
                    Intrinsic::GpuCopyImage,
                ));
            }
            methods
        }
        Ty::GpuArguments => vec![
            method(
                "dispatch_native",
                vec![
                    receiver.clone(),
                    pointer(Ty::UInt8),
                    Ty::UInt32,
                    Ty::UInt32,
                    Ty::UInt32,
                ],
                Ty::Int32,
                Intrinsic::GpuArgumentsDispatch,
            ),
            method(
                "draw_native",
                vec![receiver.clone(), pointer(Ty::UInt8), Ty::UInt32],
                Ty::Int32,
                Intrinsic::GpuArgumentsDraw,
            ),
        ],
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
        Ty::Str => vec![method(
            "at",
            vec![Ty::Str, Ty::UInt64],
            pointer(Ty::UInt8),
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

fn string_constructors(receiver: &Ty) -> Vec<(Arc<str>, FunctionDecl)> {
    [("from_str", Ty::Str), ("from_bytes", Ty::byte_span())]
        .into_iter()
        .map(|(name, arg)| {
            method(
                name,
                vec![arg],
                receiver.clone(),
                Intrinsic::StringFromBytes,
            )
        })
        .collect()
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

fn gpu_result(value: Ty, error: Ty) -> Ty {
    Ty::Result {
        value: Box::new(value),
        error: Box::new(error),
    }
}

fn gpu_methods(receiver: &Ty, element: &Ty) -> Vec<(Arc<str>, FunctionDecl)> {
    let pointer = Ty::GpuPointer {
        pointee: Box::new(element.clone()),
    };
    let span = Ty::GpuSpan {
        element: Box::new(element.clone()),
    };
    vec![
        method(
            "at",
            vec![receiver.clone(), Ty::UInt64],
            pointer,
            Intrinsic::GpuIndex,
        ),
        method(
            "slice",
            vec![receiver.clone(), Ty::UInt64, Ty::UInt64],
            span,
            Intrinsic::GpuSlice,
        ),
        method(
            "read_only",
            vec![receiver.clone()],
            receiver.clone(),
            Intrinsic::GpuReadOnly,
        ),
        method(
            "write_only",
            vec![receiver.clone()],
            receiver.clone(),
            Intrinsic::GpuWriteOnly,
        ),
    ]
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
