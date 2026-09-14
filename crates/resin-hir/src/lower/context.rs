//! Source type namespaces and builtin method signatures.
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
    pub(super) namespaces: BTreeMap<TypeId, BTreeMap<Arc<str>, FunctionId>>,
    pub(super) nominal_schemes: BTreeMap<TypeId, crate::TypeDefinition>,
    pub(super) method_owners: BTreeMap<super::scope::DeclarationId, TypeId>,
    pub(super) source_methods: BTreeMap<(TypeId, Arc<str>), SourceMethod>,
    pub(super) functions: BTreeMap<FunctionId, FunctionDecl>,
    pub(super) host_allocation_errors: BTreeMap<TypeId, FunctionId>,
    pub(super) gpu_allocators: BTreeMap<TypeId, FunctionId>,
    pub(super) gpu_pipeline_contexts: BTreeMap<TypeId, FunctionId>,
    pub(super) method_definitions: Vec<MethodDefinitions>,
}
impl Context {
    pub(crate) fn source_method(&self, owner: TypeId, name: &str) -> Option<&SourceMethod> {
        self.source_methods.get(&(owner, name.into()))
    }

    pub(crate) fn source_methods_for(
        &self,
        owner: TypeId,
    ) -> impl Iterator<Item = (&Arc<str>, &SourceMethod)> {
        self.source_methods
            .iter()
            .filter_map(move |((definition, name), method)| {
                (*definition == owner).then_some((name, method))
            })
    }

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
    pub(super) fn into_definitions(mut self) -> Vec<crate::TypeDefinition> {
        self.typer
            .definitions()
            .iter()
            .enumerate()
            .map(|(index, source)| {
                let id = TypeId::from_index(index);
                let methods = self.namespaces.remove(&id).unwrap_or_default();
                if let Some(mut definition) = self.nominal_schemes.remove(&id) {
                    definition.methods = methods;
                    if let TypeDef::Nominal { drop, .. } = source {
                        definition.drop = *drop;
                    }
                    definition
                } else {
                    super::types::definition(source, methods)
                }
            })
            .collect()
    }

    pub(crate) fn nominal_body(
        &self,
        source: &super::infer::Type,
        solver: &super::infer::Solver,
    ) -> Option<super::infer::Type> {
        use super::infer::{Head, Type};
        let (definition, arguments) = match solver.shape_hint(source) {
            Type::Node(Head::Nominal { definition }, arguments) => (definition, arguments),
            Type::Node(Head::Atom(Ty::Defined { definition }), _) => (definition, vec![]),
            _ => return None,
        };
        let scheme = self.nominal_schemes.get(&definition)?;
        Some(Type::Apply {
            body: Box::new(Type::from_hir(&scheme.body)),
            arguments: scheme
                .type_params
                .iter()
                .zip(arguments)
                .map(|(parameter, argument)| (parameter.id, argument))
                .collect(),
        })
    }

    pub(super) fn define_nominal(
        &mut self,
        definition: TypeId,
        parameters: Vec<crate::TypeParameter>,
        body: crate::Type,
    ) -> Result<(), TypeError> {
        // Legacy builtin signatures still query concrete source records. Keep that
        // view only when the completed scheme has a concrete representation.
        if let Some(concrete) =
            super::infer::Solver::default().resolve(&super::infer::Type::from_hir(&body))
        {
            match self.typer.define_type(definition, concrete) {
                Err(TypeError {
                    kind:
                        TypeErrorKind::IncompleteTypeDefinition {
                            definition: dependency,
                        },
                }) if self.nominal_schemes.contains_key(&dependency) => {}
                result => result?,
            }
        }
        let name = self.typer.definition(definition)?.name().unwrap().clone();
        self.nominal_schemes.insert(
            definition,
            crate::TypeDefinition {
                type_params: parameters,
                name,
                body,
                methods: BTreeMap::new(),
                drop: None,
            },
        );
        Ok(())
    }
}

/// Source signatures share inference handles with their declaration while a file
/// is built. The completed signature replaces these handles before another file.
#[derive(Debug, Clone)]
pub(crate) struct SourceMethod {
    pub declaration: super::scope::DeclarationId,
    pub owner_params: Vec<crate::TypeParameter>,
    pub type_params: Vec<crate::TypeParameter>,
    pub params: Vec<super::infer::Type>,
    pub result: super::infer::Type,
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
    HostAllocate {
        error: FunctionId,
    },
    GpuNew {
        allocator: FunctionId,
    },
    GpuAllocate {
        allocator: FunctionId,
    },
    GpuPipelineFactory {
        factory: FunctionId,
        graphics: bool,
    },
    GpuPipelineRecord {
        record: FunctionId,
        graphics: bool,
    },
    GpuPipelineDispatch {
        context: FunctionId,
        allocator: Option<FunctionId>,
        record: FunctionId,
    },
}

/// A builtin signature whose payload types may retain source type parameters.
#[derive(Debug, Clone)]
pub(crate) struct MethodScheme {
    pub body: FunctionBody,
    pub params: Vec<super::infer::Type>,
    pub result: super::infer::Type,
}

pub(crate) type MethodDefinitions = fn(&Ty, &Context) -> Vec<(Arc<str>, FunctionDecl)>;

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
    pub(crate) fn declare_type(&mut self, name: Arc<str>) -> TypeId {
        let ty = self.typer.reserve_type(name);
        self.namespaces.insert(ty, BTreeMap::new());
        ty
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
            .expect("nominal type has a method namespace")
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

    pub(crate) fn host_allocation_error(&self, receiver: &Ty) -> Option<FunctionId> {
        let Ty::Defined { definition } = receiver else {
            return None;
        };
        self.host_allocation_errors.get(definition).copied()
    }

    pub(crate) fn host_error(&self, factory: FunctionId) -> Ty {
        self.functions[&factory].result.clone()
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
        arguments: &[Ty],
        associated: bool,
    ) -> Option<FunctionDecl> {
        if associated
            && name == "alloc"
            && let Some(error) = self.host_allocation_error(ty)
        {
            let [_, initial] = arguments else {
                return None;
            };
            return Some(FunctionDecl {
                body: FunctionBody::HostAllocate { error },
                params: vec![Ty::UInt64, initial.clone()],
                result: gpu_result(
                    Ty::ArcSpan {
                        element: Box::new(initial.clone()),
                    },
                    self.host_error(error),
                ),
            });
        }
        if !associated
            && name == "new"
            && let Some(allocator) = self.gpu_allocator(ty)
        {
            return Some(FunctionDecl {
                body: FunctionBody::GpuNew { allocator },
                params: vec![ty.clone(), arguments.first()?.clone()],
                result: gpu_result(
                    Ty::GpuPointer {
                        pointee: Box::new(arguments.first()?.clone()),
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
            let [handle, owner, ..] = arguments else {
                return None;
            };
            if !matches!(handle, Ty::Pointer { .. }) || !matches!(owner, Ty::ArcPtr { .. }) {
                return None;
            }
            return Some(FunctionDecl {
                body: FunctionBody::Intrinsic(Intrinsic::GpuAllocateNative),
                params: vec![
                    handle.clone(),
                    owner.clone(),
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
        if associated {
            let first = arguments.first()?;
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
        if associated && let Some(error) = self.host_allocation_error(ty) {
            return Some((
                "alloc",
                format!(
                    "(count: ulong, initial: T) -> Result<ArcSpan<T>, {}>",
                    label(&self.host_error(error))
                ),
            ));
        }
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
            for (name, id) in namespace {
                if let Some(function) = self.functions.get(id) {
                    methods.insert(name.clone(), function.clone());
                }
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
        } else if matches!(from, Ty::ArcPtr { pointee } if to == &Ty::Pointer { pointee: pointee.clone() })
        {
            Some(Self::ArcAddress)
        } else if matches!(from, Ty::ArcPtr { pointee } if pointee.as_ref() == to) {
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
                vec![receiver.clone(), Ty::pointer_length(*element.clone())],
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
        Ty::Str => vec![method(
            "at",
            vec![Ty::Str, Ty::UInt64],
            pointer(Ty::UInt8),
            Intrinsic::Index,
        )],
        Ty::ArcPtr { pointee } => vec![
            method(
                "get",
                vec![pointer(receiver.clone())],
                pointer(*pointee.clone()),
                Intrinsic::ArcGet,
            ),
            method(
                "downgrade",
                vec![receiver.clone()],
                Ty::WeakPtr {
                    pointee: pointee.clone(),
                },
                Intrinsic::Downgrade,
            ),
        ],
        Ty::ArcSpan { element } => vec![
            method(
                "try_new",
                vec![Ty::UInt64, *element.clone()],
                Ty::union_of([receiver.clone(), Ty::None]),
                Intrinsic::ArcSpanTryNew,
            ),
            method(
                "get",
                vec![pointer(receiver.clone())],
                Ty::pointer_length(*element.clone()),
                Intrinsic::ArcSpanGet,
            ),
            method(
                "downgrade",
                vec![receiver.clone()],
                Ty::WeakSpan {
                    element: element.clone(),
                },
                Intrinsic::Downgrade,
            ),
        ],
        Ty::WeakSpan { element } => vec![method(
            "upgrade",
            vec![receiver.clone()],
            Ty::union_of([
                Ty::ArcSpan {
                    element: element.clone(),
                },
                Ty::None,
            ]),
            Intrinsic::Upgrade,
        )],
        Ty::WeakPtr { pointee } => vec![method(
            "upgrade",
            vec![receiver.clone()],
            Ty::union_of([
                Ty::ArcPtr {
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
impl Context {
    /// Select compiler-defined operations from the receiver's constructor without
    /// demanding concrete payload layouts during HIR construction.
    pub(crate) fn method_scheme(
        &self,
        receiver: &super::infer::Type,
        name: &str,
        arguments: &[super::infer::Type],
        associated: bool,
    ) -> Option<MethodScheme> {
        use super::infer::{Head, Type};
        let node = |head, value| Type::Node(head, vec![value]);
        let optional = |value| Type::Node(Head::Union, vec![value, Ty::None.into()]);
        let Type::Node(head, children) = receiver else {
            return None;
        };
        let host_error = match head {
            Head::Atom(ty) => self.host_allocation_error(ty),
            Head::Nominal { definition } => self.host_allocation_errors.get(definition).copied(),
            _ => None,
        };
        if associated
            && name == "alloc"
            && let Some(error) = host_error
        {
            let [_, initial] = arguments else {
                return None;
            };
            return Some(MethodScheme {
                body: FunctionBody::HostAllocate { error },
                params: vec![Ty::UInt64.into(), initial.clone()],
                result: Type::result(
                    node(Head::ArcSpan, initial.clone()),
                    self.host_error(error).into(),
                ),
            });
        }
        let element = children.first()?.clone();
        let (op, params, result) = match (head, name) {
            (Head::ArcSpan, "try_new") => (
                Intrinsic::ArcSpanTryNew,
                vec![Ty::UInt64.into(), element],
                optional(receiver.clone()),
            ),
            (Head::ArcSpan, "get") => (
                Intrinsic::ArcSpanGet,
                vec![Type::pointer(receiver.clone())],
                Type::record(vec![
                    ("data".into(), Type::pointer(element)),
                    ("length".into(), Ty::UInt64.into()),
                ]),
            ),
            (Head::ArcPtr, "get") => (
                Intrinsic::ArcGet,
                vec![Type::pointer(receiver.clone())],
                Type::pointer(element),
            ),
            (Head::ArcSpan, "downgrade") => (
                Intrinsic::Downgrade,
                vec![receiver.clone()],
                node(Head::WeakSpan, element),
            ),
            (Head::ArcPtr, "downgrade") => (
                Intrinsic::Downgrade,
                vec![receiver.clone()],
                node(Head::WeakPtr, element),
            ),
            (Head::WeakSpan, "upgrade") => (
                Intrinsic::Upgrade,
                vec![receiver.clone()],
                optional(node(Head::ArcSpan, element)),
            ),
            (Head::WeakPtr, "upgrade") => (
                Intrinsic::Upgrade,
                vec![receiver.clone()],
                optional(node(Head::ArcPtr, element)),
            ),
            _ => return None,
        };
        Some(MethodScheme {
            body: FunctionBody::Intrinsic(op),
            params,
            result,
        })
    }
}

/// Representation operations have signatures independent of library wrapper names.
pub(super) fn primitive_signature(
    operation: &str,
    parameters: &[crate::Type],
) -> Option<(crate::Intrinsic, Vec<crate::Type>, crate::Type)> {
    let [element] = parameters else {
        return None;
    };
    let pointer = crate::Type::Pointer {
        pointee: Box::new(element.clone()),
    };
    match operation {
        "pointer_index" => Some((
            crate::Intrinsic::PointerIndex,
            vec![pointer.clone(), crate::Type::UInt64, crate::Type::UInt64],
            pointer,
        )),
        "pointer_range" => Some((
            crate::Intrinsic::PointerRange,
            vec![
                pointer.clone(),
                crate::Type::UInt64,
                crate::Type::UInt64,
                crate::Type::UInt64,
            ],
            pointer,
        )),
        "pointer_bytes" => Some((
            crate::Intrinsic::PointerBytes,
            vec![pointer, crate::Type::UInt64],
            super::types::ty(&Ty::byte_span()),
        )),
        _ => None,
    }
}
