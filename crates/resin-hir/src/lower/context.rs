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
    pub(super) namespaces: BTreeMap<TypeId, BTreeMap<crate::MethodName, FunctionId>>,
    pub(super) nominal_schemes: BTreeMap<TypeId, crate::TypeDefinition>,
    pub(super) method_owners: BTreeMap<super::scope::DeclarationId, TypeId>,
    pub(super) source_methods: BTreeMap<(TypeId, crate::MethodName), SourceMethod>,
    pub(super) functions: BTreeMap<FunctionId, FunctionDecl>,
    pub(super) gpu_allocators: BTreeMap<TypeId, FunctionId>,
    pub(super) gpu_pipeline_contexts: BTreeMap<TypeId, FunctionId>,
}
impl Context {
    pub(super) fn define_source_method(&mut self, owner: TypeId, name: &str, method: SourceMethod) {
        if let Some(operator) = crate::MethodName::operator_for_method(name) {
            self.source_methods
                .insert((owner, operator), method.clone());
        }
        self.source_methods.insert((owner, name.into()), method);
    }

    pub(crate) fn source_method(
        &self,
        owner: TypeId,
        name: &crate::MethodName,
    ) -> Option<&SourceMethod> {
        self.source_methods.get(&(owner, name.clone()))
    }

    pub(crate) fn source_methods_for(
        &self,
        owner: TypeId,
    ) -> impl Iterator<Item = (&crate::MethodName, &SourceMethod)> {
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

    /// Copyability follows stored fields, not nominal identity or unused type arguments.
    /// `None` means substitution must decide. A known move-only field wins even
    /// when other fields have unresolved types.
    pub(super) fn copyability(&self, ty: &crate::Type) -> Option<bool> {
        let mut copyable = Some(true);
        let solver = super::infer::Solver::default();
        let mut pending = vec![(ty.clone(), 0)];
        let mut visited = std::collections::BTreeSet::new();
        while let Some((ty, depth)) = pending.pop() {
            match &ty {
                crate::Type::Defined { definition, .. } if visited.insert(ty.clone()) => {
                    if depth >= 256
                        || visited.len() > 65_536
                        || self
                            .definition(*definition)
                            .ok()
                            .is_none_or(|d| d.drop_hook().is_some())
                    {
                        return Some(false);
                    }
                    let Some(body) = self
                        .nominal_body(&super::infer::Type::from_hir(&ty), &solver)
                        .and_then(|body| solver.complete(&body))
                    else {
                        return Some(false);
                    };
                    pending.push((body, depth + 1));
                }
                crate::Type::Array { element, .. } | crate::Type::Error { payload: element } => {
                    pending.push((element.as_ref().clone(), depth));
                }
                crate::Type::Record { fields } => {
                    pending.extend(fields.iter().map(|field| (field.ty.clone(), depth)));
                }
                crate::Type::Union { variants } => {
                    pending.extend(variants.iter().cloned().map(|ty| (ty, depth)));
                }
                crate::Type::Parameter { .. }
                | crate::Type::Member { .. }
                | crate::Type::FunctionParameter { .. }
                | crate::Type::FunctionResult { .. }
                | crate::Type::Value { .. } => copyable = None,
                _ => {}
            }
        }
        copyable
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
                gpu_projection: None,
                gpu_pipeline: None,
                type_params: parameters,
                name,
                body,
                methods: BTreeMap::new(),
                text_view: None,
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
    pub function: bool,
    pub module: SourceModuleId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub(crate) struct FunctionDecl {
    pub body: FunctionBody,
    pub source_params: Vec<crate::Type>,
    pub params: Vec<Ty>,
    pub result: Ty,
}

/// Ordinary signatures can have a source body or a compiler-provided definition.
#[derive(Debug, Clone)]
pub(crate) enum FunctionBody {
    Ordinary,
    GpuPipelineFactory {
        factory: FunctionId,
        kind: resin_types::GpuPipelineKind,
    },
    GpuPipelineRecord {
        record: FunctionId,
        kind: resin_types::GpuPipelineKind,
    },
    GpuPipelineDispatch {
        context: FunctionId,
        allocator: Option<FunctionId>,
        record: FunctionId,
    },
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
    pub(crate) fn declare_type(&mut self, name: Arc<str>) -> TypeId {
        let ty = self.typer.reserve_type(name);
        self.namespaces.insert(ty, BTreeMap::new());
        ty
    }
    pub(crate) fn register_function(
        &mut self,
        function: FunctionId,
        source_params: Vec<crate::Type>,
        source_result: &crate::Type,
    ) -> bool {
        let solver = super::infer::Solver::default();
        let concrete = |ty: &crate::Type| solver.resolve(&super::infer::Type::from_hir(ty));
        let params = source_params
            .iter()
            .map(concrete)
            .collect::<Option<Vec<_>>>();
        let (Some(params), Some(result)) = (params, concrete(source_result)) else {
            return false;
        };
        self.functions.entry(function).or_insert(FunctionDecl {
            body: FunctionBody::Ordinary,
            source_params,
            params,
            result,
        });
        true
    }
    pub(crate) fn declared_function(&self, function: FunctionId) -> &FunctionDecl {
        &self.functions[&function]
    }
    pub(crate) fn define_method(
        &mut self,
        ty: TypeId,
        name: crate::MethodName,
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
    pub(crate) fn gpu_allocator(&self, receiver: &Ty) -> Option<FunctionId> {
        let definition = self.receiver_definition(receiver)?;
        let function = *self.gpu_allocators.get(&definition)?;
        (self.functions[&function].params[0] == *receiver).then_some(function)
    }

    pub(crate) fn gpu_error(&self, allocator: FunctionId) -> Ty {
        let (_, error) = self.functions[&allocator]
            .result
            .fallible_parts()
            .expect("validated GPU allocator");
        error.clone()
    }

    pub(crate) fn method(&self, ty: &Ty, name: &str) -> Option<FunctionDecl> {
        self.methods(ty)
            .into_iter()
            .find_map(|(n, method)| (n.as_ref() == name).then_some(method))
    }

    pub(crate) fn methods(&self, ty: &Ty) -> Vec<(Arc<str>, FunctionDecl)> {
        let mut methods = Vec::new();
        if let Some(namespace) = self
            .receiver_definition(ty)
            .and_then(|id| self.namespaces.get(&id))
        {
            for (name, id) in namespace {
                if let crate::MethodName::Named { name } = name
                    && let Some(function) = self.functions.get(id)
                {
                    methods.push((name.clone(), function.clone()));
                }
            }
        }
        methods
    }
}

impl ReceiverConversion {
    pub(crate) fn between(from: &Ty, to: &Ty) -> Option<Self> {
        if from == to {
            Some(Self::Value)
        } else if matches!((from, to), (Ty::Reference { mutable: true, referent: a }, Ty::Reference { mutable: false, referent: b }) if a == b)
        {
            Some(Self::ReadOnly)
        } else if matches!(to, Ty::Reference { referent, .. } if referent.as_ref() == from) {
            Some(Self::Borrow)
        } else if matches!(from, Ty::Reference { referent, .. } if referent.as_ref() == to) {
            Some(Self::Load)
        } else {
            None
        }
    }
}

// Primitive method signatures keep element types symbolic. Materialization belongs to LIR.
#[derive(Clone)]
pub(crate) struct IntrinsicMethod {
    pub op: Intrinsic,
    pub params: Vec<super::infer::Type>,
    pub result: super::infer::Type,
}

pub(crate) fn is_primitive_operation(name: &str) -> bool {
    matches!(
        name,
        "at" | "at_mut"
            | "lea"
            | "replace"
            | "dispatch_native"
            | "trace_rays_native"
            | "draw_native"
    )
}

pub(crate) fn primitive_operation(
    name: &str,
    arguments: &[super::infer::Type],
    solver: &super::infer::Solver,
) -> Option<IntrinsicMethod> {
    use super::infer::{Head, Type};
    let receiver = solver.head(arguments.first()?);
    let receiver = match receiver {
        Type::Node(Head::Reference { .. }, parts) => solver.head(&parts[0]),
        other => other,
    };
    let (_, mut signature) = intrinsic_methods(&receiver, solver)
        .into_iter()
        .find(|(candidate, _)| *candidate == name)?;
    if signature.op == Intrinsic::Index && matches!(receiver, Type::Node(Head::Array(_), _)) {
        signature.params[0] = Type::borrow(receiver, name == "at_mut");
    }
    Some(signature)
}

pub(crate) fn intrinsic_methods(
    receiver: &super::infer::Type,
    solver: &super::infer::Solver,
) -> Vec<(&'static str, IntrinsicMethod)> {
    use super::infer::{Head, Type};
    let receiver = solver.head(receiver);
    let pointer_receiver = matches!(receiver, Type::Node(Head::Pointer, _));
    let mut methods = Vec::new();
    if let Type::Node(Head::Pointer, parts) = &receiver {
        methods.push((
            "replace",
            IntrinsicMethod {
                op: Intrinsic::Replace,
                params: vec![receiver.clone(), parts[0].clone()],
                result: parts[0].clone(),
            },
        ));
    }
    let mut base = match receiver {
        Type::Node(Head::Pointer, parts) => solver.head(&parts[0]),
        receiver => receiver,
    };
    if matches!(base, Type::Node(Head::Atom(Ty::GpuArguments), _)) {
        methods.extend([
            (
                "trace_rays_native",
                IntrinsicMethod {
                    op: Intrinsic::GpuArgumentsTraceRays,
                    params: vec![
                        base.clone(),
                        Type::pointer(Ty::UInt8.into()),
                        Ty::UInt32.into(),
                        Ty::UInt32.into(),
                        Ty::UInt32.into(),
                    ],
                    result: Ty::Int32.into(),
                },
            ),
            (
                "dispatch_native",
                IntrinsicMethod {
                    op: Intrinsic::GpuArgumentsDispatch,
                    params: vec![
                        base.clone(),
                        Type::pointer(Ty::UInt8.into()),
                        Ty::UInt32.into(),
                        Ty::UInt32.into(),
                        Ty::UInt32.into(),
                    ],
                    result: Ty::Int32.into(),
                },
            ),
            (
                "draw_native",
                IntrinsicMethod {
                    op: Intrinsic::GpuArgumentsDraw,
                    params: vec![
                        base.clone(),
                        Type::pointer(Ty::UInt8.into()),
                        Ty::UInt32.into(),
                    ],
                    result: Ty::Int32.into(),
                },
            ),
        ]);
    }
    while let Type::Node(Head::Pointer, parts) = &base {
        base = solver.head(&parts[0]);
    }
    let index = match &base {
        Type::Node(Head::Array(_), parts) => Some((
            if pointer_receiver {
                Type::pointer(base.clone())
            } else {
                Type::reference(base.clone())
            },
            parts[0].clone(),
        )),
        Type::Node(Head::Atom(Ty::Str), _) => Some((base.clone(), Ty::UInt8.into())),
        _ => None,
    };
    if let Some((receiver, element)) = index {
        methods.push((
            "at",
            IntrinsicMethod {
                op: Intrinsic::Index,
                params: vec![receiver.clone(), Ty::UInt64.into()],
                result: Type::reference(element.clone()),
            },
        ));
        if !matches!(base, Type::Node(Head::Atom(Ty::Str), _)) {
            let mutable_receiver = if pointer_receiver {
                receiver.clone()
            } else {
                Type::borrow(base.clone(), true)
            };
            methods.push((
                "at_mut",
                IntrinsicMethod {
                    op: Intrinsic::Index,
                    params: vec![mutable_receiver, Ty::UInt64.into()],
                    result: Type::borrow(element.clone(), true),
                },
            ));
        }
        if pointer_receiver || matches!(base, Type::Node(Head::Atom(Ty::Str), _)) {
            methods.push((
                "lea",
                IntrinsicMethod {
                    op: Intrinsic::Index,
                    params: vec![receiver, Ty::UInt64.into()],
                    result: Type::pointer(element),
                },
            ));
        }
    }
    methods
}

/// Representation operations have signatures independent of library wrapper names.
pub(super) fn primitive_signature(
    operation: &str,
    parameters: &[crate::Type],
) -> Option<(crate::Intrinsic, Vec<crate::Type>, crate::Type)> {
    use crate::{Intrinsic, RecordField, Type};
    let pointer = |pointee: Type| Type::Pointer {
        pointee: Box::new(pointee),
    };
    let reference = |referent: Type| Type::Reference {
        mutable: false,
        referent: Box::new(referent),
    };
    let optional = |ty| Type::Union {
        variants: vec![Type::None, ty],
    };
    let record = |fields: &[(&str, Type)]| Type::Record {
        fields: fields
            .iter()
            .map(|(name, ty)| RecordField {
                name: (*name).into(),
                ty: ty.clone(),
            })
            .collect(),
    };
    Some(match (operation, parameters) {
        ("trace_ray", [payload]) => (
            Intrinsic::TraceRay,
            vec![
                Type::Float32,
                Type::Float32,
                Type::Float32,
                Type::Float32,
                Type::Float32,
                Type::Float32,
                Type::Float32,
                Type::Float32,
                payload.clone(),
            ],
            payload.clone(),
        ),
        ("ray_hit_info", []) => (
            Intrinsic::RayHitInfo,
            vec![],
            record(&[
                ("_0", Type::Float32),
                ("_1", Type::UInt32),
                ("_2", Type::UInt32),
                ("_3", Type::Float32),
                ("_4", Type::Float32),
            ]),
        ),
        ("pointer_index", [element]) => (
            Intrinsic::PointerIndex,
            vec![pointer(element.clone()), Type::UInt64, Type::UInt64],
            pointer(element.clone()),
        ),
        ("pointer_range", [element]) => (
            Intrinsic::PointerRange,
            vec![
                pointer(element.clone()),
                Type::UInt64,
                Type::UInt64,
                Type::UInt64,
            ],
            pointer(element.clone()),
        ),
        ("pointer_bytes", [element]) => (
            Intrinsic::PointerBytes,
            vec![pointer(element.clone()), Type::UInt64],
            super::types::ty(&Ty::byte_span()),
        ),
        ("owner_create", [element]) => (
            Intrinsic::OwnerCreate,
            vec![element.clone()],
            optional(Type::StrongOwner),
        ),
        ("owner_allocate", [element]) => (
            Intrinsic::OwnerAllocate,
            vec![Type::UInt64, element.clone()],
            optional(Type::StrongOwner),
        ),
        ("owner_data", [element]) => (
            Intrinsic::OwnerData,
            vec![reference(Type::StrongOwner)],
            pointer(element.clone()),
        ),
        ("owner_length", []) => (
            Intrinsic::OwnerLength,
            vec![reference(Type::StrongOwner)],
            Type::UInt64,
        ),
        ("owner_downgrade", []) => (
            Intrinsic::OwnerDowngrade,
            vec![reference(Type::StrongOwner)],
            Type::WeakOwner,
        ),
        ("owner_upgrade", []) => (
            Intrinsic::OwnerUpgrade,
            vec![reference(Type::WeakOwner)],
            optional(Type::StrongOwner),
        ),
        ("string_from_bytes", []) => (
            Intrinsic::StringFromBytes,
            vec![pointer(Type::UInt8), Type::UInt64],
            Type::StrongOwner,
        ),
        ("sqrt" | "sin" | "cos", [element]) => (
            match operation {
                "sqrt" => Intrinsic::Sqrt,
                "sin" => Intrinsic::Sin,
                _ => Intrinsic::Cos,
            },
            vec![element.clone()],
            element.clone(),
        ),
        ("repr", [value]) => (Intrinsic::Repr, vec![value.clone()], Type::StrongOwner),
        ("format_bytes", [arguments]) => (
            Intrinsic::FormatBytes,
            vec![pointer(Type::UInt8), Type::UInt64, arguments.clone()],
            Type::StrongOwner,
        ),
        ("weak_empty", []) => (Intrinsic::WeakEmpty, vec![], Type::WeakOwner),
        ("gpu_view_allocate", [native]) => (
            Intrinsic::GpuViewAllocate,
            vec![
                pointer(native.clone()),
                Type::StrongOwner,
                Type::UInt64,
                Type::UInt64,
                Type::Int32,
            ],
            record(&[
                (
                    "_0",
                    Type::Union {
                        variants: vec![Type::None, Type::GpuView],
                    },
                ),
                ("_1", Type::Int32),
            ]),
        ),
        ("gpu_view_offset", []) => (
            Intrinsic::GpuViewOffset,
            vec![Type::GpuView, Type::UInt64, Type::UInt64, Type::UInt64],
            Type::GpuView,
        ),
        ("gpu_view_range", [_]) => (
            Intrinsic::GpuViewRange,
            vec![Type::GpuView, Type::UInt64, Type::UInt64, Type::UInt64],
            Type::GpuView,
        ),
        ("gpu_view_restrict", []) => (
            Intrinsic::GpuViewRestrict,
            vec![Type::GpuView, Type::UInt32],
            Type::GpuView,
        ),
        ("gpu_view_load", [element]) => {
            (Intrinsic::GpuViewLoad, vec![Type::GpuView], element.clone())
        }
        ("gpu_view_store", [element]) => (
            Intrinsic::GpuViewStore,
            vec![Type::GpuView, element.clone()],
            Type::Unit,
        ),
        ("gpu_view_replace", [element]) => (
            Intrinsic::GpuViewReplace,
            vec![Type::GpuView, element.clone()],
            element.clone(),
        ),
        ("gpu_view_copy_to", [element]) => (
            Intrinsic::GpuViewCopyTo,
            vec![
                Type::GpuView,
                Type::UInt64,
                pointer(element.clone()),
                Type::UInt64,
            ],
            Type::Unit,
        ),
        ("gpu_view_copy_from", [element]) => (
            Intrinsic::GpuViewCopyFrom,
            vec![
                Type::GpuView,
                Type::UInt64,
                pointer(element.clone()),
                Type::UInt64,
            ],
            Type::Unit,
        ),
        ("gpu_view_copy_image", []) => (
            Intrinsic::GpuViewCopyImage,
            vec![
                Type::GpuView,
                Type::UInt64,
                pointer(Type::UInt8),
                pointer(Type::UInt8),
            ],
            Type::Int32,
        ),
        _ => return None,
    })
}
