//! Reserve each concrete function before translating its body, so recursion reuses IDs.
use super::{LowerError, LoweredFunction, functions, specialize};
use crate::{ApplicationNote, Error, ErrorKind, LoweringOptions, Profile};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Instance {
    definition: FunctionId,
    arguments: Vec<resin_hir::Type>,
    profile: Profile,
}

struct Request {
    instance: Instance,
    predecessor: Option<FunctionId>,
    location: Option<SourceLocation>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Nominal {
    definition: TypeId,
    arguments: Vec<resin_hir::Type>,
}

struct NominalRequest {
    instance: Nominal,
    depth: usize,
}

pub(super) struct Instances<'a> {
    source: &'a resin_hir::Module,
    options: &'a LoweringOptions,
    identities: BTreeMap<Instance, FunctionId>,
    requests: Vec<Request>,
    counts: Vec<usize>,
    type_identities: BTreeMap<Nominal, TypeId>,
    definitions: Vec<TypeDef>,
    type_requests: Vec<NominalRequest>,
    type_cursor: usize,
    active_type_depth: Option<usize>,
    typer: TyperContext,
    type_error: Option<ErrorKind>,
    entries: BTreeMap<std::sync::Arc<str>, FunctionId>,
    shaders: BTreeMap<FunctionId, ShaderEntry>,
    text_views: BTreeMap<TypeId, FunctionId>,
}

impl<'a> Instances<'a> {
    pub(super) fn new(source: &'a resin_hir::Module, options: &'a LoweringOptions) -> Self {
        Self {
            source,
            options,
            identities: BTreeMap::new(),
            requests: vec![],
            counts: vec![0; source.functions.len()],
            type_identities: BTreeMap::new(),
            definitions: vec![],
            type_requests: vec![],
            type_cursor: 0,
            active_type_depth: None,
            typer: TyperContext::new(),
            type_error: None,
            entries: BTreeMap::new(),
            shaders: BTreeMap::new(),
            text_views: BTreeMap::new(),
        }
    }

    // Whole-module clients explicitly request all ordinary definitions.
    // Production compilation instead supplies exported target roots.
    pub(super) fn reserve_roots(
        &mut self,
        cancellation: &resin_executor::Cancellation,
    ) -> Result<(), crate::BuildError> {
        for id in self
            .source
            .entries
            .values()
            .chain(self.source.shaders.keys())
        {
            cancellation.check()?;
            if self.source.functions.get(id.index()).is_none() {
                return Err(self
                    .error(
                        ErrorKind::InvalidHir {
                            message: "entry refers to a missing function".into(),
                        },
                        None,
                        None,
                    )
                    .into());
            }
        }
        for (index, function) in self.source.functions.iter().enumerate() {
            cancellation.check()?;
            if !function.signature.type_params.is_empty()
                && (function.foreign_header.is_some()
                    || self
                        .source
                        .shaders
                        .contains_key(&FunctionId::from_index(index)))
            {
                return Err(self
                    .error(
                        ErrorKind::InvalidHir {
                            message: "foreign and shader declarations require fixed signatures"
                                .into(),
                        },
                        None,
                        function.location.clone(),
                    )
                    .into());
            }
            let ray_stage = self
                .source
                .shaders
                .get(&FunctionId::from_index(index))
                .is_some_and(|shader| {
                    matches!(
                        shader.stage.as_ref(),
                        "ray_generation" | "miss" | "closest_hit"
                    )
                });
            let ray_intrinsic = function.body.as_ref().is_some_and(|body| {
                matches!(
                    body.kind,
                    resin_hir::TermKind::Intrinsic {
                        op: resin_types::Intrinsic::TraceRay | resin_types::Intrinsic::RayHitInfo,
                        ..
                    }
                )
            });
            if function.signature.type_params.is_empty() && !ray_stage && !ray_intrinsic {
                self.request(
                    FunctionId::from_index(index),
                    vec![],
                    Profile::Host,
                    None,
                    function.location.clone(),
                )?;
            }
        }
        for (name, &definition) in &self.source.entries {
            cancellation.check()?;
            if let Some(id) = self.ordinary(definition, Profile::Host) {
                self.entries.insert(name.clone(), id);
            }
        }
        for (&definition, shader) in &self.source.shaders {
            cancellation.check()?;
            self.shader(definition, shader.embedded, None, None)?;
        }
        Ok(())
    }

    pub(super) fn reserve_entries(
        &mut self,
        entries: &[crate::Entry],
        cancellation: &resin_executor::Cancellation,
    ) -> Result<(), crate::BuildError> {
        // Sorting source requests makes nominal discovery independent of caller order.
        let mut entries: Vec<_> = entries.iter().collect();
        entries.sort();
        entries.dedup();
        for entry in entries {
            cancellation.check()?;
            let location = self
                .source
                .functions
                .get(entry.function.index())
                .and_then(|function| function.location.clone());
            let arguments = entry
                .arguments
                .iter()
                .map(|ty| {
                    super::substitute::Substitution::default()
                        .normalize(ty, self)
                        .map_err(|error| self.lower_error(error, None, None))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let id = if entry.profile != Profile::Host {
                if !arguments.is_empty() {
                    return Err(self
                        .error(
                            ErrorKind::InvalidHir {
                                message: "shader entries require fixed signatures".into(),
                            },
                            None,
                            None,
                        )
                        .into());
                }
                self.shader(entry.function, false, None, location.clone())?
            } else {
                self.request(entry.function, arguments, entry.profile, None, location)?
            };
            if entry.profile == Profile::Host
                && let Some(previous) = self.entries.insert(entry.name.clone(), id)
                && previous != id
            {
                return Err(self
                    .error(
                        ErrorKind::InvalidHir {
                            message: format!("conflicting applications for entry {}", entry.name)
                                .into(),
                        },
                        None,
                        None,
                    )
                    .into());
            }
        }
        Ok(())
    }

    pub(super) fn shader(
        &mut self,
        definition: FunctionId,
        embedded: bool,
        predecessor: Option<FunctionId>,
        location: Option<SourceLocation>,
    ) -> Result<FunctionId, Error> {
        let Some(shader) = self.source.shaders.get(&definition) else {
            return Err(self.error(
                ErrorKind::InvalidHir {
                    message: "shader request requires a decorated declaration".into(),
                },
                predecessor,
                location,
            ));
        };
        let mut shader = shader.clone();
        shader.embedded = embedded;
        let profile = if shader.stage.as_ref() == "compute" {
            Profile::Compute
        } else {
            Profile::Shader
        };
        let id = self.request(definition, vec![], profile, predecessor, location)?;
        self.shaders
            .entry(id)
            .and_modify(|old| old.embedded |= embedded)
            .or_insert(shader);
        Ok(id)
    }

    pub(super) fn profile(&self, id: FunctionId) -> Profile {
        self.requests[id.index()].instance.profile
    }

    pub(super) fn request(
        &mut self,
        definition: FunctionId,
        arguments: Vec<resin_hir::Type>,
        profile: Profile,
        predecessor: Option<FunctionId>,
        location: Option<SourceLocation>,
    ) -> Result<FunctionId, Error> {
        let Some(function) = self.source.functions.get(definition.index()) else {
            return Err(self.error(
                ErrorKind::InvalidHir {
                    message: "function application refers to a missing definition".into(),
                },
                predecessor,
                location,
            ));
        };
        if !function.signature.type_params.is_empty()
            && (function.foreign_header.is_some() || self.source.shaders.contains_key(&definition))
        {
            return Err(self.error(
                ErrorKind::InvalidHir {
                    message: "foreign and shader declarations require fixed signatures".into(),
                },
                predecessor,
                location,
            ));
        }
        if arguments.len() != function.signature.type_params.len() {
            return Err(self.error(
                ErrorKind::InvalidHir {
                    message: format!(
                        "function {} expects {} type arguments, found {}",
                        function.name,
                        function.signature.type_params.len(),
                        arguments.len()
                    )
                    .into(),
                },
                predecessor,
                location,
            ));
        }
        let instance = Instance {
            definition,
            arguments,
            profile,
        };
        if let Some(&id) = self.identities.get(&instance) {
            return Ok(id);
        }
        let limit = self.options.max_monomorphs_per_function.get();
        if self.counts[definition.index()] == limit {
            return Err(self.error(
                ErrorKind::MonomorphLimit {
                    function: function.name.clone(),
                    limit,
                    arguments: self.argument_names(&instance.arguments),
                    profile,
                },
                predecessor,
                location,
            ));
        }
        let id = FunctionId::from_index(self.requests.len());
        self.counts[definition.index()] += 1;
        self.identities.insert(instance.clone(), id);
        self.requests.push(Request {
            instance,
            predecessor,
            location,
        });
        Ok(id)
    }

    pub(super) fn ordinary(&self, definition: FunctionId, profile: Profile) -> Option<FunctionId> {
        self.identities
            .get(&Instance {
                definition,
                arguments: vec![],
                profile,
            })
            .copied()
    }

    pub(super) fn reserve_types(
        &mut self,
        cancellation: &resin_executor::Cancellation,
    ) -> Result<(), crate::BuildError> {
        // Whole-module clients retain every declaration in source order. Requested
        // programs instead discover nominal declarations through their roots.
        for index in 0..self.source.types.len() {
            cancellation.check()?;
            if self.source.types[index].type_params.is_empty() {
                self.reserve_type(TypeId::from_index(index), vec![])
                    .map_err(|error| self.lower_error(error, None, None))?;
            }
        }
        self.complete_types()
            .map_err(|error| self.lower_error(error, None, None))?;
        Ok(())
    }

    fn argument_names(&self, arguments: &[resin_hir::Type]) -> Vec<Arc<str>> {
        arguments
            .iter()
            .map(|argument| resin_hir::format_type(argument, &self.source.types).into())
            .collect()
    }

    fn nominal_name(&self, instance: &Nominal) -> Arc<str> {
        resin_hir::format_type(
            &resin_hir::Type::Defined {
                definition: instance.definition,
                arguments: instance.arguments.clone(),
            },
            &self.source.types,
        )
        .into()
    }

    pub(super) fn nominal_arity(&self, definition: TypeId, count: usize) -> Result<(), LowerError> {
        let source = self.source.types.get(definition.index()).ok_or_else(|| {
            LowerError::typing(
                Span { start: 0, end: 0 },
                TypeError {
                    kind: TypeErrorKind::InvalidTypeDefinition { definition },
                },
            )
        })?;
        if source.type_params.len() != count {
            return Err(LowerError::invalid_hir(
                Span { start: 0, end: 0 },
                format!(
                    "type {} expects {} type arguments, found {count}",
                    source.name,
                    source.type_params.len()
                ),
            ));
        }
        Ok(())
    }

    pub(super) fn nominal_origin(&self, definition: TypeId) -> resin_hir::Type {
        let instance = &self.type_requests[definition.index()].instance;
        resin_hir::Type::Defined {
            definition: instance.definition,
            arguments: instance.arguments.clone(),
        }
    }

    pub(super) fn signature(
        &self,
        function: FunctionId,
    ) -> Result<resin_hir::Signature, LowerError> {
        self.source
            .functions
            .get(function.index())
            .map(|function| function.signature.clone())
            .ok_or_else(|| {
                LowerError::invalid_hir(
                    Span { start: 0, end: 0 },
                    "operation refers to a missing declaration",
                )
            })
    }

    pub(super) fn method(
        &self,
        receiver: &resin_hir::Type,
        name: &resin_hir::MethodName,
    ) -> Result<(FunctionId, resin_hir::Signature, Vec<resin_hir::Type>), LowerError> {
        let mut owner = receiver;
        while let resin_hir::MethodName::Named { .. } = name
            && let resin_hir::Type::Pointer { pointee } = owner
        {
            owner = pointee;
        }
        let missing = || LowerError {
            span: Span { start: 0, end: 0 },
            kind: ErrorKind::InvalidInstance {
                message: format!(
                    "type {} has no method {name}",
                    resin_hir::format_type(owner, &self.source.types)
                )
                .into(),
            },
        };
        let resin_hir::Type::Defined {
            definition,
            arguments,
        } = owner
        else {
            return Err(missing());
        };
        self.nominal_arity(*definition, arguments.len())?;
        let function = self.source.types[definition.index()]
            .methods
            .get(name)
            .copied()
            .ok_or_else(missing)?;
        let signature = self
            .source
            .functions
            .get(function.index())
            .ok_or_else(|| {
                LowerError::invalid_hir(
                    Span { start: 0, end: 0 },
                    "method refers to a missing function definition",
                )
            })?
            .signature
            .clone();
        let owner_params = &self.source.types[definition.index()].type_params;
        if signature.type_params.len() < owner_params.len()
            || !owner_params
                .iter()
                .zip(&signature.type_params)
                .all(|(owner, method)| owner.id == method.id)
        {
            return Err(LowerError::invalid_hir(
                Span { start: 0, end: 0 },
                "method signature does not bind its owner's type parameters",
            ));
        }
        Ok((function, signature, arguments.clone()))
    }

    pub(super) fn nominal(
        &mut self,
        definition: TypeId,
        arguments: Vec<resin_hir::Type>,
    ) -> Result<TypeId, LowerError> {
        if let Some(kind) = &self.type_error {
            return Err(LowerError {
                span: Span { start: 0, end: 0 },
                kind: kind.clone(),
            });
        }
        let id = self.reserve_type(definition, arguments)?;
        if self.active_type_depth.is_none() && self.type_cursor < self.type_requests.len() {
            self.complete_types().inspect_err(|error| {
                self.type_error = Some(error.kind.clone());
            })?;
        }
        Ok(id)
    }

    fn reserve_type(
        &mut self,
        definition: TypeId,
        arguments: Vec<resin_hir::Type>,
    ) -> Result<TypeId, LowerError> {
        self.nominal_arity(definition, arguments.len())?;
        let instance = Nominal {
            definition,
            arguments,
        };
        if let Some(&id) = self.type_identities.get(&instance) {
            return Ok(id);
        }
        let depth = self.active_type_depth.map_or(0, |depth| depth + 1);
        if depth == 256 {
            return Err(LowerError {
                span: Span { start: 0, end: 0 },
                kind: ErrorKind::TypeExpansionLimit { limit: 256 },
            });
        }
        let id = TypeId::from_index(self.definitions.len());
        self.type_identities.insert(instance.clone(), id);
        self.type_requests.push(NominalRequest { instance, depth });
        self.definitions.push(TypeDef::Nominal {
            gpu_projection: None,
            gpu_pipeline: None,
            name: self.nominal_name(&self.type_requests[id.index()].instance),
            body: None,
            drop: None,
        });
        Ok(id)
    }

    fn complete_types(&mut self) -> Result<(), LowerError> {
        // Expanding a body only reserves referenced identities. A queue closes both
        // forward and recursive edges without recursion through declaration chains.
        while self.type_cursor < self.type_requests.len() {
            self.active_type_depth = Some(self.type_requests[self.type_cursor].depth);
            let result = self.expand_type(self.type_cursor);
            self.active_type_depth = None;
            result?;
            self.type_cursor += 1;
        }
        for (index, definition) in self.definitions.iter().enumerate() {
            let body = definition.body().expect("completed nominal expansion");
            resin_types::check_references(&self.definitions, body)
                .and_then(|()| {
                    resin_types::check_layout(&self.definitions, TypeId::from_index(index), body)
                })
                .map_err(|error| LowerError::typing(Span { start: 0, end: 0 }, error))?;
        }
        self.typer = TyperContext::from_definitions(TypeTable::from(self.definitions.clone()));
        Ok(())
    }

    fn expand_type(&mut self, index: usize) -> Result<(), LowerError> {
        let instance = self.type_requests[index].instance.clone();
        let source = &self.source.types[instance.definition.index()];
        let body = super::substitute::Substitution::new(&source.type_params, &instance.arguments)?
            .ty(&source.body, self)?;
        let gpu_pipeline = source
            .gpu_pipeline
            .as_ref()
            .map(|pipeline| {
                let substitution =
                    super::substitute::Substitution::new(&source.type_params, &instance.arguments)?;
                Ok(resin_types::GpuPipeline {
                    kind: pipeline.kind,
                    root: substitution.ty(&pipeline.root, self)?,
                    owner: substitution.ty(&pipeline.owner, self)?,
                })
            })
            .transpose()?;
        let gpu_projection = source
            .gpu_projection
            .as_ref()
            .map(|projection| {
                Ok(resin_types::GpuProjection {
                    kind: projection.kind,
                    target: super::substitute::Substitution::new(
                        &source.type_params,
                        &instance.arguments,
                    )?
                    .ty(&projection.target, self)?,
                })
            })
            .transpose()?;
        let drop = source
            .drop
            .map(|hook| self.request(hook, instance.arguments.clone(), Profile::Host, None, None))
            .transpose()
            .map_err(|error| LowerError {
                span: error.span,
                kind: error.kind,
            })?;
        self.text_view(TypeId::from_index(index), &instance)?;
        self.definitions[index] = TypeDef::Nominal {
            gpu_projection,
            gpu_pipeline,
            name: self.nominal_name(&instance),
            body: Some(body),
            drop,
        };
        Ok(())
    }

    fn text_view(&mut self, id: TypeId, instance: &Nominal) -> Result<(), LowerError> {
        let source = &self.source.types[instance.definition.index()];
        let Some(function) = source.text_view else {
            return Ok(());
        };
        let signature = &self.source.functions[function.index()].signature;
        let invalid = || {
            LowerError::invalid_hir(
                signature.result.span,
                "repr_bytes must take Ref<Self> and return a borrowed byte view, with no additional type parameters",
            )
        };
        if signature.type_params.len() != instance.arguments.len() || signature.params.len() != 1 {
            return Err(invalid());
        }
        let substitution =
            super::substitute::Substitution::new(&signature.type_params, &instance.arguments)?;
        let receiver = substitution.normalize(&signature.params[0].annotation.ty, self)?;
        let expected = resin_hir::Type::Reference {
            mutable: false,
            referent: Box::new(self.nominal_origin(id)),
        };
        let result = substitution.ty(&signature.result.ty, self)?;
        if receiver != expected || result != Ty::byte_span() {
            return Err(invalid());
        }
        let hook = self
            .request(
                function,
                instance.arguments.clone(),
                Profile::Host,
                None,
                None,
            )
            .map_err(|error| LowerError {
                span: error.span,
                kind: error.kind,
            })?;
        self.text_views.insert(id, hook);
        Ok(())
    }

    pub(super) fn typer(&self) -> &TyperContext {
        &self.typer
    }
    pub(super) fn assemble(
        mut self,
        functions: Vec<LoweredFunction>,
        cancellation: &resin_executor::Cancellation,
    ) -> Result<crate::Module, crate::BuildError> {
        let mut module = crate::Module {
            entries: std::mem::take(&mut self.entries),
            foreign_headers: self
                .source
                .foreign_headers
                .iter()
                .map(|header| crate::ForeignHeader {
                    source: header.source.clone(),
                    spelling: header.spelling.clone(),
                })
                .collect(),
            shaders: std::mem::take(&mut self.shaders),
            text_views: std::mem::take(&mut self.text_views),
            types: TypeTable::from(std::mem::take(&mut self.definitions)),
            ..Default::default()
        };
        for (index, lowered) in functions.into_iter().enumerate() {
            cancellation.check()?;
            let id = FunctionId::from_index(index);
            module.functions.push(lowered.function);
            if let Some(location) = lowered.location {
                module.origins.functions.insert(id, location);
            }
            module.origins.instructions.extend(
                lowered
                    .origins
                    .into_iter()
                    .map(|((block, instruction), location)| ((id, block, instruction), location)),
            );
        }
        crate::profile::shaders(&module).map_err(|error| {
            let location = error
                .instruction
                .and_then(|(block, instruction)| {
                    module
                        .origins
                        .instructions
                        .get(&(error.function, block, instruction))
                })
                .or_else(|| module.origins.functions.get(&error.function))
                .cloned();
            vec![self.error(
                ErrorKind::UnsupportedProfile {
                    profile: Profile::Shader,
                    message: error.message,
                },
                Some(error.function),
                location,
            )]
        })?;
        Ok(module)
    }

    pub(super) fn lower(
        &mut self,
        cancellation: &resin_executor::Cancellation,
    ) -> Result<Vec<LoweredFunction>, crate::BuildError> {
        let mut completed = vec![];
        let mut errors = vec![];
        let mut index = 0;
        while index < self.requests.len() {
            cancellation.check()?;
            let id = FunctionId::from_index(index);
            match self.lower_function(id) {
                Ok(function) => completed.push(function),
                Err(error) => errors.push(error),
            }
            index += 1;
        }
        if errors.is_empty() {
            Ok(completed)
        } else {
            Err(crate::BuildError::Diagnostics { errors })
        }
    }

    fn lower_function(&mut self, id: FunctionId) -> Result<LoweredFunction, Error> {
        let instance = self.requests[id.index()].instance.clone();
        let function = &self.source.functions[instance.definition.index()];
        let body = specialize::function(function, &instance.arguments, self, id)?;
        let lowered =
            functions::lower(&body, &self.typer, instance.profile).map_err(|mut error| {
                error.applications = self.trace(Some(id));
                error
            })?;
        crate::profile::function(&self.typer, id, &lowered.function).map_err(|error| {
            let location = error
                .instruction
                .and_then(|instruction| lowered.origins.get(&instruction))
                .or(lowered.location.as_ref())
                .cloned();
            self.error(
                ErrorKind::UnsupportedProfile {
                    profile: Profile::Shader,
                    message: error.message,
                },
                Some(id),
                location,
            )
        })?;
        Ok(lowered)
    }

    pub(super) fn lower_error(
        &self,
        error: LowerError,
        current: Option<FunctionId>,
        location: Option<SourceLocation>,
    ) -> Error {
        let location = location.map(|location| SourceLocation {
            span: error.span,
            ..location
        });
        let mut result = self.error(error.kind, current, location);
        result.span = error.span;
        result
    }

    fn error(
        &self,
        kind: ErrorKind,
        predecessor: Option<FunctionId>,
        location: Option<SourceLocation>,
    ) -> Error {
        Error {
            source: location.as_ref().map(|location| location.source.clone()),
            span: location.map_or(Span { start: 0, end: 0 }, |location| location.span),
            kind,
            applications: self.trace(predecessor),
        }
    }

    fn trace(&self, mut predecessor: Option<FunctionId>) -> Vec<ApplicationNote> {
        let mut trace = vec![];
        while let Some(id) = predecessor {
            if trace.len() == 32 {
                break;
            }
            let request = &self.requests[id.index()];
            trace.push(ApplicationNote {
                function: self.source.functions[request.instance.definition.index()]
                    .name
                    .clone(),
                arguments: self.argument_names(&request.instance.arguments),
                profile: request.instance.profile,
                location: request.location.clone(),
            });
            predecessor = request.predecessor;
        }
        trace
    }
}
