//! Reserve each concrete function before translating its body, so recursion reuses IDs.
use super::{LowerError, LoweredFunction, functions, specialize};
use crate::{ApplicationNote, Error, ErrorKind, LoweringOptions, Profile};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Instance {
    definition: FunctionId,
    arguments: Vec<Ty>,
    profile: Profile,
}

struct Request {
    instance: Instance,
    predecessor: Option<FunctionId>,
    location: Option<SourceLocation>,
}

struct NominalRequest {
    definition: TypeId,
    depth: usize,
}

pub(super) struct Instances<'a> {
    source: &'a resin_hir::Module,
    options: &'a LoweringOptions,
    identities: BTreeMap<Instance, FunctionId>,
    requests: Vec<Request>,
    counts: Vec<usize>,
    type_identities: BTreeMap<TypeId, TypeId>,
    definitions: Vec<TypeDef>,
    type_requests: Vec<NominalRequest>,
    type_cursor: usize,
    active_type_depth: Option<usize>,
    typer: TyperContext,
    type_error: Option<ErrorKind>,
    entries: BTreeMap<std::sync::Arc<str>, FunctionId>,
    shaders: BTreeMap<FunctionId, ShaderEntry>,
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
        }
    }

    // Whole-module clients explicitly request all ordinary definitions.
    // Production compilation instead supplies exported target roots.
    pub(super) fn reserve_roots(&mut self) -> Result<(), Error> {
        for id in self
            .source
            .entries
            .values()
            .chain(self.source.shaders.keys())
        {
            if self.source.functions.get(id.index()).is_none() {
                return Err(self.error(
                    ErrorKind::InvalidHir {
                        message: "entry refers to a missing function".into(),
                    },
                    None,
                    None,
                ));
            }
        }
        for (index, function) in self.source.functions.iter().enumerate() {
            if !function.signature.type_params.is_empty()
                && (function.foreign_header.is_some()
                    || self
                        .source
                        .shaders
                        .contains_key(&FunctionId::from_index(index)))
            {
                return Err(self.error(
                    ErrorKind::InvalidHir {
                        message: "foreign and shader declarations require fixed signatures".into(),
                    },
                    None,
                    function.location.clone(),
                ));
            }
            if function.signature.type_params.is_empty() {
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
            if let Some(id) = self.ordinary(definition, Profile::Host) {
                self.entries.insert(name.clone(), id);
            }
        }
        for (&definition, shader) in &self.source.shaders {
            self.shader(definition, shader.embedded, None, None)?;
        }
        Ok(())
    }

    pub(super) fn reserve_entries(&mut self, entries: &[crate::Entry]) -> Result<(), Error> {
        // Sorting source requests makes nominal discovery independent of caller order.
        let mut entries: Vec<_> = entries.iter().collect();
        entries.sort();
        entries.dedup();
        for entry in entries {
            let arguments = entry
                .arguments
                .iter()
                .map(|ty| {
                    super::substitute::Substitution::default()
                        .ty(ty, self)
                        .map_err(|error| self.lower_error(error, None, None))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let id = if entry.profile == Profile::Shader {
                if !arguments.is_empty() {
                    return Err(self.error(
                        ErrorKind::InvalidHir {
                            message: "shader entries require fixed signatures".into(),
                        },
                        None,
                        None,
                    ));
                }
                self.shader(entry.function, false, None, None)?
            } else {
                self.request(entry.function, arguments, entry.profile, None, None)?
            };
            if entry.profile == Profile::Host
                && let Some(previous) = self.entries.insert(entry.name.clone(), id)
                && previous != id
            {
                return Err(self.error(
                    ErrorKind::InvalidHir {
                        message: format!("conflicting applications for entry {}", entry.name)
                            .into(),
                    },
                    None,
                    None,
                ));
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
        let id = self.request(definition, vec![], Profile::Shader, predecessor, location)?;
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
        arguments: Vec<Ty>,
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
                    arguments: instance.arguments,
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

    pub(super) fn reserve_types(&mut self) -> Result<(), LowerError> {
        // Whole-module clients retain every declaration in source order. Requested
        // programs instead discover nominal declarations through their roots.
        for index in 0..self.source.types.len() {
            self.reserve_type(TypeId::from_index(index))?;
        }
        self.complete_types()
    }

    pub(super) fn nominal(&mut self, definition: TypeId) -> Result<TypeId, LowerError> {
        if let Some(kind) = &self.type_error {
            return Err(LowerError {
                span: Span { start: 0, end: 0 },
                kind: kind.clone(),
            });
        }
        let id = self.reserve_type(definition)?;
        if self.active_type_depth.is_none() && self.type_cursor < self.type_requests.len() {
            self.complete_types().inspect_err(|error| {
                self.type_error = Some(error.kind.clone());
            })?;
        }
        Ok(id)
    }

    fn reserve_type(&mut self, definition: TypeId) -> Result<TypeId, LowerError> {
        if let Some(&id) = self.type_identities.get(&definition) {
            return Ok(id);
        }
        let Some(source) = self.source.types.get(definition.index()) else {
            return Err(LowerError::typing(
                Span { start: 0, end: 0 },
                TypeError {
                    kind: TypeErrorKind::InvalidTypeDefinition { definition },
                },
            ));
        };
        let depth = self.active_type_depth.map_or(0, |depth| depth + 1);
        if depth == 256 {
            return Err(LowerError {
                span: Span { start: 0, end: 0 },
                kind: ErrorKind::TypeExpansionLimit { limit: 256 },
            });
        }
        let id = TypeId::from_index(self.definitions.len());
        self.type_identities.insert(definition, id);
        self.type_requests
            .push(NominalRequest { definition, depth });
        self.definitions.push(TypeDef::Nominal {
            name: source.name.clone(),
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
        let definition = self.type_requests[index].definition;
        let source = &self.source.types[definition.index()];
        let body = super::substitute::Substitution::default().ty(&source.body, self)?;
        let drop = source
            .drop
            .map(|hook| self.request(hook, vec![], Profile::Host, None, None))
            .transpose()
            .map_err(|error| LowerError {
                span: error.span,
                kind: error.kind,
            })?;
        self.definitions[index] = TypeDef::Nominal {
            name: source.name.clone(),
            body: Some(body),
            drop,
        };
        Ok(())
    }

    pub(super) fn typer(&self) -> &TyperContext {
        &self.typer
    }
    pub(super) fn module(self) -> crate::Module {
        crate::Module {
            entries: self.entries,
            shaders: self.shaders,
            types: TypeTable::from(self.definitions),
            ..Default::default()
        }
    }

    pub(super) fn lower(&mut self) -> Result<Vec<LoweredFunction>, Vec<Error>> {
        let mut completed = vec![];
        let mut errors = vec![];
        let mut index = 0;
        while index < self.requests.len() {
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
            Err(errors)
        }
    }

    fn lower_function(&mut self, id: FunctionId) -> Result<LoweredFunction, Error> {
        let instance = self.requests[id.index()].instance.clone();
        let function = &self.source.functions[instance.definition.index()];
        let body = specialize::function(function, &instance.arguments, self, id)?;
        functions::lower(&body, &self.typer, instance.profile).map_err(|mut error| {
            error.applications = self.trace(Some(id));
            error
        })
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
                arguments: request.instance.arguments.clone(),
                profile: request.instance.profile,
                location: request.location.clone(),
            });
            predecessor = request.predecessor;
        }
        trace
    }
}
