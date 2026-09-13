//! Reserve each concrete function before translating its body, so recursion reuses IDs.
use super::{LowerError, LoweredFunction, functions, specialize};
use crate::{ApplicationNote, Error, ErrorKind, LoweringOptions};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Instance {
    definition: FunctionId,
    arguments: Vec<Ty>,
}

struct Request {
    instance: Instance,
    predecessor: Option<FunctionId>,
    location: Option<SourceLocation>,
}

pub(super) struct Instances<'a> {
    source: &'a resin_hir::Module,
    options: &'a LoweringOptions,
    identities: BTreeMap<Instance, FunctionId>,
    requests: Vec<Request>,
    counts: Vec<usize>,
}

impl<'a> Instances<'a> {
    pub(super) fn new(source: &'a resin_hir::Module, options: &'a LoweringOptions) -> Self {
        Self {
            source,
            options,
            identities: BTreeMap::new(),
            requests: vec![],
            counts: vec![0; source.functions.len()],
        }
    }

    // Until entry selection moves into compilation, ordinary definitions remain roots.
    // Template families enter the worklist only through completed applications.
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
                    None,
                    function.location.clone(),
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn request(
        &mut self,
        definition: FunctionId,
        arguments: Vec<Ty>,
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

    pub(super) fn ordinary(&self, definition: FunctionId) -> Option<FunctionId> {
        self.identities
            .get(&Instance {
                definition,
                arguments: vec![],
            })
            .copied()
    }

    pub(super) fn lower(
        &mut self,
        typer: &TyperContext,
    ) -> Result<Vec<LoweredFunction>, Vec<Error>> {
        let mut completed = vec![];
        let mut errors = vec![];
        let mut index = 0;
        while index < self.requests.len() {
            let id = FunctionId::from_index(index);
            match self.lower_function(id, typer) {
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

    fn lower_function(
        &mut self,
        id: FunctionId,
        typer: &TyperContext,
    ) -> Result<LoweredFunction, Error> {
        let instance = self.requests[id.index()].instance.clone();
        let function = &self.source.functions[instance.definition.index()];
        let body = specialize::function(function, &instance.arguments, self, id, typer)?;
        functions::lower(&body, typer).map_err(|mut error| {
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
                location: request.location.clone(),
            });
            predecessor = request.predecessor;
        }
        trace
    }
}
