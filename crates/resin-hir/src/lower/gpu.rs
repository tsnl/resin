//! Explicit source bridges between typed pipelines and their native owners.
use super::Generator;
use super::context::{Context, FunctionBody, FunctionDecl};
use crate::GenerateError;
use resin_source::prelude::*;
use resin_types::prelude::*;

pub(super) fn is_bridge(name: &str) -> bool {
    matches!(
        name,
        "gpu_allocator"
            | "gpu_compute_pipeline"
            | "gpu_graphics_pipeline"
            | "gpu_ray_tracing_pipeline"
            | "gpu_pipeline_context"
            | "gpu_dispatch"
            | "gpu_draw"
            | "gpu_trace_rays"
    )
}

impl Generator {
    pub(super) fn register_gpu_bridge(
        &mut self,
        owner: TypeId,
        function: FunctionId,
        decorator: &Ident,
    ) -> Result<(), GenerateError> {
        let declaration = self.typer.declared_function(function);
        let receiver_matches = declaration
            .params
            .first()
            .is_some_and(|receiver| self.typer.receiver_definition(receiver) == Some(owner));
        let body = bridge_body(function, &decorator.val, declaration)
            .filter(|_| receiver_matches)
            .ok_or_else(|| {
                GenerateError::inference(
                    decorator.span,
                    format!("invalid @{} bridge signature", decorator.val),
                )
            })?;
        if decorator.val.as_ref() == "gpu_pipeline_context"
            && self
                .typer
                .gpu_pipeline_contexts
                .insert(owner, function)
                .is_some()
        {
            return Err(GenerateError::inference(
                decorator.span,
                "a pipeline owner can declare only one GPU context accessor",
            ));
        }
        self.typer.functions.get_mut(&function).unwrap().body = body;
        Ok(())
    }
}

impl Context {
    pub(super) fn register_bridge(
        &mut self,
        function: FunctionId,
        decorator: &Ident,
    ) -> Result<(), GenerateError> {
        let invalid = || {
            GenerateError::inference(
                decorator.span,
                format!("invalid @{} bridge signature", decorator.val),
            )
        };
        let declaration = self.declared_function(function);
        let owner = declaration
            .params
            .first()
            .and_then(|receiver| self.receiver_definition(receiver))
            .ok_or_else(invalid)?;
        if decorator.val.as_ref() == "gpu_allocator" {
            if declaration.params.get(1..) != Some(&[Ty::UInt64, Ty::UInt64, Ty::Int32][..])
                || !matches!(declaration.result.fallible_parts(), Some((Ty::GpuView, _)))
            {
                return Err(invalid());
            }
            if self
                .gpu_allocators
                .insert(owner, function)
                .is_some_and(|previous| previous != function)
            {
                return Err(GenerateError::inference(
                    decorator.span,
                    "a type can declare only one GPU allocator",
                ));
            }
        } else {
            let body = bridge_body(function, &decorator.val, declaration).ok_or_else(invalid)?;
            if decorator.val.as_ref() == "gpu_pipeline_context"
                && self
                    .gpu_pipeline_contexts
                    .insert(owner, function)
                    .is_some_and(|previous| previous != function)
            {
                return Err(GenerateError::inference(
                    decorator.span,
                    "a pipeline owner can declare only one GPU context accessor",
                ));
            }
            self.functions.get_mut(&function).unwrap().body = body;
        }
        Ok(())
    }
}

fn source_value(ty: &crate::Type) -> &crate::Type {
    match ty {
        crate::Type::Reference { referent, .. } => referent,
        _ => ty,
    }
}

fn bridge_body(
    function: FunctionId,
    name: &str,
    declaration: &FunctionDecl,
) -> Option<FunctionBody> {
    let params = &declaration.params;
    if name == "gpu_pipeline_context" {
        return (declaration.source_params.len() == 1
            && matches!(
                source_value(&declaration.source_params[0]),
                crate::Type::Defined { .. }
            ))
        .then_some(FunctionBody::Ordinary);
    }
    let (value, _) = declaration.result.fallible_parts()?;
    let bytes = Ty::byte_span(false);
    match name {
        "gpu_compute_pipeline" | "gpu_graphics_pipeline" | "gpu_ray_tracing_pipeline" => {
            let kind = match name {
                "gpu_compute_pipeline" => resin_types::GpuPipelineKind::Compute,
                "gpu_graphics_pipeline" => resin_types::GpuPipelineKind::Graphics,
                _ => resin_types::GpuPipelineKind::RayTracing,
            };
            let count = match kind {
                resin_types::GpuPipelineKind::Compute => 1,
                resin_types::GpuPipelineKind::Graphics => 2,
                resin_types::GpuPipelineKind::RayTracing => 3,
            };
            (matches!(*value, Ty::Defined { .. })
                && params.len() == count + 1
                && params[1..].iter().all(|ty| *ty == bytes))
            .then_some(FunctionBody::GpuPipelineFactory {
                factory: function,
                kind,
            })
        }
        "gpu_dispatch" | "gpu_draw" | "gpu_trace_rays" => {
            let kind = match name {
                "gpu_dispatch" => resin_types::GpuPipelineKind::Compute,
                "gpu_draw" => resin_types::GpuPipelineKind::Graphics,
                _ => resin_types::GpuPipelineKind::RayTracing,
            };
            let root = if kind == resin_types::GpuPipelineKind::Graphics {
                Ty::union_of([Ty::GpuArguments, Ty::None])
            } else {
                Ty::GpuArguments
            };
            let tail = if kind == resin_types::GpuPipelineKind::Graphics {
                vec![root, Ty::UInt32]
            } else {
                vec![root, Ty::UInt32, Ty::UInt32, Ty::UInt32]
            };
            (*value == Ty::Unit
                && declaration.source_params.get(1).is_some_and(|parameter| {
                    matches!(source_value(parameter), crate::Type::Defined { .. })
                })
                && params.get(2..) == Some(tail.as_slice()))
            .then_some(FunctionBody::GpuPipelineRecord {
                record: function,
                kind,
            })
        }
        _ => None,
    }
}

impl Context {
    pub(crate) fn gpu_method_label(
        &self,
        method: &FunctionDecl,
        associated: bool,
    ) -> Option<String> {
        if !matches!(
            method.body,
            FunctionBody::GpuPipelineFactory { .. } | FunctionBody::GpuPipelineRecord { .. }
        ) {
            return None;
        }
        let label = |ty: &Ty| resin_types::format_type(ty, self.definitions());
        let receiver = if associated {
            format!("{}, ", label(&method.params[0]))
        } else {
            String::new()
        };
        let (value, error) = method.result.fallible_parts()?;
        match method.body {
            FunctionBody::GpuPipelineFactory { kind, .. } => {
                let (_, pipeline) = self.source_pipeline_type(kind)?;
                let label_kind = &pipeline.name;
                let shaders = match kind {
                    resin_types::GpuPipelineKind::Compute => "shader: @compute_shader",
                    resin_types::GpuPipelineKind::Graphics => {
                        "vertex: @vertex_shader, fragment: @fragment_shader"
                    }
                    resin_types::GpuPipelineKind::RayTracing => {
                        "ray_generation: @ray_generation_shader, miss: @miss_shader, closest_hit: @closest_hit_shader"
                    }
                };
                Some(format!(
                    "({receiver}{shaders}) -> ({label_kind}<T, {}> | Err<{}>)",
                    label(value),
                    label(error)
                ))
            }
            FunctionBody::GpuPipelineRecord { kind, .. } => {
                let (_, pipeline) = self.source_pipeline_type(kind)?;
                let label_kind = &pipeline.name;
                let dimensions = if kind == resin_types::GpuPipelineKind::Graphics {
                    "count: u32"
                } else {
                    "x: u32, y: u32, z: u32"
                };
                Some(format!(
                    "({receiver}pipeline: {label_kind}<T, {}>, arguments: _, {dimensions}) -> {}",
                    label(&method.params[1]),
                    label(&method.result)
                ))
            }
            _ => None,
        }
    }
}

/// A completed source bridge call keeps generic nominal identities in HIR.
#[derive(Clone)]
pub(crate) struct PipelineMethod {
    pub declaration: Option<super::scope::DeclarationId>,
    pub body: FunctionBody,
    pub params: Vec<crate::Type>,
    pub result: crate::Type,
}

impl Context {
    pub(crate) fn source_pipeline_method(
        &self,
        method: &FunctionDecl,
        inputs: &[crate::Type],
    ) -> Result<PipelineMethod, String> {
        match method.body {
            FunctionBody::GpuPipelineFactory { factory, kind } => {
                self.source_pipeline_factory(factory, kind, inputs)
            }
            FunctionBody::GpuPipelineRecord { record, kind } => {
                self.source_pipeline_record(record, kind, inputs)
            }
            _ => unreachable!("pipeline bridge"),
        }
    }

    fn source_pipeline_type(
        &self,
        kind: resin_types::GpuPipelineKind,
    ) -> Option<(TypeId, &crate::TypeDefinition)> {
        self.nominal_schemes.iter().find_map(|(id, source)| {
            source
                .gpu_pipeline
                .as_ref()
                .filter(|pipeline| pipeline.kind == kind)
                .map(|_| (*id, source))
        })
    }

    fn source_pipeline_factory(
        &self,
        factory: FunctionId,
        kind: resin_types::GpuPipelineKind,
        shaders: &[crate::Type],
    ) -> Result<PipelineMethod, String> {
        use crate::Type;
        let count = match kind {
            resin_types::GpuPipelineKind::Compute => 1,
            resin_types::GpuPipelineKind::Graphics => 2,
            resin_types::GpuPipelineKind::RayTracing => 3,
        };
        if shaders.len() != count {
            return Err("pipeline creation requires direct shader declarations".into());
        }
        let mut root = Type::None;
        for shader in shaders {
            let Type::Function { params, .. } = shader else {
                return Err("pipeline creation requires decorated shader declarations".into());
            };
            if let Some(Type::Pointer { pointee, .. }) = params.get(1) {
                if root != Type::None && root != **pointee {
                    return Err("pipeline shaders must use the same root type".into());
                }
                root = *pointee.clone();
            } else if kind != resin_types::GpuPipelineKind::Graphics {
                return Err("shader root parameter must be a pointer".into());
            }
        }
        if kind == resin_types::GpuPipelineKind::RayTracing {
            let Type::Function {
                params: miss,
                result: miss_result,
            } = &shaders[1]
            else {
                unreachable!()
            };
            let Type::Function {
                params: hit,
                result: hit_result,
            } = &shaders[2]
            else {
                unreachable!()
            };
            if miss.first() != hit.first()
                || miss.first() != Some(miss_result.as_ref())
                || hit.first() != Some(hit_result.as_ref())
            {
                return Err("ray shaders must use the same payload type".into());
            }
        }
        let native = self.declared_function(factory);
        let (owner, error) = native
            .result
            .fallible_parts()
            .expect("validated pipeline factory");
        let (definition, _) = self
            .source_pipeline_type(kind)
            .ok_or("no source pipeline type is registered for this shader stage")?;
        let value = Type::Defined {
            definition,
            arguments: vec![root, super::types::ty(owner)],
        };
        let mut params = vec![native.source_params[0].clone()];
        params.extend_from_slice(shaders);
        Ok(PipelineMethod {
            declaration: None,
            body: native.body.clone(),
            params,
            result: Type::Union {
                variants: vec![
                    value,
                    Type::Error {
                        payload: Box::new(super::types::ty(error)),
                    },
                ],
            },
        })
    }

    fn source_pipeline_record(
        &self,
        record: FunctionId,
        kind: resin_types::GpuPipelineKind,
        inputs: &[crate::Type],
    ) -> Result<PipelineMethod, String> {
        use crate::Type;
        let pipeline = inputs
            .first()
            .ok_or("recording requires a typed pipeline")?;
        let Type::Defined {
            definition,
            arguments,
        } = pipeline
        else {
            return Err("recording requires a registered source pipeline type".into());
        };
        let source = self
            .nominal_schemes
            .get(definition)
            .ok_or("pipeline source declaration is missing")?;
        let metadata = source
            .gpu_pipeline
            .as_ref()
            .ok_or("recording requires a registered source pipeline type")?;
        if metadata.kind != kind {
            return Err(match kind {
                resin_types::GpuPipelineKind::Compute => "dispatch requires a compute pipeline",
                resin_types::GpuPipelineKind::Graphics => "draw requires a graphics pipeline",
                resin_types::GpuPipelineKind::RayTracing => {
                    "trace_rays requires a ray tracing pipeline"
                }
            }
            .into());
        }
        let apply = |ty| {
            super::gpu_projections::substitute(ty, &source.type_params, arguments.clone())
                .ok_or("incomplete pipeline type arguments")
        };
        let root = apply(&metadata.root)?;
        let owner = apply(&metadata.owner)?;
        let native = self.declared_function(record);
        if &owner != source_value(&native.source_params[1]) {
            return Err("pipeline owner does not match the command recorder".into());
        }
        let owner_id = self
            .receiver_definition(&native.params[1])
            .ok_or("pipeline owner requires a source nominal facade")?;
        let context = *self
            .gpu_pipeline_contexts
            .get(&owner_id)
            .ok_or("pipeline owner has no GPU context accessor")?;
        let getter = self.declared_function(context);
        if getter.params != [native.params[1].clone()] {
            return Err("pipeline context accessor has the wrong owner type".into());
        }
        let allocator = self
            .gpu_allocator(&getter.result)
            .ok_or("pipeline context requires a registered GPU allocator")?;
        let (_, error) = native
            .result
            .fallible_parts()
            .expect("validated pipeline record");
        if *error != self.gpu_error(allocator) {
            return Err(
                "pipeline recording and GPU allocation must use the same error type".into(),
            );
        }
        let input = if root == Type::None {
            Type::None
        } else {
            self.source_projection(&root, 0)?
        };
        let pipeline = if let Type::Reference { mutable, .. } = &native.source_params[1] {
            Type::Reference {
                mutable: *mutable,
                referent: Box::new(pipeline.clone()),
            }
        } else {
            pipeline.clone()
        };
        let mut params = vec![native.source_params[0].clone(), pipeline, input];
        params.extend(native.params[3..].iter().map(super::types::ty));
        Ok(PipelineMethod {
            declaration: None,
            body: FunctionBody::GpuPipelineDispatch {
                context,
                allocator: (root != Type::None).then_some(allocator),
                record,
            },
            params,
            result: super::types::ty(&native.result),
        })
    }

    fn source_projection(&self, target: &crate::Type, depth: usize) -> Result<crate::Type, String> {
        use crate::Type;
        if depth >= 128 {
            return Err("GPU projection type exceeds the depth limit".into());
        }
        if let Some(projection) = self.registered_projection(target)? {
            return Ok(projection);
        }
        match target {
            Type::Defined { .. } => {
                let body = super::gpu_projections::body(self, target)
                    .ok_or("incomplete shader root declaration")?;
                self.source_projection(&body, depth + 1)
            }
            Type::Record { fields } => Ok(Type::Record {
                fields: fields
                    .iter()
                    .map(|field| {
                        Ok(crate::RecordField {
                            name: field.name.clone(),
                            ty: self.source_projection(&field.ty, depth + 1)?,
                        })
                    })
                    .collect::<Result<_, String>>()?,
            }),
            Type::Array { element, length } => Ok(Type::Array {
                element: Box::new(self.source_projection(element, depth + 1)?),
                length: *length,
            }),
            Type::Pointer { .. } => {
                Err("shader pointer requires an explicitly registered GPU projection".into())
            }
            _ => Ok(target.clone()),
        }
    }

    fn registered_projection(&self, target: &crate::Type) -> Result<Option<crate::Type>, String> {
        use crate::Type;
        let mut candidates = Vec::new();
        for (definition, source) in &self.nominal_schemes {
            let Some(projection) = &source.gpu_projection else {
                continue;
            };
            let element = match (&projection.target, target) {
                (Type::Pointer { .. }, Type::Pointer { pointee, .. }) => Some(*pointee.clone()),
                (
                    Type::Defined {
                        definition: expected,
                        ..
                    },
                    Type::Defined {
                        definition: actual,
                        arguments,
                    },
                ) if expected == actual && arguments.len() == 1 => Some(arguments[0].clone()),
                _ => None,
            };
            if let Some(element) = element {
                let arguments = vec![element];
                if super::gpu_projections::substitute(
                    &projection.target,
                    &source.type_params,
                    arguments.clone(),
                )
                .as_ref()
                    == Some(target)
                {
                    candidates.push(Type::Defined {
                        definition: *definition,
                        arguments,
                    });
                }
            }
        }
        if candidates.len() <= 1 {
            return Ok(candidates.pop());
        }
        let mut names = candidates
            .iter()
            .map(|candidate| {
                let Type::Defined { definition, .. } = candidate else {
                    unreachable!("registered projection candidate")
                };
                format!("`{}`", self.nominal_schemes[definition].name)
            })
            .collect::<Vec<_>>();
        names.sort();
        Err(format!(
            "shader storage has ambiguous GPU projections: {}",
            names.join(", ")
        ))
    }
}
