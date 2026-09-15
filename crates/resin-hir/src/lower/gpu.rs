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
            | "gpu_pipeline_context"
            | "gpu_dispatch"
            | "gpu_draw"
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

fn bridge_body(
    function: FunctionId,
    name: &str,
    declaration: &FunctionDecl,
) -> Option<FunctionBody> {
    let params = &declaration.params;
    if name == "gpu_pipeline_context" {
        return matches!(params.as_slice(), [Ty::Defined { .. }])
            .then_some(FunctionBody::Defined(function));
    }
    let Ty::Result { value, .. } = &declaration.result else {
        return None;
    };
    let bytes = Ty::byte_span();
    match name {
        "gpu_compute_pipeline" | "gpu_graphics_pipeline" => {
            let graphics = name == "gpu_graphics_pipeline";
            let count = if graphics { 2 } else { 1 };
            (matches!(**value, Ty::Defined { .. })
                && params.len() == count + 1
                && params[1..].iter().all(|ty| *ty == bytes))
            .then_some(FunctionBody::GpuPipelineFactory {
                factory: function,
                graphics,
            })
        }
        "gpu_dispatch" | "gpu_draw" => {
            let graphics = name == "gpu_draw";
            let root = if graphics {
                Ty::union_of([Ty::GpuArguments, Ty::None])
            } else {
                Ty::GpuArguments
            };
            let tail = if graphics {
                vec![root, Ty::UInt32]
            } else {
                vec![root, Ty::UInt32, Ty::UInt32, Ty::UInt32]
            };
            (**value == Ty::Unit
                && matches!(params.get(1), Some(Ty::Defined { .. }))
                && params.get(2..) == Some(tail.as_slice()))
            .then_some(FunctionBody::GpuPipelineRecord {
                record: function,
                graphics,
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
        let Ty::Result { value, error } = &method.result else {
            return None;
        };
        match method.body {
            FunctionBody::GpuPipelineFactory { graphics, .. } => {
                let (_, pipeline) = self.source_pipeline_type(graphics)?;
                let kind = &pipeline.name;
                let shaders = if graphics {
                    "vertex: @vertex_shader, fragment: @fragment_shader"
                } else {
                    "shader: @compute_shader"
                };
                Some(format!(
                    "({receiver}{shaders}) -> Result<{kind}<T, {}>, {}>",
                    label(value),
                    label(error)
                ))
            }
            FunctionBody::GpuPipelineRecord { graphics, .. } => {
                let (_, pipeline) = self.source_pipeline_type(graphics)?;
                let kind = &pipeline.name;
                let dimensions = if graphics {
                    "count: uint"
                } else {
                    "x: uint, y: uint, z: uint"
                };
                Some(format!(
                    "({receiver}pipeline: {kind}<T, {}>, arguments: _, {dimensions}) -> {}",
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
            FunctionBody::GpuPipelineFactory { factory, graphics } => {
                self.source_pipeline_factory(factory, graphics, inputs)
            }
            FunctionBody::GpuPipelineRecord { record, graphics } => {
                self.source_pipeline_record(record, graphics, inputs)
            }
            _ => unreachable!("pipeline bridge"),
        }
    }

    fn source_pipeline_type(&self, graphics: bool) -> Option<(TypeId, &crate::TypeDefinition)> {
        let kind = if graphics {
            resin_types::GpuPipelineKind::Graphics
        } else {
            resin_types::GpuPipelineKind::Compute
        };
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
        graphics: bool,
        shaders: &[crate::Type],
    ) -> Result<PipelineMethod, String> {
        use crate::Type;
        let count = if graphics { 2 } else { 1 };
        if shaders.len() != count {
            return Err("pipeline creation requires direct shader declarations".into());
        }
        let mut root = Type::None;
        for shader in shaders {
            let Type::Function { params, .. } = shader else {
                return Err("pipeline creation requires decorated shader declarations".into());
            };
            if let Some(Type::Pointer { pointee }) = params.get(1) {
                if root != Type::None && root != **pointee {
                    return Err("vertex and fragment shaders must use the same root type".into());
                }
                root = *pointee.clone();
            } else if !graphics {
                return Err("compute shader root parameter must be a pointer".into());
            }
        }
        let native = self.declared_function(factory);
        let Ty::Result {
            value: owner,
            error,
        } = &native.result
        else {
            unreachable!()
        };
        let (definition, _) = self
            .source_pipeline_type(graphics)
            .ok_or("no source pipeline type is registered for this shader stage")?;
        let value = Type::Defined {
            definition,
            arguments: vec![root, super::types::ty(owner)],
        };
        let mut params = vec![super::types::ty(&native.params[0])];
        params.extend_from_slice(shaders);
        Ok(PipelineMethod {
            body: native.body.clone(),
            params,
            result: Type::Result {
                value: Box::new(value),
                error: Box::new(super::types::ty(error)),
            },
        })
    }

    fn source_pipeline_record(
        &self,
        record: FunctionId,
        graphics: bool,
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
        let expected_kind = if graphics {
            resin_types::GpuPipelineKind::Graphics
        } else {
            resin_types::GpuPipelineKind::Compute
        };
        if metadata.kind != expected_kind {
            return Err(if graphics {
                "draw requires a graphics pipeline"
            } else {
                "dispatch requires a compute pipeline"
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
        if owner != super::types::ty(&native.params[1]) {
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
        let Ty::Result { error, .. } = &native.result else {
            unreachable!()
        };
        if **error != self.gpu_error(allocator) {
            return Err(
                "pipeline recording and GPU allocation must use the same error type".into(),
            );
        }
        let input = if root == Type::None {
            Type::None
        } else {
            self.source_projection(&root, 0)?
        };
        let mut params = vec![super::types::ty(&native.params[0]), pipeline.clone(), input];
        params.extend(native.params[3..].iter().map(super::types::ty));
        Ok(PipelineMethod {
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
                (Type::Pointer { .. }, Type::Pointer { pointee }) => Some(*pointee.clone()),
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
