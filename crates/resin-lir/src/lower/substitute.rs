//! Normalize type identities, then materialize only the nominal layouts an operation uses.
//! Neither operation performs inference or numeric defaulting.
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::collections::BTreeMap;

const TYPE_DEPTH_LIMIT: usize = 256;
const TYPE_SIZE_LIMIT: usize = 65536;

#[derive(Default)]
pub(super) struct Substitution {
    arguments: BTreeMap<resin_hir::TypeParameterId, resin_hir::Type>,
}

impl Substitution {
    pub(super) fn new(
        parameters: &[resin_hir::TypeParameter],
        arguments: &[resin_hir::Type],
    ) -> Result<Self, super::LowerError> {
        let mut result = Self::default();
        for (parameter, argument) in parameters.iter().zip(arguments) {
            if result
                .arguments
                .insert(parameter.id, argument.clone())
                .is_some()
            {
                return Err(super::LowerError::invalid_hir(
                    parameter.name.span,
                    "duplicate type parameter identity",
                ));
            }
        }
        Ok(result)
    }

    pub(super) fn ty(
        &self,
        source: &resin_hir::Type,
        instances: &mut super::instances::Instances<'_>,
    ) -> Result<Ty, super::LowerError> {
        materialize(&self.normalize(source, instances)?, instances)
    }

    pub(super) fn normalize(
        &self,
        source: &resin_hir::Type,
        instances: &mut super::instances::Instances<'_>,
    ) -> Result<resin_hir::Type, super::LowerError> {
        self.normalize_at(source, 0, &mut { TYPE_SIZE_LIMIT }, instances)
    }

    fn normalize_at(
        &self,
        source: &resin_hir::Type,
        depth: usize,
        remaining: &mut usize,
        instances: &mut super::instances::Instances<'_>,
    ) -> Result<resin_hir::Type, super::LowerError> {
        consume_node(depth, remaining)?;
        Ok(match source {
            resin_hir::Type::Parameter { parameter } => {
                let argument = self.arguments.get(parameter).ok_or_else(|| {
                    super::LowerError::invalid_hir(
                        Span { start: 0, end: 0 },
                        "unbound type parameter; annotate the application",
                    )
                })?;
                // Charge substituted structure before cloning: recursive requests may double it.
                *remaining += 1;
                check_size(argument, depth, remaining)?;
                argument.clone()
            }
            resin_hir::Type::Member { base, name } => {
                let base = self.normalize_at(base, depth + 1, remaining, instances)?;
                let base = materialize(&base, instances)?;
                let result = expression(&member(instances.typer(), &base, name)?.ty, instances);
                check_size(&result, depth, remaining)?;
                result
            }
            resin_hir::Type::Defined {
                definition,
                arguments,
            } => {
                instances.nominal_arity(*definition, arguments.len())?;
                resin_hir::Type::Defined {
                    definition: *definition,
                    arguments: arguments
                        .iter()
                        .map(|argument| {
                            self.normalize_at(argument, depth + 1, remaining, instances)
                        })
                        .collect::<Result<_, _>>()?,
                }
            }
            resin_hir::Type::Type => resin_hir::Type::Type,
            resin_hir::Type::Unit => resin_hir::Type::Unit,
            resin_hir::Type::None => resin_hir::Type::None,
            resin_hir::Type::Bool => resin_hir::Type::Bool,
            resin_hir::Type::Int8 => resin_hir::Type::Int8,
            resin_hir::Type::Int16 => resin_hir::Type::Int16,
            resin_hir::Type::Int32 => resin_hir::Type::Int32,
            resin_hir::Type::Int64 => resin_hir::Type::Int64,
            resin_hir::Type::UInt8 => resin_hir::Type::UInt8,
            resin_hir::Type::UInt16 => resin_hir::Type::UInt16,
            resin_hir::Type::UInt32 => resin_hir::Type::UInt32,
            resin_hir::Type::UInt64 => resin_hir::Type::UInt64,
            resin_hir::Type::Float32 => resin_hir::Type::Float32,
            resin_hir::Type::Float64 => resin_hir::Type::Float64,
            resin_hir::Type::Str => resin_hir::Type::Str,
            resin_hir::Type::GpuArguments => resin_hir::Type::GpuArguments,
            resin_hir::Type::Foreign { name } => resin_hir::Type::Foreign { name: name.clone() },
            resin_hir::Type::Pointer { pointee } => resin_hir::Type::Pointer {
                pointee: Box::new(self.normalize_at(pointee, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::GpuPointer { pointee } => resin_hir::Type::GpuPointer {
                pointee: Box::new(self.normalize_at(pointee, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::Arc { pointee } => resin_hir::Type::Arc {
                pointee: Box::new(self.normalize_at(pointee, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::Weak { pointee } => resin_hir::Type::Weak {
                pointee: Box::new(self.normalize_at(pointee, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::Span { element } => resin_hir::Type::Span {
                element: Box::new(self.normalize_at(element, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::GpuSpan { element } => resin_hir::Type::GpuSpan {
                element: Box::new(self.normalize_at(element, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::GpuComputePipeline { root, owner } => {
                resin_hir::Type::GpuComputePipeline {
                    root: Box::new(self.normalize_at(root, depth + 1, remaining, instances)?),
                    owner: Box::new(self.normalize_at(owner, depth + 1, remaining, instances)?),
                }
            }
            resin_hir::Type::GpuGraphicsPipeline { root, owner } => {
                resin_hir::Type::GpuGraphicsPipeline {
                    root: Box::new(self.normalize_at(root, depth + 1, remaining, instances)?),
                    owner: Box::new(self.normalize_at(owner, depth + 1, remaining, instances)?),
                }
            }
            resin_hir::Type::Function { param, result } => resin_hir::Type::Function {
                param: Box::new(self.normalize_at(param, depth + 1, remaining, instances)?),
                result: Box::new(self.normalize_at(result, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::Result { value, error } => resin_hir::Type::Result {
                value: Box::new(self.normalize_at(value, depth + 1, remaining, instances)?),
                error: Box::new(self.normalize_at(error, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::Array { element, length } => resin_hir::Type::Array {
                element: Box::new(self.normalize_at(element, depth + 1, remaining, instances)?),
                length: *length,
            },
            resin_hir::Type::Record { fields } => resin_hir::Type::Record {
                fields: fields
                    .iter()
                    .map(|field| {
                        Ok(resin_hir::RecordField {
                            name: field.name.clone(),
                            ty: self.normalize_at(&field.ty, depth + 1, remaining, instances)?,
                        })
                    })
                    .collect::<Result<_, super::LowerError>>()?,
            },
            resin_hir::Type::Union { variants } => {
                let mut members = vec![];
                for variant in variants {
                    match self.normalize_at(variant, depth + 1, remaining, instances)? {
                        resin_hir::Type::Union { variants } => members.extend(variants),
                        ty => members.push(ty),
                    }
                }
                members.sort();
                members.dedup();
                if members.len() == 1 {
                    members.pop().unwrap()
                } else {
                    resin_hir::Type::Union { variants: members }
                }
            }
        })
    }
}

fn materialize(
    source: &resin_hir::Type,
    instances: &mut super::instances::Instances<'_>,
) -> Result<Ty, super::LowerError> {
    Ok(match source {
        resin_hir::Type::Parameter { .. } | resin_hir::Type::Member { .. } => {
            unreachable!("normalized type")
        }
        resin_hir::Type::Defined {
            definition,
            arguments,
        } => Ty::Defined {
            definition: instances.nominal(*definition, arguments.clone())?,
        },
        resin_hir::Type::Type => Ty::Type,
        resin_hir::Type::Unit => Ty::Unit,
        resin_hir::Type::None => Ty::None,
        resin_hir::Type::Bool => Ty::Bool,
        resin_hir::Type::Int8 => Ty::Int8,
        resin_hir::Type::Int16 => Ty::Int16,
        resin_hir::Type::Int32 => Ty::Int32,
        resin_hir::Type::Int64 => Ty::Int64,
        resin_hir::Type::UInt8 => Ty::UInt8,
        resin_hir::Type::UInt16 => Ty::UInt16,
        resin_hir::Type::UInt32 => Ty::UInt32,
        resin_hir::Type::UInt64 => Ty::UInt64,
        resin_hir::Type::Float32 => Ty::Float32,
        resin_hir::Type::Float64 => Ty::Float64,
        resin_hir::Type::Str => Ty::Str,
        resin_hir::Type::GpuArguments => Ty::GpuArguments,
        resin_hir::Type::Foreign { name } => Ty::Foreign { name: name.clone() },
        resin_hir::Type::Pointer { pointee } => Ty::Pointer {
            pointee: Box::new(materialize(pointee, instances)?),
        },
        resin_hir::Type::GpuPointer { pointee } => Ty::GpuPointer {
            pointee: Box::new(materialize(pointee, instances)?),
        },
        resin_hir::Type::Arc { pointee } => Ty::Arc {
            pointee: Box::new(materialize(pointee, instances)?),
        },
        resin_hir::Type::Weak { pointee } => Ty::Weak {
            pointee: Box::new(materialize(pointee, instances)?),
        },
        resin_hir::Type::Span { element } => Ty::Span {
            element: Box::new(materialize(element, instances)?),
        },
        resin_hir::Type::GpuSpan { element } => Ty::GpuSpan {
            element: Box::new(materialize(element, instances)?),
        },
        resin_hir::Type::GpuComputePipeline { root, owner } => Ty::GpuComputePipeline {
            root: Box::new(materialize(root, instances)?),
            owner: Box::new(materialize(owner, instances)?),
        },
        resin_hir::Type::GpuGraphicsPipeline { root, owner } => Ty::GpuGraphicsPipeline {
            root: Box::new(materialize(root, instances)?),
            owner: Box::new(materialize(owner, instances)?),
        },
        resin_hir::Type::Function { param, result } => Ty::Function {
            param: Box::new(materialize(param, instances)?),
            result: Box::new(materialize(result, instances)?),
        },
        resin_hir::Type::Result { value, error } => {
            let value = materialize(value, instances)?;
            let error = materialize(error, instances)?;
            if error.variants().is_none() {
                return Err(super::LowerError {
                    span: Span { start: 0, end: 0 },
                    kind: crate::ErrorKind::InvalidInstance {
                        message: "Result errors must be structs or unions of structs".into(),
                    },
                });
            }
            Ty::Result {
                value: Box::new(value),
                error: Box::new(error),
            }
        }
        resin_hir::Type::Array { element, length } => Ty::Array {
            element: Box::new(materialize(element, instances)?),
            length: *length,
        },
        resin_hir::Type::Record { fields } => Ty::Record {
            fields: fields
                .iter()
                .map(|field| {
                    Ok(RecordField {
                        name: field.name.clone(),
                        ty: materialize(&field.ty, instances)?,
                    })
                })
                .collect::<Result<_, super::LowerError>>()?,
        },
        resin_hir::Type::Union { variants } => Ty::union_of(
            variants
                .iter()
                .map(|variant| materialize(variant, instances))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    })
}

// Restore nominal origins when a determining member lookup supplies an argument.
fn expression(source: &Ty, instances: &super::instances::Instances<'_>) -> resin_hir::Type {
    match source {
        Ty::Defined { definition } => instances.nominal_origin(*definition),
        Ty::Type => resin_hir::Type::Type,
        Ty::Unit => resin_hir::Type::Unit,
        Ty::None => resin_hir::Type::None,
        Ty::Bool => resin_hir::Type::Bool,
        Ty::Int8 => resin_hir::Type::Int8,
        Ty::Int16 => resin_hir::Type::Int16,
        Ty::Int32 => resin_hir::Type::Int32,
        Ty::Int64 => resin_hir::Type::Int64,
        Ty::UInt8 => resin_hir::Type::UInt8,
        Ty::UInt16 => resin_hir::Type::UInt16,
        Ty::UInt32 => resin_hir::Type::UInt32,
        Ty::UInt64 => resin_hir::Type::UInt64,
        Ty::Float32 => resin_hir::Type::Float32,
        Ty::Float64 => resin_hir::Type::Float64,
        Ty::Str => resin_hir::Type::Str,
        Ty::GpuArguments => resin_hir::Type::GpuArguments,
        Ty::Foreign { name } => resin_hir::Type::Foreign { name: name.clone() },
        Ty::Pointer { pointee } => resin_hir::Type::Pointer {
            pointee: Box::new(expression(pointee, instances)),
        },
        Ty::GpuPointer { pointee } => resin_hir::Type::GpuPointer {
            pointee: Box::new(expression(pointee, instances)),
        },
        Ty::Arc { pointee } => resin_hir::Type::Arc {
            pointee: Box::new(expression(pointee, instances)),
        },
        Ty::Weak { pointee } => resin_hir::Type::Weak {
            pointee: Box::new(expression(pointee, instances)),
        },
        Ty::Span { element } => resin_hir::Type::Span {
            element: Box::new(expression(element, instances)),
        },
        Ty::GpuSpan { element } => resin_hir::Type::GpuSpan {
            element: Box::new(expression(element, instances)),
        },
        Ty::GpuComputePipeline { root, owner } => resin_hir::Type::GpuComputePipeline {
            root: Box::new(expression(root, instances)),
            owner: Box::new(expression(owner, instances)),
        },
        Ty::GpuGraphicsPipeline { root, owner } => resin_hir::Type::GpuGraphicsPipeline {
            root: Box::new(expression(root, instances)),
            owner: Box::new(expression(owner, instances)),
        },
        Ty::Function { param, result } => resin_hir::Type::Function {
            param: Box::new(expression(param, instances)),
            result: Box::new(expression(result, instances)),
        },
        Ty::Result { value, error } => resin_hir::Type::Result {
            value: Box::new(expression(value, instances)),
            error: Box::new(expression(error, instances)),
        },
        Ty::Array { element, length } => resin_hir::Type::Array {
            element: Box::new(expression(element, instances)),
            length: *length,
        },
        Ty::Record { fields } => resin_hir::Type::Record {
            fields: fields
                .iter()
                .map(|field| resin_hir::RecordField {
                    name: field.name.clone(),
                    ty: expression(&field.ty, instances),
                })
                .collect(),
        },
        Ty::Union { variants } => {
            // Concrete IDs follow layout discovery; identity keys follow source
            // origins. Restore canonical ordering when crossing back to keys.
            let mut variants: Vec<_> = variants
                .iter()
                .map(|variant| expression(variant, instances))
                .collect();
            variants.sort();
            variants.dedup();
            if variants.len() == 1 {
                variants.pop().unwrap()
            } else {
                resin_hir::Type::Union { variants }
            }
        }
    }
}

pub(super) fn member(
    typer: &TyperContext,
    base: &Ty,
    name: &str,
) -> Result<FieldAccess, super::LowerError> {
    let mut base = base;
    while let Some(pointee) = base.deref_target() {
        base = pointee;
    }
    typer
        .type_field(base, name)
        .map_err(|error| super::LowerError::typing(Span { start: 0, end: 0 }, error))
}

fn expansion_limit() -> super::LowerError {
    super::LowerError {
        span: Span { start: 0, end: 0 },
        kind: crate::ErrorKind::TypeExpansionLimit {
            limit: TYPE_DEPTH_LIMIT,
        },
    }
}

fn check_size(
    ty: &resin_hir::Type,
    depth: usize,
    remaining: &mut usize,
) -> Result<(), super::LowerError> {
    consume_node(depth, remaining)?;
    match ty {
        resin_hir::Type::Defined { arguments, .. } => {
            for argument in arguments {
                check_size(argument, depth + 1, remaining)?;
            }
        }
        resin_hir::Type::Parameter { .. } | resin_hir::Type::Member { .. } => {
            unreachable!("normalized argument")
        }
        resin_hir::Type::Pointer { pointee } => check_size(pointee, depth + 1, remaining)?,
        resin_hir::Type::GpuPointer { pointee } => check_size(pointee, depth + 1, remaining)?,
        resin_hir::Type::Arc { pointee } => check_size(pointee, depth + 1, remaining)?,
        resin_hir::Type::Weak { pointee } => check_size(pointee, depth + 1, remaining)?,
        resin_hir::Type::Span { element } => check_size(element, depth + 1, remaining)?,
        resin_hir::Type::GpuSpan { element } => check_size(element, depth + 1, remaining)?,
        resin_hir::Type::GpuComputePipeline { root, owner } => {
            check_size(root, depth + 1, remaining)?;
            check_size(owner, depth + 1, remaining)?;
        }
        resin_hir::Type::GpuGraphicsPipeline { root, owner } => {
            check_size(root, depth + 1, remaining)?;
            check_size(owner, depth + 1, remaining)?;
        }
        resin_hir::Type::Function { param, result } => {
            check_size(param, depth + 1, remaining)?;
            check_size(result, depth + 1, remaining)?;
        }
        resin_hir::Type::Result { value, error } => {
            check_size(value, depth + 1, remaining)?;
            check_size(error, depth + 1, remaining)?;
        }
        resin_hir::Type::Array { element, .. } => check_size(element, depth + 1, remaining)?,
        resin_hir::Type::Record { fields } => {
            for field in fields {
                check_size(&field.ty, depth + 1, remaining)?;
            }
        }
        resin_hir::Type::Union { variants } => {
            for variant in variants {
                check_size(variant, depth + 1, remaining)?;
            }
        }
        resin_hir::Type::Type
        | resin_hir::Type::Unit
        | resin_hir::Type::None
        | resin_hir::Type::Bool
        | resin_hir::Type::Int8
        | resin_hir::Type::Int16
        | resin_hir::Type::Int32
        | resin_hir::Type::Int64
        | resin_hir::Type::UInt8
        | resin_hir::Type::UInt16
        | resin_hir::Type::UInt32
        | resin_hir::Type::UInt64
        | resin_hir::Type::Float32
        | resin_hir::Type::Float64
        | resin_hir::Type::Str
        | resin_hir::Type::GpuArguments
        | resin_hir::Type::Foreign { .. } => {}
    }
    Ok(())
}

fn consume_node(depth: usize, remaining: &mut usize) -> Result<(), super::LowerError> {
    if depth > TYPE_DEPTH_LIMIT {
        return Err(expansion_limit());
    }
    *remaining = remaining.checked_sub(1).ok_or(super::LowerError {
        span: Span { start: 0, end: 0 },
        kind: crate::ErrorKind::TypeSizeLimit {
            limit: TYPE_SIZE_LIMIT,
        },
    })?;
    Ok(())
}
