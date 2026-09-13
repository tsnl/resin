//! Substitute closed arguments without performing type inference or numeric defaulting.
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::collections::BTreeMap;

const TYPE_DEPTH_LIMIT: usize = 256;
const TYPE_SIZE_LIMIT: usize = 65536;

#[derive(Default)]
pub(super) struct Substitution {
    arguments: BTreeMap<resin_hir::TypeParameterId, Ty>,
}

impl Substitution {
    pub(super) fn new(
        parameters: &[resin_hir::TypeParameter],
        arguments: &[Ty],
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
        self.ty_at(source, 0, &mut { TYPE_SIZE_LIMIT }, instances)
    }

    fn ty_at(
        &self,
        source: &resin_hir::Type,
        depth: usize,
        remaining: &mut usize,
        instances: &mut super::instances::Instances<'_>,
    ) -> Result<Ty, super::LowerError> {
        consume_node(depth, remaining)?;
        Ok(match source {
            resin_hir::Type::Parameter { parameter } => {
                let argument = self.arguments.get(parameter).ok_or_else(|| {
                    super::LowerError::invalid_hir(
                        Span { start: 0, end: 0 },
                        "unbound type parameter; annotate the application",
                    )
                })?;
                // Charge substituted structure before cloning it. A small source tree can
                // double an earlier argument at every recursive request.
                *remaining += 1;
                check_size(argument, depth, remaining)?;
                argument.clone()
            }
            resin_hir::Type::Member { base, name } => {
                let base = self.ty_at(base, depth + 1, remaining, instances)?;
                let result = member(instances.typer(), &base, name)?.ty;
                check_size(&result, depth, remaining)?;
                result
            }
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
            resin_hir::Type::Defined { definition } => Ty::Defined {
                definition: instances.nominal(*definition)?,
            },
            resin_hir::Type::Pointer { pointee } => Ty::Pointer {
                pointee: Box::new(self.ty_at(pointee, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::Span { element } => Ty::Span {
                element: Box::new(self.ty_at(element, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::GpuPointer { pointee } => Ty::GpuPointer {
                pointee: Box::new(self.ty_at(pointee, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::GpuSpan { element } => Ty::GpuSpan {
                element: Box::new(self.ty_at(element, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::Arc { pointee } => Ty::Arc {
                pointee: Box::new(self.ty_at(pointee, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::Weak { pointee } => Ty::Weak {
                pointee: Box::new(self.ty_at(pointee, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::GpuComputePipeline { root, owner } => Ty::GpuComputePipeline {
                root: Box::new(self.ty_at(root, depth + 1, remaining, instances)?),
                owner: Box::new(self.ty_at(owner, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::GpuGraphicsPipeline { root, owner } => Ty::GpuGraphicsPipeline {
                root: Box::new(self.ty_at(root, depth + 1, remaining, instances)?),
                owner: Box::new(self.ty_at(owner, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::Function { param, result } => Ty::Function {
                param: Box::new(self.ty_at(param, depth + 1, remaining, instances)?),
                result: Box::new(self.ty_at(result, depth + 1, remaining, instances)?),
            },
            resin_hir::Type::Result { value, error } => {
                let value = self.ty_at(value, depth + 1, remaining, instances)?;
                let error = self.ty_at(error, depth + 1, remaining, instances)?;
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
                element: Box::new(self.ty_at(element, depth + 1, remaining, instances)?),
                length: *length,
            },
            resin_hir::Type::Record { fields } => Ty::Record {
                fields: fields
                    .iter()
                    .map(|f| {
                        Ok(RecordField {
                            name: f.name.clone(),
                            ty: self.ty_at(&f.ty, depth + 1, remaining, instances)?,
                        })
                    })
                    .collect::<Result<_, super::LowerError>>()?,
            },
            resin_hir::Type::Union { variants } => Ty::union_of(
                variants
                    .iter()
                    .map(|ty| self.ty_at(ty, depth + 1, remaining, instances))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        })
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

fn check_size(ty: &Ty, depth: usize, remaining: &mut usize) -> Result<(), super::LowerError> {
    consume_node(depth, remaining)?;
    match ty {
        Ty::Pointer { pointee }
        | Ty::GpuPointer { pointee }
        | Ty::Arc { pointee }
        | Ty::Weak { pointee } => check_size(pointee, depth + 1, remaining)?,
        Ty::Span { element } | Ty::GpuSpan { element } | Ty::Array { element, .. } => {
            check_size(element, depth + 1, remaining)?
        }
        Ty::GpuComputePipeline { root, owner } | Ty::GpuGraphicsPipeline { root, owner } => {
            check_size(root, depth + 1, remaining)?;
            check_size(owner, depth + 1, remaining)?;
        }
        Ty::Function { param, result } => {
            check_size(param, depth + 1, remaining)?;
            check_size(result, depth + 1, remaining)?;
        }
        Ty::Result { value, error } => {
            check_size(value, depth + 1, remaining)?;
            check_size(error, depth + 1, remaining)?;
        }
        Ty::Record { fields } => {
            for field in fields {
                check_size(&field.ty, depth + 1, remaining)?;
            }
        }
        Ty::Union { variants } => {
            for variant in variants {
                check_size(variant, depth + 1, remaining)?;
            }
        }
        Ty::Type
        | Ty::Unit
        | Ty::None
        | Ty::Bool
        | Ty::Int8
        | Ty::Int16
        | Ty::Int32
        | Ty::Int64
        | Ty::UInt8
        | Ty::UInt16
        | Ty::UInt32
        | Ty::UInt64
        | Ty::Float32
        | Ty::Float64
        | Ty::Str
        | Ty::Foreign { .. }
        | Ty::Defined { .. }
        | Ty::GpuArguments => {}
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
