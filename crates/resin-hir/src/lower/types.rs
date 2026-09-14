//! Complete the frontend's solved types as HIR expressions.
use resin_types::prelude::*;
use std::{collections::BTreeMap, sync::Arc};

pub(super) fn ty(source: &Ty) -> crate::Type {
    match source {
        Ty::Type => crate::Type::Type,
        Ty::Unit => crate::Type::Unit,
        Ty::None => crate::Type::None,
        Ty::Bool => crate::Type::Bool,
        Ty::Int8 => crate::Type::Int8,
        Ty::Int16 => crate::Type::Int16,
        Ty::Int32 => crate::Type::Int32,
        Ty::Int64 => crate::Type::Int64,
        Ty::UInt8 => crate::Type::UInt8,
        Ty::UInt16 => crate::Type::UInt16,
        Ty::UInt32 => crate::Type::UInt32,
        Ty::UInt64 => crate::Type::UInt64,
        Ty::Float32 => crate::Type::Float32,
        Ty::Float64 => crate::Type::Float64,
        Ty::Str => crate::Type::Str,
        Ty::GpuArguments => crate::Type::GpuArguments,
        Ty::Foreign { name } => crate::Type::Foreign { name: name.clone() },
        Ty::Defined { definition } => crate::Type::Defined {
            arguments: vec![],
            definition: *definition,
        },
        Ty::Pointer { pointee } => crate::Type::Pointer {
            pointee: Box::new(ty(pointee)),
        },
        Ty::GpuPointer { pointee } => crate::Type::GpuPointer {
            pointee: Box::new(ty(pointee)),
        },
        Ty::GpuSpan { element } => crate::Type::GpuSpan {
            element: Box::new(ty(element)),
        },
        Ty::ArcPtr { pointee } => crate::Type::ArcPtr {
            pointee: Box::new(ty(pointee)),
        },
        Ty::ArcSpan { element } => crate::Type::ArcSpan {
            element: Box::new(ty(element)),
        },
        Ty::WeakSpan { element } => crate::Type::WeakSpan {
            element: Box::new(ty(element)),
        },
        Ty::WeakPtr { pointee } => crate::Type::WeakPtr {
            pointee: Box::new(ty(pointee)),
        },
        Ty::GpuComputePipeline { root, owner } => crate::Type::GpuComputePipeline {
            root: Box::new(ty(root)),
            owner: Box::new(ty(owner)),
        },
        Ty::GpuGraphicsPipeline { root, owner } => crate::Type::GpuGraphicsPipeline {
            root: Box::new(ty(root)),
            owner: Box::new(ty(owner)),
        },
        Ty::Function { params, result } => crate::Type::Function {
            params: params.iter().map(ty).collect(),
            result: Box::new(ty(result)),
        },
        Ty::Result { value, error } => crate::Type::Result {
            value: Box::new(ty(value)),
            error: Box::new(ty(error)),
        },
        Ty::Array { element, length } => crate::Type::Array {
            element: Box::new(ty(element)),
            length: *length,
        },
        Ty::Record { fields } => crate::Type::Record {
            fields: fields
                .iter()
                .map(|f| crate::RecordField {
                    name: f.name.clone(),
                    ty: ty(&f.ty),
                })
                .collect(),
        },
        Ty::Union { variants } => crate::Type::Union {
            variants: variants.iter().map(ty).collect(),
        },
    }
}

pub(super) fn definition(
    source: &TypeDef,
    methods: BTreeMap<Arc<str>, FunctionId>,
) -> crate::TypeDefinition {
    let TypeDef::Nominal {
        name,
        body: Some(body),
        drop,
    } = source
    else {
        unreachable!("completed source type declaration");
    };
    crate::TypeDefinition {
        type_params: vec![],
        name: name.clone(),
        body: ty(body),
        methods,
        drop: *drop,
    }
}
