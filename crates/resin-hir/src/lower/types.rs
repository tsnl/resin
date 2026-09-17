//! Complete the frontend's solved types as HIR expressions.
use resin_types::prelude::*;
use std::collections::BTreeMap;

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
        Ty::GpuView => crate::Type::GpuView,
        Ty::GpuPipelineContract => crate::Type::GpuPipelineContract,
        Ty::GpuArguments => crate::Type::GpuArguments,
        Ty::Foreign { name } => crate::Type::Foreign { name: name.clone() },
        Ty::Defined { definition } => crate::Type::Defined {
            arguments: vec![],
            definition: *definition,
        },
        Ty::Reference { referent } => crate::Type::Reference {
            referent: Box::new(ty(referent)),
        },
        Ty::Pointer { pointee } => crate::Type::Pointer {
            pointee: Box::new(ty(pointee)),
        },
        Ty::StrongOwner => crate::Type::StrongOwner,
        Ty::WeakOwner => crate::Type::WeakOwner,
        Ty::Function { params, result } => crate::Type::Function {
            params: params.iter().map(ty).collect(),
            result: Box::new(ty(result)),
        },
        Ty::Error { payload } => crate::Type::Error {
            payload: Box::new(ty(payload)),
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
    methods: BTreeMap<crate::MethodName, FunctionId>,
) -> crate::TypeDefinition {
    let TypeDef::Nominal {
        name,
        body: Some(body),
        drop,
        ..
    } = source
    else {
        unreachable!("completed source type declaration");
    };
    crate::TypeDefinition {
        gpu_projection: None,
        gpu_pipeline: None,
        type_params: vec![],
        name: name.clone(),
        body: ty(body),
        methods,
        text_view: None,
        drop: *drop,
    }
}
