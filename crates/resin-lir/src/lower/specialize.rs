//! Translate HIR type expressions and a requested body into concrete data.
use super::concrete;
use resin_types::prelude::*;

pub(super) fn ty(source: &resin_hir::Type) -> Ty {
    match source {
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
            definition: *definition,
        },
        resin_hir::Type::Pointer { pointee } => Ty::Pointer {
            pointee: Box::new(ty(pointee)),
        },
        resin_hir::Type::Span { element } => Ty::Span {
            element: Box::new(ty(element)),
        },
        resin_hir::Type::GpuPointer { pointee } => Ty::GpuPointer {
            pointee: Box::new(ty(pointee)),
        },
        resin_hir::Type::GpuSpan { element } => Ty::GpuSpan {
            element: Box::new(ty(element)),
        },
        resin_hir::Type::Arc { pointee } => Ty::Arc {
            pointee: Box::new(ty(pointee)),
        },
        resin_hir::Type::Weak { pointee } => Ty::Weak {
            pointee: Box::new(ty(pointee)),
        },
        resin_hir::Type::GpuComputePipeline { root, owner } => Ty::GpuComputePipeline {
            root: Box::new(ty(root)),
            owner: Box::new(ty(owner)),
        },
        resin_hir::Type::GpuGraphicsPipeline { root, owner } => Ty::GpuGraphicsPipeline {
            root: Box::new(ty(root)),
            owner: Box::new(ty(owner)),
        },
        resin_hir::Type::Function { param, result } => Ty::Function {
            param: Box::new(ty(param)),
            result: Box::new(ty(result)),
        },
        resin_hir::Type::Result { value, error } => Ty::Result {
            value: Box::new(ty(value)),
            error: Box::new(ty(error)),
        },
        resin_hir::Type::Array { element, length } => Ty::Array {
            element: Box::new(ty(element)),
            length: *length,
        },
        resin_hir::Type::Record { fields } => Ty::Record {
            fields: fields
                .iter()
                .map(|f| RecordField {
                    name: f.name.clone(),
                    ty: ty(&f.ty),
                })
                .collect(),
        },
        resin_hir::Type::Union { variants } => Ty::Union {
            variants: variants.iter().map(ty).collect(),
        },
    }
}

fn constant(value: &resin_hir::Constant) -> Value {
    match value {
        resin_hir::Constant::Unit => Value::Unit,
        resin_hir::Constant::None => Value::None,
        resin_hir::Constant::Bool { value } => Value::Bool { value: *value },
        resin_hir::Constant::Int8 { value } => Value::Int8 { value: *value },
        resin_hir::Constant::Int16 { value } => Value::Int16 { value: *value },
        resin_hir::Constant::Int32 { value } => Value::Int32 { value: *value },
        resin_hir::Constant::Int64 { value } => Value::Int64 { value: *value },
        resin_hir::Constant::UInt8 { value } => Value::UInt8 { value: *value },
        resin_hir::Constant::UInt16 { value } => Value::UInt16 { value: *value },
        resin_hir::Constant::UInt32 { value } => Value::UInt32 { value: *value },
        resin_hir::Constant::UInt64 { value } => Value::UInt64 { value: *value },
        resin_hir::Constant::Float32 { value } => Value::Float32 { value: *value },
        resin_hir::Constant::Float64 { value } => Value::Float64 { value: *value },
        resin_hir::Constant::Type { ty: value } => Value::Type { ty: ty(value) },
        resin_hir::Constant::Str { value } => Value::Str {
            value: value.clone(),
        },
    }
}

pub(super) fn definitions(source: &[resin_hir::TypeDefinition]) -> Result<TypeTable, TypeError> {
    let definitions: TypeTable = source
        .iter()
        .map(|definition| TypeDef::Nominal {
            name: definition.name.clone(),
            body: Some(ty(&definition.body)),
            drop: definition.drop,
        })
        .collect::<Vec<_>>()
        .into();
    for (index, definition) in definitions.iter().enumerate() {
        let body = definition.body().expect("completed nominal body");
        resin_types::check_references(&definitions, body)?;
        resin_types::check_layout(&definitions, TypeId::from_index(index), body)?;
    }
    Ok(definitions)
}

pub(super) fn function(source: &resin_hir::Function) -> concrete::Function {
    let params = source
        .signature
        .params
        .iter()
        .map(|p| concrete::Parameter {
            binding: p.binding,
            name: p.name.clone(),
            ty: ty(&p.annotation.ty),
        })
        .collect::<Vec<_>>();
    concrete::Function {
        location: source.location.clone(),
        name: source.name.clone(),
        foreign: source.foreign_header.as_ref().map(|header| Foreign {
            header: header.clone(),
            params: params.iter().map(|p| p.ty.clone()).collect(),
        }),
        signature: concrete::Signature {
            params,
            result: ty(&source.signature.result.ty),
        },
        body: source.body.as_ref().map(term),
    }
}

fn term(source: &resin_hir::Term) -> concrete::Term {
    concrete::Term {
        span: source.span,
        ty: ty(&source.ty),
        kind: kind(&source.kind),
    }
}

fn boxed(source: &resin_hir::Term) -> Box<concrete::Term> {
    Box::new(term(source))
}

fn arguments(source: &resin_hir::Arguments) -> concrete::Arguments {
    concrete::Arguments {
        receiver: source.receiver.as_deref().map(boxed),
        argument: boxed(&source.argument),
        params: source.params.iter().map(ty).collect(),
    }
}

fn statement(source: &resin_hir::Statement) -> concrete::Statement {
    match source {
        resin_hir::Statement::Define {
            binding,
            name,
            init,
        } => concrete::Statement::Define {
            binding: *binding,
            name: name.clone(),
            init: term(init),
        },
        resin_hir::Statement::Declare {
            binding,
            name,
            ty: annotation,
        } => concrete::Statement::Declare {
            binding: *binding,
            name: name.clone(),
            ty: ty(&annotation.ty),
        },
        resin_hir::Statement::Expr { term: source } => {
            concrete::Statement::Expr { term: term(source) }
        }
    }
}

fn arm(source: &resin_hir::MatchArm) -> concrete::MatchArm {
    let tag = match &source.tag {
        resin_hir::Case::Ok => Case::Ok,
        resin_hir::Case::Err => Case::Err,
        resin_hir::Case::Type { ty: value } => Case::Type(ty(value)),
    };
    concrete::MatchArm {
        tag,
        binding: source.binding,
        body: term(&source.body),
    }
}

fn receiver(source: resin_hir::ReceiverConversion) -> concrete::ReceiverConversion {
    match source {
        resin_hir::ReceiverConversion::Value => concrete::ReceiverConversion::Value,
        resin_hir::ReceiverConversion::Address => concrete::ReceiverConversion::Address,
        resin_hir::ReceiverConversion::Load => concrete::ReceiverConversion::Load,
        resin_hir::ReceiverConversion::ArcAddress => concrete::ReceiverConversion::ArcAddress,
        resin_hir::ReceiverConversion::ArcLoad => concrete::ReceiverConversion::ArcLoad,
    }
}

fn kind(source: &resin_hir::TermKind) -> concrete::TermKind {
    match source {
        resin_hir::TermKind::Constant { value } => concrete::TermKind::Constant {
            value: constant(value),
        },
        resin_hir::TermKind::Local { binding, name } => concrete::TermKind::Local {
            binding: *binding,
            name: name.clone(),
        },
        resin_hir::TermKind::Function { function } => concrete::TermKind::Function {
            function: *function,
        },
        resin_hir::TermKind::Shader { function, stage } => concrete::TermKind::Shader {
            function: *function,
            stage: stage.clone(),
        },
        resin_hir::TermKind::Unwrap { value } => concrete::TermKind::Unwrap {
            value: boxed(value),
        },
        resin_hir::TermKind::Try { value } => concrete::TermKind::Try {
            value: boxed(value),
        },
        resin_hir::TermKind::Match { value, arms } => concrete::TermKind::Match {
            value: boxed(value),
            arms: arms.iter().map(arm).collect(),
        },
        resin_hir::TermKind::If { cond, then, els } => concrete::TermKind::If {
            cond: boxed(cond),
            then: boxed(then),
            els: boxed(els),
        },
        resin_hir::TermKind::While { cond, body } => concrete::TermKind::While {
            cond: boxed(cond),
            body: boxed(body),
        },
        resin_hir::TermKind::Block { stmts, tail } => concrete::TermKind::Block {
            stmts: stmts.iter().map(statement).collect(),
            tail: boxed(tail),
        },
        resin_hir::TermKind::Record { fields } => concrete::TermKind::Record {
            fields: fields
                .iter()
                .map(|(name, value)| (name.clone(), term(value)))
                .collect(),
        },
        resin_hir::TermKind::Array { elems } => concrete::TermKind::Array {
            elems: elems.iter().map(term).collect(),
        },
        resin_hir::TermKind::Builtin { name, args } => concrete::TermKind::Builtin {
            name: name.clone(),
            args: args.iter().map(term).collect(),
        },
        resin_hir::TermKind::Call { func, arg } => concrete::TermKind::Call {
            func: boxed(func),
            arg: boxed(arg),
        },
        resin_hir::TermKind::Pack { args } => concrete::TermKind::Pack {
            args: arguments(args),
        },
        resin_hir::TermKind::Intrinsic { op, args } => concrete::TermKind::Intrinsic {
            op: *op,
            args: arguments(args),
        },
        resin_hir::TermKind::Adapt { conversion, arg } => concrete::TermKind::Adapt {
            conversion: receiver(*conversion),
            arg: boxed(arg),
        },
        resin_hir::TermKind::Convert { conversion, arg } => concrete::TermKind::Convert {
            conversion: conversion.clone(),
            arg: boxed(arg),
        },
        resin_hir::TermKind::ArcNew { value } => concrete::TermKind::ArcNew {
            value: boxed(value),
        },
        resin_hir::TermKind::GpuNew { allocator, args } => concrete::TermKind::GpuNew {
            allocator: *allocator,
            args: arguments(args),
        },
        resin_hir::TermKind::GpuAllocate { allocator, args } => concrete::TermKind::GpuAllocate {
            allocator: *allocator,
            args: arguments(args),
        },
        resin_hir::TermKind::GpuPipelineCreate {
            factory,
            shaders,
            args,
        } => concrete::TermKind::GpuPipelineCreate {
            factory: *factory,
            shaders: shaders.clone(),
            args: arguments(args),
        },
        resin_hir::TermKind::GpuPipelineDispatch {
            context,
            allocator,
            record,
            args,
        } => concrete::TermKind::GpuPipelineDispatch {
            context: *context,
            allocator: *allocator,
            record: *record,
            args: arguments(args),
        },
        resin_hir::TermKind::WeakEmpty { pointee } => concrete::TermKind::WeakEmpty {
            pointee: ty(pointee),
        },
        resin_hir::TermKind::Result { failure, arg } => concrete::TermKind::Result {
            failure: *failure,
            arg: boxed(arg),
        },
        resin_hir::TermKind::Absurd { arg } => concrete::TermKind::Absurd { arg: boxed(arg) },
        resin_hir::TermKind::Assign { place, value } => concrete::TermKind::Assign {
            place: boxed(place),
            value: boxed(value),
        },
        resin_hir::TermKind::Address { place } => concrete::TermKind::Address {
            place: boxed(place),
        },
        resin_hir::TermKind::Deref { pointer } => concrete::TermKind::Deref {
            pointer: boxed(pointer),
        },
        resin_hir::TermKind::Field { base, access } => concrete::TermKind::Field {
            base: boxed(base),
            access: FieldAccess {
                ty: ty(&access.ty),
                index: access.index,
                steps: access.steps.clone(),
            },
        },
    }
}
