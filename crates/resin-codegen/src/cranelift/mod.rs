//! Host object translation. Each job owns its Cranelift module and function builders.
//! Only completed object bytes leave the worker; no Cranelift state enters a cache.

use crate::{Error, GenerationError, NativeInputs, NativeObject, NativeOptimization};
use cranelift_codegen::{
    ir,
    settings::{self, Configurable},
};
use cranelift_module::{Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use resin_executor::Cancellation;
use resin_lir::Verified;
use resin_types::prelude::*;
use std::collections::BTreeSet;

mod aggregate;
mod entry;
mod foreign;
mod formatting;
mod function;
mod gpu;
mod gpu_pipeline;
mod gpu_projection;
mod lifecycle;
mod numeric;
mod runtime;
mod types;

pub(super) fn generate(
    checked: Verified<'_>,
    entry: &str,
    optimization: NativeOptimization,
    inputs: &NativeInputs,
    cancellation: &Cancellation,
) -> Result<NativeObject, GenerationError> {
    cancellation.check()?;
    let source = checked.module();
    let entry = *source.entries.get(entry).ok_or_else(|| {
        unsupported(format!(
            "entry `{entry}` is not exported; add export {{ {entry} }};"
        ))
    })?;
    entry::validate(source, entry)?;
    let mut module = create_module(optimization)?;
    let types = types::Types::new(checked, &mut module, inputs)?;
    let reachable = reachable(source, &types)?;
    let mut functions = vec![None; source.functions.len()];
    for &index in &reachable {
        let input = &source.functions[index];
        let signature = types.signature(
            &module,
            &input.locals[..input.parameter_count]
                .iter()
                .map(|local| local.ty.clone())
                .collect::<Vec<_>>(),
            &input.result,
        );
        functions[index] = Some(
            module
                .declare_function(&format!("resin_native_{index}"), Linkage::Local, &signature)
                .map_err(failure)?,
        );
    }
    lifecycle::define(&mut module, &types, entry, &functions, cancellation)?;
    for index in reachable {
        cancellation.check()?;
        if source.functions[index].foreign.is_some() {
            let symbol = inputs
                .foreign
                .get(&FunctionId::from_index(index))
                .ok_or_else(|| {
                    unsupported(format!("missing foreign adapter for function {index}"))
                })?;
            foreign::define(
                &mut module,
                checked,
                &types,
                index,
                &functions,
                symbol,
                cancellation,
            )?;
        } else {
            function::define(
                &mut module,
                checked,
                &types,
                index,
                &functions,
                cancellation,
            )?;
        }
    }
    entry::define(&mut module, &types, entry, &functions)?;
    cancellation.check()?;
    let bytes = module.finish().emit().map_err(failure)?;
    cancellation.check()?;
    Ok(NativeObject {
        bytes: bytes.into(),
    })
}

fn create_module(optimization: NativeOptimization) -> Result<ObjectModule, Error> {
    let mut settings = settings::builder();
    settings
        .set(
            "opt_level",
            match optimization {
                NativeOptimization::None => "none",
                NativeOptimization::Speed => "speed",
            },
        )
        .map_err(failure)?;
    settings.set("is_pic", "true").map_err(failure)?;
    let isa = cranelift_native::builder_with_options(false)
        .map_err(failure)?
        .finish(settings::Flags::new(settings))
        .map_err(failure)?;
    if isa.pointer_type() != ir::types::I64 {
        return Err(unsupported("only 64-bit hosts are supported"));
    }
    let builder = ObjectBuilder::new(isa, "resin", cranelift_module::default_libcall_names())
        .map_err(failure)?;
    Ok(ObjectModule::new(builder))
}

fn reachable(
    module: &resin_lir::Module,
    types: &types::Types<'_>,
) -> Result<BTreeSet<usize>, Error> {
    // LIR construction has already selected the complete dependency graph, including
    // drop hooks and native GPU bridges whose identities are not Function instructions.
    let mut reached = BTreeSet::new();
    for (index, function) in module.functions.iter().enumerate() {
        if function.profile != resin_lir::Profile::Host {
            continue;
        }
        types.validate_value(&function.result)?;
        for local in &function.locals {
            types.validate_value(&local.ty)?;
        }
        reached.insert(index);
    }
    Ok(reached)
}

fn scalar_type(ty: &Ty) -> Result<ir::Type, Error> {
    Ok(match ty {
        Ty::Unit | Ty::None | Ty::Bool | Ty::Int8 | Ty::UInt8 => ir::types::I8,
        Ty::Int16 | Ty::UInt16 => ir::types::I16,
        Ty::Int32 | Ty::UInt32 => ir::types::I32,
        Ty::Int64 | Ty::UInt64 | Ty::Type | Ty::StrongOwner | Ty::WeakOwner | Ty::GpuArguments => {
            ir::types::I64
        }
        Ty::Float32 => ir::types::F32,
        Ty::Float64 => ir::types::F64,
        Ty::Pointer { .. } | Ty::Function { .. } => ir::types::I64,
        _ => {
            return Err(unsupported(format!(
                "type {ty:?} has no direct scalar representation"
            )));
        }
    })
}

fn unsupported(message: impl std::fmt::Display) -> Error {
    Error(format!("native code generation: {message}"))
}

fn failure(error: impl std::fmt::Display) -> Error {
    Error(format!("Cranelift object generation failed: {error}"))
}
