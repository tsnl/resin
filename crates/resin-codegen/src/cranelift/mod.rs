//! Host object translation. Each job owns its Cranelift module and function builders.
//! Only completed object bytes leave the worker; no Cranelift state enters a cache.

use crate::{Error, GenerationError, NativeObject, NativeOptimization};
use cranelift_codegen::{
    ir,
    settings::{self, Configurable},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use resin_executor::Cancellation;
use resin_lir::{Instr, Verified};
use resin_types::prelude::*;
use std::collections::BTreeSet;

mod function;
mod numeric;

pub(super) fn generate(
    checked: Verified<'_>,
    entry: &str,
    optimization: NativeOptimization,
    cancellation: &Cancellation,
) -> Result<NativeObject, GenerationError> {
    cancellation.check()?;
    let source = checked.module();
    let entry = *source
        .entries
        .get(entry)
        .ok_or_else(|| unsupported(format!("entry `{entry}` is not exported")))?;
    validate_entry(source, entry)?;
    let reachable = reachable(source, entry)?;
    let mut module = create_module(optimization)?;
    let mut functions = vec![None; source.functions.len()];
    for &index in &reachable {
        let input = &source.functions[index];
        let signature = signature(
            &module,
            &input.locals[..input.parameter_count]
                .iter()
                .map(|local| local.ty.clone())
                .collect::<Vec<_>>(),
            &input.result,
        )?;
        functions[index] = Some(
            module
                .declare_function(&format!("resin_native_{index}"), Linkage::Local, &signature)
                .map_err(failure)?,
        );
    }
    for index in reachable {
        cancellation.check()?;
        function::define(&mut module, checked, index, &functions, cancellation)?;
    }
    define_main(
        &mut module,
        functions[entry.index()].unwrap(),
        &source.functions[entry.index()].result,
    )?;
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

fn validate_entry(module: &resin_lir::Module, entry: FunctionId) -> Result<(), Error> {
    if !module.foreign_headers.is_empty() {
        return Err(unsupported("C header dependencies require the C backend"));
    }
    let function = &module.functions[entry.index()];
    if function.parameter_count != 0 || !matches!(function.result, Ty::Unit | Ty::Int32) {
        return Err(unsupported(
            "the entry must take no arguments and return int or unit",
        ));
    }
    Ok(())
}

fn reachable(module: &resin_lir::Module, entry: FunctionId) -> Result<BTreeSet<usize>, Error> {
    let mut pending = vec![entry.index()];
    let mut reached = BTreeSet::new();
    while let Some(index) = pending.pop() {
        if !reached.insert(index) {
            continue;
        }
        let function = &module.functions[index];
        if function.foreign.is_some() || function.profile != resin_lir::Profile::Host {
            return Err(Error::at(
                module,
                index,
                None,
                unsupported("foreign and shader functions require the C/SPIR-V backend"),
            ));
        }
        scalar_type(&function.result)?;
        for local in &function.locals {
            scalar_type(&local.ty)?;
        }
        for block in &function.blocks {
            for instruction in &block.instrs {
                if let Instr::Function { function } = instruction {
                    pending.push(function.index());
                }
            }
        }
    }
    Ok(reached)
}

fn signature(module: &ObjectModule, params: &[Ty], result: &Ty) -> Result<ir::Signature, Error> {
    let mut signature = module.make_signature();
    for ty in params {
        signature.params.push(ir::AbiParam::new(scalar_type(ty)?));
    }
    signature
        .returns
        .push(ir::AbiParam::new(scalar_type(result)?));
    Ok(signature)
}

fn scalar_type(ty: &Ty) -> Result<ir::Type, Error> {
    Ok(match ty {
        Ty::Unit | Ty::None | Ty::Bool | Ty::Int8 | Ty::UInt8 => ir::types::I8,
        Ty::Int16 | Ty::UInt16 => ir::types::I16,
        Ty::Int32 | Ty::UInt32 => ir::types::I32,
        Ty::Int64 | Ty::UInt64 => ir::types::I64,
        Ty::Float32 => ir::types::F32,
        Ty::Float64 => ir::types::F64,
        Ty::Pointer { pointee } => {
            scalar_type(pointee)?;
            ir::types::I64
        }
        Ty::Function { params, result } => {
            for ty in params {
                scalar_type(ty)?;
            }
            scalar_type(result)?;
            ir::types::I64
        }
        _ => {
            return Err(unsupported(format!(
                "type {ty:?} is outside the scalar prototype"
            )));
        }
    })
}

fn define_main(module: &mut ObjectModule, entry: FuncId, result: &Ty) -> Result<(), Error> {
    use ir::InstBuilder;
    let mut context = module.make_context();
    context
        .func
        .signature
        .returns
        .push(ir::AbiParam::new(ir::types::I32));
    let main = module
        .declare_function("main", Linkage::Export, &context.func.signature)
        .map_err(failure)?;
    let callee = module.declare_func_in_func(entry, &mut context.func);
    let mut frontend = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut context.func, &mut frontend);
    let start = builder.create_block();
    builder.switch_to_block(start);
    let call = builder.ins().call(callee, &[]);
    let value = if *result == Ty::Unit {
        builder.ins().iconst(ir::types::I32, 0)
    } else {
        builder.inst_results(call)[0]
    };
    builder.ins().return_(&[value]);
    builder.seal_all_blocks();
    builder.finalize(module.target_config());
    module.define_function(main, &mut context).map_err(failure)
}

fn unsupported(message: impl std::fmt::Display) -> Error {
    Error(format!("Cranelift prototype: {message}"))
}

fn failure(error: impl std::fmt::Display) -> Error {
    Error(format!("Cranelift object generation failed: {error}"))
}
