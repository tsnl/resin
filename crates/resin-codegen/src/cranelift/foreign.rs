//! Bridge Resin's scalar calling convention to the explicit C adapter ABI.
use super::{failure, types::Types, unsupported};
use crate::{Error, GenerationError};
use cranelift_codegen::ir::{self, InstBuilder};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::ObjectModule;
use resin_executor::Cancellation;
use resin_types::prelude::*;

pub(super) fn define(
    module: &mut ObjectModule,
    checked: resin_lir::Verified<'_>,
    types: &Types<'_>,
    index: usize,
    functions: &[Option<FuncId>],
    symbol: &str,
    cancellation: &Cancellation,
) -> Result<(), GenerationError> {
    cancellation.check()?;
    let input = &checked.module().functions[index];
    let parameters = input.locals[..input.parameter_count]
        .iter()
        .map(|local| local.ty.clone())
        .collect::<Vec<_>>();
    let mut native = module.make_signature();
    for parameter in &parameters {
        native.params.push(parameter_abi(types.shape(parameter))?);
    }
    if !matches!(types.shape(&input.result), Ty::Unit) {
        native
            .returns
            .push(parameter_abi(types.shape(&input.result))?);
    }
    let adapter = module
        .declare_function(symbol, Linkage::Import, &native)
        .map_err(failure)?;
    let mut context = module.make_context();
    context.func.signature = types.signature(module, &parameters, &input.result);
    let mut frontend = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut context.func, &mut frontend);
    let block = builder.create_block();
    builder.append_block_params_for_function_params(block);
    builder.switch_to_block(block);
    let args = builder.block_params(block).to_vec();
    let adapter = module.declare_func_in_func(adapter, builder.func);
    let call = builder.ins().call(adapter, &args);
    let result = if matches!(types.shape(&input.result), Ty::Unit) {
        builder.ins().iconst(ir::types::I8, 0)
    } else {
        builder.inst_results(call)[0]
    };
    builder.ins().return_(&[result]);
    builder.seal_all_blocks();
    builder.finalize(module.target_config());
    cancellation.check()?;
    module
        .define_function(
            functions[index].expect("declared foreign bridge"),
            &mut context,
        )
        .map_err(failure)?;
    Ok(())
}

fn parameter_abi(ty: &Ty) -> Result<ir::AbiParam, Error> {
    let scalar = match ty {
        Ty::Bool | Ty::Int8 | Ty::UInt8 => ir::types::I8,
        Ty::Int16 | Ty::UInt16 => ir::types::I16,
        Ty::Int32 | Ty::UInt32 => ir::types::I32,
        Ty::Int64 | Ty::UInt64 | Ty::Pointer { .. } => ir::types::I64,
        Ty::Float32 => ir::types::F32,
        Ty::Float64 => ir::types::F64,
        _ => {
            return Err(unsupported(format!(
                "foreign adapter requires a scalar C type, found {ty:?}"
            )));
        }
    };
    let parameter = ir::AbiParam::new(scalar);
    Ok(match ty {
        Ty::Int8 | Ty::Int16 => parameter.sext(),
        Ty::Bool | Ty::UInt8 | Ty::UInt16 => parameter.uext(),
        _ => parameter,
    })
}
