//! C process startup surrounding a call through Resin's internal function ABI.

use super::{
    failure,
    function::{Body, Operand},
    runtime,
    types::Types,
};
use crate::Error;
use cranelift_codegen::ir::{self, InstBuilder, condcodes::IntCC};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::ObjectModule;
use resin_types::prelude::*;

pub(super) fn validate(module: &resin_lir::Module, entry: FunctionId) -> Result<(), Error> {
    let function = &module.functions[entry.index()];
    let parameters = &function.locals[..function.parameter_count];
    let process_inputs = matches!(parameters, [argc, argv, envp]
        if argc.ty == Ty::Int32 && string_array(&argv.ty) && string_array(&envp.ty));
    if function.foreign.is_some()
        || function.profile != resin_lir::Profile::Host
        || !(parameters.is_empty() || process_inputs)
        || !matches!(function.result, Ty::Unit | Ty::Int32 | Ty::Result { .. })
    {
        let entry = function.name.as_deref().unwrap_or("<entry>");
        return Err(Error(format!(
            "entry function `{entry}` must be a Resin function, take () or (int, Ptr<Ptr<ubyte>>, Ptr<Ptr<ubyte>>), and return int, (), or Result of either"
        )));
    }
    if matches!(&function.result, Ty::Result { value, .. } if !matches!(value.as_ref(), Ty::Unit | Ty::Int32))
    {
        return Err(Error(
            "Result entry points must have an int or () success type".into(),
        ));
    }
    Ok(())
}

pub(super) fn define(
    module: &mut ObjectModule,
    types: &Types<'_>,
    entry: FunctionId,
    functions: &[Option<FuncId>],
) -> Result<(), Error> {
    validate(types.module, entry)?;
    let input = &types.module.functions[entry.index()];
    let mut context = module.make_context();
    context.func.signature.params.extend([
        ir::AbiParam::new(ir::types::I32),
        ir::AbiParam::new(ir::types::I64),
    ]);
    context
        .func
        .signature
        .returns
        .push(ir::AbiParam::new(ir::types::I32));
    let main = module
        .declare_function("main", Linkage::Export, &context.func.signature)
        .map_err(failure)?;
    let mut frontend = FunctionBuilderContext::new();
    let builder = FunctionBuilder::new(&mut context.func, &mut frontend);
    let mut body = Body::new(builder, module, types, functions, input);
    let start = body.builder.create_block();
    body.builder.append_block_params_for_function_params(start);
    body.builder.switch_to_block(start);
    let arguments = body.builder.block_params(start).to_vec();
    body.register_process_cleanup()?;
    let result = body.call_entry(entry, arguments[0], arguments[1])?;
    body.finish_entry(&input.result, result)?;
    body.builder.seal_all_blocks();
    body.builder.finalize(body.module.target_config());
    module.define_function(main, &mut context).map_err(failure)
}

impl Body<'_, '_> {
    fn register_process_cleanup(&mut self) -> Result<(), Error> {
        let signature = self.module.make_signature();
        let name = self
            .types
            .runtime
            .get("resin_cleanup")
            .map_or("resin_cleanup", |name| name.as_ref());
        let cleanup = self
            .module
            .declare_function(name, Linkage::Import, &signature)
            .map_err(failure)?;
        let reference = self.module.declare_func_in_func(cleanup, self.builder.func);
        let address = self.builder.ins().func_addr(ir::types::I64, reference);
        self.runtime_call("atexit", &[(ir::types::I64, address)], Some(ir::types::I32))?;
        Ok(())
    }

    fn call_entry(
        &mut self,
        entry: FunctionId,
        argc: ir::Value,
        argv: ir::Value,
    ) -> Result<ir::Value, Error> {
        let params = self.input.locals[..self.input.parameter_count]
            .iter()
            .map(|local| local.ty.clone())
            .collect::<Vec<_>>();
        let result = self.input.result.clone();
        let function = self.functions[entry.index()].expect("declared entry");
        let reference = self
            .module
            .declare_func_in_func(function, self.builder.func);
        let value = self.builder.ins().func_addr(ir::types::I64, reference);
        let mut arguments = vec![Operand {
            ty: Ty::Function {
                params: params.clone(),
                result: Box::new(result),
            },
            value,
            function: Some(function),
            live: None,
        }];
        if params.len() == 3 {
            let owned_argv = self.allocate(&params[1]);
            let owned_envp = self.allocate(&params[2]);
            let count = self
                .runtime_call(
                    "resin_process_init",
                    &[
                        (ir::types::I32, argc),
                        (ir::types::I64, argv),
                        (ir::types::I64, owned_argv),
                        (ir::types::I64, owned_envp),
                    ],
                    Some(ir::types::I32),
                )?
                .unwrap();
            let argv = self.read(&params[1], owned_argv);
            let envp = self.read(&params[2], owned_envp);
            arguments.extend(
                params
                    .into_iter()
                    .zip([count, argv, envp])
                    .map(|(ty, value)| Operand {
                        ty,
                        value,
                        function: None,
                        live: None,
                    }),
            );
        }
        self.call(&arguments)
    }

    fn finish_entry(&mut self, ty: &Ty, result: ir::Value) -> Result<(), Error> {
        let status = match ty {
            Ty::Unit => self.builder.ins().iconst(ir::types::I32, 0),
            Ty::Int32 => result,
            Ty::Result { value, error } => {
                let failed = self.builder.create_block();
                let succeeded = self.builder.create_block();
                let tag =
                    self.builder
                        .ins()
                        .load(ir::types::I32, ir::MemFlagsData::new(), result, 0);
                let is_error = self.builder.ins().icmp_imm_u(IntCC::Equal, tag, 1);
                self.builder
                    .ins()
                    .brif(is_error, failed, &[], succeeded, &[]);
                self.builder.switch_to_block(failed);
                let payload = self.offset(result, self.types.layout(ty).offsets[1]);
                self.print_entry_error(error, payload)?;
                self.drop_value(ty, result)?;
                let status = self.builder.ins().iconst(ir::types::I32, 1);
                self.builder.ins().return_(&[status]);
                self.builder.switch_to_block(succeeded);
                if value.as_ref() == &Ty::Unit {
                    self.builder.ins().iconst(ir::types::I32, 0)
                } else {
                    let payload = self.offset(result, self.types.layout(ty).offsets[1]);
                    self.read(value, payload)
                }
            }
            _ => unreachable!("validated entry result"),
        };
        self.builder.ins().return_(&[status]);
        Ok(())
    }

    fn print_entry_error(&mut self, ty: &Ty, payload: ir::Value) -> Result<(), Error> {
        if let Ty::Defined { definition } = ty {
            let name = self.types.table[definition.index()]
                .name()
                .expect("named entry error");
            return self.write_entry_error(&format!("unhandled error: {name}\n"));
        }
        let tag = self
            .builder
            .ins()
            .load(ir::types::I32, ir::MemFlagsData::new(), payload, 0);
        let complete = self.builder.create_block();
        for definition in ty.variants().expect("verified entry error set") {
            let selected = self.builder.create_block();
            let next = self.builder.create_block();
            let matches =
                self.builder
                    .ins()
                    .icmp_imm_u(IntCC::Equal, tag, i64::from(definition.tag()));
            self.builder.ins().brif(matches, selected, &[], next, &[]);
            self.builder.switch_to_block(selected);
            let name = self.types.table[definition.index()]
                .name()
                .expect("named entry error");
            self.write_entry_error(&format!("unhandled error: {name}\n"))?;
            self.builder.ins().jump(complete, &[]);
            self.builder.switch_to_block(next);
        }
        self.write_entry_error("unhandled error: invalid error tag\n")?;
        self.builder.ins().jump(complete, &[]);
        self.builder.switch_to_block(complete);
        Ok(())
    }

    fn write_entry_error(&mut self, text: &str) -> Result<(), Error> {
        let data = runtime::data(self.module, &mut self.builder, text.as_bytes())?;
        let length = self.builder.ins().iconst(ir::types::I64, text.len() as i64);
        let stream = self.builder.ins().iconst(ir::types::I32, 1);
        self.runtime_call(
            "resin_stream_write",
            &[
                (ir::types::I32, stream),
                (ir::types::I64, data),
                (ir::types::I64, length),
            ],
            Some(ir::types::I32),
        )?;
        Ok(())
    }
}

fn string_array(ty: &Ty) -> bool {
    matches!(ty, Ty::Pointer { pointee } if matches!(pointee.as_ref(), Ty::Pointer { pointee } if pointee.as_ref() == &Ty::UInt8))
}
