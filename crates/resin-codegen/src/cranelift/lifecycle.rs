//! Ownership operations have a separate lifetime from storage snapshots. Copying
//! bits transfers a value; these helpers explicitly acquire or release ownership.
use super::{
    failure,
    function::{Body, Operand},
    types::Types,
};
use crate::{Error, GenerationError};
use cranelift_codegen::ir::{self, InstBuilder, condcodes::IntCC};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Module};
use cranelift_object::ObjectModule;
use resin_executor::Cancellation;
use resin_types::prelude::*;

pub(super) fn define(
    module: &mut ObjectModule,
    types: &Types<'_>,
    entry: FunctionId,
    functions: &[Option<FuncId>],
    cancellation: &Cancellation,
) -> Result<(), GenerationError> {
    for (ty, lifecycle) in &types.lifecycle {
        for (id, retaining) in [(lifecycle.retain, true), (lifecycle.drop, false)] {
            cancellation.check()?;
            let mut context = module.make_context();
            context
                .func
                .signature
                .params
                .push(ir::AbiParam::new(ir::types::I64));
            let mut frontend = FunctionBuilderContext::new();
            let builder = FunctionBuilder::new(&mut context.func, &mut frontend);
            let mut body = Body::new(
                builder,
                module,
                types,
                functions,
                &types.module.functions[entry.index()],
            );
            let block = body.builder.create_block();
            body.builder.append_block_params_for_function_params(block);
            body.builder.switch_to_block(block);
            let address = body.builder.block_params(block)[0];
            body.visit_ownership(ty, address, retaining)?;
            body.builder.ins().return_(&[]);
            body.builder.seal_all_blocks();
            let config = body.module.target_config();
            body.builder.finalize(config);
            module.define_function(id, &mut context).map_err(failure)?;
        }
    }
    Ok(())
}

impl Body<'_, '_> {
    pub fn mark_local(&mut self, local: LocalId, initialized: bool) {
        if let Some(address) = self.local_live[local.index()] {
            let value = self
                .builder
                .ins()
                .iconst(ir::types::I8, i64::from(initialized));
            self.builder
                .ins()
                .store(ir::MemFlagsData::new(), value, address, 0);
        }
    }

    pub fn mark_address(
        &mut self,
        live: Option<ir::Value>,
        initialized: bool,
    ) -> Result<(), Error> {
        if let Some(address) = live {
            self.when(address, |body| {
                let value = body
                    .builder
                    .ins()
                    .iconst(ir::types::I8, i64::from(initialized));
                body.builder
                    .ins()
                    .store(ir::MemFlagsData::new(), value, address, 0);
                Ok(())
            })?;
        }
        Ok(())
    }

    pub fn drop_local(&mut self, local: LocalId) -> Result<(), Error> {
        if let Some(flag) = self.local_live[local.index()] {
            let initialized =
                self.builder
                    .ins()
                    .load(ir::types::I8, ir::MemFlagsData::new(), flag, 0);
            self.when(initialized, |body| {
                body.destroy_storage(
                    &body.input.locals[local.index()].ty,
                    body.locals[local.index()],
                )
            })?;
            self.mark_local(local, false);
        }
        Ok(())
    }

    pub fn drop_stored(
        &mut self,
        ty: &Ty,
        address: ir::Value,
        live: Option<ir::Value>,
    ) -> Result<(), Error> {
        if !ty.needs_drop(self.types.table) {
            return Ok(());
        }
        let Some(flag) = live else {
            return self.destroy_storage(ty, address);
        };
        let known = self.builder.create_block();
        let unknown = self.builder.create_block();
        let ready = self.builder.create_block();
        self.builder.append_block_param(ready, ir::types::I8);
        self.builder.ins().brif(flag, known, &[], unknown, &[]);
        self.builder.switch_to_block(known);
        let initialized = self
            .builder
            .ins()
            .load(ir::types::I8, ir::MemFlagsData::new(), flag, 0);
        self.builder
            .ins()
            .jump(ready, &[ir::BlockArg::Value(initialized)]);
        self.builder.switch_to_block(unknown);
        let initialized = self.builder.ins().iconst(ir::types::I8, 1);
        self.builder
            .ins()
            .jump(ready, &[ir::BlockArg::Value(initialized)]);
        self.builder.switch_to_block(ready);
        let initialized = self.builder.block_params(ready)[0];
        self.when(initialized, |body| body.destroy_storage(ty, address))
    }

    fn destroy_storage(&mut self, ty: &Ty, address: ir::Value) -> Result<(), Error> {
        let value = if let Some(scalar) = self.types.scalar(ty) {
            self.builder
                .ins()
                .load(scalar, ir::MemFlagsData::new(), address, 0)
        } else {
            address
        };
        self.drop_value(ty, value)
    }

    pub fn retain(&mut self, ty: &Ty, value: ir::Value) -> Result<(), Error> {
        self.lifecycle_call(ty, value, true)
    }

    pub fn drop_value(&mut self, ty: &Ty, value: ir::Value) -> Result<(), Error> {
        self.lifecycle_call(ty, value, false)
    }

    fn lifecycle_call(&mut self, ty: &Ty, value: ir::Value, retaining: bool) -> Result<(), Error> {
        let Some(lifecycle) = self.types.lifecycle.get(ty) else {
            return Ok(());
        };
        let function = if retaining {
            lifecycle.retain
        } else {
            lifecycle.drop
        };
        let address = if self.types.scalar(ty).is_some() {
            let address = self.allocate(ty);
            self.write(ty, address, value);
            address
        } else {
            value
        };
        let reference = self
            .module
            .declare_func_in_func(function, self.builder.func);
        self.builder.ins().call(reference, &[address]);
        Ok(())
    }

    fn visit_ownership(
        &mut self,
        ty: &Ty,
        address: ir::Value,
        retaining: bool,
    ) -> Result<(), Error> {
        match ty {
            Ty::StrongOwner | Ty::GpuArguments | Ty::GpuView | Ty::GpuPipelineContract => {
                let owner =
                    self.builder
                        .ins()
                        .load(ir::types::I64, ir::MemFlagsData::new(), address, 0);
                self.runtime_call(
                    if retaining {
                        "resin_arc_retain"
                    } else {
                        "resin_arc_release"
                    },
                    &[(ir::types::I64, owner)],
                    None,
                )?;
            }
            Ty::WeakOwner => {
                let owner =
                    self.builder
                        .ins()
                        .load(ir::types::I64, ir::MemFlagsData::new(), address, 0);
                self.runtime_call(
                    if retaining {
                        "resin_weak_retain"
                    } else {
                        "resin_weak_release"
                    },
                    &[(ir::types::I64, owner)],
                    None,
                )?;
            }
            Ty::Defined { definition } => {
                let definition = &self.types.table[definition.index()];
                if !retaining && let Some(function) = definition.drop_hook() {
                    let reference = self.module.declare_func_in_func(
                        self.functions[function.index()].unwrap(),
                        self.builder.func,
                    );
                    self.builder.ins().call(reference, &[address]);
                }
                self.visit_child(definition.body().unwrap(), address, retaining)?;
            }
            Ty::Record { fields } => {
                for position in 0..fields.len() {
                    let index = if retaining {
                        position
                    } else {
                        fields.len() - position - 1
                    };
                    let child = self.offset(address, self.types.layout(ty).offsets[index]);
                    self.visit_child(&fields[index].ty, child, retaining)?;
                }
            }
            Ty::Array { element, length } if element.needs_drop(self.types.table) => {
                let count = self.builder.ins().iconst(ir::types::I64, *length as i64);
                self.each(count, |body, index| {
                    let index = if retaining {
                        index
                    } else {
                        let end = body.builder.ins().iadd_imm_s(count, -1);
                        body.builder.ins().isub(end, index)
                    };
                    let child = body.pointer_offset(address, index, element);
                    body.visit_child(element, child, retaining)
                })?;
            }
            Ty::Union { .. } | Ty::Result { .. } => {
                let tag =
                    self.builder
                        .ins()
                        .load(ir::types::I32, ir::MemFlagsData::new(), address, 0);
                for (case, payload) in ty.payloads().unwrap() {
                    if !payload.needs_drop(self.types.table) {
                        continue;
                    }
                    let selected = self.builder.ins().icmp_imm_u(
                        IntCC::Equal,
                        tag,
                        i64::from(case.tag(self.types.table)),
                    );
                    self.when(selected, |body| {
                        let child = body.offset(address, body.types.layout(ty).offsets[1]);
                        body.visit_child(&payload, child, retaining)
                    })?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn visit_child(&mut self, ty: &Ty, address: ir::Value, retaining: bool) -> Result<(), Error> {
        if !ty.needs_drop(self.types.table) {
            return Ok(());
        }
        let value = if let Some(scalar) = self.types.scalar(ty) {
            self.builder
                .ins()
                .load(scalar, ir::MemFlagsData::new(), address, 0)
        } else {
            address
        };
        self.lifecycle_call(ty, value, retaining)
    }

    pub fn when(
        &mut self,
        condition: ir::Value,
        emit: impl FnOnce(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let selected = self.builder.create_block();
        let merge = self.builder.create_block();
        self.builder
            .ins()
            .brif(condition, selected, &[], merge, &[]);
        self.builder.switch_to_block(selected);
        emit(self)?;
        self.builder.ins().jump(merge, &[]);
        self.builder.switch_to_block(merge);
        Ok(())
    }

    pub fn each(
        &mut self,
        count: ir::Value,
        mut emit: impl FnMut(&mut Self, ir::Value) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let condition = self.builder.create_block();
        let body = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.append_block_param(condition, ir::types::I64);
        let zero = self.builder.ins().iconst(ir::types::I64, 0);
        self.builder
            .ins()
            .jump(condition, &[ir::BlockArg::Value(zero)]);
        self.builder.switch_to_block(condition);
        let index = self.builder.block_params(condition)[0];
        let more = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, index, count);
        self.builder.ins().brif(more, body, &[], done, &[]);
        self.builder.switch_to_block(body);
        emit(self, index)?;
        let next = self.builder.ins().iadd_imm_s(index, 1);
        self.builder
            .ins()
            .jump(condition, &[ir::BlockArg::Value(next)]);
        self.builder.switch_to_block(done);
        Ok(())
    }

    pub fn owner_allocate(
        &mut self,
        element: &Ty,
        args: &[Operand],
        result: &Ty,
    ) -> Result<ir::Value, Error> {
        let layout = self.types.layout(element);
        let stride = self
            .builder
            .ins()
            .iconst(ir::types::I64, layout.size as i64);
        let align = self
            .builder
            .ins()
            .iconst(ir::types::I64, layout.align as i64);
        let destroy = if let Some(lifecycle) = self.types.lifecycle.get(element) {
            let function = self
                .module
                .declare_func_in_func(lifecycle.drop, self.builder.func);
            self.builder.ins().func_addr(ir::types::I64, function)
        } else {
            self.builder.ins().iconst(ir::types::I64, 0)
        };
        let owner = self
            .runtime_call(
                "resin_arc_span_try_new",
                &[
                    (ir::types::I64, args[0].value),
                    (ir::types::I64, stride),
                    (ir::types::I64, align),
                    (ir::types::I64, destroy),
                ],
                Some(ir::types::I64),
            )?
            .unwrap();
        self.when(owner, |body| {
            let data = body
                .runtime_call(
                    "resin_arc_data",
                    &[(ir::types::I64, owner)],
                    Some(ir::types::I64),
                )?
                .unwrap();
            body.each(args[0].value, |body, index| {
                let destination = body.pointer_offset(data, index, element);
                body.write(element, destination, args[1].value);
                body.retain(element, args[1].value)
            })
        })?;
        self.optional_owner(result, owner)
    }

    pub fn optional_owner(&mut self, ty: &Ty, owner: ir::Value) -> Result<ir::Value, Error> {
        let destination = self.allocate(ty);
        let present = self.builder.create_block();
        let absent = self.builder.create_block();
        let merge = self.builder.create_block();
        self.builder.ins().brif(owner, present, &[], absent, &[]);
        self.builder.switch_to_block(present);
        let payload = ty.without_none().unwrap();
        let some = self.variant(
            ty,
            &Case::Type(payload.clone()),
            &Operand {
                ty: payload,
                value: owner,
                function: None,
                live: None,
            },
        );
        self.write(ty, destination, some);
        self.builder.ins().jump(merge, &[]);
        self.builder.switch_to_block(absent);
        let unit = self.builder.ins().iconst(ir::types::I8, 0);
        let none = self.variant(
            ty,
            &Case::Type(Ty::None),
            &Operand {
                ty: Ty::None,
                value: unit,
                function: None,
                live: None,
            },
        );
        self.write(ty, destination, none);
        self.builder.ins().jump(merge, &[]);
        self.builder.switch_to_block(merge);
        Ok(destination)
    }

    pub fn owner_operation(
        &mut self,
        instruction: &resin_lir::Instr,
        args: &[Operand],
        result: &Ty,
    ) -> Result<ir::Value, Error> {
        let handle =
            self.builder
                .ins()
                .load(ir::types::I64, ir::MemFlagsData::new(), args[0].value, 0);
        let name = match instruction {
            resin_lir::Instr::OwnerData { .. } => "resin_arc_data",
            resin_lir::Instr::OwnerLength => "resin_arc_span_length",
            resin_lir::Instr::OwnerDowngrade => "resin_weak_retain",
            resin_lir::Instr::OwnerUpgrade => "resin_weak_upgrade",
            _ => unreachable!(),
        };
        let output = if matches!(instruction, resin_lir::Instr::OwnerDowngrade) {
            None
        } else {
            Some(ir::types::I64)
        };
        let value = self
            .runtime_call(name, &[(ir::types::I64, handle)], output)?
            .unwrap_or(handle);
        if matches!(instruction, resin_lir::Instr::OwnerUpgrade) {
            self.optional_owner(result, value)
        } else {
            Ok(value)
        }
    }
}
