//! Apply completed host-to-shader projection plans into runtime-owned storage.
use super::function::{Body, Operand};
use crate::Error;
use cranelift_codegen::ir::{
    self, InstBuilder,
    condcodes::IntCC,
    types::{I32, I64},
};
use resin_types::{GpuProjectionOperation, GpuProjectionPlan, prelude::*};

pub(super) struct Projected {
    pub tag: ir::Value,
    pub owner: ir::Value,
    pub error: ir::Value,
}

impl Body<'_, '_> {
    pub(super) fn gpu_project(
        &mut self,
        allocator: FunctionId,
        plan: &GpuProjectionPlan,
        gpu: &Operand,
        source: &Operand,
        error: &Ty,
    ) -> Result<Projected, Error> {
        let copied_gpu = self.gpu_copy(gpu)?;
        let bytes = self.gpu_word(self.types.layout(&plan.target).size);
        let alignment = self.gpu_word(self.types.layout(&plan.target).align);
        let memory = self.builder.ins().iconst(I32, 0);
        let allocation = self.gpu_call(allocator, &[copied_gpu, bytes, alignment, memory])?;
        let tag = self
            .builder
            .ins()
            .load(I32, ir::MemFlagsData::new(), allocation.value, 0);
        let ok = self.builder.create_block();
        let err = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.append_block_param(done, I64);
        self.builder
            .append_block_param(done, self.types.value_type(error));
        self.builder.ins().brif(tag, err, &[], ok, &[]);
        self.builder.switch_to_block(ok);
        let view = self.payload(&allocation, &Case::Ok, &Ty::GpuView)?;
        let zero = self.gpu_word(0);
        let root_view = self.gpu_offset(view, zero, bytes, alignment)?;
        self.gpu_host(root_view, &plan.target, 3)?;
        let owner = self
            .runtime_call("resin_gpu_projection_new_ref", &[(I64, view)], Some(I64))?
            .unwrap();
        self.drop_value(&Ty::GpuView, view)?;
        let root = self
            .runtime_call("resin_gpu_projection_root", &[(I64, owner)], Some(I64))?
            .unwrap();
        // Projection recursively borrows the source; runtime projection calls retain
        // every GPU owner they place in the completed argument set.
        self.gpu_project_value(plan, source.value, root, owner)?;
        let empty_error = self.allocate(error);
        let empty_error = self.read(error, empty_error);
        self.builder.ins().jump(
            done,
            &[ir::BlockArg::Value(owner), ir::BlockArg::Value(empty_error)],
        );
        self.builder.switch_to_block(err);
        let error_value = self.payload(&allocation, &Case::Err, error)?;
        let no_owner = self.gpu_word(0);
        self.builder.ins().jump(
            done,
            &[
                ir::BlockArg::Value(no_owner),
                ir::BlockArg::Value(error_value),
            ],
        );
        self.builder.switch_to_block(done);
        Ok(Projected {
            tag,
            owner: self.builder.block_params(done)[0],
            error: self.builder.block_params(done)[1],
        })
    }

    fn gpu_project_value(
        &mut self,
        plan: &GpuProjectionPlan,
        source: ir::Value,
        destination: ir::Value,
        owner: ir::Value,
    ) -> Result<(), Error> {
        match &plan.operation {
            GpuProjectionOperation::Copy => self.write(&plan.target, destination, source),
            GpuProjectionOperation::Pointer { element } => {
                let bytes = self.gpu_word(self.types.layout(element).size);
                let alignment = self.gpu_word(self.types.layout(element).align);
                let pointer = self
                    .runtime_call(
                        "resin_gpu_projection_pointer_ref",
                        &[(I64, owner), (I64, source), (I64, bytes), (I64, alignment)],
                        Some(I64),
                    )?
                    .unwrap();
                self.builder
                    .ins()
                    .store(ir::MemFlagsData::new(), pointer, destination, 0);
            }
            GpuProjectionOperation::Sequence { element } => {
                let source_type = self.types.shape(&plan.source);
                let count_address = self.offset(source, self.types.layout(source_type).offsets[1]);
                let count = self
                    .builder
                    .ins()
                    .load(I64, ir::MemFlagsData::new(), count_address, 0);
                let size = self.types.layout(element).size as u64;
                let invalid = self.builder.ins().icmp_imm_u(
                    IntCC::UnsignedGreaterThan,
                    count,
                    (u64::MAX / size) as i64,
                );
                self.guard(invalid, "GPU projection length overflow")?;
                let bytes = self.builder.ins().imul_imm_s(count, size as i64);
                let alignment = self.gpu_word(self.types.layout(element).align);
                let pointer = self
                    .runtime_call(
                        "resin_gpu_projection_pointer_ref",
                        &[(I64, owner), (I64, source), (I64, bytes), (I64, alignment)],
                        Some(I64),
                    )?
                    .unwrap();
                let offsets = self
                    .types
                    .layout(self.types.shape(&plan.target))
                    .offsets
                    .clone();
                let data = self.offset(destination, offsets[0]);
                let length = self.offset(destination, offsets[1]);
                self.builder
                    .ins()
                    .store(ir::MemFlagsData::new(), pointer, data, 0);
                self.builder
                    .ins()
                    .store(ir::MemFlagsData::new(), count, length, 0);
            }
            GpuProjectionOperation::Record { fields } => {
                let source_offsets = self
                    .types
                    .layout(self.types.shape(&plan.source))
                    .offsets
                    .clone();
                let target_offsets = self
                    .types
                    .layout(self.types.shape(&plan.target))
                    .offsets
                    .clone();
                for (index, field) in fields.iter().enumerate() {
                    let address = self.offset(source, source_offsets[index]);
                    let source = self.read(&field.source, address);
                    let destination = self.offset(destination, target_offsets[index]);
                    self.gpu_project_value(field, source, destination, owner)?;
                }
            }
            GpuProjectionOperation::Array { element, length } => {
                let count = self.gpu_word(*length);
                self.each(count, |body, index| {
                    let source = body.pointer_offset(source, index, &element.source);
                    let source = body.read(&element.source, source);
                    let destination = body.pointer_offset(destination, index, &element.target);
                    body.gpu_project_value(element, source, destination, owner)
                })?;
            }
        }
        Ok(())
    }
}
