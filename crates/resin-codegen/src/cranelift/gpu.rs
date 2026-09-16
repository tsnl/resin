//! Host GPU operations consume completed projection and pipeline contracts from LIR.
use super::function::{Body, Operand};
use crate::Error;
use cranelift_codegen::ir::{
    self, InstBuilder,
    condcodes::IntCC,
    types::{I32, I64},
};
use resin_lir::Instr;
use resin_types::prelude::*;

impl Body<'_, '_> {
    pub fn gpu_instruction(
        &mut self,
        instruction: &Instr,
        args: &[Operand],
        result: &Ty,
    ) -> Result<ir::Value, Error> {
        match instruction {
            Instr::GpuViewAllocate => self.gpu_allocate(args, result),
            Instr::GpuViewOffset => {
                self.gpu_offset(args[0].value, args[1].value, args[2].value, args[3].value)
            }
            Instr::GpuViewRange { element } => {
                let start = self.builder.ins().icmp(
                    IntCC::UnsignedGreaterThan,
                    args[2].value,
                    args[1].value,
                );
                let available = self.builder.ins().isub(args[1].value, args[2].value);
                let length =
                    self.builder
                        .ins()
                        .icmp(IntCC::UnsignedGreaterThan, args[3].value, available);
                let invalid = self.builder.ins().bor(start, length);
                self.guard(invalid, "GPU view out of bounds")?;
                let offset = self.gpu_bytes(element, args[2].value)?;
                let bytes = self.gpu_bytes(element, args[3].value)?;
                let alignment = self.gpu_word(self.types.layout(element).align);
                self.gpu_offset(args[0].value, offset, bytes, alignment)
            }
            Instr::GpuViewRestrict => {
                let value = self.read(&Ty::GpuView, args[0].value);
                let access = self
                    .builder
                    .ins()
                    .load(I32, ir::MemFlagsData::new(), value, 16);
                let restricted = self.builder.ins().band(access, args[1].value);
                self.builder
                    .ins()
                    .store(ir::MemFlagsData::new(), restricted, value, 16);
                Ok(value)
            }
            Instr::GpuViewLoad { element } => {
                let address = self.gpu_host(args[0].value, element, 1)?;
                Ok(self.read(element, address))
            }
            Instr::GpuViewStore | Instr::GpuViewReplace => {
                let replace = matches!(instruction, Instr::GpuViewReplace);
                let address =
                    self.gpu_host(args[0].value, &args[1].ty, if replace { 3 } else { 2 })?;
                let old = replace.then(|| self.read(&args[1].ty, address));
                self.write(&args[1].ty, address, args[1].value);
                Ok(old.unwrap_or_else(|| self.builder.ins().iconst(ir::types::I8, 0)))
            }
            Instr::GpuViewCopyTo => self.gpu_copy_to(args),
            Instr::GpuViewCopyImage => {
                // ResinGpuSpan is a 24-byte view followed by its 64-bit length.
                let slot = self.builder.create_sized_stack_slot(ir::StackSlotData::new(
                    ir::StackSlotKind::ExplicitSlot,
                    32,
                    3,
                ));
                let span = self.builder.ins().stack_addr(I64, slot, 0);
                self.copy_storage(&Ty::GpuView, span, args[0].value);
                self.builder
                    .ins()
                    .store(ir::MemFlagsData::new(), args[1].value, span, 24);
                Ok(self
                    .runtime_call(
                        "resin_gpu_copy_image_to_span_ref",
                        &[(I64, args[2].value), (I64, args[3].value), (I64, span)],
                        Some(I32),
                    )?
                    .unwrap())
            }
            Instr::GpuArgumentsDispatch | Instr::GpuArgumentsDraw => {
                let mut values = vec![(I64, args[1].value), (I64, args[0].value)];
                values.extend(args[2..].iter().map(|arg| (I32, arg.value)));
                let name = if matches!(instruction, Instr::GpuArgumentsDispatch) {
                    "resin_gpu_projected_dispatch"
                } else {
                    "resin_gpu_projected_draw"
                };
                Ok(self.runtime_call(name, &values, Some(I32))?.unwrap())
            }
            Instr::GpuComputePipeline {
                factory, shader, ..
            } => self.gpu_create_pipeline(*factory, &[*shader], &args[0], result),
            Instr::GpuGraphicsPipeline {
                factory,
                vertex,
                fragment,
                ..
            } => self.gpu_create_pipeline(*factory, &[*vertex, *fragment], &args[0], result),
            Instr::GpuDispatch {
                context,
                allocator,
                record,
                projection,
            } => self.gpu_record(
                (*context, Some(*allocator), *record),
                Some(projection),
                args,
                result,
            ),
            Instr::GpuDraw {
                context,
                allocator,
                record,
                projection,
            } => self.gpu_record(
                (*context, *allocator, *record),
                projection.as_ref(),
                args,
                result,
            ),
            _ => unreachable!("GPU instruction dispatch"),
        }
    }

    pub(super) fn gpu_word(&mut self, value: usize) -> ir::Value {
        self.builder.ins().iconst(I64, value as i64)
    }

    pub(super) fn gpu_bytes(&mut self, element: &Ty, count: ir::Value) -> Result<ir::Value, Error> {
        let stride = self.types.layout(element).size as u64;
        let invalid = self.builder.ins().icmp_imm_u(
            IntCC::UnsignedGreaterThan,
            count,
            (u64::MAX / stride) as i64,
        );
        self.guard(invalid, "GPU allocation or view size overflow")?;
        Ok(self.builder.ins().imul_imm_s(count, stride as i64))
    }

    pub(super) fn gpu_offset(
        &mut self,
        view: ir::Value,
        offset: ir::Value,
        bytes: ir::Value,
        alignment: ir::Value,
    ) -> Result<ir::Value, Error> {
        let output = self.allocate(&Ty::GpuView);
        self.runtime_call(
            "resin_gpu_ptr_offset_into",
            &[
                (I64, view),
                (I64, offset),
                (I64, bytes),
                (I64, alignment),
                (I64, output),
            ],
            None,
        )?;
        Ok(output)
    }

    pub(super) fn gpu_host(
        &mut self,
        view: ir::Value,
        ty: &Ty,
        access: i64,
    ) -> Result<ir::Value, Error> {
        let bytes = self.gpu_word(self.types.layout(ty).size);
        let alignment = self.gpu_word(self.types.layout(ty).align);
        let access = self.builder.ins().iconst(I32, access);
        Ok(self
            .runtime_call(
                "resin_gpu_ptr_host_ref",
                &[(I64, view), (I64, bytes), (I64, alignment), (I32, access)],
                Some(I64),
            )?
            .unwrap())
    }

    fn gpu_allocate(&mut self, args: &[Operand], result: &Ty) -> Result<ir::Value, Error> {
        let output = self.allocate(&Ty::GpuView);
        let status = self
            .runtime_call(
                "resin_gpu_ptr_allocate",
                &[
                    (I64, args[0].value),
                    (I64, args[1].value),
                    (I64, args[2].value),
                    (I64, args[3].value),
                    (I32, args[4].value),
                    (I64, output),
                ],
                Some(I32),
            )?
            .unwrap();
        let Ty::Record { fields } = result else {
            unreachable!("verified allocation result")
        };
        let union = self.allocate(&fields[0].ty);
        let good = self.builder.ins().iconst(
            I32,
            i64::from(Case::Type(Ty::GpuView).tag(self.types.table)),
        );
        let bad = self
            .builder
            .ins()
            .iconst(I32, i64::from(Case::Type(Ty::None).tag(self.types.table)));
        let successful = self.builder.ins().icmp_imm_s(IntCC::Equal, status, 0);
        let tag = self.builder.ins().select(successful, good, bad);
        self.builder
            .ins()
            .store(ir::MemFlagsData::new(), tag, union, 0);
        let payload = self.offset(union, self.types.layout(&fields[0].ty).offsets[1]);
        self.write(&Ty::GpuView, payload, output);
        Ok(self.construct(
            &[
                Operand {
                    ty: fields[0].ty.clone(),
                    value: union,
                    function: None,
                    live: None,
                },
                Operand {
                    ty: Ty::Int32,
                    value: status,
                    function: None,
                    live: None,
                },
            ],
            result,
        ))
    }

    fn gpu_copy_to(&mut self, args: &[Operand]) -> Result<ir::Value, Error> {
        let Ty::Pointer { pointee } = &args[2].ty else {
            unreachable!("verified GPU copy")
        };
        let bytes = self.gpu_bytes(pointee, args[1].value)?;
        let invalid =
            self.builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, args[3].value, args[1].value);
        self.guard(invalid, "GPU copy destination is too short")?;
        let copy = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.ins().brif(bytes, copy, &[], done, &[]);
        self.builder.switch_to_block(copy);
        let alignment = self.gpu_word(self.types.layout(pointee).align);
        let access = self.builder.ins().iconst(I32, 1);
        let source = self
            .runtime_call(
                "resin_gpu_ptr_host_ref",
                &[
                    (I64, args[0].value),
                    (I64, bytes),
                    (I64, alignment),
                    (I32, access),
                ],
                Some(I64),
            )?
            .unwrap();
        self.runtime_call(
            "memmove",
            &[(I64, args[2].value), (I64, source), (I64, bytes)],
            Some(I64),
        )?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        Ok(self.builder.ins().iconst(ir::types::I8, 0))
    }
}
