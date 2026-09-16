//! Lower the verified formatting transport to the runtime's fixed C layout.

use super::function::{Body, Operand};
use crate::Error;
use cranelift_codegen::ir::{self, InstBuilder};
use cranelift_module::Module;
use resin_types::prelude::*;

// ResinPrintArg is repr(C): u32 kind at 0, then an 8-aligned union at 8.
// Its largest member is { pointer, usize }, so each argument is 24 bytes.
// Both the Rust runtime and resin_runtime/print.h define this same 64-bit ABI.
const ARGUMENT_SIZE: usize = 24;
const ARGUMENT_ALIGNMENT: u8 = 3;

impl Body<'_, '_> {
    pub fn format_bytes(&mut self, args: &[Operand], result: &Ty) -> Result<ir::Value, Error> {
        let invalid = || Error("format_bytes expects a byte pointer, length and tuple".into());
        let [data, length, arguments] = args else {
            return Err(invalid());
        };
        if result != &Ty::StrongOwner {
            return Err(invalid());
        }
        let fields = match &arguments.ty {
            Ty::Unit => &[][..],
            Ty::Record { fields } => fields.as_slice(),
            _ => return Err(invalid()),
        };
        let pointer = self.print_arguments(fields.len())?;
        for (index, field) in fields.iter().enumerate() {
            if field.name.as_ref() != format!("_{index}") {
                return Err(invalid());
            }
            let address = self.offset(
                arguments.value,
                self.types.layout(&arguments.ty).offsets[index],
            );
            let value = self.read(&field.ty, address);
            let destination = self.offset(pointer, index * ARGUMENT_SIZE);
            self.print_argument(&field.ty, value, destination)?;
        }
        let count = self
            .builder
            .ins()
            .iconst(ir::types::I64, fields.len() as i64);
        Ok(self
            .runtime_call(
                "resin_format",
                &[
                    (ir::types::I64, data.value),
                    (ir::types::I64, length.value),
                    (ir::types::I64, pointer),
                    (ir::types::I64, count),
                ],
                Some(ir::types::I64),
            )?
            .unwrap())
    }

    pub fn copy_bytes(&mut self, args: &[Operand], result: &Ty) -> Result<ir::Value, Error> {
        let [data, length] = args else {
            return Err(Error("byte copy expects pointer and length".into()));
        };
        if result != &Ty::StrongOwner {
            return Err(Error("byte copy produces an owner".into()));
        }
        Ok(self
            .runtime_call(
                "resin_string_from_str",
                &[(ir::types::I64, data.value), (ir::types::I64, length.value)],
                Some(ir::types::I64),
            )?
            .unwrap())
    }

    fn print_arguments(&mut self, count: usize) -> Result<ir::Value, Error> {
        if count == 0 {
            return Ok(self.builder.ins().iconst(ir::types::I64, 0));
        }
        let bytes = count
            .checked_mul(ARGUMENT_SIZE)
            .and_then(|bytes| u32::try_from(bytes).ok())
            .ok_or_else(|| Error("format argument storage is too large".into()))?;
        let slot = self.builder.create_sized_stack_slot(ir::StackSlotData::new(
            ir::StackSlotKind::ExplicitSlot,
            bytes,
            ARGUMENT_ALIGNMENT,
        ));
        let address = self.builder.ins().stack_addr(ir::types::I64, slot, 0);
        self.builder.emit_small_memset(
            self.module.target_config(),
            address,
            0,
            u64::from(bytes),
            8,
            ir::MemFlagsData::new(),
        );
        Ok(address)
    }

    fn print_argument(
        &mut self,
        ty: &Ty,
        value: ir::Value,
        destination: ir::Value,
    ) -> Result<(), Error> {
        let (kind, value) = match ty {
            Ty::Unit => (0, self.builder.ins().iconst(ir::types::I64, 0)),
            Ty::Bool => (1, self.builder.ins().uextend(ir::types::I64, value)),
            Ty::Int8 | Ty::Int16 | Ty::Int32 => {
                (2, self.builder.ins().sextend(ir::types::I64, value))
            }
            Ty::Int64 => (2, value),
            Ty::UInt8 | Ty::UInt16 | Ty::UInt32 => {
                (3, self.builder.ins().uextend(ir::types::I64, value))
            }
            Ty::UInt64 => (3, value),
            Ty::Float32 => (4, self.builder.ins().fpromote(ir::types::F64, value)),
            Ty::Float64 => (5, value),
            Ty::Str | Ty::Record { .. } => {
                let offsets = &self.types.layout(ty).offsets;
                let data = self.builder.ins().load(
                    ir::types::I64,
                    ir::MemFlagsData::new(),
                    value,
                    offsets[0] as i32,
                );
                let length = self.builder.ins().load(
                    ir::types::I64,
                    ir::MemFlagsData::new(),
                    value,
                    offsets[1] as i32,
                );
                self.builder
                    .ins()
                    .store(ir::MemFlagsData::new(), length, destination, 16);
                (6, data)
            }
            Ty::Pointer { .. } => (7, value),
            _ => return Err(Error(format!("cannot format {ty:?}"))),
        };
        let kind = self.builder.ins().iconst(ir::types::I32, kind);
        self.builder
            .ins()
            .store(ir::MemFlagsData::new(), kind, destination, 0);
        self.builder
            .ins()
            .store(ir::MemFlagsData::new(), value, destination, 8);
        Ok(())
    }
}
