//! Plain value representation. Aggregate operands own snapshots, and projections
//! borrow storage only when their source operand is an explicit language pointer.
use super::function::{Body, Operand, scalar_literal};
use crate::Error;
use cranelift_codegen::ir::{self, InstBuilder, condcodes::IntCC};
use cranelift_module::Module;
use resin_types::prelude::*;

impl Body<'_, '_> {
    pub fn literal(&mut self, value: &Value, ty: &Ty) -> Result<ir::Value, Error> {
        match value {
            Value::Type { ty } => Ok(self.builder.ins().iconst(
                ir::types::I64,
                self.types.table.id(ty).unwrap().index() as i64,
            )),
            Value::Str { value } => {
                let data = self
                    .module
                    .declare_data_in_func(self.types.literal(value), self.builder.func);
                let pointer = self.builder.ins().symbol_value(ir::types::I64, data);
                let length = self
                    .builder
                    .ins()
                    .iconst(ir::types::I64, value.len() as i64);
                let address = self.allocate(ty);
                self.builder
                    .ins()
                    .store(ir::MemFlagsData::new(), pointer, address, 0);
                self.builder
                    .ins()
                    .store(ir::MemFlagsData::new(), length, address, 8);
                Ok(address)
            }
            Value::Record { value } => {
                let Ty::Record { fields } = self.types.shape(ty) else {
                    unreachable!()
                };
                let args = value
                    .fields
                    .iter()
                    .zip(fields)
                    .map(|(field, declaration)| {
                        Ok(Operand {
                            ty: declaration.ty.clone(),
                            value: self.literal(&field.value, &declaration.ty)?,
                            function: None,
                            live: None,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(self.construct(&args, ty))
            }
            Value::Array { value } => {
                let args = value
                    .elements
                    .iter()
                    .map(|element| {
                        Ok(Operand {
                            ty: value.element_ty.clone(),
                            value: self.literal(element, &value.element_ty)?,
                            function: None,
                            live: None,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(self.construct(&args, ty))
            }
            _ => scalar_literal(&mut self.builder, value),
        }
    }

    pub fn construct(&mut self, args: &[Operand], ty: &Ty) -> ir::Value {
        let address = self.allocate(ty);
        for (index, arg) in args.iter().enumerate() {
            let offset = if let Ty::Array { element, .. } = self.types.shape(ty) {
                index * self.types.layout(element).size
            } else {
                self.types.layout(ty).offsets[index]
            };
            let destination = self.offset(address, offset);
            self.write(&arg.ty, destination, arg.value);
        }
        address
    }

    pub fn access_static(&mut self, base: &Operand, index: usize, result: &Ty) -> ir::Value {
        let (ty, pointer) = self.projected_type(&base.ty);
        let offset = match ty {
            Ty::Array { element, .. } => index * self.types.layout(element).size,
            _ => self.types.layout(ty).offsets[index],
        };
        let address = self.offset(base.value, offset);
        if pointer {
            address
        } else {
            self.read(result, address)
        }
    }

    pub fn access_dynamic(&mut self, args: &[Operand], result: &Ty) -> Result<ir::Value, Error> {
        let (ty, pointer) = self.projected_type(&args[0].ty);
        let ty = ty.clone();
        let (data, length, stride) = match &ty {
            Ty::Array { element, length } => (
                args[0].value,
                self.builder.ins().iconst(ir::types::I64, *length as i64),
                self.types.layout(element).size,
            ),
            Ty::Str => {
                let data = self.builder.ins().load(
                    ir::types::I64,
                    ir::MemFlagsData::new(),
                    args[0].value,
                    0,
                );
                let length = self.builder.ins().load(
                    ir::types::I64,
                    ir::MemFlagsData::new(),
                    args[0].value,
                    8,
                );
                (data, length, 1)
            }
            _ => unreachable!("verified index"),
        };
        // Legacy array-call indexing accepts every integer width. Preserve signed
        // negatives while normalizing to the host address width before checking.
        let index = if self.types.scalar(&args[1].ty) == Some(ir::types::I64) {
            args[1].value
        } else if matches!(args[1].ty, Ty::Int8 | Ty::Int16 | Ty::Int32) {
            self.builder.ins().sextend(ir::types::I64, args[1].value)
        } else {
            self.builder.ins().uextend(ir::types::I64, args[1].value)
        };
        let invalid = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, index, length);
        self.guard(invalid, "array index out of bounds")?;
        let offset = self.builder.ins().imul_imm_s(index, stride as i64);
        let address = self.builder.ins().iadd(data, offset);
        Ok(if pointer || ty == Ty::Str {
            address
        } else {
            self.read(result, address)
        })
    }

    fn projected_type<'a>(&'a self, ty: &'a Ty) -> (&'a Ty, bool) {
        if let Ty::Pointer { pointee } = self.types.shape(ty) {
            (self.types.shape(pointee), true)
        } else {
            (self.types.shape(ty), false)
        }
    }

    pub fn pointer_bytes(&mut self, args: &[Operand], result: &Ty) -> Result<ir::Value, Error> {
        let Ty::Pointer { pointee } = self.types.shape(&args[0].ty) else {
            unreachable!()
        };
        let stride = self.types.layout(pointee).size as u64;
        let invalid = self.builder.ins().icmp_imm_u(
            IntCC::UnsignedGreaterThan,
            args[1].value,
            (u64::MAX / stride) as i64,
        );
        self.guard(invalid, "span byte length overflow")?;
        let bytes = self.builder.ins().imul_imm_s(args[1].value, stride as i64);
        let address = self.allocate(result);
        self.builder
            .ins()
            .store(ir::MemFlagsData::new(), args[0].value, address, 0);
        self.builder
            .ins()
            .store(ir::MemFlagsData::new(), bytes, address, 8);
        Ok(address)
    }

    pub fn variant(&mut self, ty: &Ty, tag: &Case, payload: &Operand) -> ir::Value {
        if matches!(tag, Case::Type(member) if member == ty) {
            return payload.value;
        }
        let address = self.allocate(ty);
        let code = self
            .builder
            .ins()
            .iconst(ir::types::I32, i64::from(tag.tag(self.types.table)));
        self.builder
            .ins()
            .store(ir::MemFlagsData::new(), code, address, 0);
        let destination = self.offset(address, self.types.layout(ty).offsets[1]);
        self.write(&payload.ty, destination, payload.value);
        address
    }

    pub fn is_variant(&mut self, value: &Operand, tag: &Case) -> ir::Value {
        let ty = if let Ty::Pointer { pointee } = &value.ty {
            pointee.as_ref()
        } else {
            &value.ty
        };
        if matches!(tag, Case::Type(member) if member == ty) {
            return self.builder.ins().iconst(ir::types::I8, 1);
        }
        let actual =
            self.builder
                .ins()
                .load(ir::types::I32, ir::MemFlagsData::new(), value.value, 0);
        self.builder
            .ins()
            .icmp_imm_u(IntCC::Equal, actual, i64::from(tag.tag(self.types.table)))
    }

    pub fn payload(
        &mut self,
        value: &Operand,
        tag: &Case,
        result: &Ty,
    ) -> Result<ir::Value, Error> {
        if matches!(tag, Case::Type(member) if member == &value.ty) {
            return Ok(value.value);
        }
        let valid = self.is_variant(value, tag);
        let invalid = self.builder.ins().icmp_imm_s(IntCC::Equal, valid, 0);
        self.guard(invalid, "invalid union tag")?;
        let address = self.offset(value.value, self.types.layout(&value.ty).offsets[1]);
        Ok(self.read(result, address))
    }

    pub fn widen(&mut self, value: &Operand, to: &Ty) -> Result<ir::Value, Error> {
        if value.ty == *to {
            return Ok(value.value);
        }
        if matches!(to, Ty::Union { variants } if variants.contains(&value.ty)) {
            return Ok(self.variant(to, &Case::Type(value.ty.clone()), value));
        }
        let destination = self.allocate(to);
        let merge = self.builder.create_block();
        for (case, payload) in value.ty.payloads().unwrap_or_default() {
            let target = if matches!(&case, Case::Type(member) if member == to) {
                Some(to.clone())
            } else {
                to.payload(&case)
            };
            let Some(target) = target else {
                continue;
            };
            let selected = self.builder.create_block();
            let next = self.builder.create_block();
            let condition = self.is_variant(value, &case);
            self.builder.ins().brif(condition, selected, &[], next, &[]);
            self.builder.switch_to_block(selected);
            let payload_value = self.payload(value, &case, &payload)?;
            let widened = self.widen(
                &Operand {
                    ty: payload,
                    value: payload_value,
                    function: None,
                    live: None,
                },
                &target,
            )?;
            let completed = self.variant(
                to,
                &case,
                &Operand {
                    ty: target,
                    value: widened,
                    function: None,
                    live: None,
                },
            );
            self.write(to, destination, completed);
            self.builder.ins().jump(merge, &[]);
            self.builder.switch_to_block(next);
        }
        self.builder.ins().trap(ir::TrapCode::unwrap_user(6));
        self.builder.switch_to_block(merge);
        Ok(self.read(to, destination))
    }
}
