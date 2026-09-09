//! Shader values retain Resin's aggregate structure. Device accesses use the
//! shared host/device layout, including byte arrays with a one-byte stride.
use crate::Error;
use resin_types::prelude::*;
use rspirv::{dr::Operand, spirv::*};

use super::{Context, build_error};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Representation {
    Value,
    Buffer,
}

impl Context<'_> {
    pub(super) fn shape<'a>(&'a self, mut ty: &'a Ty) -> &'a Ty {
        while let Ty::Defined { definition } = ty {
            ty = self.module.types[definition.index()].body().unwrap();
        }
        ty
    }

    pub(super) fn validate(&mut self, ty: &Ty) -> Result<(), Error> {
        if !self.validated.insert(ty.clone()) {
            return Ok(());
        }
        match ty {
            Ty::Unit
            | Ty::None
            | Ty::Bool
            | Ty::Int32
            | Ty::UInt8
            | Ty::UInt32
            | Ty::UInt64
            | Ty::Float32
            | Ty::Arc { .. }
            | Ty::Weak { .. } => {}
            Ty::Str => return Err(super::str_storage_error()),
            Ty::Pointer { pointee } => self.validate_buffer(pointee)?,
            Ty::Span { element } => self.validate_buffer(element)?,
            Ty::Array { element, length } => {
                if *length == 0 {
                    return Err(Error("shader arrays must not be empty".into()));
                }
                self.validate(element)?;
            }
            Ty::Record { fields } => {
                for field in fields {
                    self.validate(&field.ty)?;
                }
            }
            Ty::Defined { definition } => {
                self.validate(self.module.types[definition.index()].body().unwrap())?;
            }
            Ty::Union { .. } | Ty::Result { .. } => {
                for (_, payload) in ty.payloads().unwrap() {
                    self.validate(&payload)?;
                }
            }
            _ => {
                return Err(Error(format!(
                    "shader profile does not support type {ty:?}"
                )));
            }
        }
        Ok(())
    }

    fn validate_buffer(&mut self, ty: &Ty) -> Result<(), Error> {
        crate::layout::layout(self.module, ty)?;
        self.validate(ty)
    }

    pub(super) fn ty(&mut self, ty: &Ty) -> Result<Word, Error> {
        self.type_id(ty, Representation::Value)
    }

    fn type_id(&mut self, ty: &Ty, representation: Representation) -> Result<Word, Error> {
        let representation = if matches!(
            self.shape(ty),
            Ty::Record { .. }
                | Ty::Array { .. }
                | Ty::Span { .. }
                | Ty::Union { .. }
                | Ty::Result { .. }
        ) {
            representation
        } else {
            Representation::Value
        };
        let key = (ty.clone(), representation);
        if let Some(&id) = self.types.get(&key) {
            return Ok(id);
        }
        let id = match ty {
            Ty::Unit | Ty::None | Ty::UInt32 => self.builder.type_int(32, 0),
            Ty::Bool => self.builder.type_bool(),
            Ty::Int32 => self.builder.type_int(32, 1),
            Ty::UInt8 => {
                self.builder.capability(Capability::Int8);
                self.builder.capability(Capability::StorageBuffer8BitAccess);
                self.builder.type_int(8, 0)
            }
            Ty::UInt64 | Ty::Pointer { .. } | Ty::Arc { .. } | Ty::Weak { .. } => {
                self.builder.type_int(64, 0)
            }
            Ty::Float32 => self.builder.type_float(32, None),
            Ty::Defined { definition } => {
                let definition = &self.module.types[definition.index()];
                let body = definition.body().unwrap();
                let id = self.type_id(body, representation)?;
                if matches!(self.shape(body), Ty::Record { .. } | Ty::Array { .. })
                    && self.named_types.insert(id)
                    && let Some(name) = definition.name()
                {
                    let name = match representation {
                        Representation::Value => name.to_string(),
                        Representation::Buffer => format!("{name}.storage"),
                    };
                    self.builder.name(id, name);
                }
                id
            }
            Ty::Span { .. } => {
                let word = self.ty(&Ty::UInt64)?;
                self.structure(ty, vec![word, word], representation)?
            }
            Ty::Record { fields } => {
                let mut members = fields
                    .iter()
                    .map(|field| self.type_id(&field.ty, representation))
                    .collect::<Result<Vec<_>, _>>()?;
                if members.is_empty() {
                    members.push(self.ty(&Ty::UInt32)?);
                }
                self.structure(ty, members, representation)?
            }
            Ty::Union { .. } | Ty::Result { .. } => {
                let mut members = vec![self.ty(&Ty::UInt32)?];
                for (_, payload) in ty.payloads().unwrap() {
                    members.push(self.type_id(&payload, representation)?);
                }
                self.structure(ty, members, representation)?
            }
            Ty::Array { element, length } => {
                let element_type = self.type_id(element, representation)?;
                let count = self.constant_u32(
                    u32::try_from(*length)
                        .map_err(|_| Error("shader array is too large".into()))?,
                );
                // Explicit IDs avoid accidentally sharing decorations with another
                // structurally equal type that has a different storage contract.
                let id = self.builder.id();
                self.builder.type_array_id(Some(id), element_type, count);
                if representation == Representation::Buffer {
                    let layout = crate::layout::layout(self.module, element)?;
                    let stride = u32::try_from(layout.size)
                        .map_err(|_| Error("shader array stride is too large".into()))?;
                    self.builder.decorate(
                        id,
                        Decoration::ArrayStride,
                        [Operand::LiteralBit32(stride)],
                    );
                }
                id
            }
            Ty::Str => return Err(super::str_storage_error()),
            _ => {
                return Err(Error(format!(
                    "shader profile does not support type {ty:?}"
                )));
            }
        };
        self.types.insert(key, id);
        Ok(id)
    }

    fn structure(
        &mut self,
        ty: &Ty,
        members: Vec<Word>,
        representation: Representation,
    ) -> Result<Word, Error> {
        let id = self.builder.id();
        self.builder.type_struct_id(Some(id), members);
        if representation == Representation::Buffer {
            let layout = crate::layout::layout(self.module, ty)?;
            for (member, offset) in layout.offsets.into_iter().enumerate() {
                let offset = u32::try_from(offset)
                    .map_err(|_| Error("shader record offset is too large".into()))?;
                self.builder.member_decorate(
                    id,
                    member as u32,
                    Decoration::Offset,
                    [Operand::LiteralBit32(offset)],
                );
            }
        }
        if let Ty::Record { fields } = ty {
            for (member, field) in fields.iter().enumerate() {
                self.builder
                    .member_name(id, member as u32, field.name.as_ref());
            }
        }
        Ok(id)
    }

    pub(super) fn pointer_type(&mut self, storage: StorageClass, ty: &Ty) -> Result<Word, Error> {
        let value = self.ty(ty)?;
        Ok(self.builder.type_pointer(None, storage, value))
    }

    pub(super) fn zero(&mut self, ty: &Ty) -> Result<Word, Error> {
        let ty = self.ty(ty)?;
        Ok(self.builder.constant_null(ty))
    }

    pub(super) fn constant_u32(&mut self, value: u32) -> Word {
        if let Some(&id) = self.u32_constants.get(&value) {
            return id;
        }
        let ty = self.builder.type_int(32, 0);
        let id = self.builder.constant_bit32(ty, value);
        self.u32_constants.insert(value, id);
        id
    }

    pub(super) fn constant_u64(&mut self, value: u64) -> Word {
        if let Some(&id) = self.u64_constants.get(&value) {
            return id;
        }
        let ty = self.builder.type_int(64, 0);
        let id = self.builder.constant_bit64(ty, value);
        self.u64_constants.insert(value, id);
        id
    }

    pub(super) fn constant_bool(&mut self, value: bool) -> Word {
        let ty = self.builder.type_bool();
        if value {
            self.builder.constant_true(ty)
        } else {
            self.builder.constant_false(ty)
        }
    }

    pub(super) fn physical_load(&mut self, pointee: &Ty, address: Word) -> Result<Word, Error> {
        let layout = crate::layout::layout(self.module, pointee)?;
        let storage_type = self.type_id(pointee, Representation::Buffer)?;
        let pointer_type =
            self.builder
                .type_pointer(None, StorageClass::PhysicalStorageBuffer, storage_type);
        let pointer = self
            .builder
            .convert_u_to_ptr(pointer_type, None, address)
            .map_err(build_error)?;
        let value_type = self.ty(pointee)?;
        let value = self
            .builder
            .load(
                storage_type,
                None,
                pointer,
                Some(MemoryAccess::ALIGNED),
                [Operand::LiteralBit32(layout.align as u32)],
            )
            .map_err(build_error)?;
        self.copy_representation(storage_type, value_type, value)
    }

    pub(super) fn physical_store(
        &mut self,
        pointee: &Ty,
        address: Word,
        value: Word,
    ) -> Result<(), Error> {
        let layout = crate::layout::layout(self.module, pointee)?;
        let storage_type = self.type_id(pointee, Representation::Buffer)?;
        let value_type = self.ty(pointee)?;
        let value = self.copy_representation(value_type, storage_type, value)?;
        let pointer_type =
            self.builder
                .type_pointer(None, StorageClass::PhysicalStorageBuffer, storage_type);
        let pointer = self
            .builder
            .convert_u_to_ptr(pointer_type, None, address)
            .map_err(build_error)?;
        self.builder
            .store(
                pointer,
                value,
                Some(MemoryAccess::ALIGNED),
                [Operand::LiteralBit32(layout.align as u32)],
            )
            .map_err(build_error)
    }

    // Explicit Offset/ArrayStride decorations describe buffers and cannot occur
    // in Function storage. OpCopyLogical converts matching aggregate shapes
    // recursively without imposing a layout on ordinary shader values.
    fn copy_representation(&mut self, from: Word, to: Word, value: Word) -> Result<Word, Error> {
        if from == to {
            return Ok(value);
        }
        self.builder
            .copy_logical(to, None, value)
            .map_err(build_error)
    }
}
