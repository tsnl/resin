//! Compute interface and shared variables. No workgroup objects enter the Resin ABI.
use super::{Context, build_error};
use crate::Error;
use resin_types::prelude::*;
use rspirv::{dr::Operand, spirv::*};

pub(super) struct Interface {
    pub width: Word,
    pub group: Word,
    pub lane: Word,
}

impl Interface {
    pub fn declare(context: &mut Context<'_>) -> Result<Self, Error> {
        let uint = context.ty(&Ty::UInt32)?;
        let width = context.builder.spec_constant_bit32(uint, 1);
        context
            .builder
            .decorate(width, Decoration::SpecId, [Operand::LiteralBit32(0)]);
        context.builder.name(width, "compute_workgroup_size");
        let vector = context.builder.type_vector(uint, 3);
        let pointer = context
            .builder
            .type_pointer(None, StorageClass::Input, vector);
        let mut builtin = |kind, name| {
            let id = context
                .builder
                .variable(pointer, None, StorageClass::Input, None);
            context
                .builder
                .decorate(id, Decoration::BuiltIn, [Operand::BuiltIn(kind)]);
            context.builder.name(id, name);
            id
        };
        let group = builtin(BuiltIn::WorkgroupId, "workgroup_id");
        let lane = builtin(BuiltIn::LocalInvocationId, "local_invocation_id");
        Ok(Self { width, group, lane })
    }
}

impl Context<'_> {
    pub(super) fn local_storage(&self, root: Word) -> StorageClass {
        if self.shared.contains(&root) {
            StorageClass::Workgroup
        } else {
            StorageClass::Function
        }
    }

    pub(super) fn shared_variable(&mut self, ty: &Ty) -> Result<Word, Error> {
        // The first schedule reserves all group locals. Use a conservative
        // eight-byte scalar slot (including padding) and Vulkan's portable 16 KiB
        // floor, instead of relying on driver rejection for excessive scratch.
        let bytes = self
            .shared_size(ty)
            .and_then(|size| self.shared_bytes.checked_add(size));
        self.shared_bytes = bytes.filter(|size| *size <= 16384).ok_or_else(|| Error::unsupported("compute workgroup storage exceeds the prototype's portable 16 KiB budget; use a smaller batch".into()))?;
        let pointer = self.pointer_type(StorageClass::Workgroup, ty)?;
        let function = self.builder.selected_function();
        let block = self.builder.selected_block();
        self.builder.select_function(None).map_err(build_error)?;
        let variable = self
            .builder
            .variable(pointer, None, StorageClass::Workgroup, None);
        self.builder
            .select_function(function)
            .map_err(build_error)?;
        self.builder.select_block(block).map_err(build_error)?;
        self.shared.push(variable);
        Ok(variable)
    }

    fn shared_size(&self, ty: &Ty) -> Option<usize> {
        match self.shape(ty) {
            Ty::Array { element, length } => {
                self.shared_size(element)?.checked_mul((*length).max(1))
            }
            Ty::Record { fields } => fields
                .iter()
                .try_fold(0usize, |size, field| {
                    size.checked_add(self.shared_size(&field.ty)?)
                })
                .map(|size| size.max(8)),
            Ty::Union { .. } => ty.payloads()?.iter().try_fold(8usize, |size, (_, ty)| {
                size.checked_add(self.shared_size(ty)?)
            }),
            _ => Some(8),
        }
    }

    pub(super) fn barrier(&mut self) -> Result<(), Error> {
        let scope = self.constant_u32(Scope::Workgroup as u32);
        // Publish both compiler scratch and explicit writes through device pointers.
        let semantics = self.constant_u32(
            (MemorySemantics::ACQUIRE_RELEASE
                | MemorySemantics::WORKGROUP_MEMORY
                | MemorySemantics::UNIFORM_MEMORY)
                .bits(),
        );
        self.builder
            .control_barrier(scope, scope, semantics)
            .map_err(build_error)
    }
}
