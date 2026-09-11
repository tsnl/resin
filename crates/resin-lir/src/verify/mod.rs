//! Certify stack and control-flow invariants before target lowering.
use crate::{Module, ModuleTypes, VerifyError};
use flow::check_function;
use resin_types::prelude::*;

mod error;
mod flow;
mod gpu;
mod instructions;
mod module;
mod pipeline;
mod rules;
mod table;

pub(super) fn analyze(module: &Module) -> Result<ModuleTypes, VerifyError> {
    module::check(module)?;
    let functions = module
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| check_function(module, FunctionId::from_index(index), function))
        .collect::<Result<Vec<_>, _>>()?;
    let types = table::collect(module, &functions);
    Ok(ModuleTypes { functions, types })
}

//
// Stack effects shared by construction and verification
//

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StackEffect {
    pub(crate) pops: usize,
    pub(crate) pushes: usize,
}

pub(crate) fn stack_effect(instr: &crate::Instr) -> StackEffect {
    use crate::Instr;
    match instr {
        Instr::WeakEmpty { .. }
        | Instr::TakeLocal { .. }
        | Instr::Shader { .. }
        | Instr::Push { .. }
        | Instr::Function { .. }
        | Instr::LocalAddress { .. } => StackEffect { pops: 0, pushes: 1 },
        Instr::TransferLoad
        | Instr::ArcNew
        | Instr::ArcData
        | Instr::Downgrade
        | Instr::Upgrade
        | Instr::AccessStatic { .. }
        | Instr::MakeVariant { .. }
        | Instr::ExcludeNone
        | Instr::IsVariant { .. }
        | Instr::VariantPayload { .. }
        | Instr::Widen { .. }
        | Instr::Load
        | Instr::Ascribe { .. }
        | Instr::Eliminate { .. }
        | Instr::NumericCast { .. }
        | Instr::PointerCast { .. } => StackEffect { pops: 1, pushes: 1 },
        Instr::GpuReadOnly
        | Instr::GpuWriteOnly
        | Instr::GpuComputePipeline { .. }
        | Instr::GpuGraphicsPipeline { .. } => StackEffect { pops: 1, pushes: 1 },
        Instr::GpuDispatch { .. } => StackEffect { pops: 6, pushes: 1 },
        Instr::GpuDraw { .. } => StackEffect { pops: 4, pushes: 1 },
        Instr::GpuNew { .. } | Instr::GpuAllocate { .. } | Instr::GpuCopyTo => {
            StackEffect { pops: 2, pushes: 1 }
        }
        Instr::GpuSlice | Instr::GpuArgumentsDraw | Instr::GpuCopyImage => {
            StackEffect { pops: 3, pushes: 1 }
        }
        Instr::GpuAllocateNative | Instr::GpuArgumentsDispatch => {
            StackEffect { pops: 5, pushes: 1 }
        }
        Instr::AccessDynamic | Instr::Store | Instr::Replace => StackEffect { pops: 2, pushes: 1 },
        Instr::ForgetLocal { .. } | Instr::DropLocal { .. } => StackEffect { pops: 0, pushes: 0 },
        Instr::Discard | Instr::SetLocal { .. } => StackEffect { pops: 1, pushes: 0 },
        Instr::MakeRecord { fields } => StackEffect {
            pops: fields.len(),
            pushes: 1,
        },
        Instr::MakeArray { elements, .. } => StackEffect {
            pops: *elements,
            pushes: 1,
        },
        Instr::Call => StackEffect { pops: 2, pushes: 1 },
        Instr::CallBuiltin { params, .. } => StackEffect {
            pops: params.len(),
            pushes: 1,
        },
    }
}

#[cfg(test)]
mod tests;
