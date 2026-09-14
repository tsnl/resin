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
    let typer = TyperContext::from_definitions(module.types.clone());
    for (index, function) in module.functions.iter().enumerate() {
        crate::profile::function(&typer, FunctionId::from_index(index), function)
            .map_err(crate::profile::Error::verify)?;
        if function.profile == crate::Profile::Shader {
            crate::profile::stack_types(&typer, FunctionId::from_index(index), &functions[index])
                .map_err(crate::profile::Error::verify)?;
        }
    }
    let shaders = crate::profile::shaders(module).map_err(crate::profile::Error::verify)?;
    Ok(ModuleTypes {
        functions,
        types,
        shaders,
    })
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
        | Instr::ArcSpanData
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
        Instr::GpuDraw { .. } | Instr::PointerRange => StackEffect { pops: 4, pushes: 1 },
        Instr::GpuNew { .. }
        | Instr::GpuAllocate { .. }
        | Instr::GpuCopyTo
        | Instr::PointerBytes
        | Instr::ArcSpanTryNew { .. }
        | Instr::HostAllocate { .. } => StackEffect { pops: 2, pushes: 1 },
        Instr::PointerIndex | Instr::GpuSlice | Instr::GpuArgumentsDraw | Instr::GpuCopyImage => {
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
        Instr::Call { arguments } => StackEffect {
            pops: arguments.saturating_add(1),
            pushes: 1,
        },
        Instr::CallBuiltin { params, .. } => StackEffect {
            pops: params.len(),
            pushes: 1,
        },
    }
}

#[cfg(test)]
mod tests;
