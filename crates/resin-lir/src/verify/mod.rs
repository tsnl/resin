//! Certify stack and control-flow invariants before target lowering.
use crate::{Module, ModuleTypes, VerificationError, VerifyError};
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
    match analyze_cancellable(module, &resin_executor::Cancellation::new()) {
        Ok(analysis) => Ok(analysis),
        Err(VerificationError::Invalid { error }) => Err(error),
        Err(VerificationError::Execution { .. }) => unreachable!("private token is not cancelled"),
    }
}

pub(super) fn analyze_cancellable(
    module: &Module,
    cancellation: &resin_executor::Cancellation,
) -> Result<ModuleTypes, VerificationError> {
    cancellation.check()?;
    module::check(module)?;
    let mut functions = Vec::with_capacity(module.functions.len());
    for (index, function) in module.functions.iter().enumerate() {
        cancellation.check()?;
        functions.push(check_function(
            module,
            FunctionId::from_index(index),
            function,
        )?);
    }
    cancellation.check()?;
    let types = table::collect(module, &functions);
    let typer = TyperContext::from_definitions(module.types.clone());
    for (index, function) in module.functions.iter().enumerate() {
        cancellation.check()?;
        crate::profile::function(&typer, FunctionId::from_index(index), function)
            .map_err(crate::profile::Error::verify)?;
        if function.profile == crate::Profile::Shader {
            crate::profile::stack_types(&typer, FunctionId::from_index(index), &functions[index])
                .map_err(crate::profile::Error::verify)?;
        }
    }
    cancellation.check()?;
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
        Instr::GpuViewRange { .. } => StackEffect { pops: 4, pushes: 1 },
        Instr::GpuViewLoad { .. } => StackEffect { pops: 1, pushes: 1 },
        Instr::GpuViewRestrict | Instr::GpuViewStore | Instr::GpuViewReplace => {
            StackEffect { pops: 2, pushes: 1 }
        }
        Instr::GpuViewOffset
        | Instr::GpuViewCopyTo
        | Instr::GpuViewCopyFrom
        | Instr::GpuViewCopyImage => StackEffect { pops: 4, pushes: 1 },
        Instr::GpuViewAllocate => StackEffect { pops: 5, pushes: 1 },

        Instr::WeakEmpty
        | Instr::TakeLocal { .. }
        | Instr::TakeField { .. }
        | Instr::Push { .. }
        | Instr::Function { .. }
        | Instr::LocalRef { .. } => StackEffect { pops: 0, pushes: 1 },
        Instr::Borrow
        | Instr::TransferLoad
        | Instr::OwnerData { .. }
        | Instr::OwnerLength
        | Instr::OwnerDowngrade
        | Instr::OwnerUpgrade
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
        Instr::GpuComputePipeline { .. } | Instr::GpuGraphicsPipeline { .. } => {
            StackEffect { pops: 1, pushes: 1 }
        }
        Instr::GpuDispatch { .. } => StackEffect { pops: 6, pushes: 1 },
        Instr::GpuDraw { .. } | Instr::PointerRange => StackEffect { pops: 4, pushes: 1 },
        Instr::OwnerCreate { .. } => StackEffect { pops: 1, pushes: 1 },
        Instr::PointerBytes | Instr::OwnerAllocate { .. } => StackEffect { pops: 2, pushes: 1 },
        Instr::PointerIndex | Instr::GpuArgumentsDraw => StackEffect { pops: 3, pushes: 1 },
        Instr::GpuArgumentsDispatch => StackEffect { pops: 5, pushes: 1 },
        Instr::AccessDynamic | Instr::Store | Instr::Replace => StackEffect { pops: 2, pushes: 1 },
        Instr::ForgetLocal { .. } | Instr::DropLocal { .. } => StackEffect { pops: 0, pushes: 0 },
        Instr::Discard | Instr::SetLocal { .. } | Instr::SetField { .. } => {
            StackEffect { pops: 1, pushes: 0 }
        }
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
