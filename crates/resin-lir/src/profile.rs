//! Concrete shader operations and dependency graphs, shared by construction and certification.
use crate::{
    BlockId, Function, Instr, Module, Profile, VerifyError, VerifyErrorKind, VerifyLocation,
};
use resin_types::prelude::*;
use std::{collections::BTreeMap, sync::Arc};

pub(crate) struct Error {
    pub(crate) function: FunctionId,
    pub(crate) instruction: Option<(BlockId, usize)>,
    pub(crate) message: Arc<str>,
}

impl Error {
    pub(crate) fn verify(self) -> VerifyError {
        let location = match self.instruction {
            Some((basic_block, instruction)) => VerifyLocation::Instruction {
                function: self.function,
                basic_block,
                instruction,
            },
            None => VerifyLocation::Function {
                function: self.function,
            },
        };
        VerifyError {
            location,
            kind: VerifyErrorKind::UnsupportedProfile {
                profile: Profile::Shader,
                message: self.message,
            },
        }
    }
}

pub(crate) fn function(
    typer: &TyperContext,
    id: FunctionId,
    function: &Function,
) -> Result<(), Error> {
    if function.profile == Profile::Host {
        return Ok(());
    }
    let error = |message: String| Error {
        function: id,
        instruction: None,
        message: message.into(),
    };
    if function.foreign.is_some() {
        return Err(error(format!(
            "shader cannot call foreign function {}",
            function.name.as_deref().unwrap_or("<unnamed>")
        )));
    }
    for (block, body) in function.blocks.iter().enumerate() {
        for (index, op) in body.instrs.iter().enumerate() {
            instruction(typer, op).map_err(|message| Error {
                function: id,
                instruction: Some((BlockId::from_index(block), index)),
                message: message.into(),
            })?;
        }
    }
    for ty in function
        .locals
        .iter()
        .map(|local| &local.ty)
        .chain([&function.result])
    {
        if ty.needs_drop(typer.definitions()) {
            return Err(error("shader cannot consume managed values: reference counting and destruction are host-only".into()));
        }
        resin_types::shader::value_type(typer.definitions(), ty).map_err(error)?;
    }
    Ok(())
}

/// Function references and local addresses are expression categories. Their types
/// need not be storable; actual signatures and locals use the value/storage rules.
pub(crate) fn expression_type(typer: &TyperContext, ty: &Ty) -> Result<(), String> {
    match ty {
        Ty::Function { params, result } => {
            for param in params {
                resin_types::shader::value_type(typer.definitions(), param)?;
            }
            resin_types::shader::value_type(typer.definitions(), result)
        }
        Ty::Reference {
            referent: pointee, ..
        } => resin_types::shader::value_type(typer.definitions(), pointee),
        _ => resin_types::shader::value_type(typer.definitions(), ty),
    }
}

fn instruction(typer: &TyperContext, op: &Instr) -> Result<(), String> {
    match op {
        Instr::TraceRay { payload } => resin_types::shader::ray_payload(typer.definitions(), payload),
        Instr::RayHitInfo => Ok(()),
        Instr::CallBuiltin { name, params, result } => {
            let signature = resin_types::shader::builtin_instance(typer, name, params)?;
            typer.same(&signature.result, result).map_err(|error| error.to_string())
        }
        Instr::Push { value } => literal(value),
        Instr::PointerCast { .. } => Err("shader pointer casts are unsupported; use typed pointers and indexing instead".into()),
        Instr::ForgetLocal { .. } | Instr::Discard | Instr::TakeLocal { .. }
        | Instr::SetLocal { .. } | Instr::LocalRef { .. } | Instr::Function { .. }
        | Instr::Borrow | Instr::ReadOnly | Instr::Call { .. } | Instr::TransferLoad | Instr::Load
        | Instr::TakeField { .. } | Instr::SetField { .. } | Instr::Store | Instr::Replace | Instr::MakeVariant { .. } | Instr::IsVariant { .. }
        | Instr::VariantPayload { .. } | Instr::ExcludeNone | Instr::Widen { .. }
        | Instr::NumericCast { .. } | Instr::Ascribe { .. }
        | Instr::MakeArray { .. } | Instr::MakeRecord { .. } | Instr::AccessStatic { .. }
        | Instr::AccessDynamic | Instr::PointerIndex | Instr::Eliminate { .. } => Ok(()),
        Instr::OwnerData { .. } | Instr::OwnerLength | Instr::OwnerAllocate { .. } | Instr::OwnerCreate { .. } | Instr::OwnerDowngrade | Instr::OwnerUpgrade
        | Instr::WeakEmpty | Instr::DropLocal { .. } => Err("shader cannot consume managed values: reference counting and destruction are host-only".into()),
        Instr::GpuViewAllocate | Instr::GpuViewRange { .. } | Instr::GpuViewOffset | Instr::GpuViewRestrict | Instr::GpuViewLoad { .. } | Instr::GpuViewStore | Instr::GpuViewReplace | Instr::GpuViewCopyTo | Instr::GpuViewCopyFrom | Instr::GpuViewCopyImage
        | Instr::GpuRayTracingPipeline { .. } | Instr::GpuComputePipeline { .. } | Instr::GpuGraphicsPipeline { .. }
        | Instr::GpuDispatch { .. } | Instr::GpuDraw { .. } | Instr::GpuArgumentsTraceRays | Instr::GpuArgumentsDispatch
        | Instr::GpuArgumentsDraw
        | Instr::PointerBytes | Instr::PointerRange => {
            Err(format!("shader profile does not support {op:?}"))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Visit {
    Unseen,
    Active,
    Complete,
}

struct Edge {
    function: FunctionId,
    instruction: (BlockId, usize),
}

fn edges(function: &Function) -> Vec<Edge> {
    function
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(block, body)| {
            body.instrs
                .iter()
                .enumerate()
                .filter_map(move |(index, op)| match op {
                    Instr::Function { function } => Some(Edge {
                        function: *function,
                        instruction: (BlockId::from_index(block), index),
                    }),
                    _ => None,
                })
        })
        .collect()
}

/// Dependency order is a completed result used by emission, not a visited-key flag.
/// Walk iteratively so long acyclic helper chains do not consume the Rust call stack.
pub(crate) fn shaders(module: &Module) -> Result<BTreeMap<FunctionId, Vec<FunctionId>>, Error> {
    let graph: Vec<_> = module.functions.iter().map(edges).collect();
    let mut result = BTreeMap::new();
    for &root in module.shaders.keys() {
        let mut states = vec![Visit::Unseen; graph.len()];
        let mut pending = vec![(root, 0)];
        let mut order = vec![];
        while let Some((id, cursor)) = pending.last_mut() {
            if id.index() >= graph.len() {
                return Err(Error {
                    function: *id,
                    instruction: None,
                    message: "invalid shader function".into(),
                });
            }
            states[id.index()] = Visit::Active;
            if let Some(edge) = graph[id.index()].get(*cursor) {
                *cursor += 1;
                let Some(&state) = states.get(edge.function.index()) else {
                    return Err(Error {
                        function: *id,
                        instruction: Some(edge.instruction),
                        message: "invalid shader function reference".into(),
                    });
                };
                if state == Visit::Active {
                    let name = module
                        .functions
                        .get(edge.function.index())
                        .and_then(|f| f.name.as_deref())
                        .unwrap_or("<invalid>");
                    return Err(Error {
                        function: *id,
                        instruction: Some(edge.instruction),
                        message: format!("recursive shader call graph at {name}").into(),
                    });
                }
                if state == Visit::Unseen {
                    pending.push((edge.function, 0));
                }
            } else {
                states[id.index()] = Visit::Complete;
                order.push(*id);
                pending.pop();
            }
        }
        let stage = module.shaders[&root].stage.as_ref();
        for &id in &order {
            for block in &module.functions[id.index()].blocks {
                for instruction in &block.instrs {
                    let allowed = match instruction {
                        Instr::TraceRay { .. } => stage == "ray_generation",
                        Instr::RayHitInfo => stage == "closest_hit",
                        _ => true,
                    };
                    if !allowed {
                        return Err(Error { function: id, instruction: None, message: "ray operation is not available in this shader stage (nested tracing is unsupported)".into() });
                    }
                }
            }
        }
        result.insert(root, order);
    }
    Ok(result)
}

fn literal(value: &Value) -> Result<(), String> {
    match value {
        Value::Unit
        | Value::None
        | Value::Bool { .. }
        | Value::Int32 { .. }
        | Value::UInt8 { .. }
        | Value::UInt32 { .. }
        | Value::Int64 { .. }
        | Value::UInt64 { .. } => Ok(()),
        Value::Float32 { value } if value.is_finite() => Ok(()),
        Value::Record { value } => {
            for field in &value.fields {
                literal(&field.value)?;
            }
            Ok(())
        }
        Value::Str { .. } => resin_types::shader::value_type(&[], &Ty::Str),
        _ => Err("unsupported shader literal".into()),
    }
}

pub(crate) fn stack_types(
    typer: &TyperContext,
    id: FunctionId,
    flow: &crate::FunctionTypes,
) -> Result<(), Error> {
    for (block, inputs) in flow.inputs.iter().enumerate() {
        for ty in inputs {
            expression_type(typer, ty).map_err(|message| Error {
                function: id,
                instruction: Some((BlockId::from_index(block), 0)),
                message: message.into(),
            })?;
        }
        let mut stack = inputs.clone();
        for (instruction, result) in flow.results[block].iter().enumerate() {
            let count = flow.operand_count(BlockId::from_index(block), instruction);
            let args = stack.split_off(stack.len() - count);
            let error = |message: String| Error {
                function: id,
                instruction: Some((BlockId::from_index(block), instruction)),
                message: message.into(),
            };
            if args
                .iter()
                .chain(result)
                .any(|ty| ty.needs_drop(typer.definitions()))
            {
                return Err(error("shader cannot consume managed values: reference counting and destruction are host-only".into()));
            }
            if let Some(ty) = result {
                expression_type(typer, ty).map_err(error)?;
                stack.push(ty.clone());
            }
        }
    }
    Ok(())
}
