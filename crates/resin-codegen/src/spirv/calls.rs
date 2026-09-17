//! Local borrows cross calls as a root object plus a projection path. Vulkan
//! permits Function pointers to whole objects as arguments, but not arbitrary
//! pointers produced by access chains. Specialize static paths and pass dynamic
//! indices separately; the callee reconstructs the exact alias without copying.
use super::{
    Context, build_error, function,
    symbols::{LocalAddress, LocalIndex, Slot},
};
use crate::Error;
use resin_types::prelude::*;
use rspirv::spirv::{Decoration, StorageClass, Word};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum Index {
    Static { index: u32 },
    Dynamic { ty: Ty },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct LocalParameter {
    pub root: Ty,
    pub indices: Vec<Index>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct Signature {
    function: usize,
    parameters: Vec<Option<LocalParameter>>,
}

impl LocalParameter {
    pub(super) fn types(&self, context: &mut Context<'_>) -> Result<Vec<Word>, Error> {
        let mut types = vec![context.pointer_type(StorageClass::Function, &self.root)?];
        for index in &self.indices {
            if let Index::Dynamic { ty } = index {
                types.push(context.ty(ty)?);
            }
        }
        Ok(types)
    }

    pub(super) fn argument(
        &self,
        context: &mut Context<'_>,
        arguments: &mut impl Iterator<Item = Word>,
        ty: Ty,
    ) -> Slot {
        let root = arguments.next().unwrap();
        context.builder.decorate(root, Decoration::Aliased, []);
        let indices = self
            .indices
            .iter()
            .map(|index| match index {
                Index::Static { index } => LocalIndex::Static { index: *index },
                Index::Dynamic { ty } => LocalIndex::Dynamic {
                    id: arguments.next().unwrap(),
                    ty: ty.clone(),
                },
            })
            .collect();
        Slot {
            ty,
            id: root,
            local: Some(LocalAddress {
                root,
                root_type: self.root.clone(),
                indices,
            }),
        }
    }
}

pub(super) fn local_call(
    context: &mut Context<'_>,
    args: &[Slot],
    result: &Ty,
) -> Result<(Slot, Word), Error> {
    let function = context
        .functions
        .iter()
        .position(|id| *id == args[0].id)
        .ok_or_else(|| Error::unsupported("shader calls require a named function".into()))?;
    let signature = Signature {
        function,
        parameters: args[1..]
            .iter()
            .map(|arg| {
                arg.local.as_ref().map(|local| LocalParameter {
                    root: local.root_type.clone(),
                    indices: local
                        .indices
                        .iter()
                        .map(|index| match index {
                            LocalIndex::Static { index } => Index::Static { index: *index },
                            LocalIndex::Dynamic { ty, .. } => Index::Dynamic { ty: ty.clone() },
                        })
                        .collect(),
                })
            })
            .collect(),
    };
    let callee = specialize(context, signature)?;
    let mut arguments = Vec::new();
    for arg in &args[1..] {
        if let Some(local) = &arg.local {
            arguments.push(local.root);
            for index in &local.indices {
                if let LocalIndex::Dynamic { id, .. } = index {
                    arguments.push(*id);
                }
            }
        } else {
            arguments.push(arg.id);
        }
    }
    let ty = context.ty(result)?;
    let id = context
        .builder
        .function_call(ty, None, callee, arguments)
        .map_err(build_error)?;
    Ok((Slot::value(result.clone(), id), callee))
}

fn specialize(context: &mut Context<'_>, signature: Signature) -> Result<Word, Error> {
    if let Some(id) = context.local_functions.get(&signature) {
        return Ok(*id);
    }
    // Local projection specializations are backend representations of already
    // verified, acyclic calls. Bound their additional work and emission stack.
    if context.local_functions.len() + context.local_call_depth >= 16_384
        || context.local_call_depth >= 32
    {
        return Err(Error::unsupported(
            "shader local-reference specialization limit exceeded".into(),
        ));
    }
    context.local_call_depth += 1;
    let selected_function = context.builder.selected_function();
    let selected_block = context.builder.selected_block();
    context.builder.select_function(None).map_err(build_error)?;
    let id = context.builder.id();
    let module = context.module;
    let analysis = context.analysis;
    let may_fail = function::lower(
        context,
        &module.functions[signature.function],
        &analysis[signature.function],
        signature.function,
        id,
        &signature.parameters,
    )?;
    context.local_call_depth -= 1;
    context.fallibility.insert(id, may_fail);
    context.local_functions.insert(signature, id);
    context
        .builder
        .select_function(selected_function)
        .map_err(build_error)?;
    context
        .builder
        .select_block(selected_block)
        .map_err(build_error)?;
    Ok(id)
}
