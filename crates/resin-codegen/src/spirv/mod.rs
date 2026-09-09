//! Lower verified, structured LIR directly to a Vulkan SPIR-V module.
use std::collections::{HashMap, HashSet};

use crate::Error;
use resin_types::prelude::*;
use rspirv::{binary::Assemble, dr::Builder, spirv::*};

mod entry;
mod function;
mod ops;
mod reachable;
mod symbols;
mod types;

pub(super) fn generate(
    checked: resin_lir::Verified<'_>,
    entry: FunctionId,
    stage: Stage,
) -> Result<Vec<u8>, Error> {
    let module = checked.module();
    let analysis = checked.analysis();
    let reachable = reachable::functions(module, entry.index())?;
    let mut context = Context::new(module, &analysis.types);
    for &index in &reachable {
        register_function_types(&mut context, index, &analysis.functions[index])?;
    }
    for index in reachable {
        function::lower(
            &mut context,
            &module.functions[index],
            &analysis.functions[index],
            index,
        )?;
    }
    entry::lower(&mut context, entry, stage)?;
    Ok(context
        .builder
        .module()
        .assemble()
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect())
}

/// All IDs and types belong to one shader module. LIR pointers are u64 device
/// addresses; Function-storage pointers never escape the function that owns them.
struct Context<'a> {
    builder: Builder,
    module: &'a resin_lir::Module,
    table: &'a TypeTable,
    functions: Vec<Word>,
    failed: Word,
    glsl: Word,
    types: HashMap<(Ty, types::Representation), Word>,
    u32_constants: HashMap<u32, Word>,
    u64_constants: HashMap<u64, Word>,
    named_types: HashSet<Word>,
    validated: HashSet<Ty>,
}

impl<'a> Context<'a> {
    fn new(module: &'a resin_lir::Module, table: &'a TypeTable) -> Self {
        let mut builder = Builder::new();
        builder.set_version(1, 6);
        builder.capability(Capability::Shader);
        builder.capability(Capability::Int64);
        builder.capability(Capability::PhysicalStorageBufferAddresses);
        builder.memory_model(
            AddressingModel::PhysicalStorageBuffer64,
            MemoryModel::GLSL450,
        );
        let glsl = builder.ext_inst_import("GLSL.std.450");
        let bool_type = builder.type_bool();
        let bool_pointer = builder.type_pointer(None, StorageClass::Private, bool_type);
        let initial = builder.constant_false(bool_type);
        let failed = builder.variable(bool_pointer, None, StorageClass::Private, Some(initial));
        builder.name(failed, "failed");
        let functions = module.functions.iter().map(|_| builder.id()).collect();
        Self {
            builder,
            module,
            table,
            functions,
            failed,
            glsl,
            types: HashMap::new(),
            u32_constants: HashMap::new(),
            u64_constants: HashMap::new(),
            named_types: HashSet::new(),
            validated: HashSet::new(),
        }
    }

    fn tag(&self, case: &Case) -> u32 {
        case.tag(self.table)
    }
}

fn register_function_types(
    context: &mut Context<'_>,
    index: usize,
    flow: &resin_lir::FunctionTypes,
) -> Result<(), Error> {
    let function = &context.module.functions[index];
    for ty in function
        .locals
        .iter()
        .map(|local| &local.ty)
        .chain([&function.result])
    {
        context
            .validate(ty)
            .map_err(|error| Error::at(context.module, index, None, error))?;
        context.ty(ty)?;
    }
    for ty in flow
        .inputs
        .iter()
        .flatten()
        .chain(flow.results.iter().flatten().flatten())
    {
        match ty {
            Ty::Function { .. } => continue,
            Ty::Pointer { pointee } => context.validate(pointee)?,
            _ => context.validate(ty)?,
        }
        context.ty(ty)?;
    }
    Ok(())
}

fn str_storage_error() -> Error {
    Error(
        "shader string literals need device-backed storage; pass a Span<ubyte> in the shader root"
            .into(),
    )
}

fn build_error(error: rspirv::dr::Error) -> Error {
    Error(format!("cannot construct SPIR-V: {error}"))
}
