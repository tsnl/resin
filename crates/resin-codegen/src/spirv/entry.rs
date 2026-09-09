//! Vulkan stage interfaces and the small entry wrapper around a Resin function.
use crate::Error;
use resin_types::{prelude::*, shader::Interface};
use rspirv::{dr::Operand, spirv::*};

use super::{Context, build_error};

pub(super) fn lower(
    context: &mut Context<'_>,
    entry: FunctionId,
    stage: Stage,
) -> Result<(), Error> {
    let function = &context.module.functions[entry.index()];
    let param = &function.locals[0].ty;
    let result = &function.result;
    let typer = TyperContext::from_definitions(context.module.types.clone());
    let interface =
        resin_types::shader::validate(&typer, param, result, false, stage.name()).map_err(Error)?;
    let variables = Variables::declare(context, &interface)?;
    let void = context.builder.type_void();
    let function_type = context.builder.type_function(void, []);
    let wrapper = context
        .builder
        .begin_function(void, None, FunctionControl::NONE, function_type)
        .map_err(build_error)?;
    context.builder.name(wrapper, "main");
    context.builder.begin_block(None).map_err(build_error)?;
    let input = variables.argument(context, &interface, param)?;
    let result_type = context.ty(result)?;
    let output = context
        .builder
        .function_call(result_type, None, context.functions[entry.index()], [input])
        .map_err(build_error)?;
    variables.finish(context, &interface, result, output)?;
    context.builder.ret().map_err(build_error)?;
    context.builder.end_function().map_err(build_error)?;
    declare_entry(context, wrapper, stage, &variables.interfaces);
    Ok(())
}

struct Variables {
    interfaces: Vec<Word>,
    root: Option<Word>,
    inputs: Vec<Word>,
    outputs: Vec<Word>,
    vector: Word,
}

impl Variables {
    fn declare(context: &mut Context<'_>, interface: &Interface) -> Result<Self, Error> {
        let float = context.ty(&Ty::Float32)?;
        let mut variables = Self {
            interfaces: vec![context.failed],
            root: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            vector: context.builder.type_vector(float, 4),
        };
        match interface {
            Interface::Compute { .. } => {
                let uint = context.ty(&Ty::UInt32)?;
                let vector = context.builder.type_vector(uint, 3);
                variables.input_builtin(context, vector, BuiltIn::WorkgroupId, "workgroup_id");
                variables.input_builtin(
                    context,
                    vector,
                    BuiltIn::LocalInvocationId,
                    "local_invocation_id",
                );
                variables.push_constant(context)?;
            }
            Interface::Vertex { root, .. } => {
                let int = context.ty(&Ty::Int32)?;
                variables.input_builtin(context, int, BuiltIn::VertexIndex, "vertex_index");
                let position =
                    variables.variable(context, variables.vector, StorageClass::Output, "position");
                context.builder.decorate(
                    position,
                    Decoration::BuiltIn,
                    [Operand::BuiltIn(BuiltIn::Position)],
                );
                variables.outputs.push(position);
                variables.output_location(context, "color");
                if *root {
                    variables.push_constant(context)?;
                }
            }
            Interface::Fragment { root, .. } => {
                let color =
                    variables.variable(context, variables.vector, StorageClass::Input, "color");
                context
                    .builder
                    .decorate(color, Decoration::Location, [Operand::LiteralBit32(0)]);
                variables.inputs.push(color);
                variables.output_location(context, "output_color");
                if *root {
                    variables.push_constant(context)?;
                }
            }
        }
        Ok(variables)
    }

    fn variable(
        &mut self,
        context: &mut Context<'_>,
        ty: Word,
        storage: StorageClass,
        name: &str,
    ) -> Word {
        let pointer = context.builder.type_pointer(None, storage, ty);
        let variable = context.builder.variable(pointer, None, storage, None);
        context.builder.name(variable, name);
        self.interfaces.push(variable);
        variable
    }

    fn input_builtin(&mut self, context: &mut Context<'_>, ty: Word, builtin: BuiltIn, name: &str) {
        let variable = self.variable(context, ty, StorageClass::Input, name);
        context
            .builder
            .decorate(variable, Decoration::BuiltIn, [Operand::BuiltIn(builtin)]);
        self.inputs.push(variable);
    }

    fn output_location(&mut self, context: &mut Context<'_>, name: &str) {
        let variable = self.variable(context, self.vector, StorageClass::Output, name);
        context
            .builder
            .decorate(variable, Decoration::Location, [Operand::LiteralBit32(0)]);
        self.outputs.push(variable);
    }

    fn push_constant(&mut self, context: &mut Context<'_>) -> Result<(), Error> {
        let word = context.ty(&Ty::UInt64)?;
        let structure = context.builder.id();
        context.builder.type_struct_id(Some(structure), [word]);
        context.builder.name(structure, "Push");
        context.builder.member_name(structure, 0, "root");
        context.builder.decorate(structure, Decoration::Block, []);
        context.builder.member_decorate(
            structure,
            0,
            Decoration::Offset,
            [Operand::LiteralBit32(0)],
        );
        self.root = Some(self.variable(context, structure, StorageClass::PushConstant, "push"));
        Ok(())
    }

    fn argument(
        &self,
        context: &mut Context<'_>,
        interface: &Interface,
        param: &Ty,
    ) -> Result<Word, Error> {
        let input = match interface {
            Interface::Compute { .. } => self.compute_index(context)?,
            Interface::Vertex { index, .. } => {
                let ty = context.ty(index)?;
                context
                    .builder
                    .load(ty, None, self.inputs[0], None, [])
                    .map_err(build_error)?
            }
            Interface::Fragment { color, .. } => {
                let input = context
                    .builder
                    .load(self.vector, None, self.inputs[0], None, [])
                    .map_err(build_error)?;
                self.color_record(context, color, input)?
            }
        };
        let Some(root) = self.root else {
            return Ok(input);
        };
        let pointer = context.pointer_type(StorageClass::PushConstant, &Ty::UInt64)?;
        let zero = context.constant_u32(0);
        let address = context
            .builder
            .access_chain(pointer, None, root, [zero])
            .map_err(build_error)?;
        let word = context.ty(&Ty::UInt64)?;
        let value = context
            .builder
            .load(word, None, address, None, [])
            .map_err(build_error)?;
        let ty = context.ty(param)?;
        context
            .builder
            .composite_construct(ty, None, [input, value])
            .map_err(build_error)
    }

    fn compute_index(&self, context: &mut Context<'_>) -> Result<Word, Error> {
        let uint = context.ty(&Ty::UInt32)?;
        let vector = context.builder.type_vector(uint, 3);
        let word = context.ty(&Ty::UInt64)?;
        let mut coordinates = Vec::new();
        for &input in &self.inputs {
            let value = context
                .builder
                .load(vector, None, input, None, [])
                .map_err(build_error)?;
            let x = context
                .builder
                .composite_extract(uint, None, value, [0])
                .map_err(build_error)?;
            coordinates.push(
                context
                    .builder
                    .u_convert(word, None, x)
                    .map_err(build_error)?,
            );
        }
        // Widen before multiplication: the full global index need not fit u32.
        let width = context.constant_u64(64);
        let base = context
            .builder
            .i_mul(word, None, coordinates[0], width)
            .map_err(build_error)?;
        context
            .builder
            .i_add(word, None, base, coordinates[1])
            .map_err(build_error)
    }

    fn color_record(
        &self,
        context: &mut Context<'_>,
        ty: &Ty,
        vector: Word,
    ) -> Result<Word, Error> {
        let float = context.ty(&Ty::Float32)?;
        let mut fields = Vec::new();
        for index in 0..4 {
            fields.push(
                context
                    .builder
                    .composite_extract(float, None, vector, [index])
                    .map_err(build_error)?,
            );
        }
        let ty = context.ty(ty)?;
        context
            .builder
            .composite_construct(ty, None, fields)
            .map_err(build_error)
    }

    fn finish(
        &self,
        context: &mut Context<'_>,
        interface: &Interface,
        result: &Ty,
        output: Word,
    ) -> Result<(), Error> {
        if matches!(interface, Interface::Compute { .. }) {
            return Ok(());
        }
        let bool_type = context.ty(&Ty::Bool)?;
        let failed = context
            .builder
            .load(bool_type, None, context.failed, None, [])
            .map_err(build_error)?;
        let end = context.builder.id();
        let success = context.builder.id();
        context
            .builder
            .selection_merge(end, SelectionControl::NONE)
            .map_err(build_error)?;
        context
            .builder
            .branch_conditional(failed, end, success, [])
            .map_err(build_error)?;
        context
            .builder
            .begin_block(Some(success))
            .map_err(build_error)?;
        match interface {
            Interface::Vertex {
                position, color, ..
            } => {
                for (index, ty) in [position, color].into_iter().enumerate() {
                    let field_type = context.ty(ty)?;
                    let value = context
                        .builder
                        .composite_extract(field_type, None, output, [index as u32])
                        .map_err(build_error)?;
                    self.store_vector(context, self.outputs[index], value)?;
                }
            }
            Interface::Fragment { .. } => {
                debug_assert!(matches!(context.shape(result), Ty::Record { .. }));
                self.store_vector(context, self.outputs[0], output)?;
            }
            Interface::Compute { .. } => unreachable!(),
        }
        context.builder.branch(end).map_err(build_error)?;
        context
            .builder
            .begin_block(Some(end))
            .map_err(build_error)?;
        Ok(())
    }

    fn store_vector(
        &self,
        context: &mut Context<'_>,
        destination: Word,
        value: Word,
    ) -> Result<(), Error> {
        let float = context.ty(&Ty::Float32)?;
        let mut fields = Vec::new();
        for index in 0..4 {
            fields.push(
                context
                    .builder
                    .composite_extract(float, None, value, [index])
                    .map_err(build_error)?,
            );
        }
        let vector = context
            .builder
            .composite_construct(self.vector, None, fields)
            .map_err(build_error)?;
        context
            .builder
            .store(destination, vector, None, [])
            .map_err(build_error)
    }
}

fn declare_entry(context: &mut Context<'_>, entry: Word, stage: Stage, interfaces: &[Word]) {
    let model = match stage {
        Stage::Compute => ExecutionModel::GLCompute,
        Stage::Vertex => ExecutionModel::Vertex,
        Stage::Fragment => ExecutionModel::Fragment,
    };
    context
        .builder
        .entry_point(model, entry, "main", interfaces);
    match stage {
        Stage::Compute => {
            context
                .builder
                .execution_mode(entry, ExecutionMode::LocalSize, [64, 1, 1])
        }
        Stage::Fragment => {
            context
                .builder
                .execution_mode(entry, ExecutionMode::OriginUpperLeft, [])
        }
        Stage::Vertex => {}
    }
}
