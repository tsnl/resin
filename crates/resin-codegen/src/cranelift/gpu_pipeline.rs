//! Source pipeline factories and recorders retain their checked Resin signatures.
use super::function::{Body, Operand};
use crate::Error;
use cranelift_codegen::ir::{
    self, InstBuilder,
    condcodes::IntCC,
    types::{I32, I64},
};
use cranelift_module::Module;
use resin_types::prelude::*;

impl Body<'_, '_> {
    pub(super) fn gpu_call(
        &mut self,
        function: FunctionId,
        args: &[ir::Value],
    ) -> Result<Operand, Error> {
        let input = &self.types.module.functions[function.index()];
        let params = input.locals[..input.parameter_count]
            .iter()
            .map(|local| local.ty.clone())
            .collect::<Vec<_>>();
        let result = input.result.clone();
        let mut operands = vec![Operand {
            ty: Ty::Function {
                params: params.clone(),
                result: Box::new(result.clone()),
            },
            value: self.gpu_word(0),
            function: self.functions[function.index()],
            live: None,
        }];
        operands.extend(params.into_iter().zip(args).map(|(ty, &value)| Operand {
            ty,
            value,
            function: None,
            live: None,
        }));
        Ok(Operand {
            ty: result,
            value: self.call(&operands)?,
            function: None,
            live: None,
        })
    }

    pub(super) fn gpu_copy(&mut self, value: &Operand) -> Result<ir::Value, Error> {
        self.retain(&value.ty, value.value)?;
        Ok(value.value)
    }

    pub(super) fn gpu_create_pipeline(
        &mut self,
        factory: FunctionId,
        shaders: &[FunctionId],
        gpu: &Operand,
        result: &Ty,
    ) -> Result<ir::Value, Error> {
        let mut args = vec![self.gpu_copy(gpu)?];
        for (index, &shader) in shaders.iter().enumerate() {
            let (data, length) = self.types.shader(shader);
            let reference = self.module.declare_data_in_func(data, self.builder.func);
            let pointer = self.builder.ins().symbol_value(I64, reference);
            let span =
                self.allocate(&self.types.module.functions[factory.index()].locals[index + 1].ty);
            let length = self.gpu_word(length);
            self.builder
                .ins()
                .store(ir::MemFlagsData::new(), pointer, span, 0);
            self.builder
                .ins()
                .store(ir::MemFlagsData::new(), length, span, 8);
            args.push(span);
        }
        let created = self.gpu_call(factory, &args)?;
        let Ty::Result {
            value: pipeline,
            error,
        } = result
        else {
            unreachable!("verified pipeline result")
        };
        let metadata = resin_types::gpu_pipeline_contract(&self.types.module.types, pipeline)
            .expect("verified pipeline")
            .clone();
        let output = self.allocate(result);
        let destination = self.offset(output, self.types.layout(result).offsets[1]);
        let tag = self
            .builder
            .ins()
            .load(I32, ir::MemFlagsData::new(), created.value, 0);
        self.builder
            .ins()
            .store(ir::MemFlagsData::new(), tag, output, 0);
        let ok = self.builder.create_block();
        let err = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.ins().brif(tag, err, &[], ok, &[]);
        self.builder.switch_to_block(ok);
        let owner = self.payload(&created, &Case::Ok, &metadata.owner)?;
        let owner = if self.types.scalar(&metadata.owner).is_some() {
            owner
        } else {
            self.builder
                .ins()
                .load(I64, ir::MemFlagsData::new(), owner, 0)
        };
        self.builder
            .ins()
            .store(ir::MemFlagsData::new(), owner, destination, 0);
        for (offset, number) in [
            (8, self.types.table.id(&metadata.root).unwrap().index()),
            (12, self.types.table.id(&metadata.owner).unwrap().index()),
            (16, pipeline_kind(metadata.kind)),
        ] {
            let value = self.builder.ins().iconst(I32, number as i64);
            self.builder
                .ins()
                .store(ir::MemFlagsData::new(), value, destination, offset);
        }
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(err);
        let error_value = self.payload(&created, &Case::Err, error)?;
        self.write(error, destination, error_value);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        Ok(output)
    }

    pub(super) fn gpu_record(
        &mut self,
        functions: (FunctionId, Option<FunctionId>, FunctionId),
        projection: Option<&resin_types::GpuProjectionPlan>,
        args: &[Operand],
        result: &Ty,
    ) -> Result<ir::Value, Error> {
        let (context, allocator, recorder) = functions;
        let metadata = resin_types::gpu_pipeline_contract(&self.types.module.types, &args[1].ty)
            .expect("verified pipeline")
            .clone();
        for (offset, number) in [
            (8, self.types.table.id(&metadata.root).unwrap().index()),
            (12, self.types.table.id(&metadata.owner).unwrap().index()),
            (16, pipeline_kind(metadata.kind)),
        ] {
            let value =
                self.builder
                    .ins()
                    .load(I32, ir::MemFlagsData::new(), args[1].value, offset);
            let invalid = self
                .builder
                .ins()
                .icmp_imm_u(IntCC::NotEqual, value, number as i64);
            self.guard(
                invalid,
                "GPU pipeline contract does not match its source wrapper",
            )?;
        }
        let handle = self
            .builder
            .ins()
            .load(I64, ir::MemFlagsData::new(), args[1].value, 0);
        let owner = if self.types.scalar(&metadata.owner).is_some() {
            handle
        } else {
            let value = self.allocate(&metadata.owner);
            self.builder
                .ins()
                .store(ir::MemFlagsData::new(), handle, value, 0);
            value
        };
        let owner = Operand {
            ty: metadata.owner.clone(),
            value: owner,
            function: None,
            live: None,
        };
        let Some(allocator) = allocator else {
            return self.gpu_record_call(recorder, args, &owner, None);
        };
        let copied_owner = self.gpu_copy(&owner)?;
        let gpu = self.gpu_call(context, &[copied_owner])?;
        let Ty::Result { error, .. } = result else {
            unreachable!("verified recording result")
        };
        let projected = self.gpu_project(
            allocator,
            projection.expect("root projection"),
            &gpu,
            &args[2],
            error,
        )?;
        self.drop_value(&gpu.ty, gpu.value)?;
        let output = self.allocate(result);
        let ok = self.builder.create_block();
        let err = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.ins().brif(projected.tag, err, &[], ok, &[]);
        self.builder.switch_to_block(ok);
        let recorded = self.gpu_record_call(recorder, args, &owner, Some(projected.owner))?;
        self.write(result, output, recorded);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(err);
        self.builder
            .ins()
            .store(ir::MemFlagsData::new(), projected.tag, output, 0);
        let destination = self.offset(output, self.types.layout(result).offsets[1]);
        self.write(error, destination, projected.error);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        Ok(output)
    }

    fn gpu_record_call(
        &mut self,
        recorder: FunctionId,
        args: &[Operand],
        owner: &Operand,
        projection: Option<ir::Value>,
    ) -> Result<ir::Value, Error> {
        let root_type = &self.types.module.functions[recorder.index()].locals[2].ty;
        let root = if matches!(root_type, Ty::Union { .. }) {
            let ty = if projection.is_some() {
                Ty::GpuArguments
            } else {
                Ty::None
            };
            let value = projection.unwrap_or_else(|| self.builder.ins().iconst(ir::types::I8, 0));
            self.variant(
                root_type,
                &Case::Type(ty.clone()),
                &Operand {
                    ty,
                    value,
                    function: None,
                    live: None,
                },
            )
        } else {
            projection.expect("compute projection")
        };
        let mut values = vec![self.gpu_copy(&args[0])?, self.gpu_copy(owner)?, root];
        for arg in &args[3..] {
            values.push(self.gpu_copy(arg)?);
        }
        Ok(self.gpu_call(recorder, &values)?.value)
    }
}

fn pipeline_kind(kind: resin_types::GpuPipelineKind) -> usize {
    match kind {
        resin_types::GpuPipelineKind::Compute => 0,
        resin_types::GpuPipelineKind::Graphics => 1,
    }
}
