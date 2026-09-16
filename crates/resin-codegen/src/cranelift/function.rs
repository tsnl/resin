use super::{failure, numeric, types::Types, unsupported};
use crate::{Error, GenerationError};
use cranelift_codegen::ir::{self, InstBuilder};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Module};
use cranelift_object::ObjectModule;
use resin_executor::Cancellation;
use resin_lir::{BlockId, FunctionTypes, Instr, Terminator, Verified};
use resin_types::prelude::*;

#[derive(Clone)]
pub(super) struct Operand {
    pub ty: Ty,
    /// Scalar bits, or the address of an independently owned aggregate snapshot.
    pub value: ir::Value,
    pub function: Option<FuncId>,
    /// Optional address of a managed local's initialization flag; null means initialized.
    pub live: Option<ir::Value>,
}

/// Per-function translation state. Helpers consume completed type/layout facts;
/// all allocations here belong to the emitted function, not the compiler cache.
pub(super) struct Body<'a, 'b> {
    pub builder: FunctionBuilder<'a>,
    pub module: &'b mut ObjectModule,
    pub types: &'b Types<'b>,
    pub functions: &'b [Option<FuncId>],
    pub locals: Vec<ir::Value>,
    pub local_live: Vec<Option<ir::Value>>,
    pub input: &'b resin_lir::Function,
    result: Option<ir::Value>,
}

#[derive(Clone, Copy)]
enum Exit {
    Return,
    Merge { block: usize },
    LoopTest { body: usize, next: usize },
    Continue { condition: usize },
}

pub(super) fn define(
    module: &mut ObjectModule,
    checked: Verified<'_>,
    types: &Types<'_>,
    index: usize,
    functions: &[Option<FuncId>],
    cancellation: &Cancellation,
) -> Result<(), GenerationError> {
    let input = &checked.module().functions[index];
    let flow = &checked.analysis().functions[index];
    let mut context = module.make_context();
    context.func.signature = types.signature(
        module,
        &input.locals[..input.parameter_count]
            .iter()
            .map(|local| local.ty.clone())
            .collect::<Vec<_>>(),
        &input.result,
    );
    let mut frontend = FunctionBuilderContext::new();
    let builder = FunctionBuilder::new(&mut context.func, &mut frontend);
    let mut body = Body {
        builder,
        module,
        types,
        functions,
        locals: vec![],
        local_live: vec![],
        input,
        result: None,
    };
    body.generate(flow, index, cancellation)?;
    body.builder.seal_all_blocks();
    let config = body.module.target_config();
    body.builder.finalize(config);
    cancellation.check()?;
    module
        .define_function(functions[index].unwrap(), &mut context)
        .map_err(|error| failure(format!("{error:?}")))?;
    Ok(())
}

impl<'a, 'b> Body<'a, 'b> {
    pub fn new(
        builder: FunctionBuilder<'a>,
        module: &'b mut ObjectModule,
        types: &'b Types<'b>,
        functions: &'b [Option<FuncId>],
        input: &'b resin_lir::Function,
    ) -> Self {
        Self {
            builder,
            module,
            types,
            functions,
            locals: vec![],
            local_live: vec![],
            input,
            result: None,
        }
    }
}

impl Body<'_, '_> {
    fn generate(
        &mut self,
        flow: &FunctionTypes,
        index: usize,
        cancellation: &Cancellation,
    ) -> Result<(), GenerationError> {
        let entry = self.builder.create_block();
        self.builder.append_block_params_for_function_params(entry);
        self.builder.switch_to_block(entry);
        let params = self.builder.block_params(entry).to_vec();
        let indirect = self.types.scalar(&self.input.result).is_none();
        self.result = indirect.then(|| params[0]);
        for (index, local) in self.input.locals.iter().enumerate() {
            let address = self.allocate(&local.ty);
            if index < self.input.parameter_count {
                self.write(&local.ty, address, params[index + usize::from(indirect)]);
            }
            self.locals.push(address);
            let live = if local.ty.needs_drop(self.types.table) {
                let flag = self.allocate(&Ty::Bool);
                let initialized = self
                    .builder
                    .ins()
                    .iconst(ir::types::I8, i64::from(index < self.input.parameter_count));
                self.builder
                    .ins()
                    .store(ir::MemFlagsData::new(), initialized, flag, 0);
                Some(flag)
            } else {
                None
            };
            self.local_live.push(live);
        }
        let blocks = self.blocks(flow);
        self.builder
            .ins()
            .jump(blocks[self.input.entry.index()], &[]);
        let exits = exits(self.input);
        for (block_index, block) in self.input.blocks.iter().enumerate() {
            cancellation.check()?;
            self.builder.switch_to_block(blocks[block_index]);
            let incoming = self.builder.block_params(blocks[block_index]).to_vec();
            let mut incoming = incoming.into_iter();
            let mut operands = flow.inputs[block_index]
                .iter()
                .map(|ty| {
                    let value = incoming.next().unwrap();
                    let value = if self.types.scalar(ty).is_some() {
                        value
                    } else {
                        self.read(ty, value)
                    };
                    let live = self
                        .tracks_initialization(ty)
                        .then(|| incoming.next().unwrap());
                    Operand {
                        ty: ty.clone(),
                        value,
                        function: None,
                        live,
                    }
                })
                .collect::<Vec<_>>();
            let mut diverged = false;
            for (position, instruction) in block.instrs.iter().enumerate() {
                if matches!(instruction, Instr::Eliminate { .. }) {
                    self.runtime_call("abort", &[], None)?;
                    self.builder.ins().trap(ir::TrapCode::unwrap_user(1));
                    diverged = true;
                    break;
                }
                let count = flow.operand_count(BlockId::from_index(block_index), position);
                let args = operands.split_off(operands.len() - count);
                let result = flow.results[block_index][position].as_ref();
                if let Some(ty) = result {
                    self.types.validate_value(ty)?;
                }
                let value = self
                    .instruction(instruction, &args, result)
                    .map_err(|error| {
                        Error::at(
                            self.types.module,
                            index,
                            Some((block_index, position)),
                            error,
                        )
                    })?;
                if let Some(ty) = result {
                    operands.push(Operand {
                        ty: ty.clone(),
                        value: value.expect("verified instruction has a result"),
                        function: if let Instr::Function { function } = instruction {
                            self.functions[function.index()]
                        } else {
                            None
                        },
                        live: if let Instr::LocalAddress { local } = instruction {
                            self.local_live[local.index()]
                        } else {
                            None
                        },
                    });
                }
            }
            if !diverged {
                self.terminate(&blocks, &block.terminator, exits[block_index], &operands);
            }
        }
        Ok(())
    }

    fn blocks(&mut self, flow: &FunctionTypes) -> Vec<ir::Block> {
        flow.inputs
            .iter()
            .map(|inputs| {
                let block = self.builder.create_block();
                for input in inputs {
                    self.builder
                        .append_block_param(block, self.types.value_type(input));
                    if self.tracks_initialization(input) {
                        self.builder.append_block_param(block, ir::types::I64);
                    }
                }
                block
            })
            .collect()
    }

    pub fn allocate(&mut self, ty: &Ty) -> ir::Value {
        let layout = self.types.layout(ty);
        self.allocate_bytes(
            u32::try_from(layout.size).expect("validated stack size"),
            layout.align as u32,
        )
    }

    pub fn allocate_bytes(&mut self, size: u32, align: u32) -> ir::Value {
        let slot = self.builder.create_sized_stack_slot(ir::StackSlotData::new(
            ir::StackSlotKind::ExplicitSlot,
            size,
            align.trailing_zeros() as u8,
        ));
        let address = self.builder.ins().stack_addr(ir::types::I64, slot, 0);
        self.builder.emit_small_memset(
            self.module.target_config(),
            address,
            0,
            u64::from(size),
            1,
            ir::MemFlagsData::new(),
        );
        address
    }

    pub fn offset(&mut self, address: ir::Value, offset: usize) -> ir::Value {
        if offset == 0 {
            address
        } else {
            self.builder.ins().iadd_imm_s(address, offset as i64)
        }
    }

    pub fn copy_storage(&mut self, ty: &Ty, destination: ir::Value, source: ir::Value) {
        self.builder.emit_small_memory_copy(
            self.module.target_config(),
            destination,
            source,
            self.types.layout(ty).size as u64,
            1,
            1,
            false,
            ir::MemFlagsData::new(),
        );
    }

    /// Transfer bits into an independently owned snapshot. Lifecycle retains are
    /// a separate operation so Take/TransferLoad can share this representation.
    pub fn read(&mut self, ty: &Ty, address: ir::Value) -> ir::Value {
        if let Some(scalar) = self.types.scalar(ty) {
            return self
                .builder
                .ins()
                .load(scalar, ir::MemFlagsData::new(), address, 0);
        }
        let snapshot = self.allocate(ty);
        self.copy_storage(ty, snapshot, address);
        snapshot
    }

    pub fn write(&mut self, ty: &Ty, address: ir::Value, value: ir::Value) {
        if self.types.scalar(ty).is_some() {
            self.builder
                .ins()
                .store(ir::MemFlagsData::new(), value, address, 0);
        } else {
            self.copy_storage(ty, address, value);
        }
    }

    fn edge_values(&mut self, operands: &[Operand]) -> Vec<ir::BlockArg> {
        let mut values = Vec::new();
        for operand in operands {
            // Copy on both sides of an edge to preserve parallel aggregate swaps.
            let value = if self.types.scalar(&operand.ty).is_some() {
                operand.value
            } else {
                self.read(&operand.ty, operand.value)
            };
            values.push(ir::BlockArg::Value(value));
            if self.tracks_initialization(&operand.ty) {
                let live = operand
                    .live
                    .unwrap_or_else(|| self.builder.ins().iconst(ir::types::I64, 0));
                values.push(ir::BlockArg::Value(live));
            }
        }
        values
    }

    fn tracks_initialization(&self, ty: &Ty) -> bool {
        matches!(ty, Ty::Pointer { pointee } if pointee.needs_drop(self.types.table))
    }

    fn terminate(
        &mut self,
        blocks: &[ir::Block],
        terminator: &Terminator,
        exit: Exit,
        operands: &[Operand],
    ) {
        if matches!(terminator, Terminator::Return) {
            if let Some(destination) = self.result {
                self.write(&self.input.result, destination, operands[0].value);
                self.builder.ins().return_(&[]);
            } else {
                self.builder.ins().return_(&[operands[0].value]);
            }
            return;
        }
        let mut values = self.edge_values(operands);
        match *terminator {
            Terminator::Return => unreachable!(),
            Terminator::Merge => {
                let Exit::Merge { block } = exit else {
                    unreachable!("verified merge")
                };
                self.builder.ins().jump(blocks[block], &values);
            }
            Terminator::Continue => {
                let Exit::Continue { condition } = exit else {
                    unreachable!("verified continue")
                };
                self.builder.ins().jump(blocks[condition], &values);
            }
            Terminator::LoopTest => {
                let Exit::LoopTest { body, next } = exit else {
                    unreachable!("verified loop test")
                };
                values.pop();
                self.builder.ins().brif(
                    operands.last().unwrap().value,
                    blocks[body],
                    &values,
                    blocks[next],
                    &values,
                );
            }
            Terminator::If { then, els, .. } => {
                values.pop();
                self.builder.ins().brif(
                    operands.last().unwrap().value,
                    blocks[then.index()],
                    &values,
                    blocks[els.index()],
                    &values,
                );
            }
            Terminator::Loop { condition, .. } => {
                self.builder.ins().jump(blocks[condition.index()], &values);
            }
        }
    }

    fn instruction(
        &mut self,
        instruction: &Instr,
        args: &[Operand],
        result: Option<&Ty>,
    ) -> Result<Option<ir::Value>, Error> {
        let value = match instruction {
            Instr::Push { value } => self.literal(value, result.unwrap())?,
            Instr::LocalAddress { local } => self.locals[local.index()],
            Instr::TakeLocal { local } => {
                let value = self.read(result.unwrap(), self.locals[local.index()]);
                self.mark_local(*local, false);
                value
            }
            Instr::SetLocal { local } => {
                self.drop_local(*local)?;
                self.write(&args[0].ty, self.locals[local.index()], args[0].value);
                self.mark_local(*local, true);
                return Ok(None);
            }
            Instr::DropLocal { local } => {
                self.drop_local(*local)?;
                return Ok(None);
            }
            Instr::ForgetLocal { local } => {
                self.mark_local(*local, false);
                return Ok(None);
            }
            Instr::Discard => {
                self.drop_value(&args[0].ty, args[0].value)?;
                return Ok(None);
            }
            Instr::Load | Instr::TransferLoad => {
                let value = self.read(result.unwrap(), args[0].value);
                if matches!(instruction, Instr::Load) {
                    self.retain(result.unwrap(), value)?;
                }
                value
            }
            Instr::Store => {
                self.drop_stored(&args[1].ty, args[0].value, args[0].live)?;
                self.retain(&args[1].ty, args[1].value)?;
                self.write(&args[1].ty, args[0].value, args[1].value);
                self.mark_address(args[0].live, true)?;
                args[1].value
            }
            Instr::Replace => {
                let previous = self.read(result.unwrap(), args[0].value);
                self.write(&args[1].ty, args[0].value, args[1].value);
                previous
            }
            Instr::PointerCast { .. } | Instr::Ascribe { .. } => args[0].value,
            Instr::NumericCast { ty } => numeric::convert(
                self.module,
                &mut self.builder,
                &args[0],
                ty,
                self.types
                    .runtime
                    .get("resin_fail")
                    .map_or("resin_fail", |name| name.as_ref()),
            )?,
            Instr::Function { function } => {
                let reference = self.module.declare_func_in_func(
                    self.functions[function.index()].unwrap(),
                    self.builder.func,
                );
                self.builder.ins().func_addr(ir::types::I64, reference)
            }
            Instr::Call { .. } => self.call(args)?,
            Instr::CallBuiltin { name, .. } => match name.as_ref() {
                "format_bytes" => self.format_bytes(args, result.unwrap())?,
                "string_from_bytes" => self.copy_bytes(args, result.unwrap())?,
                _ => numeric::builtin(
                    self.module,
                    &mut self.builder,
                    name,
                    args,
                    self.types
                        .runtime
                        .get("resin_fail")
                        .map_or("resin_fail", |name| name.as_ref()),
                )?,
            },
            Instr::PointerIndex => self.pointer_index(args)?,
            Instr::PointerRange => self.pointer_range(args)?,
            Instr::PointerBytes => self.pointer_bytes(args, result.unwrap())?,
            Instr::GpuViewAllocate
            | Instr::GpuViewOffset
            | Instr::GpuViewRange { .. }
            | Instr::GpuViewRestrict
            | Instr::GpuViewLoad { .. }
            | Instr::GpuViewStore
            | Instr::GpuViewReplace
            | Instr::GpuViewCopyTo
            | Instr::GpuViewCopyImage
            | Instr::GpuComputePipeline { .. }
            | Instr::GpuGraphicsPipeline { .. }
            | Instr::GpuDispatch { .. }
            | Instr::GpuDraw { .. }
            | Instr::GpuArgumentsDispatch
            | Instr::GpuArgumentsDraw => {
                self.gpu_instruction(instruction, args, result.unwrap())?
            }
            Instr::OwnerAllocate { element } => {
                self.owner_allocate(element, args, result.unwrap())?
            }
            Instr::OwnerData { .. }
            | Instr::OwnerLength
            | Instr::OwnerDowngrade
            | Instr::OwnerUpgrade => self.owner_operation(instruction, args, result.unwrap())?,
            Instr::WeakEmpty => self.builder.ins().iconst(ir::types::I64, 0),
            Instr::MakeRecord { .. } | Instr::MakeArray { .. } => {
                self.construct(args, result.unwrap())
            }
            Instr::AccessStatic { index } => self.access_static(&args[0], *index, result.unwrap()),
            Instr::AccessDynamic => self.access_dynamic(args, result.unwrap())?,
            Instr::MakeVariant { ty, tag } => self.variant(ty, tag, &args[0]),
            Instr::IsVariant { tag } => self.is_variant(&args[0], tag),
            Instr::VariantPayload { tag } => self.payload(&args[0], tag, result.unwrap())?,
            Instr::Widen { ty } => self.widen(&args[0], ty)?,
            Instr::ExcludeNone => {
                let invalid = self.is_variant(&args[0], &Case::Type(Ty::None));
                self.guard(invalid, "cannot unwrap None")?;
                self.widen(&args[0], result.unwrap())?
            }
            _ => {
                return Err(unsupported(format!(
                    "instruction {instruction:?} is not implemented"
                )));
            }
        };
        if matches!(
            instruction,
            Instr::AccessStatic { .. } | Instr::AccessDynamic
        ) && !matches!(self.types.shape(&args[0].ty), Ty::Pointer { .. })
        {
            self.retain(result.unwrap(), value)?;
            self.drop_value(&args[0].ty, args[0].value)?;
        }
        if matches!(
            instruction,
            Instr::IsVariant { .. }
                | Instr::CallBuiltin { .. }
                | Instr::OwnerAllocate { .. }
                | Instr::OwnerData { .. }
                | Instr::OwnerLength
                | Instr::OwnerDowngrade
                | Instr::OwnerUpgrade
                | Instr::GpuViewAllocate
                | Instr::GpuViewLoad { .. }
                | Instr::GpuViewStore
                | Instr::GpuViewReplace
                | Instr::GpuViewCopyTo
                | Instr::GpuViewCopyImage
                | Instr::GpuComputePipeline { .. }
                | Instr::GpuGraphicsPipeline { .. }
                | Instr::GpuDispatch { .. }
                | Instr::GpuDraw { .. }
                | Instr::GpuArgumentsDispatch
                | Instr::GpuArgumentsDraw
        ) {
            for arg in args {
                self.drop_value(&arg.ty, arg.value)?;
            }
        }
        Ok(Some(value))
    }

    pub fn call(&mut self, args: &[Operand]) -> Result<ir::Value, Error> {
        let Ty::Function { params, result } = self.types.shape(&args[0].ty) else {
            unreachable!("verified callee")
        };
        let result_address = self
            .types
            .scalar(result)
            .is_none()
            .then(|| self.allocate(result));
        let mut values = result_address.into_iter().collect::<Vec<_>>();
        values.extend(args[1..].iter().map(|arg| arg.value));
        let instruction = if let Some(function) = args[0].function {
            let reference = self
                .module
                .declare_func_in_func(function, self.builder.func);
            self.builder.ins().call(reference, &values)
        } else {
            let invalid =
                self.builder
                    .ins()
                    .icmp_imm_s(ir::condcodes::IntCC::Equal, args[0].value, 0);
            self.guard(invalid, "calling an uninitialized function")?;
            let signature = self.types.signature(self.module, params, result);
            let reference = self.builder.import_signature(signature);
            self.builder
                .ins()
                .call_indirect(reference, args[0].value, &values)
        };
        Ok(result_address.unwrap_or_else(|| self.builder.inst_results(instruction)[0]))
    }

    fn pointer_index(&mut self, args: &[Operand]) -> Result<ir::Value, Error> {
        let invalid = self.builder.ins().icmp(
            ir::condcodes::IntCC::UnsignedGreaterThanOrEqual,
            args[2].value,
            args[1].value,
        );
        self.guard(invalid, "array index out of bounds")?;
        let Ty::Pointer { pointee } = self.types.shape(&args[0].ty) else {
            unreachable!()
        };
        Ok(self.pointer_offset(args[0].value, args[2].value, pointee))
    }

    fn pointer_range(&mut self, args: &[Operand]) -> Result<ir::Value, Error> {
        let invalid_start = self.builder.ins().icmp(
            ir::condcodes::IntCC::UnsignedGreaterThan,
            args[2].value,
            args[1].value,
        );
        let available = self.builder.ins().isub(args[1].value, args[2].value);
        let invalid_count = self.builder.ins().icmp(
            ir::condcodes::IntCC::UnsignedGreaterThan,
            args[3].value,
            available,
        );
        let invalid = self.builder.ins().bor(invalid_start, invalid_count);
        self.guard(invalid, "span slice out of bounds")?;
        let Ty::Pointer { pointee } = self.types.shape(&args[0].ty) else {
            unreachable!()
        };
        Ok(self.pointer_offset(args[0].value, args[2].value, pointee))
    }

    pub fn pointer_offset(
        &mut self,
        pointer: ir::Value,
        index: ir::Value,
        pointee: &Ty,
    ) -> ir::Value {
        let offset = self
            .builder
            .ins()
            .imul_imm_s(index, self.types.layout(pointee).size as i64);
        self.builder.ins().iadd(pointer, offset)
    }
}

// Each structured child has one lexical exit. Assign destinations without turning
// LIR into a second public graph or inferring stack types again.
fn exits(function: &resin_lir::Function) -> Vec<Exit> {
    let mut exits = vec![Exit::Return; function.blocks.len()];
    let mut pending = vec![(function.entry.index(), Exit::Return)];
    while let Some((index, exit)) = pending.pop() {
        exits[index] = exit;
        match function.blocks[index].terminator {
            Terminator::If { then, els, next } => {
                let destination = next.map_or(exit, |block| Exit::Merge {
                    block: block.index(),
                });
                pending.push((then.index(), destination));
                pending.push((els.index(), destination));
                if let Some(next) = next {
                    pending.push((next.index(), exit));
                }
            }
            Terminator::Loop {
                condition,
                body,
                next,
            } => {
                let next_index = next.map_or_else(
                    || match exit {
                        Exit::Merge { block } => block,
                        _ => unreachable!("verified tail loop merges a selection"),
                    },
                    |block| block.index(),
                );
                pending.push((
                    condition.index(),
                    Exit::LoopTest {
                        body: body.index(),
                        next: next_index,
                    },
                ));
                pending.push((
                    body.index(),
                    Exit::Continue {
                        condition: condition.index(),
                    },
                ));
                if let Some(next) = next {
                    pending.push((next.index(), exit));
                }
            }
            _ => {}
        }
    }
    exits
}

pub(super) fn scalar_literal(
    builder: &mut FunctionBuilder<'_>,
    value: &Value,
) -> Result<ir::Value, Error> {
    Ok(match value {
        Value::Unit | Value::None => builder.ins().iconst(ir::types::I8, 0),
        Value::Bool { value } => builder.ins().iconst(ir::types::I8, i64::from(*value)),
        Value::Int8 { value } => builder.ins().iconst(ir::types::I8, i64::from(*value)),
        Value::Int16 { value } => builder.ins().iconst(ir::types::I16, i64::from(*value)),
        Value::Int32 { value } => builder.ins().iconst(ir::types::I32, i64::from(*value)),
        Value::Int64 { value } => builder.ins().iconst(ir::types::I64, *value),
        Value::UInt8 { value } => builder.ins().iconst(ir::types::I8, i64::from(*value)),
        Value::UInt16 { value } => builder.ins().iconst(ir::types::I16, i64::from(*value)),
        Value::UInt32 { value } => builder.ins().iconst(ir::types::I32, i64::from(*value)),
        Value::UInt64 { value } => builder.ins().iconst(ir::types::I64, *value as i64),
        Value::Float32 { value } => builder
            .ins()
            .f32const(ir::immediates::Ieee32::with_bits(value.to_bits())),
        Value::Float64 { value } => builder
            .ins()
            .f64const(ir::immediates::Ieee64::with_bits(value.to_bits())),
        // Dynamic addresses belong to a compiler process, never to a portable object.
        _ => return Err(unsupported(format!("literal {value:?} is not implemented"))),
    })
}
