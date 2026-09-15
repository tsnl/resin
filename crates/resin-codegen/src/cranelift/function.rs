use super::{failure, numeric, scalar_type, signature, unsupported};
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
    pub value: ir::Value,
    function: Option<FuncId>,
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
    index: usize,
    functions: &[Option<FuncId>],
    cancellation: &Cancellation,
) -> Result<(), GenerationError> {
    let input = &checked.module().functions[index];
    let flow = &checked.analysis().functions[index];
    let mut context = module.make_context();
    context.func.signature = signature(
        module,
        &input.locals[..input.parameter_count]
            .iter()
            .map(|local| local.ty.clone())
            .collect::<Vec<_>>(),
        &input.result,
    )?;
    let mut frontend = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut context.func, &mut frontend);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let blocks = blocks(&mut builder, flow)?;
    let locals = locals(&mut builder, input)?;
    builder.ins().jump(blocks[input.entry.index()], &[]);
    let exits = exits(input);
    for (block_index, block) in input.blocks.iter().enumerate() {
        cancellation.check()?;
        builder.switch_to_block(blocks[block_index]);
        let mut operands = builder
            .block_params(blocks[block_index])
            .iter()
            .zip(&flow.inputs[block_index])
            .map(|(&value, ty)| Operand {
                ty: ty.clone(),
                value,
                function: None,
            })
            .collect::<Vec<_>>();
        let mut diverged = false;
        for (position, instruction) in block.instrs.iter().enumerate() {
            if matches!(instruction, Instr::Eliminate { .. }) {
                builder.ins().trap(ir::TrapCode::unwrap_user(1));
                diverged = true;
                break;
            }
            let count = flow.operand_count(BlockId::from_index(block_index), position);
            let args = operands.split_off(operands.len() - count);
            let result = flow.results[block_index][position].as_ref();
            let value = instruction_value(
                &mut builder,
                module,
                functions,
                &locals,
                instruction,
                &args,
                result,
            )
            .map_err(|error| {
                Error::at(
                    checked.module(),
                    index,
                    Some((block_index, position)),
                    error,
                )
            })?;
            if let Some(ty) = result {
                operands.push(Operand {
                    ty: ty.clone(),
                    value: value.expect("verified instruction has a value"),
                    function: if let Instr::Function { function } = instruction {
                        functions[function.index()]
                    } else {
                        None
                    },
                });
            }
        }
        if !diverged {
            terminate(
                &mut builder,
                &blocks,
                &block.terminator,
                exits[block_index],
                &operands,
            );
        }
    }
    builder.seal_all_blocks();
    builder.finalize(module.target_config());
    cancellation.check()?;
    module
        .define_function(functions[index].unwrap(), &mut context)
        .map_err(failure)?;
    Ok(())
}

fn blocks(
    builder: &mut FunctionBuilder<'_>,
    flow: &FunctionTypes,
) -> Result<Vec<ir::Block>, Error> {
    flow.inputs
        .iter()
        .map(|inputs| {
            let block = builder.create_block();
            for input in inputs {
                builder.append_block_param(block, scalar_type(input)?);
            }
            Ok(block)
        })
        .collect()
}

fn locals(
    builder: &mut FunctionBuilder<'_>,
    input: &resin_lir::Function,
) -> Result<Vec<ir::StackSlot>, Error> {
    let parameters = builder
        .block_params(builder.current_block().unwrap())
        .to_vec();
    input
        .locals
        .iter()
        .enumerate()
        .map(|(index, local)| {
            let ty = scalar_type(&local.ty)?;
            let slot = builder.create_sized_stack_slot(ir::StackSlotData::new(
                ir::StackSlotKind::ExplicitSlot,
                ty.bytes(),
                ty.bytes().trailing_zeros() as u8,
            ));
            // C generation zero-initializes plain storage too; pointer reads of storage
            // that source initialization permits observe the same initial bits.
            let value = if index < input.parameter_count {
                parameters[index]
            } else {
                numeric::zero(builder, ty)
            };
            builder.ins().stack_store(ir::types::I64, value, slot, 0);
            Ok(slot)
        })
        .collect()
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

fn terminate(
    builder: &mut FunctionBuilder<'_>,
    blocks: &[ir::Block],
    terminator: &Terminator,
    exit: Exit,
    operands: &[Operand],
) {
    let mut values = operands
        .iter()
        .map(|operand| ir::BlockArg::Value(operand.value))
        .collect::<Vec<_>>();
    match *terminator {
        Terminator::Return => {
            builder.ins().return_(&[operands[0].value]);
        }
        Terminator::Merge => {
            let Exit::Merge { block } = exit else {
                unreachable!("verified merge")
            };
            builder.ins().jump(blocks[block], &values);
        }
        Terminator::Continue => {
            let Exit::Continue { condition } = exit else {
                unreachable!("verified continue")
            };
            builder.ins().jump(blocks[condition], &values);
        }
        Terminator::LoopTest => {
            let Exit::LoopTest { body, next } = exit else {
                unreachable!("verified loop test")
            };
            values.pop();
            builder.ins().brif(
                operands.last().unwrap().value,
                blocks[body],
                &values,
                blocks[next],
                &values,
            );
        }
        Terminator::If { then, els, .. } => {
            values.pop();
            builder.ins().brif(
                operands.last().unwrap().value,
                blocks[then.index()],
                &values,
                blocks[els.index()],
                &values,
            );
        }
        Terminator::Loop { condition, .. } => {
            builder.ins().jump(blocks[condition.index()], &values);
        }
    }
}

fn instruction_value(
    builder: &mut FunctionBuilder<'_>,
    module: &mut ObjectModule,
    functions: &[Option<FuncId>],
    locals: &[ir::StackSlot],
    instruction: &Instr,
    args: &[Operand],
    result: Option<&Ty>,
) -> Result<Option<ir::Value>, Error> {
    let ty = result.map(scalar_type).transpose()?;
    let value = match instruction {
        Instr::Push { value } => literal(builder, value)?,
        Instr::LocalAddress { local } => {
            builder
                .ins()
                .stack_addr(ir::types::I64, locals[local.index()], 0)
        }
        Instr::TakeLocal { local } => {
            builder
                .ins()
                .stack_load(ir::types::I64, ty.unwrap(), locals[local.index()], 0)
        }
        Instr::SetLocal { local } => {
            builder
                .ins()
                .stack_store(ir::types::I64, args[0].value, locals[local.index()], 0);
            return Ok(None);
        }
        Instr::DropLocal { .. } | Instr::ForgetLocal { .. } | Instr::Discard => return Ok(None),
        Instr::Load | Instr::TransferLoad => {
            builder
                .ins()
                .load(ty.unwrap(), ir::MemFlagsData::new(), args[0].value, 0)
        }
        Instr::Store => {
            builder
                .ins()
                .store(ir::MemFlagsData::new(), args[1].value, args[0].value, 0);
            args[1].value
        }
        Instr::Replace => {
            let old = builder
                .ins()
                .load(ty.unwrap(), ir::MemFlagsData::new(), args[0].value, 0);
            builder
                .ins()
                .store(ir::MemFlagsData::new(), args[1].value, args[0].value, 0);
            old
        }
        Instr::PointerCast { .. } => args[0].value,
        Instr::Ascribe { ty } if *ty == args[0].ty => args[0].value,
        Instr::NumericCast { ty } => numeric::convert(builder, &args[0], ty)?,
        Instr::Function { function } => {
            let reference =
                module.declare_func_in_func(functions[function.index()].unwrap(), builder.func);
            builder.ins().func_addr(ir::types::I64, reference)
        }
        Instr::Call { .. } => call(builder, module, args)?,
        Instr::CallBuiltin { name, .. } => numeric::builtin(builder, name, args)?,
        Instr::PointerIndex => pointer_index(builder, args)?,
        Instr::PointerRange => pointer_range(builder, args)?,
        _ => {
            return Err(unsupported(format!(
                "instruction {instruction:?} is not implemented"
            )));
        }
    };
    Ok(Some(value))
}

fn call(
    builder: &mut FunctionBuilder<'_>,
    module: &mut ObjectModule,
    args: &[Operand],
) -> Result<ir::Value, Error> {
    let Ty::Function { params, result } = &args[0].ty else {
        unreachable!("verified callee")
    };
    let values = args[1..].iter().map(|arg| arg.value).collect::<Vec<_>>();
    let instruction = if let Some(function) = args[0].function {
        let reference = module.declare_func_in_func(function, builder.func);
        builder.ins().call(reference, &values)
    } else {
        builder
            .ins()
            .trapz(args[0].value, ir::TrapCode::unwrap_user(1));
        let reference = builder.import_signature(signature(module, params, result)?);
        builder
            .ins()
            .call_indirect(reference, args[0].value, &values)
    };
    Ok(builder.inst_results(instruction)[0])
}

fn pointer_index(builder: &mut FunctionBuilder<'_>, args: &[Operand]) -> Result<ir::Value, Error> {
    let invalid = builder.ins().icmp(
        ir::condcodes::IntCC::UnsignedGreaterThanOrEqual,
        args[2].value,
        args[1].value,
    );
    builder.ins().trapnz(invalid, ir::TrapCode::unwrap_user(2));
    pointer_offset(builder, &args[0], args[2].value)
}

fn pointer_range(builder: &mut FunctionBuilder<'_>, args: &[Operand]) -> Result<ir::Value, Error> {
    let invalid_start = builder.ins().icmp(
        ir::condcodes::IntCC::UnsignedGreaterThan,
        args[2].value,
        args[1].value,
    );
    let available = builder.ins().isub(args[1].value, args[2].value);
    let invalid_count = builder.ins().icmp(
        ir::condcodes::IntCC::UnsignedGreaterThan,
        args[3].value,
        available,
    );
    let invalid = builder.ins().bor(invalid_start, invalid_count);
    builder.ins().trapnz(invalid, ir::TrapCode::unwrap_user(2));
    pointer_offset(builder, &args[0], args[2].value)
}

fn pointer_offset(
    builder: &mut FunctionBuilder<'_>,
    base: &Operand,
    index: ir::Value,
) -> Result<ir::Value, Error> {
    let Ty::Pointer { pointee } = &base.ty else {
        unreachable!("verified pointer")
    };
    let offset = builder
        .ins()
        .imul_imm_s(index, i64::from(scalar_type(pointee)?.bytes()));
    Ok(builder.ins().iadd(base.value, offset))
}

fn literal(builder: &mut FunctionBuilder<'_>, value: &Value) -> Result<ir::Value, Error> {
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
