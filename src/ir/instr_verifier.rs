//! Stack and control-flow verification for typed IR instructions.
//!
//! IR generation must call [`verify`] before handing a module to any backend.

use std::{collections::VecDeque, fmt};

use super::{BlockId, Function, FunctionId, Instr, Module, RecordField, Terminator, Ty, Value};

/// The height component of an instruction's stack effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackEffect {
    pub pops: usize,
    pub pushes: usize,
}

impl Instr {
    pub fn stack_effect(&self) -> StackEffect {
        match self {
            Self::Push { .. }
            | Self::LocalAddress { .. }
            | Self::GlobalAddress { .. }
            | Self::NonLocalAddress { .. } => StackEffect { pops: 0, pushes: 1 },
            Self::AccessStatic { .. } | Self::Load => StackEffect { pops: 1, pushes: 1 },
            Self::AccessDynamic | Self::Store => StackEffect { pops: 2, pushes: 1 },
            Self::Discard => StackEffect { pops: 1, pushes: 0 },
            Self::MakeRecord { fields } => StackEffect {
                pops: fields.len(),
                pushes: 1,
            },
            Self::MakeArray { elements, .. } => StackEffect {
                pops: *elements,
                pushes: 1,
            },
            Self::MakeClosure { captures, .. } => StackEffect {
                pops: *captures,
                pushes: 1,
            },
            Self::Call { args } => StackEffect {
                pops: args + 1,
                pushes: 1,
            },
            Self::CallBuiltin { params, .. } => StackEffect {
                pops: params.len(),
                pushes: 1,
            },
        }
    }
}

impl Terminator {
    pub const fn stack_effect(&self) -> StackEffect {
        match self {
            Self::Break { .. } => StackEffect { pops: 0, pushes: 0 },
            Self::Branch { .. } | Self::Return => StackEffect { pops: 1, pushes: 0 },
        }
    }
}

/// The location at which IR verification failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyError {
    pub function: FunctionId,
    pub basic_block: BlockId,
    pub instruction: Option<usize>,
    pub kind: VerifyErrorKind,
}

/// A violation of the typed stack or control-flow invariants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyErrorKind {
    InvalidLocal { local: usize },
    InvalidGlobal { global: usize },
    InvalidNonLocal { nonlocal: usize },
    InvalidFunction { function: usize },
    InvalidBasicBlock { basic_block: usize },
    UnreachableBasicBlock,
    StackUnderflow { needed: usize, available: usize },
    InvalidImmediate,
    TypeMismatch { expected: Ty, found: Ty },
    ExpectedPointer { found: Ty },
    ExpectedAggregate { found: Ty },
    ExpectedArray { found: Ty },
    ExpectedFunction { found: Ty },
    ExpectedInteger { found: Ty },
    StaticIndexOutOfBounds { index: usize, length: usize },
    ArgumentCount { expected: usize, found: usize },
    ConflictingBasicBlockStack { expected: Vec<Ty>, found: Vec<Ty> },
    InvalidReturnStack { expected: Ty, found: Vec<Ty> },
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid IR in function {}, basic block {}",
            self.function.index(),
            self.basic_block.index()
        )?;
        if let Some(instruction) = self.instruction {
            write!(f, ", instruction {instruction}")?;
        }
        write!(f, ": {:?}", self.kind)
    }
}

impl std::error::Error for VerifyError {}

/// Verify every reachable basic block and every control-flow edge in a module.
///
/// The inferred entry stack for each basic block is kept only for the duration of
/// verification. Multiple incoming edges must infer exactly the same types.
pub fn verify(module: &Module) -> Result<(), VerifyError> {
    for (index, function) in module.functions.iter().enumerate() {
        verify_function(module, FunctionId::from_index(index), function)?;
    }
    Ok(())
}

fn verify_function(
    module: &Module,
    function_id: FunctionId,
    function: &Function,
) -> Result<(), VerifyError> {
    let entry = function.entry;
    let location = Location::terminator(function_id, entry);

    if function.blocks.get(entry.index()).is_none() {
        return Err(location.error(VerifyErrorKind::InvalidBasicBlock {
            basic_block: entry.index(),
        }));
    }

    for local in &function.params {
        if function.locals.get(local.index()).is_none() {
            return Err(location.error(VerifyErrorKind::InvalidLocal {
                local: local.index(),
            }));
        }
    }

    let mut entries = vec![None; function.blocks.len()];
    entries[entry.index()] = Some(Vec::new());
    let mut pending = VecDeque::from([entry]);

    while let Some(basic_block_id) = pending.pop_front() {
        let basic_block = &function.blocks[basic_block_id.index()];
        let mut stack = entries[basic_block_id.index()]
            .clone()
            .expect("queued basic blocks always have an inferred input stack");

        for (instruction, instr) in basic_block.instrs.iter().enumerate() {
            let location = Location::instruction(function_id, basic_block_id, instruction);
            verify_instr(module, function, instr, &mut stack, location)?;
        }

        let location = Location::terminator(function_id, basic_block_id);
        match basic_block.terminator {
            Terminator::Break { target } => propagate(
                function,
                target,
                stack,
                &mut entries,
                &mut pending,
                location,
            )?,
            Terminator::Branch { then, els } => {
                let condition = pop(&mut stack, 1, location)?
                    .pop()
                    .expect("one stack value was requested");
                expect_type(Ty::Bool, condition, location)?;
                propagate(
                    function,
                    then,
                    stack.clone(),
                    &mut entries,
                    &mut pending,
                    location,
                )?;
                propagate(function, els, stack, &mut entries, &mut pending, location)?;
            }
            Terminator::Return => {
                if stack.as_slice() != [function.result.clone()] {
                    return Err(location.error(VerifyErrorKind::InvalidReturnStack {
                        expected: function.result.clone(),
                        found: stack,
                    }));
                }
            }
        }
    }

    if let Some(basic_block) = entries.iter().position(Option::is_none) {
        return Err(
            Location::terminator(function_id, BlockId::from_index(basic_block))
                .error(VerifyErrorKind::UnreachableBasicBlock),
        );
    }

    Ok(())
}

fn verify_instr(
    module: &Module,
    function: &Function,
    instr: &Instr,
    stack: &mut Vec<Ty>,
    location: Location,
) -> Result<(), VerifyError> {
    match instr {
        Instr::Push { value } => stack.push(immediate_ty(value, location)?),
        Instr::LocalAddress { local } => {
            let local = function.locals.get(local.index()).ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidLocal {
                    local: local.index(),
                })
            })?;
            stack.push(Ty::Pointer {
                pointee: Box::new(local.ty.clone()),
            });
        }
        Instr::GlobalAddress { global } => {
            let global = module.globals.get(global.index()).ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidGlobal {
                    global: global.index(),
                })
            })?;
            stack.push(Ty::Pointer {
                pointee: Box::new(global.ty.clone()),
            });
        }
        Instr::NonLocalAddress { nonlocal } => {
            let nonlocal = function.nonlocals.get(nonlocal.index()).ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidNonLocal {
                    nonlocal: nonlocal.index(),
                })
            })?;
            stack.push(Ty::Pointer {
                pointee: Box::new(nonlocal.ty.clone()),
            });
        }
        Instr::AccessStatic { index } => {
            let source = pop_one(stack, location)?;
            stack.push(project_static(source, *index, location)?);
        }
        Instr::AccessDynamic => {
            let index = pop_one(stack, location)?;
            if !index.is_integer() {
                return Err(location.error(VerifyErrorKind::ExpectedInteger { found: index }));
            }
            let source = pop_one(stack, location)?;
            stack.push(project_dynamic(source, location)?);
        }
        Instr::Load => {
            let address = pop_one(stack, location)?;
            let Ty::Pointer { pointee } = address else {
                return Err(location.error(VerifyErrorKind::ExpectedPointer { found: address }));
            };
            stack.push(*pointee);
        }
        Instr::Store => {
            let value = pop_one(stack, location)?;
            let address = pop_one(stack, location)?;
            let Ty::Pointer { pointee } = address else {
                return Err(location.error(VerifyErrorKind::ExpectedPointer { found: address }));
            };
            expect_type(*pointee, value.clone(), location)?;
            stack.push(value);
        }
        Instr::Discard => {
            pop_one(stack, location)?;
        }
        Instr::MakeRecord { fields } => {
            let values = pop(stack, fields.len(), location)?;
            stack.push(Ty::Record {
                fields: fields
                    .iter()
                    .cloned()
                    .zip(values)
                    .map(|(name, ty)| RecordField { name, ty })
                    .collect(),
            });
        }
        Instr::MakeArray { elements, element } => {
            let values = pop(stack, *elements, location)?;
            for value in values {
                expect_type(element.clone(), value, location)?;
            }
            stack.push(Ty::Array {
                element: Box::new(element.clone()),
                length: *elements,
            });
        }
        Instr::MakeClosure {
            function: target,
            captures,
        } => {
            let target_function = module.functions.get(target.index()).ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidFunction {
                    function: target.index(),
                })
            })?;
            if *captures != target_function.nonlocals.len() {
                return Err(location.error(VerifyErrorKind::ArgumentCount {
                    expected: target_function.nonlocals.len(),
                    found: *captures,
                }));
            }
            let values = pop(stack, *captures, location)?;
            let expected: Vec<_> = target_function
                .nonlocals
                .iter()
                .map(|nonlocal| nonlocal.ty.clone())
                .collect();
            expect_types(&expected, &values, location)?;
            stack.push(function_ty(target_function, location)?);
        }
        Instr::Call { args } => {
            let values = pop(stack, *args, location)?;
            let callee = pop_one(stack, location)?;
            let Ty::Function { params, result } = callee else {
                return Err(location.error(VerifyErrorKind::ExpectedFunction { found: callee }));
            };
            if *args != params.len() {
                return Err(location.error(VerifyErrorKind::ArgumentCount {
                    expected: params.len(),
                    found: *args,
                }));
            }
            expect_types(&params, &values, location)?;
            stack.push(*result);
        }
        Instr::CallBuiltin { params, result, .. } => {
            let values = pop(stack, params.len(), location)?;
            expect_types(params, &values, location)?;
            stack.push(result.clone());
        }
    }
    Ok(())
}

fn immediate_ty(value: &Value, location: Location) -> Result<Ty, VerifyError> {
    let ty = match value {
        Value::Unit => Ty::Unit,
        Value::Bool { .. } => Ty::Bool,
        Value::Int8 { .. } => Ty::Int8,
        Value::Int16 { .. } => Ty::Int16,
        Value::Int32 { .. } => Ty::Int32,
        Value::Int64 { .. } => Ty::Int64,
        Value::UInt8 { .. } => Ty::UInt8,
        Value::UInt16 { .. } => Ty::UInt16,
        Value::UInt32 { .. } => Ty::UInt32,
        Value::UInt64 { .. } => Ty::UInt64,
        Value::Float32 { .. } => Ty::Float32,
        Value::Float64 { .. } => Ty::Float64,
        Value::Array { value } => {
            for element in &value.elements {
                let found = immediate_ty(element, location)?;
                expect_type(value.element_ty.clone(), found, location)?;
            }
            Ty::Array {
                element: Box::new(value.element_ty.clone()),
                length: value.elements.len(),
            }
        }
        Value::Record { value } => Ty::Record {
            fields: value
                .fields
                .iter()
                .map(|field| {
                    immediate_ty(&field.value, location).map(|ty| RecordField {
                        name: field.name.clone(),
                        ty,
                    })
                })
                .collect::<Result<_, _>>()?,
        },
        Value::StaticAddress { .. } | Value::DynamicAddress { .. } | Value::Closure { .. } => {
            return Err(location.error(VerifyErrorKind::InvalidImmediate));
        }
    };
    Ok(ty)
}

fn project_static(source: Ty, index: usize, location: Location) -> Result<Ty, VerifyError> {
    match source {
        Ty::Pointer { pointee } => Ok(Ty::Pointer {
            pointee: Box::new(project_static(*pointee, index, location)?),
        }),
        Ty::Record { fields } => fields
            .get(index)
            .map(|field| field.ty.clone())
            .ok_or_else(|| {
                location.error(VerifyErrorKind::StaticIndexOutOfBounds {
                    index,
                    length: fields.len(),
                })
            }),
        Ty::Array { element, length } => {
            if index >= length {
                Err(location.error(VerifyErrorKind::StaticIndexOutOfBounds { index, length }))
            } else {
                Ok(*element)
            }
        }
        found => Err(location.error(VerifyErrorKind::ExpectedAggregate { found })),
    }
}

fn project_dynamic(source: Ty, location: Location) -> Result<Ty, VerifyError> {
    match source {
        Ty::Pointer { pointee } => match *pointee {
            Ty::Array { element, .. } => Ok(Ty::Pointer { pointee: element }),
            found => Err(location.error(VerifyErrorKind::ExpectedArray { found })),
        },
        Ty::Array { element, .. } => Ok(*element),
        found => Err(location.error(VerifyErrorKind::ExpectedArray { found })),
    }
}

fn function_ty(function: &Function, location: Location) -> Result<Ty, VerifyError> {
    Ok(Ty::Function {
        params: local_types(function, &function.params, location)?,
        result: Box::new(function.result.clone()),
    })
}

fn local_types(
    function: &Function,
    locals: &[super::LocalId],
    location: Location,
) -> Result<Vec<Ty>, VerifyError> {
    locals
        .iter()
        .map(|local| {
            function
                .locals
                .get(local.index())
                .map(|local| local.ty.clone())
                .ok_or_else(|| {
                    location.error(VerifyErrorKind::InvalidLocal {
                        local: local.index(),
                    })
                })
        })
        .collect()
}

fn expect_types(expected: &[Ty], found: &[Ty], location: Location) -> Result<(), VerifyError> {
    if expected.len() != found.len() {
        return Err(location.error(VerifyErrorKind::ArgumentCount {
            expected: expected.len(),
            found: found.len(),
        }));
    }
    for (expected, found) in expected.iter().cloned().zip(found.iter().cloned()) {
        expect_type(expected, found, location)?;
    }
    Ok(())
}

fn expect_type(expected: Ty, found: Ty, location: Location) -> Result<(), VerifyError> {
    if expected == found {
        Ok(())
    } else {
        Err(location.error(VerifyErrorKind::TypeMismatch { expected, found }))
    }
}

fn pop_one(stack: &mut Vec<Ty>, location: Location) -> Result<Ty, VerifyError> {
    stack.pop().ok_or_else(|| {
        location.error(VerifyErrorKind::StackUnderflow {
            needed: 1,
            available: 0,
        })
    })
}

fn pop(stack: &mut Vec<Ty>, count: usize, location: Location) -> Result<Vec<Ty>, VerifyError> {
    if stack.len() < count {
        return Err(location.error(VerifyErrorKind::StackUnderflow {
            needed: count,
            available: stack.len(),
        }));
    }
    Ok(stack.split_off(stack.len() - count))
}

fn propagate(
    function: &Function,
    target: BlockId,
    stack: Vec<Ty>,
    entries: &mut [Option<Vec<Ty>>],
    pending: &mut VecDeque<BlockId>,
    location: Location,
) -> Result<(), VerifyError> {
    if function.blocks.get(target.index()).is_none() {
        return Err(location.error(VerifyErrorKind::InvalidBasicBlock {
            basic_block: target.index(),
        }));
    }

    match &entries[target.index()] {
        None => {
            entries[target.index()] = Some(stack);
            pending.push_back(target);
        }
        Some(expected) if expected != &stack => {
            return Err(location.error(VerifyErrorKind::ConflictingBasicBlockStack {
                expected: expected.clone(),
                found: stack,
            }));
        }
        Some(_) => {}
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct Location {
    function: FunctionId,
    basic_block: BlockId,
    instruction: Option<usize>,
}

impl Location {
    const fn instruction(function: FunctionId, basic_block: BlockId, instruction: usize) -> Self {
        Self {
            function,
            basic_block,
            instruction: Some(instruction),
        }
    }

    const fn terminator(function: FunctionId, basic_block: BlockId) -> Self {
        Self {
            function,
            basic_block,
            instruction: None,
        }
    }

    fn error(self, kind: VerifyErrorKind) -> VerifyError {
        VerifyError {
            function: self.function,
            basic_block: self.basic_block,
            instruction: self.instruction,
            kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BasicBlock, Local, LocalId, NonLocal, NonLocalId};

    #[test]
    fn chained_assignment_preserves_the_value() {
        let function = Function {
            nonlocals: vec![],
            params: vec![],
            result: Ty::Int32,
            locals: vec![Local { ty: Ty::Int32 }, Local { ty: Ty::Int32 }],
            entry: BlockId::from_index(0),
            blocks: vec![BasicBlock {
                instrs: vec![
                    Instr::LocalAddress {
                        local: LocalId::from_index(0),
                    },
                    Instr::LocalAddress {
                        local: LocalId::from_index(1),
                    },
                    Instr::Push {
                        value: Value::Int32 { value: 1 },
                    },
                    Instr::Store,
                    Instr::Store,
                ],
                terminator: Terminator::Return,
            }],
        };

        verify(&Module {
            globals: vec![],
            functions: vec![function],
        })
        .unwrap();
    }

    #[test]
    fn conflicting_join_stacks_are_rejected() {
        let function = Function {
            nonlocals: vec![],
            params: vec![],
            result: Ty::Int32,
            locals: vec![],
            entry: BlockId::from_index(0),
            blocks: vec![
                BasicBlock {
                    instrs: vec![Instr::Push {
                        value: Value::Bool { value: true },
                    }],
                    terminator: Terminator::Branch {
                        then: BlockId::from_index(1),
                        els: BlockId::from_index(2),
                    },
                },
                BasicBlock {
                    instrs: vec![Instr::Push {
                        value: Value::Int32 { value: 1 },
                    }],
                    terminator: Terminator::Break {
                        target: BlockId::from_index(3),
                    },
                },
                BasicBlock {
                    instrs: vec![Instr::Push {
                        value: Value::Float32 { value: 1.0 },
                    }],
                    terminator: Terminator::Break {
                        target: BlockId::from_index(3),
                    },
                },
                BasicBlock {
                    instrs: vec![],
                    terminator: Terminator::Return,
                },
            ],
        };

        let error = verify(&Module {
            globals: vec![],
            functions: vec![function],
        })
        .unwrap_err();

        assert!(matches!(
            error.kind,
            VerifyErrorKind::ConflictingBasicBlockStack { .. }
        ));
    }

    #[test]
    fn indirect_calls_use_the_callee_on_the_stack() {
        let target = Function {
            nonlocals: vec![NonLocal { ty: Ty::Int32 }],
            params: vec![LocalId::from_index(0)],
            result: Ty::Int32,
            locals: vec![Local { ty: Ty::Int32 }],
            entry: BlockId::from_index(0),
            blocks: vec![BasicBlock {
                instrs: vec![
                    Instr::NonLocalAddress {
                        nonlocal: NonLocalId::from_index(0),
                    },
                    Instr::Load,
                ],
                terminator: Terminator::Return,
            }],
        };
        let caller = Function {
            nonlocals: vec![],
            params: vec![],
            result: Ty::Int32,
            locals: vec![],
            entry: BlockId::from_index(0),
            blocks: vec![BasicBlock {
                instrs: vec![
                    Instr::Push {
                        value: Value::Int32 { value: 9 },
                    },
                    Instr::MakeClosure {
                        function: FunctionId::from_index(0),
                        captures: 1,
                    },
                    Instr::Push {
                        value: Value::Int32 { value: 4 },
                    },
                    Instr::Call { args: 1 },
                ],
                terminator: Terminator::Return,
            }],
        };

        verify(&Module {
            globals: vec![],
            functions: vec![target, caller],
        })
        .unwrap();
    }

    #[test]
    fn loop_backedges_must_match_the_header_stack() {
        let function = Function {
            nonlocals: vec![],
            params: vec![],
            result: Ty::Unit,
            locals: vec![],
            entry: BlockId::from_index(0),
            blocks: vec![
                BasicBlock {
                    instrs: vec![],
                    terminator: Terminator::Break {
                        target: BlockId::from_index(1),
                    },
                },
                BasicBlock {
                    instrs: vec![Instr::Push {
                        value: Value::Bool { value: true },
                    }],
                    terminator: Terminator::Branch {
                        then: BlockId::from_index(2),
                        els: BlockId::from_index(3),
                    },
                },
                BasicBlock {
                    instrs: vec![],
                    terminator: Terminator::Break {
                        target: BlockId::from_index(1),
                    },
                },
                BasicBlock {
                    instrs: vec![Instr::Push { value: Value::Unit }],
                    terminator: Terminator::Return,
                },
            ],
        };

        verify(&Module {
            globals: vec![],
            functions: vec![function],
        })
        .unwrap();
    }
}
