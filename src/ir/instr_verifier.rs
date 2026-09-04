//! Stack and control-flow verification for typed IR instructions.
//!
//! IR generation must call [`verify`] before handing a module to any backend.

use std::{collections::VecDeque, fmt};

use super::{
    BlockId, Function, FunctionId, GlobalId, Instr, Module, RecordField, Terminator, Ty, TypeId,
    Value,
};

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
    pub location: VerifyLocation,
    pub kind: VerifyErrorKind,
}

/// The IR entity containing a verification error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyLocation {
    TypeDefinition {
        definition: TypeId,
    },
    Global {
        global: GlobalId,
    },
    Function {
        function: FunctionId,
    },
    BasicBlock {
        function: FunctionId,
        basic_block: BlockId,
    },
    Instruction {
        function: FunctionId,
        basic_block: BlockId,
        instruction: usize,
    },
}

/// A violation of the typed stack or control-flow invariants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyErrorKind {
    InvalidTypeDefinition { definition: usize },
    RecursiveTypeWithoutIndirection { definition: TypeId },
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
        write!(f, "invalid IR in ")?;
        match self.location {
            VerifyLocation::TypeDefinition { definition } => {
                write!(f, "type definition {}", definition.index())?;
            }
            VerifyLocation::Global { global } => write!(f, "global {}", global.index())?,
            VerifyLocation::Function { function } => {
                write!(f, "function {}", function.index())?;
            }
            VerifyLocation::BasicBlock {
                function,
                basic_block,
            } => write!(
                f,
                "function {}, basic block {}",
                function.index(),
                basic_block.index()
            )?,
            VerifyLocation::Instruction {
                function,
                basic_block,
                instruction,
            } => write!(
                f,
                "function {}, basic block {}, instruction {instruction}",
                function.index(),
                basic_block.index()
            )?,
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
    verify_module_types(module)?;

    for (index, global) in module.globals.iter().enumerate() {
        validate_ty(
            module,
            &global.ty,
            Location::global(GlobalId::from_index(index)),
        )?;
    }

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
    let function_location = Location::function(function_id);

    validate_ty(module, &function.result, function_location)?;
    for local in &function.locals {
        validate_ty(module, &local.ty, function_location)?;
    }
    for nonlocal in &function.nonlocals {
        validate_ty(module, &nonlocal.ty, function_location)?;
    }

    let location = Location::basic_block(function_id, entry);

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

        let location = Location::basic_block(function_id, basic_block_id);
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
                let condition_shape = resolve_shape(module, condition.clone(), location)?;
                if condition_shape != Ty::Bool {
                    return Err(location.error(VerifyErrorKind::TypeMismatch {
                        expected: Ty::Bool,
                        found: condition,
                    }));
                }
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
            Location::basic_block(function_id, BlockId::from_index(basic_block))
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
        Instr::Push { value } => stack.push(immediate_ty(module, value, location)?),
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
            stack.push(project_static(module, source, *index, location)?);
        }
        Instr::AccessDynamic => {
            let index = pop_one(stack, location)?;
            if !is_integer(module, &index, location)? {
                return Err(location.error(VerifyErrorKind::ExpectedInteger { found: index }));
            }
            let source = pop_one(stack, location)?;
            stack.push(project_dynamic(module, source, location)?);
        }
        Instr::Load => {
            let address = pop_one(stack, location)?;
            let shape = resolve_shape(module, address.clone(), location)?;
            let Ty::Pointer { pointee } = shape else {
                return Err(location.error(VerifyErrorKind::ExpectedPointer { found: address }));
            };
            stack.push(*pointee);
        }
        Instr::Store => {
            let value = pop_one(stack, location)?;
            let address = pop_one(stack, location)?;
            let shape = resolve_shape(module, address.clone(), location)?;
            let Ty::Pointer { pointee } = shape else {
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
            validate_ty(module, element, location)?;
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
            let shape = resolve_shape(module, callee.clone(), location)?;
            let Ty::Function { params, result } = shape else {
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
            for param in params {
                validate_ty(module, param, location)?;
            }
            validate_ty(module, result, location)?;
            let values = pop(stack, params.len(), location)?;
            expect_types(params, &values, location)?;
            stack.push(result.clone());
        }
    }
    Ok(())
}

fn immediate_ty(module: &Module, value: &Value, location: Location) -> Result<Ty, VerifyError> {
    let ty = match value {
        Value::Type { ty } => {
            validate_ty(module, ty, location)?;
            Ty::Type
        }
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
            validate_ty(module, &value.element_ty, location)?;
            for element in &value.elements {
                let found = immediate_ty(module, element, location)?;
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
                    immediate_ty(module, &field.value, location).map(|ty| RecordField {
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

fn project_static(
    module: &Module,
    source: Ty,
    index: usize,
    location: Location,
) -> Result<Ty, VerifyError> {
    match source {
        Ty::Pointer { pointee } => Ok(Ty::Pointer {
            pointee: Box::new(project_static(module, *pointee, index, location)?),
        }),
        Ty::Defined { .. } => project_static(
            module,
            resolve_shape(module, source, location)?,
            index,
            location,
        ),
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

fn project_dynamic(module: &Module, source: Ty, location: Location) -> Result<Ty, VerifyError> {
    match source {
        Ty::Pointer { pointee } => match resolve_shape(module, *pointee, location)? {
            Ty::Array { element, .. } => Ok(Ty::Pointer { pointee: element }),
            found => Err(location.error(VerifyErrorKind::ExpectedArray { found })),
        },
        Ty::Defined { .. } => {
            project_dynamic(module, resolve_shape(module, source, location)?, location)
        }
        Ty::Array { element, .. } => Ok(*element),
        found => Err(location.error(VerifyErrorKind::ExpectedArray { found })),
    }
}

fn verify_module_types(module: &Module) -> Result<(), VerifyError> {
    for (index, definition) in module.types.iter().enumerate() {
        let definition_id = TypeId::from_index(index);
        let location = Location::type_definition(definition_id);
        validate_ty(module, &definition.body, location)?;

        let mut active = vec![definition_id];
        validate_finite_representation(module, &definition.body, &mut active, location)?;
    }
    Ok(())
}

/// Check that every nominal reference points into this module's type table.
///
/// This deliberately does not expand definitions: the finite Rust `Ty` tree is
/// the result of evaluating a source type expression in a context where all of
/// its enclosing nominal identities have already been reserved.
fn validate_ty(module: &Module, ty: &Ty, location: Location) -> Result<(), VerifyError> {
    match ty {
        Ty::Defined { definition } => {
            if module.types.get(definition.index()).is_none() {
                return Err(location.error(VerifyErrorKind::InvalidTypeDefinition {
                    definition: definition.index(),
                }));
            }
        }
        Ty::Pointer { pointee } => validate_ty(module, pointee, location)?,
        Ty::Array { element, .. } => validate_ty(module, element, location)?,
        Ty::Record { fields } => {
            for field in fields {
                validate_ty(module, &field.ty, location)?;
            }
        }
        Ty::Function { params, result } => {
            for param in params {
                validate_ty(module, param, location)?;
            }
            validate_ty(module, result, location)?;
        }
        Ty::Type
        | Ty::Unit
        | Ty::Bool
        | Ty::Int8
        | Ty::Int16
        | Ty::Int32
        | Ty::Int64
        | Ty::UInt8
        | Ty::UInt16
        | Ty::UInt32
        | Ty::UInt64
        | Ty::Float32
        | Ty::Float64 => {}
    }
    Ok(())
}

/// Reject nominal cycles whose representation contains itself inline.
///
/// Pointer and function values have a fixed-size representation independent of
/// their referents/signatures, so they terminate the layout walk. Records and
/// arrays contain their children inline and therefore do not.
fn validate_finite_representation(
    module: &Module,
    ty: &Ty,
    active: &mut Vec<TypeId>,
    location: Location,
) -> Result<(), VerifyError> {
    match ty {
        Ty::Defined { definition } => {
            if active.contains(definition) {
                return Err(
                    location.error(VerifyErrorKind::RecursiveTypeWithoutIndirection {
                        definition: *definition,
                    }),
                );
            }
            let body = &module
                .types
                .get(definition.index())
                .ok_or_else(|| {
                    location.error(VerifyErrorKind::InvalidTypeDefinition {
                        definition: definition.index(),
                    })
                })?
                .body;
            active.push(*definition);
            validate_finite_representation(module, body, active, location)?;
            active.pop();
        }
        Ty::Array { element, .. } => {
            validate_finite_representation(module, element, active, location)?;
        }
        Ty::Record { fields } => {
            for field in fields {
                validate_finite_representation(module, &field.ty, active, location)?;
            }
        }
        Ty::Pointer { .. }
        | Ty::Function { .. }
        | Ty::Type
        | Ty::Unit
        | Ty::Bool
        | Ty::Int8
        | Ty::Int16
        | Ty::Int32
        | Ty::Int64
        | Ty::UInt8
        | Ty::UInt16
        | Ty::UInt32
        | Ty::UInt64
        | Ty::Float32
        | Ty::Float64 => {}
    }
    Ok(())
}

fn resolve_shape(module: &Module, mut ty: Ty, location: Location) -> Result<Ty, VerifyError> {
    let mut visited = Vec::new();
    while let Ty::Defined { definition } = ty {
        if visited.contains(&definition) {
            return Err(
                location.error(VerifyErrorKind::RecursiveTypeWithoutIndirection { definition })
            );
        }
        visited.push(definition);
        ty = module
            .types
            .get(definition.index())
            .ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidTypeDefinition {
                    definition: definition.index(),
                })
            })?
            .body
            .clone();
    }
    Ok(ty)
}

fn is_integer(module: &Module, ty: &Ty, location: Location) -> Result<bool, VerifyError> {
    Ok(resolve_shape(module, ty.clone(), location)?.is_integer())
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
struct Location(VerifyLocation);

impl Location {
    const fn type_definition(definition: TypeId) -> Self {
        Self(VerifyLocation::TypeDefinition { definition })
    }

    const fn global(global: GlobalId) -> Self {
        Self(VerifyLocation::Global { global })
    }

    const fn function(function: FunctionId) -> Self {
        Self(VerifyLocation::Function { function })
    }

    const fn instruction(function: FunctionId, basic_block: BlockId, instruction: usize) -> Self {
        Self(VerifyLocation::Instruction {
            function,
            basic_block,
            instruction,
        })
    }

    const fn basic_block(function: FunctionId, basic_block: BlockId) -> Self {
        Self(VerifyLocation::BasicBlock {
            function,
            basic_block,
        })
    }

    fn error(self, kind: VerifyErrorKind) -> VerifyError {
        VerifyError {
            location: self.0,
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
            types: vec![],
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
            types: vec![],
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
            types: vec![],
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
            types: vec![],
            globals: vec![],
            functions: vec![function],
        })
        .unwrap();
    }
}
