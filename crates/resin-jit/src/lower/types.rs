use resin_core::{Accessor, ElementType as IrElementType};
use resin_dsl::{ElementOperator, ElementType as DslElementType};

use crate::error::CompileError;

pub(crate) fn shape_u32(shape: &[usize]) -> Result<Box<[u32]>, CompileError> {
    shape
        .iter()
        .map(|&dim| {
            u32::try_from(dim).map_err(|_| CompileError::ShapeOverflow(dim))
        })
        .collect()
}

pub(crate) fn map_element_type(element_type: DslElementType) -> Result<IrElementType, CompileError> {
    match element_type {
        DslElementType::F32 => Ok(IrElementType::F4),
        DslElementType::U32 => Ok(IrElementType::U4),
    }
}

pub(crate) fn map_element_operator(
    operator: ElementOperator,
) -> Result<resin_core::ElementOperator, CompileError> {
    use resin_core::{
        BinaryAssocElementOperator, BinaryBitwiseOperator, BinaryCompareOperator,
        BinaryElementOperator, ElementOperator as IrOp, UnaryElementOperator,
    };

    match operator {
        ElementOperator::Neg => Ok(IrOp::Unary(UnaryElementOperator::Neg)),
        ElementOperator::Log => Ok(IrOp::Unary(UnaryElementOperator::Log)),
        ElementOperator::Exp => Ok(IrOp::Unary(UnaryElementOperator::Exp)),
        ElementOperator::Relu => Ok(IrOp::Unary(UnaryElementOperator::Relu)),
        ElementOperator::Abs => Ok(IrOp::Unary(UnaryElementOperator::Abs)),
        ElementOperator::Sqrt => Ok(IrOp::Unary(UnaryElementOperator::Sqrt)),
        ElementOperator::Floor => Ok(IrOp::Unary(UnaryElementOperator::Floor)),
        ElementOperator::Cast => Ok(IrOp::Unary(UnaryElementOperator::Convert)),
        ElementOperator::Bitcast => Ok(IrOp::Unary(UnaryElementOperator::Bitcast)),
        ElementOperator::Mul => Ok(IrOp::Binary(BinaryElementOperator::Assoc(
            BinaryAssocElementOperator::Mul,
        ))),
        ElementOperator::Div => Ok(IrOp::Binary(BinaryElementOperator::Div)),
        ElementOperator::Add => Ok(IrOp::Binary(BinaryElementOperator::Assoc(
            BinaryAssocElementOperator::Add,
        ))),
        ElementOperator::Sub => Ok(IrOp::Binary(BinaryElementOperator::Sub)),
        ElementOperator::Min => Ok(IrOp::Binary(BinaryElementOperator::Assoc(
            BinaryAssocElementOperator::Min,
        ))),
        ElementOperator::Max => Ok(IrOp::Binary(BinaryElementOperator::Assoc(
            BinaryAssocElementOperator::Max,
        ))),
        ElementOperator::Pow => Ok(IrOp::Binary(BinaryElementOperator::Pow)),
        ElementOperator::CmpEq => Ok(IrOp::Compare(BinaryCompareOperator::Eq)),
        ElementOperator::CmpNe => Ok(IrOp::Compare(BinaryCompareOperator::Ne)),
        ElementOperator::CmpLt => Ok(IrOp::Compare(BinaryCompareOperator::Lt)),
        ElementOperator::CmpLe => Ok(IrOp::Compare(BinaryCompareOperator::Le)),
        ElementOperator::CmpGt => Ok(IrOp::Compare(BinaryCompareOperator::Gt)),
        ElementOperator::CmpGe => Ok(IrOp::Compare(BinaryCompareOperator::Ge)),
        ElementOperator::BitAnd => Ok(IrOp::Bitwise(BinaryBitwiseOperator::Band)),
        ElementOperator::BitOr => Ok(IrOp::Bitwise(BinaryBitwiseOperator::Bor)),
        ElementOperator::BitXor => Ok(IrOp::Bitwise(BinaryBitwiseOperator::Bxor)),
        ElementOperator::Shl => Ok(IrOp::Bitwise(BinaryBitwiseOperator::Shl)),
        ElementOperator::Shr => Ok(IrOp::Bitwise(BinaryBitwiseOperator::Shr)),
        ElementOperator::Rem
        | ElementOperator::Matmul => Err(CompileError::UnsupportedOperator(operator)),
    }
}

pub(crate) fn rpn_for_elementwise(
    operator: ElementOperator,
    arity: usize,
) -> Result<resin_ir::ElementRpnExpr, CompileError> {
    use resin_ir::RpnAtom;

    let mut atoms = Vec::with_capacity(arity + 1);
    for i in 0..arity {
        atoms.push(RpnAtom::Arg(i as u32));
    }
    atoms.push(RpnAtom::Op(map_element_operator(operator)?));
    Ok(resin_ir::ElementRpnExpr { atoms })
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ViewKey {
    pub buffer: usize,
    pub accessor: Accessor,
}