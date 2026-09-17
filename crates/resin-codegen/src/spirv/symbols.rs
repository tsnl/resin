use crate::Error;
use resin_lir::Instr;
use resin_types::prelude::*;
use rspirv::spirv::Word;

/// Function addresses cannot become integer addresses in Vulkan shaders. Keep
/// their projection paths until a load or store needs an OpAccessChain.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct LocalAddress {
    pub root: Word,
    pub root_type: Ty,
    pub indices: Vec<LocalIndex>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum LocalIndex {
    Static { index: u32 },
    Dynamic { id: Word, ty: Ty },
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Slot {
    pub ty: Ty,
    pub id: Word,
    pub local: Option<LocalAddress>,
}

impl Slot {
    pub fn value(ty: Ty, id: Word) -> Self {
        Self {
            ty,
            id,
            local: None,
        }
    }

    pub fn symbolic(&self) -> bool {
        self.local.is_some() || matches!(self.ty, Ty::Function { .. })
    }
}

pub(super) fn agree(left: &[Slot], right: &[Slot]) -> Result<(), Error> {
    if left
        .iter()
        .zip(right)
        .any(|(a, b)| (a.symbolic() || b.symbolic()) && (a.local != b.local || a.id != b.id))
    {
        return Err(Error::unsupported(
            "shader cannot merge distinct local addresses or function values".into(),
        ));
    }
    Ok(())
}

pub(super) fn check(instruction: &Instr, args: &[Slot]) -> Result<(), Error> {
    if args
        .iter()
        .enumerate()
        .any(|(i, arg)| arg.local.is_some() && !permits_local_address(instruction, i))
    {
        return Err(Error::unsupported(
            "shader-local addresses cannot escape through values, casts, or calls".into(),
        ));
    }
    Ok(())
}

fn permits_local_address(instruction: &Instr, operand: usize) -> bool {
    matches!(instruction, Instr::Discard)
        || operand == 0
            && matches!(
                instruction,
                Instr::Load
                    | Instr::ReadOnly
                    | Instr::TransferLoad
                    | Instr::IsVariant { .. }
                    | Instr::Store
                    | Instr::Replace
                    | Instr::AccessStatic { .. }
                    | Instr::AccessDynamic
            )
}
