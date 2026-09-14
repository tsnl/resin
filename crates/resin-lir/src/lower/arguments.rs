use super::FunctionLowering;
use super::LowerError;
use crate::Instr;
use crate::lower::concrete::ReceiverConversion;
use crate::lower::concrete::Term;
use resin_types::prelude::*;

impl FunctionLowering<'_> {
    pub(super) fn gen_receiver(
        &mut self,
        receiver: &Term,
        conversion: ReceiverConversion,
        to: &Ty,
    ) -> Result<(), LowerError> {
        match conversion {
            ReceiverConversion::Value => {
                self.gen_term(receiver, Some(to))?;
            }
            ReceiverConversion::Address => match self.gen_operand(receiver)? {
                super::places::Operand::Place { .. } => {}
                super::places::Operand::Value(ty) => {
                    let local = self.save_top(&ty);
                    self.emit(Instr::LocalAddress { local });
                }
            },
            ReceiverConversion::Load => {
                self.gen_term(receiver, None)?;
                self.emit(Instr::Load);
            }
        }
        Ok(())
    }
}
