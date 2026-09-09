use super::Generator;
use crate::Instr;
use resin_common::prelude::*;
use resin_hir::ReceiverConversion;
use resin_hir::Term;

impl Generator {
    pub(super) fn hold_arc_address(&mut self, term: &Term) -> Result<Ty, GenerateError> {
        let ty = self.gen_term(term, None)?;
        let Ty::Arc { pointee } = &ty else {
            return Err(GenerateError::inference(term.span, "expected Arc<T>"));
        };
        let pointee = *pointee.clone();
        let owner = self.save_top(&ty);
        self.load_local(owner);
        self.emit(Instr::ArcData);
        Ok(Ty::Pointer {
            pointee: Box::new(pointee),
        })
    }

    pub(super) fn gen_receiver(
        &mut self,
        receiver: &Term,
        conversion: ReceiverConversion,
        to: &Ty,
    ) -> Result<(), GenerateError> {
        match conversion {
            conversion @ (ReceiverConversion::ArcAddress | ReceiverConversion::ArcLoad) => {
                self.hold_arc_address(receiver)?;
                if matches!(conversion, ReceiverConversion::ArcLoad) {
                    self.emit(Instr::Load);
                }
            }
            ReceiverConversion::Value => {
                self.gen_term(receiver, Some(to))?;
            }
            ReceiverConversion::Address => {
                self.check_place_initialized(receiver)?;
                match self.gen_operand(receiver)? {
                    super::places::Operand::Place(_) => {}
                    super::places::Operand::Value(ty) => {
                        let local = self.save_top(&ty);
                        self.emit(Instr::LocalAddress { local });
                    }
                }
            }
            ReceiverConversion::Load => {
                self.gen_term(receiver, None)?;
                self.emit(Instr::Load);
            }
        }
        Ok(())
    }

    pub(super) fn unpack_argument(
        &mut self,
        arg: &Term,
        params: &[Ty],
    ) -> Result<(), GenerateError> {
        self.gen_term(arg, Some(&Ty::parameter(params)))?;
        match params.len() {
            0 => self.emit(Instr::Discard),
            1 => {}
            count => {
                let saved = self.save_top(&Ty::parameter(params));
                for index in 0..count {
                    self.emit(Instr::LocalAddress { local: saved });
                    self.emit(Instr::AccessStatic { index });
                    self.emit(Instr::TransferLoad);
                }
                self.emit(Instr::ForgetLocal { local: saved });
            }
        }
        Ok(())
    }
}
