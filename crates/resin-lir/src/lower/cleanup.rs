use super::Generator;
use crate::Instr;
use resin_common::prelude::*;

impl Generator {
    pub(super) fn cleanup(&mut self, first_scope: usize, result: &Ty) {
        let owned: Vec<_> = self.owned[first_scope..]
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().rev().copied())
            .filter(|id| {
                self.function
                    .as_ref()
                    .unwrap()
                    .local_type(*id)
                    .needs_drop(self.typer.definitions())
            })
            .collect();
        if owned.is_empty() {
            return;
        }
        let saved = self.save_top(result);
        for local in owned {
            self.emit(Instr::DropLocal { local });
        }
        if result.needs_drop(self.typer.definitions()) {
            self.emit(Instr::TakeLocal { local: saved });
        } else {
            self.load_local(saved);
        }
    }
}
