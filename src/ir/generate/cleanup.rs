use std::sync::Arc;

use crate::ast::Term;
use crate::ir::{Instr, Ty};

use super::{GenerateError, Generator, Scopes};

#[derive(Clone)]
pub(super) struct Deferred {
    body: Arc<Term>,
    scopes: Scopes,
}

impl Generator {
    pub(super) fn defer(&mut self, body: Arc<Term>) {
        self.defers
            .last_mut()
            .expect("inside a block")
            .push(Deferred {
                body,
                scopes: self.scopes.clone(),
            });
    }

    pub(super) fn cleanup(&mut self, first_scope: usize, result: &Ty) -> Result<(), GenerateError> {
        let pending: Vec<_> = self.defers[first_scope..]
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().rev().cloned())
            .collect();
        if pending.is_empty() {
            return Ok(());
        }
        let saved = self.save_top(result);
        for deferred in pending {
            self.gen_deferred(deferred)?;
        }
        self.load_local(saved);
        Ok(())
    }

    fn gen_deferred(&mut self, mut deferred: Deferred) -> Result<(), GenerateError> {
        deferred.scopes.update_initialization(&self.scopes);
        let mut before = std::mem::replace(&mut self.scopes, deferred.scopes);
        let in_defer = std::mem::replace(&mut self.in_defer, true);
        let result = self.gen_term(&deferred.body, None);
        before.update_initialization(&self.scopes);
        self.scopes = before;
        self.in_defer = in_defer;
        result?;
        self.emit(Instr::Discard);
        Ok(())
    }
}
