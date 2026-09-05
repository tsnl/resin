use std::{collections::HashMap, sync::Arc};

use crate::ast::{Ident, Term, Type};
use crate::ir::{FunctionId, Instr, LocalId, NonLocalId, Terminator, Ty};

use super::builder::FunctionBuilder;
use super::scope::{Initialization, ValueBinding, ValueBindingKind};
use super::{GenerateError, Generator};

pub(super) struct FunctionState {
    pub(super) builder: FunctionBuilder,
    remotes: HashMap<Remote, NonLocalId>,
    captures: Vec<Remote>,
}

impl FunctionState {
    pub(super) fn new(name: Option<Arc<str>>) -> Self {
        Self {
            builder: FunctionBuilder::new(name),
            remotes: HashMap::new(),
            captures: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Remote {
    pub(super) owner_depth: usize,
    pub(super) value: RemoteValue,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum RemoteValue {
    Local(LocalId),
    CurrentClosure,
}

impl Generator {
    pub(super) fn gen_lambda(
        &mut self,
        params: &[(Ident, Type)],
        body: &Term,
        expected: Option<&Ty>,
        name: Option<&Arc<str>>,
    ) -> Result<Ty, GenerateError> {
        let expected_fn = expected.and_then(|ty| match ty {
            Ty::Function { param, result } => Some((param.as_ref(), result.as_ref())),
            _ => None,
        });

        let enclosing_scopes = self.scopes.clone();
        let recursive_binding = name
            .and_then(|name| self.scopes.lookup_value(name))
            .filter(|binding| matches!(binding.kind, ValueBindingKind::Local(_)))
            .cloned();
        self.functions.push(FunctionState::new(name.cloned()));
        self.scopes.push();
        if let (Some(name), Some(binding)) = (name, recursive_binding) {
            self.scopes
                .define_value(
                    name.clone(),
                    ValueBinding {
                        kind: ValueBindingKind::CurrentClosure,
                        ty: binding.ty,
                        initialization: Initialization::Initialized,
                        depth: self.current_depth(),
                    },
                )
                .expect("fresh recursive-name scope");
        }
        // Parameters may shadow the recursive name.
        self.scopes.push();

        let param_ty = self.bind_params(params)?;
        if let Some((expected_param, _)) = expected_fn {
            self.typer
                .same(expected_param, &param_ty)
                .map_err(|err| GenerateError::typing(body.span, err))?;
        }

        let body_expected = self
            .evaluator()
            .result_type(body)
            .or_else(|| expected_fn.map(|(_, result)| result.clone()));
        let body_ty = self.gen_term(body, body_expected.as_ref())?;
        self.function().result(body_ty.clone());
        self.terminate(Terminator::Return);
        self.scopes.pop();
        self.scopes.pop();
        // Generating a delayed body cannot initialize its enclosing bindings.
        self.scopes = enclosing_scopes;
        self.finish_lambda()?;
        Ok(self.typer.type_lambda(&param_ty, &body_ty))
    }

    fn bind_params(&mut self, params: &[(Ident, Type)]) -> Result<Ty, GenerateError> {
        let mut param_tys = Vec::with_capacity(params.len());
        for (_, ann) in params {
            param_tys.push(self.evaluator().ty(ann)?);
        }
        let param_ty = Ty::parameter(&param_tys);
        let parameter = LocalId::from_index(0);
        self.function().parameter(
            if params.len() == 1 {
                Some(params[0].0.val.clone())
            } else {
                None
            },
            param_ty.clone(),
        );
        for (index, ((name, _), ty)) in params.iter().zip(&param_tys).enumerate() {
            let local = if params.len() == 1 {
                parameter
            } else {
                let local = self.alloc_local(ty.clone(), Some(name.val.clone()));
                self.emit(Instr::LocalAddress { local });
                self.emit(Instr::LocalAddress { local: parameter });
                self.emit(Instr::AccessStatic { index });
                self.emit(Instr::Load);
                self.emit(Instr::Store);
                self.emit(Instr::Discard);
                local
            };
            self.bind_value(
                name,
                ValueBinding {
                    kind: ValueBindingKind::Local(local),
                    ty: Some(ty.clone()),
                    initialization: Initialization::Initialized,
                    depth: self.current_depth(),
                },
            )?;
        }
        Ok(param_ty)
    }

    fn finish_lambda(&mut self) -> Result<(), GenerateError> {
        let FunctionState {
            builder, captures, ..
        } = self.functions.pop().expect("lambda builder");
        let n_captures = captures.len();
        let function = builder.finish();
        let id = FunctionId::from_index(self.module.functions.len());
        self.module.functions.push(function);
        for remote in captures {
            self.emit_capture_value(remote)?;
        }
        self.emit(Instr::MakeClosure {
            function: id,
            captures: n_captures,
        });
        Ok(())
    }

    fn emit_capture_value(&mut self, remote: Remote) -> Result<(), GenerateError> {
        let current = self.current_depth();
        if remote.owner_depth == current {
            match remote.value {
                RemoteValue::Local(local) => self.emit(Instr::LocalAddress { local }),
                RemoteValue::CurrentClosure => {
                    self.emit(Instr::CurrentClosure);
                    return Ok(());
                }
            }
        } else {
            let nonlocal = self.functions[current]
                .remotes
                .get(&remote)
                .copied()
                .expect("enclosing function captures the same remote");
            self.emit(Instr::NonLocalAddress { nonlocal });
        }
        self.emit(Instr::Load);
        Ok(())
    }

    pub(super) fn capture(&mut self, remote: Remote, ty: &Ty) -> NonLocalId {
        let current = self.current_depth();
        for depth in remote.owner_depth + 1..=current {
            if self.functions[depth].remotes.contains_key(&remote) {
                continue;
            }
            let owner = &self.functions[remote.owner_depth].builder;
            let name = match remote.value {
                RemoteValue::Local(local) => owner.local_name(local),
                RemoteValue::CurrentClosure => owner.name(),
            };
            let function = &mut self.functions[depth];
            let nonlocal = function.builder.nonlocal(ty.clone(), name);
            function.remotes.insert(remote, nonlocal);
            function.captures.push(remote);
        }
        self.functions[current].remotes[&remote]
    }
}
