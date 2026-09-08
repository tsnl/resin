//! Second expression pass: lower a concrete typed tree to stack IR.
use super::{
    GenerateError, Generator,
    typed::{Statement, StatementKind, Term, TermKind},
};
use crate::ir::{Instr, Ty, Value};

impl Generator {
    pub(super) fn lower_term(&mut self, term: &Term) -> Result<Ty, GenerateError> {
        let span = term.span;
        let expected = &term.ty;
        match &term.kind {
            TermKind::Error(error) => Err(error.clone()),
            TermKind::Unit => {
                self.emit(Instr::Push { value: Value::Unit });
                Ok(Ty::Unit)
            }
            TermKind::None => {
                self.emit(Instr::Push { value: Value::None });
                Ok(Ty::None)
            }
            TermKind::Num { value } => {
                let (value, ty) = self.evaluator().number(span, value, Some(expected))?;
                self.emit(Instr::Push { value });
                Ok(ty)
            }
            TermKind::String { value } => {
                self.emit(Instr::Push {
                    value: Value::Bytes {
                        value: value.as_bytes().into(),
                    },
                });
                Ok(Ty::byte_span())
            }
            TermKind::Var { name } => self.gen_var(name),
            TermKind::Type { ty } => {
                self.emit(Instr::Push {
                    value: Value::Type { ty: ty.ty.clone() },
                });
                Ok(Ty::Type)
            }
            TermKind::Unwrap { value } => {
                self.gen_term(value, None)?;
                self.emit(Instr::ExcludeNone);
                Ok(expected.clone())
            }
            TermKind::Try { value } => self.gen_try(span, value),
            TermKind::Match { value, arms } => self.gen_match(span, value, arms, expected),
            TermKind::If { cond, then, els } => self.gen_if(cond, then, els, expected),
            TermKind::While { cond, body } => self.gen_while(cond, body),
            TermKind::Block { stmts, tail } => self.gen_block(stmts, tail, expected),
            TermKind::Record { fields } => self.gen_record(fields, expected),
            TermKind::Array { elems } => self.gen_array(elems, expected),
            TermKind::Builtin { name, args } => self.gen_builtin(span, name, args, expected),
            TermKind::MethodCall {
                receiver,
                receiver_type,
                name,
                arg,
            } => self.gen_method_call(receiver.as_deref(), &receiver_type.ty, name, arg),
            TermKind::Call { func, arg } => self.gen_call(func, arg),
            TermKind::Ascribe { ty, arg } => self.gen_ascription(span, &ty.ty, arg),
            TermKind::Result { failure, arg } => self.gen_result(span, *failure, arg, expected),
            TermKind::Absurd { arg } => {
                self.gen_term(arg, Some(&Ty::union([])))?;
                self.emit(Instr::Eliminate {
                    result: expected.clone(),
                });
                Ok(expected.clone())
            }
            TermKind::Layout { ty, size } => {
                let layout = crate::ir::layout::layout(self.typer.definitions(), &ty.ty)
                    .map_err(|e| GenerateError::inference(span, e.to_string()))?;
                self.emit(Instr::Push {
                    value: Value::UInt64 {
                        value: if *size { layout.size } else { layout.align } as u64,
                    },
                });
                Ok(Ty::UInt64)
            }
            TermKind::Assign { place, value } => self.gen_assign(place, value),
            TermKind::Address { place } => self.gen_place(place),
            TermKind::Deref { pointer } => {
                if matches!(pointer.ty, Ty::Arc { .. }) {
                    self.hold_arc_address(pointer)?;
                } else {
                    self.gen_term(pointer, None)?;
                }
                self.emit(Instr::Load);
                Ok(expected.clone())
            }
            TermKind::Field { base, name } => self.gen_field_value(span, base, name),
        }
    }

    pub(super) fn lower_statement(&mut self, statement: &Statement) -> Result<(), GenerateError> {
        self.with_context(statement.context, |g| match &statement.kind {
            StatementKind::Error(error) => Err(error.clone()),
            StatementKind::Define {
                binding,
                name,
                init,
            } => g.gen_define(binding.expect("checked declaration"), name, init),
            StatementKind::Declare { binding, name, ty } => {
                g.gen_declare(*binding, name, ty.ty.clone())
            }
            StatementKind::TypeDefinition => Ok(()),
            StatementKind::Expr { term } => {
                g.gen_term(term, None)?;
                g.emit(Instr::Discard);
                Ok(())
            }
        })
    }
}
