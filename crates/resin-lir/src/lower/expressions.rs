//! Every HIR constructor has an explicit storage/control-flow translation here.
use super::Generator;
use super::LowerError;
use crate::Instr;
use resin_hir::{Arguments, Statement, Term, TermKind};
use resin_types::prelude::*;

impl Generator {
    pub(super) fn lower_term(&mut self, term: &Term) -> Result<Ty, LowerError> {
        let span = term.span;
        let expected = &term.ty;
        match &term.kind {
            TermKind::Constant { value } => self.emit(Instr::Push {
                value: value.clone(),
            }),
            TermKind::Function { function } => self.emit(Instr::Function {
                function: *function,
            }),
            TermKind::Shader { function, stage } => self.emit(Instr::Shader {
                function: *function,
                stage: stage.clone(),
            }),
            TermKind::Local { binding, name } => return self.gen_var(*binding, name),
            TermKind::Unwrap { value } => {
                self.gen_term(value, None)?;
                self.emit(Instr::ExcludeNone);
            }
            TermKind::Try { value } => return self.gen_try(span, value),
            TermKind::Match { value, arms } => return self.gen_match(value, arms, expected),
            TermKind::If { cond, then, els } => return self.gen_if(cond, then, els, expected),
            TermKind::While { cond, body } => return self.gen_while(cond, body),
            TermKind::Block { stmts, tail } => return self.gen_block(stmts, tail, expected),
            TermKind::Record { fields } => return self.gen_record(fields, expected),
            TermKind::Array { elems } => return self.gen_array(elems, expected),
            TermKind::Builtin { name, args } => {
                return self.gen_builtin(name, args, expected);
            }
            TermKind::Call { func, arg } => return self.gen_call(func, arg),
            TermKind::Pack { args } => self.gen_pack(args)?,
            TermKind::Intrinsic { op, args } => self.gen_intrinsic(*op, args, expected)?,
            TermKind::Adapt { conversion, arg } => self.gen_receiver(arg, *conversion, expected)?,
            TermKind::Convert { conversion, arg } => {
                return self.gen_conversion(term, arg, conversion);
            }
            TermKind::ArcNew { value } => {
                let Ty::Arc { pointee } = expected else {
                    unreachable!("checked Arc constructor")
                };
                self.gen_term(value, Some(pointee))?;
                self.emit(Instr::ArcNew);
            }
            TermKind::WeakEmpty { pointee } => self.emit(Instr::WeakEmpty {
                pointee: pointee.clone(),
            }),
            TermKind::Result { failure, arg } => {
                return self.gen_result(span, *failure, arg, expected);
            }
            TermKind::Absurd { arg } => {
                self.gen_term(arg, Some(&Ty::union([])))?;
                self.emit(Instr::Eliminate {
                    result: expected.clone(),
                });
            }
            TermKind::Assign { place, value } => return self.gen_assign(place, value),
            TermKind::Address { place } => return self.gen_place(place),
            TermKind::Deref { pointer } => {
                if matches!(pointer.ty, Ty::Arc { .. }) {
                    self.hold_arc_address(pointer)?;
                } else {
                    self.gen_term(pointer, None)?;
                }
                self.emit(Instr::Load);
            }
            TermKind::Field { base, access } => return self.gen_field_value(base, access),
        }
        Ok(expected.clone())
    }

    pub(super) fn lower_statement(&mut self, statement: &Statement) -> Result<(), LowerError> {
        match statement {
            Statement::Define {
                binding,
                name,
                init,
            } => self.gen_define(*binding, name, init),
            Statement::Declare { binding, name, ty } => {
                self.gen_declare(*binding, name, ty.ty.clone())
            }
            Statement::Expr { term } => {
                self.gen_term(term, None)?;
                self.emit(Instr::Discard);
                Ok(())
            }
        }
    }

    fn gen_arguments(&mut self, args: &Arguments) -> Result<(), LowerError> {
        if let Some(receiver) = &args.receiver {
            self.gen_term(receiver, None)?;
        }
        self.unpack_argument(&args.argument, &args.params)
    }

    fn gen_pack(&mut self, args: &Arguments) -> Result<(), LowerError> {
        self.gen_arguments(args)?;
        let count = args.params.len() + usize::from(args.receiver.is_some());
        if count > 1 {
            self.emit(Instr::MakeRecord {
                fields: (0..count).map(|i| format!("_{i}").into()).collect(),
            });
        }
        Ok(())
    }

    fn gen_intrinsic(
        &mut self,
        op: Intrinsic,
        args: &Arguments,
        result: &Ty,
    ) -> Result<(), LowerError> {
        self.gen_arguments(args)?;
        match op {
            Intrinsic::StringFromStr => self.emit(Instr::CallBuiltin {
                name: "string_from_str".into(),
                params: args.params.clone(),
                result: result.clone(),
            }),
            Intrinsic::Replace => self.emit(Instr::Replace),
            Intrinsic::Index => self.emit(Instr::AccessDynamic),
            Intrinsic::ArcGet => {
                self.emit(Instr::Load);
                self.emit(Instr::ArcData);
            }
            Intrinsic::Downgrade => self.emit(Instr::Downgrade),
            Intrinsic::Upgrade => self.emit(Instr::Upgrade),
        }
        Ok(())
    }
}
