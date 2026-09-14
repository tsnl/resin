//! Every HIR constructor has an explicit storage/control-flow translation here.
use super::FunctionLowering;
use super::LowerError;
use crate::Instr;
use crate::lower::concrete::{Arguments, Statement, Term, TermKind};
use resin_types::prelude::*;

impl FunctionLowering<'_> {
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
            TermKind::Call { func, args } => return self.gen_call(func, args),
            TermKind::Intrinsic {
                op,
                type_args,
                args,
            } => self.gen_intrinsic(*op, type_args, args, expected)?,
            TermKind::Adapt { conversion, arg } => self.gen_receiver(arg, *conversion, expected)?,
            TermKind::Convert { conversion, arg } => {
                return self.gen_conversion(term, arg, conversion);
            }
            TermKind::GpuNew { allocator, args } => {
                let Ty::Result { value, .. } = expected else {
                    unreachable!("GPU allocation result")
                };
                let Ty::GpuPointer { pointee } = &**value else {
                    unreachable!("GPU allocation pointer")
                };
                self.gen_arguments(args)?;
                self.emit(Instr::GpuNew {
                    allocator: *allocator,
                    element: *pointee.clone(),
                });
            }
            TermKind::GpuAllocate { allocator, args } => {
                let Ty::Result { value, .. } = expected else {
                    unreachable!("GPU allocation result")
                };
                let Ty::GpuSpan { element } = &**value else {
                    unreachable!("GPU allocation span")
                };
                self.gen_arguments(args)?;
                self.emit(Instr::GpuAllocate {
                    allocator: *allocator,
                    element: *element.clone(),
                });
            }
            TermKind::GpuPipelineCreate {
                factory,
                shaders,
                args,
            } => {
                self.gen_arguments(args)?;
                let Ty::Result {
                    value: pipeline, ..
                } = expected
                else {
                    unreachable!("pipeline creation result")
                };
                self.emit(match shaders.as_slice() {
                    [shader] => Instr::GpuComputePipeline {
                        pipeline: *pipeline.clone(),
                        factory: *factory,
                        shader: *shader,
                    },
                    [vertex, fragment] => Instr::GpuGraphicsPipeline {
                        pipeline: *pipeline.clone(),
                        factory: *factory,
                        vertex: *vertex,
                        fragment: *fragment,
                    },
                    _ => unreachable!("checked pipeline stages"),
                });
            }
            TermKind::GpuPipelineDispatch {
                projection,
                context,
                allocator,
                record,
                args,
            } => {
                self.gen_arguments(args)?;
                self.emit(if args.values.len() == 6 {
                    Instr::GpuDispatch {
                        projection: projection.clone().expect("compute projection"),
                        context: *context,
                        allocator: allocator.expect("compute root"),
                        record: *record,
                    }
                } else {
                    Instr::GpuDraw {
                        projection: projection.clone(),
                        context: *context,
                        allocator: *allocator,
                        record: *record,
                    }
                });
            }
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
                self.gen_term(pointer, None)?;
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
                self.gen_declare(*binding, name, ty.clone())
            }
            Statement::Expr { term } => {
                self.gen_term(term, None)?;
                self.emit(Instr::Discard);
                Ok(())
            }
        }
    }

    fn gen_arguments(&mut self, args: &Arguments) -> Result<(), LowerError> {
        if args.values.len() != args.params.len() {
            return Err(LowerError::invalid_hir(
                self.source_span,
                "intrinsic argument count does not match signature",
            ));
        }
        for (arg, param) in args.values.iter().zip(&args.params) {
            self.gen_term(arg, Some(param))?;
        }
        Ok(())
    }

    fn gen_intrinsic(
        &mut self,
        op: Intrinsic,
        type_args: &[Ty],
        args: &Arguments,
        result: &Ty,
    ) -> Result<(), LowerError> {
        self.gen_arguments(args)?;
        match op {
            Intrinsic::GpuPointerProjection
            | Intrinsic::GpuSequenceProjection
            | Intrinsic::GpuPipelineType => {
                return Err(LowerError::invalid_hir(
                    self.source_span,
                    "GPU projection contract reached storage lowering",
                ));
            }
            Intrinsic::GpuElementLayout => self.emit(Instr::GpuElementLayout {
                element: type_args[0].clone(),
            }),
            Intrinsic::GpuViewLoad => self.emit(Instr::GpuViewLoad {
                element: result.clone(),
            }),
            Intrinsic::GpuViewAllocate => self.emit(Instr::GpuViewAllocate),
            Intrinsic::GpuViewIndex => self.emit(Instr::GpuViewIndex {
                element: type_args[0].clone(),
            }),
            Intrinsic::GpuViewRange => self.emit(Instr::GpuViewRange {
                element: type_args[0].clone(),
            }),
            Intrinsic::GpuViewOffset => self.emit(Instr::GpuViewOffset),
            Intrinsic::GpuViewRestrict => self.emit(Instr::GpuViewRestrict),
            Intrinsic::GpuViewStore => self.emit(Instr::GpuViewStore),
            Intrinsic::GpuViewReplace => self.emit(Instr::GpuViewReplace),
            Intrinsic::GpuViewCopyTo => self.emit(Instr::GpuViewCopyTo),
            Intrinsic::GpuViewCopyImage => self.emit(Instr::GpuViewCopyImage),
            Intrinsic::FormatBytes => self.emit(Instr::CallBuiltin {
                name: "format_bytes".into(),
                params: args.params.clone(),
                result: result.clone(),
            }),
            Intrinsic::StringFromBytes => self.emit(Instr::CallBuiltin {
                name: "string_from_bytes".into(),
                params: args.params.clone(),
                result: result.clone(),
            }),
            Intrinsic::Replace => self.emit(Instr::Replace),
            Intrinsic::PointerIndex => self.emit(Instr::PointerIndex),
            Intrinsic::PointerRange => self.emit(Instr::PointerRange),
            Intrinsic::PointerBytes => self.emit(Instr::PointerBytes),
            Intrinsic::Index => self.emit(Instr::AccessDynamic),
            Intrinsic::GpuIndex => self.emit(Instr::AccessDynamic),
            Intrinsic::GpuSlice => self.emit(Instr::GpuSlice),
            Intrinsic::GpuReadOnly => self.emit(Instr::GpuReadOnly),
            Intrinsic::GpuWriteOnly => self.emit(Instr::GpuWriteOnly),
            Intrinsic::GpuAllocateNative => self.emit(Instr::GpuAllocateNative),
            Intrinsic::GpuCopyTo => self.emit(Instr::GpuCopyTo),
            Intrinsic::GpuArgumentsDispatch => self.emit(Instr::GpuArgumentsDispatch),
            Intrinsic::GpuArgumentsDraw => self.emit(Instr::GpuArgumentsDraw),
            Intrinsic::GpuCopyImage => self.emit(Instr::GpuCopyImage),
            Intrinsic::OwnerAllocate => self.emit(Instr::OwnerAllocate {
                element: args.params[1].clone(),
            }),
            Intrinsic::OwnerData => {
                let Ty::Pointer { pointee } = result else {
                    unreachable!("owner payload pointer")
                };
                self.emit(Instr::OwnerData {
                    pointee: *pointee.clone(),
                });
            }
            Intrinsic::OwnerLength => self.emit(Instr::OwnerLength),
            Intrinsic::OwnerDowngrade => self.emit(Instr::OwnerDowngrade),
            Intrinsic::OwnerUpgrade => self.emit(Instr::OwnerUpgrade),
            Intrinsic::WeakEmpty => self.emit(Instr::WeakEmpty),
        }
        Ok(())
    }
}
