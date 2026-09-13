//! Translate one completed HIR application into a concrete expression tree.
use super::{concrete, instances::Instances, substitute::Substitution};
use crate::Error;
use resin_source::prelude::*;
use resin_types::prelude::*;

pub(super) fn function(
    source: &resin_hir::Function,
    arguments: &[Ty],
    instances: &mut Instances<'_>,
    current: FunctionId,
) -> Result<concrete::Function, Error> {
    let substitution = Substitution::new(&source.signature.type_params, arguments)
        .map_err(|error| instances.lower_error(error, Some(current), source.location.clone()))?;
    let span = source
        .location
        .as_ref()
        .map_or(Span { start: 0, end: 0 }, |location| location.span);
    Specialization {
        substitution,
        instances,
        current,
        location: source.location.clone(),
        span,
    }
    .function(source)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Value,
    Place,
}

struct Specialization<'a, 'source> {
    substitution: Substitution,
    instances: &'a mut Instances<'source>,
    current: FunctionId,
    location: Option<SourceLocation>,
    span: Span,
}

impl Specialization<'_, '_> {
    fn ty(&mut self, source: &resin_hir::Type) -> Result<Ty, Error> {
        self.substitution
            .ty(source, self.instances)
            .map_err(|mut error| {
                error.span = self.span;
                self.instances
                    .lower_error(error, Some(self.current), self.location.clone())
            })
    }

    fn request(&mut self, function: FunctionId, arguments: Vec<Ty>) -> Result<FunctionId, Error> {
        self.request_profile(function, arguments, self.instances.profile(self.current))
    }

    fn host_bridge(&mut self, function: FunctionId) -> Result<FunctionId, Error> {
        self.request_profile(function, vec![], crate::Profile::Host)
    }

    fn request_profile(
        &mut self,
        function: FunctionId,
        arguments: Vec<Ty>,
        profile: crate::Profile,
    ) -> Result<FunctionId, Error> {
        let location = self.location.as_ref().map(|location| SourceLocation {
            span: self.span,
            ..location.clone()
        });
        self.instances
            .request(function, arguments, profile, Some(self.current), location)
            .map_err(|mut error| {
                error.span = self.span;
                error
            })
    }

    fn shader(&mut self, function: FunctionId) -> Result<FunctionId, Error> {
        let location = self.location.as_ref().map(|location| SourceLocation {
            span: self.span,
            ..location.clone()
        });
        self.instances
            .shader(function, true, Some(self.current), location)
    }

    fn function(&mut self, source: &resin_hir::Function) -> Result<concrete::Function, Error> {
        let params = source
            .signature
            .params
            .iter()
            .map(|parameter| {
                Ok(concrete::Parameter {
                    binding: parameter.binding,
                    name: parameter.name.clone(),
                    ty: self.ty(&parameter.annotation.ty)?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(concrete::Function {
            location: source.location.clone(),
            name: source.name.clone(),
            foreign: source.foreign_header.as_ref().map(|header| Foreign {
                header: header.clone(),
                params: params
                    .iter()
                    .map(|parameter| parameter.ty.clone())
                    .collect(),
            }),
            signature: concrete::Signature {
                params,
                result: self.ty(&source.signature.result.ty)?,
            },
            body: source
                .body
                .as_ref()
                .map(|body| self.term(body))
                .transpose()?,
        })
    }

    fn constant(&mut self, value: &resin_hir::Constant) -> Result<Value, Error> {
        Ok(match value {
            resin_hir::Constant::Unit => Value::Unit,
            resin_hir::Constant::None => Value::None,
            resin_hir::Constant::Bool { value } => Value::Bool { value: *value },
            resin_hir::Constant::Int8 { value } => Value::Int8 { value: *value },
            resin_hir::Constant::Int16 { value } => Value::Int16 { value: *value },
            resin_hir::Constant::Int32 { value } => Value::Int32 { value: *value },
            resin_hir::Constant::Int64 { value } => Value::Int64 { value: *value },
            resin_hir::Constant::UInt8 { value } => Value::UInt8 { value: *value },
            resin_hir::Constant::UInt16 { value } => Value::UInt16 { value: *value },
            resin_hir::Constant::UInt32 { value } => Value::UInt32 { value: *value },
            resin_hir::Constant::UInt64 { value } => Value::UInt64 { value: *value },
            resin_hir::Constant::Float32 { value } => Value::Float32 { value: *value },
            resin_hir::Constant::Float64 { value } => Value::Float64 { value: *value },
            resin_hir::Constant::Type { ty: value } => Value::Type {
                ty: self.ty(value)?,
            },
            resin_hir::Constant::Str { value } => Value::Str {
                value: value.clone(),
            },
        })
    }

    fn term(&mut self, source: &resin_hir::Term) -> Result<concrete::Term, Error> {
        self.complete_term(source, Access::Value)
    }

    fn place(&mut self, source: &resin_hir::Term) -> Result<Box<concrete::Term>, Error> {
        let access = match source.kind {
            resin_hir::TermKind::Local { .. }
            | resin_hir::TermKind::Field { .. }
            | resin_hir::TermKind::Deref { .. } => Access::Place,
            _ => Access::Value,
        };
        self.complete_term(source, access).map(Box::new)
    }

    fn complete_term(
        &mut self,
        source: &resin_hir::Term,
        access: Access,
    ) -> Result<concrete::Term, Error> {
        let previous_span = std::mem::replace(&mut self.span, source.span);
        let result = (|| {
            let ty = self.ty(&source.ty)?;
            if self.instances.profile(self.current) == crate::Profile::Shader {
                if access == Access::Value && ty.needs_drop(self.instances.typer().definitions()) {
                    return Err(self.profile_error("shader cannot consume managed values: reference counting and destruction are host-only".into()));
                }
                crate::profile::expression_type(self.instances.typer(), &ty)
                    .map_err(|message| self.profile_error(message))?;
            }
            Ok(concrete::Term {
                span: source.span,
                ty,
                kind: self.kind(&source.kind, &source.ty)?,
            })
        })();
        self.span = previous_span;
        result
    }

    fn profile_error(&self, message: String) -> Error {
        self.instances.lower_error(
            super::LowerError {
                span: self.span,
                kind: crate::ErrorKind::UnsupportedProfile {
                    profile: crate::Profile::Shader,
                    message: message.into(),
                },
            },
            Some(self.current),
            self.location.clone(),
        )
    }

    fn boxed(&mut self, source: &resin_hir::Term) -> Result<Box<concrete::Term>, Error> {
        Ok(Box::new(self.term(source)?))
    }

    fn arguments(&mut self, source: &resin_hir::Arguments) -> Result<concrete::Arguments, Error> {
        Ok(concrete::Arguments {
            receiver: source
                .receiver
                .as_deref()
                .map(|source| self.boxed(source))
                .transpose()?,
            argument: self.boxed(&source.argument)?,
            params: source
                .params
                .iter()
                .map(|value| self.ty(value))
                .collect::<Result<_, _>>()?,
        })
    }

    fn statement(&mut self, source: &resin_hir::Statement) -> Result<concrete::Statement, Error> {
        Ok(match source {
            resin_hir::Statement::Define {
                binding,
                name,
                init,
            } => concrete::Statement::Define {
                binding: *binding,
                name: name.clone(),
                init: self.term(init)?,
            },
            resin_hir::Statement::Declare {
                binding,
                name,
                ty: annotation,
            } => concrete::Statement::Declare {
                binding: *binding,
                name: name.clone(),
                ty: self.ty(&annotation.ty)?,
            },
            resin_hir::Statement::Expr { term: source } => concrete::Statement::Expr {
                term: self.term(source)?,
            },
        })
    }

    fn arm(&mut self, source: &resin_hir::MatchArm) -> Result<concrete::MatchArm, Error> {
        let tag = match &source.tag {
            resin_hir::Case::Ok => Case::Ok,
            resin_hir::Case::Err => Case::Err,
            resin_hir::Case::Type { ty: value } => Case::Type(self.ty(value)?),
        };
        Ok(concrete::MatchArm {
            tag,
            binding: source.binding,
            body: self.term(&source.body)?,
        })
    }

    fn receiver(&self, source: resin_hir::ReceiverConversion) -> concrete::ReceiverConversion {
        match source {
            resin_hir::ReceiverConversion::Value => concrete::ReceiverConversion::Value,
            resin_hir::ReceiverConversion::Address => concrete::ReceiverConversion::Address,
            resin_hir::ReceiverConversion::Load => concrete::ReceiverConversion::Load,
            resin_hir::ReceiverConversion::ArcAddress => concrete::ReceiverConversion::ArcAddress,
            resin_hir::ReceiverConversion::ArcLoad => concrete::ReceiverConversion::ArcLoad,
        }
    }

    fn steps(&mut self, source: &[Conv]) -> Result<Vec<Conv>, Error> {
        source
            .iter()
            .map(|step| {
                Ok(match step {
                    Conv::Wrap { definition } => Conv::Wrap {
                        definition: self.nominal(*definition)?,
                    },
                    Conv::Unwrap { definition } => Conv::Unwrap {
                        definition: self.nominal(*definition)?,
                    },
                    other => *other,
                })
            })
            .collect()
    }

    fn nominal(&mut self, definition: TypeId) -> Result<TypeId, Error> {
        self.instances.nominal(definition).map_err(|mut error| {
            error.span = self.span;
            self.instances
                .lower_error(error, Some(self.current), self.location.clone())
        })
    }

    fn conversion(&mut self, conversion: &ExplicitConversion) -> Result<ExplicitConversion, Error> {
        Ok(match conversion {
            ExplicitConversion::Ascribe(steps) => ExplicitConversion::Ascribe(self.steps(steps)?),
            other => other.clone(),
        })
    }

    fn builtin(
        &mut self,
        name: &std::sync::Arc<str>,
        args: &[resin_hir::Term],
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        let params = args
            .iter()
            .map(|arg| self.ty(&arg.ty))
            .collect::<Result<Vec<_>, _>>()?;
        let expected = self.ty(expected)?;
        let signature = if self.instances.profile(self.current) == crate::Profile::Shader {
            resin_types::shader::builtin_instance(self.instances.typer(), name, &params)
                .map_err(|message| self.profile_error(message))?
        } else {
            self.instances
                .typer()
                .builtin_instance(name, &params)
                .map_err(|error| {
                    self.instances.lower_error(
                        super::LowerError::typing(self.span, error),
                        Some(self.current),
                        self.location.clone(),
                    )
                })?
        };
        self.instances
            .typer()
            .same(&expected, &signature.result)
            .map_err(|error| {
                self.instances.lower_error(
                    super::LowerError::typing(self.span, error),
                    Some(self.current),
                    self.location.clone(),
                )
            })?;
        let args = args
            .iter()
            .map(|arg| self.term(arg))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(concrete::TermKind::Builtin {
            name: name.clone(),
            args,
        })
    }

    fn kind(
        &mut self,
        source: &resin_hir::TermKind,
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        Ok(match source {
            resin_hir::TermKind::Constant { value } => concrete::TermKind::Constant {
                value: self.constant(value)?,
            },
            resin_hir::TermKind::Local { binding, name } => concrete::TermKind::Local {
                binding: *binding,
                name: name.clone(),
            },
            resin_hir::TermKind::Function {
                function,
                type_args,
            } => {
                let arguments = type_args
                    .iter()
                    .map(|ty| self.ty(ty))
                    .collect::<Result<_, _>>()?;
                concrete::TermKind::Function {
                    function: self.request(*function, arguments)?,
                }
            }
            resin_hir::TermKind::Shader { function, stage } => concrete::TermKind::Shader {
                function: self.shader(*function)?,
                stage: stage.clone(),
            },
            resin_hir::TermKind::Unwrap { value } => concrete::TermKind::Unwrap {
                value: self.boxed(value)?,
            },
            resin_hir::TermKind::Try { value } => concrete::TermKind::Try {
                value: self.boxed(value)?,
            },
            resin_hir::TermKind::Match { value, arms } => concrete::TermKind::Match {
                value: self.boxed(value)?,
                arms: arms
                    .iter()
                    .map(|value| self.arm(value))
                    .collect::<Result<_, _>>()?,
            },
            resin_hir::TermKind::If { cond, then, els } => concrete::TermKind::If {
                cond: self.boxed(cond)?,
                then: self.boxed(then)?,
                els: self.boxed(els)?,
            },
            resin_hir::TermKind::While { cond, body } => concrete::TermKind::While {
                cond: self.boxed(cond)?,
                body: self.boxed(body)?,
            },
            resin_hir::TermKind::Block { stmts, tail } => concrete::TermKind::Block {
                stmts: stmts
                    .iter()
                    .map(|value| self.statement(value))
                    .collect::<Result<_, _>>()?,
                tail: self.boxed(tail)?,
            },
            resin_hir::TermKind::Record { fields } => concrete::TermKind::Record {
                fields: fields
                    .iter()
                    .map(|(name, value)| Ok((name.clone(), self.term(value)?)))
                    .collect::<Result<_, Error>>()?,
            },
            resin_hir::TermKind::Array { elems } => concrete::TermKind::Array {
                elems: elems
                    .iter()
                    .map(|value| self.term(value))
                    .collect::<Result<_, _>>()?,
            },
            resin_hir::TermKind::Builtin { name, args } => self.builtin(name, args, expected)?,
            resin_hir::TermKind::Call { func, arg } => concrete::TermKind::Call {
                func: self.boxed(func)?,
                arg: self.boxed(arg)?,
            },
            resin_hir::TermKind::Pack { args } => concrete::TermKind::Pack {
                args: self.arguments(args)?,
            },
            resin_hir::TermKind::Intrinsic { op, args } => concrete::TermKind::Intrinsic {
                op: *op,
                args: self.arguments(args)?,
            },
            resin_hir::TermKind::Adapt { conversion, arg } => concrete::TermKind::Adapt {
                conversion: self.receiver(*conversion),
                arg: if *conversion == resin_hir::ReceiverConversion::Address {
                    self.place(arg)?
                } else {
                    self.boxed(arg)?
                },
            },
            resin_hir::TermKind::Convert { conversion, arg } => concrete::TermKind::Convert {
                conversion: self.conversion(conversion)?,
                arg: self.boxed(arg)?,
            },
            resin_hir::TermKind::ArcNew { value } => concrete::TermKind::ArcNew {
                value: self.boxed(value)?,
            },
            resin_hir::TermKind::GpuNew { allocator, args } => concrete::TermKind::GpuNew {
                allocator: self.host_bridge(*allocator)?,
                args: self.arguments(args)?,
            },
            resin_hir::TermKind::GpuAllocate { allocator, args } => {
                concrete::TermKind::GpuAllocate {
                    allocator: self.host_bridge(*allocator)?,
                    args: self.arguments(args)?,
                }
            }
            resin_hir::TermKind::GpuPipelineCreate {
                factory,
                shaders,
                args,
            } => concrete::TermKind::GpuPipelineCreate {
                factory: self.host_bridge(*factory)?,
                shaders: shaders
                    .iter()
                    .map(|id| self.shader(*id))
                    .collect::<Result<_, _>>()?,
                args: self.arguments(args)?,
            },
            resin_hir::TermKind::GpuPipelineDispatch {
                context,
                allocator,
                record,
                args,
            } => concrete::TermKind::GpuPipelineDispatch {
                context: self.host_bridge(*context)?,
                allocator: allocator.map(|id| self.host_bridge(id)).transpose()?,
                record: self.host_bridge(*record)?,
                args: self.arguments(args)?,
            },
            resin_hir::TermKind::WeakEmpty { pointee } => concrete::TermKind::WeakEmpty {
                pointee: self.ty(pointee)?,
            },
            resin_hir::TermKind::Result { failure, arg } => concrete::TermKind::Result {
                failure: *failure,
                arg: self.boxed(arg)?,
            },
            resin_hir::TermKind::Absurd { arg } => concrete::TermKind::Absurd {
                arg: self.boxed(arg)?,
            },
            resin_hir::TermKind::Assign { place, value } => concrete::TermKind::Assign {
                place: self.place(place)?,
                value: self.boxed(value)?,
            },
            resin_hir::TermKind::Address { place } => concrete::TermKind::Address {
                place: self.place(place)?,
            },
            resin_hir::TermKind::Deref { pointer } => concrete::TermKind::Deref {
                pointer: self.boxed(pointer)?,
            },
            resin_hir::TermKind::Field { base, access } => concrete::TermKind::Field {
                base: self.place(base)?,
                access: FieldAccess {
                    ty: self.ty(&access.ty)?,
                    index: access.index,
                    steps: self.steps(&access.steps)?,
                },
            },
        })
    }
}
