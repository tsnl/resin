//! Translate one completed HIR application into a concrete expression tree.
use super::{concrete, instances::Instances, substitute::Substitution};
use crate::Error;
use resin_source::prelude::*;
use resin_types::prelude::*;

pub(super) fn function(
    source: &resin_hir::Function,
    arguments: &[resin_hir::Type],
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

    fn argument(&mut self, source: &resin_hir::Type) -> Result<resin_hir::Type, Error> {
        self.substitution
            .normalize(source, self.instances)
            .map_err(|mut error| {
                error.span = self.span;
                self.instances
                    .lower_error(error, Some(self.current), self.location.clone())
            })
    }

    fn request(
        &mut self,
        function: FunctionId,
        arguments: Vec<resin_hir::Type>,
    ) -> Result<FunctionId, Error> {
        self.request_profile(function, arguments, self.instances.profile(self.current))
    }

    fn host_bridge(&mut self, function: FunctionId) -> Result<FunctionId, Error> {
        self.request_profile(function, vec![], crate::Profile::Host)
    }

    fn request_profile(
        &mut self,
        function: FunctionId,
        arguments: Vec<resin_hir::Type>,
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
            foreign: source.foreign_header.as_ref().map(|header| crate::Foreign {
                header: crate::ForeignHeader {
                    source: header.source.clone(),
                    spelling: header.spelling.clone(),
                },
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
        if let resin_hir::TermKind::Read { place }
        | resin_hir::TermKind::Move { place }
        | resin_hir::TermKind::ReadOwned { place } = &source.kind
        {
            return self.place(place);
        }
        let access = match source.kind {
            resin_hir::TermKind::Local { .. }
            | resin_hir::TermKind::Field { .. }
            | resin_hir::TermKind::Deref { .. }
            | resin_hir::TermKind::Use { .. } => Access::Place,
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
                kind: match &source.kind {
                    resin_hir::TermKind::Use { arg } => {
                        self.reference_use(arg, &source.ty, access)?
                    }
                    kind => self.kind(kind, &source.ty)?,
                },
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

    fn parallel_parameter(
        &mut self,
        source: &resin_hir::ParallelParameter,
    ) -> Result<concrete::ParallelParameter, Error> {
        Ok(concrete::ParallelParameter {
            binding: source.binding,
            name: source.name.clone(),
            ty: self.ty(&source.ty)?,
        })
    }

    fn require_host_parallel(&self) -> Result<(), Error> {
        if self.instances.profile(self.current) == crate::Profile::Shader {
            return Err(self.profile_error("parallel blocks currently have a host backend only; cooperative workgroup lowering is not implemented".into()));
        }
        Ok(())
    }

    fn arguments(&mut self, source: &resin_hir::Arguments) -> Result<concrete::Arguments, Error> {
        Ok(concrete::Arguments {
            values: source
                .values
                .iter()
                .map(|arg| self.term(arg))
                .collect::<Result<_, _>>()?,
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
            resin_hir::Case::Error { .. } | resin_hir::Case::Wildcard => {
                return Err(self
                    .instance_error("wildcard must be expanded before concrete arm translation"));
            }
            resin_hir::Case::Type { ty: value } => Case::Type(self.ty(value)?),
        };
        Ok(concrete::MatchArm {
            error_payload: None,
            tag,
            binding: source.binding,
            body: self.term(&source.body)?,
        })
    }

    fn match_expression(
        &mut self,
        value: &resin_hir::Term,
        arms: &[resin_hir::MatchArm],
    ) -> Result<concrete::TermKind, Error> {
        let value = self.boxed(value)?;
        let mut tags: Vec<_> = value.ty.members().into_iter().map(Case::Type).collect();
        let mut completed = vec![];
        for (index, arm) in arms.iter().enumerate() {
            if let resin_hir::Case::Error { payload } = &arm.tag {
                let payload = self.ty(payload)?;
                let body = self.term(&arm.body)?;
                let mut matched = 0;
                tags.retain(|tag| {
                    if matches!(tag, Case::Type(Ty::Error { .. })) {
                        completed.push(concrete::MatchArm {
                            tag: tag.clone(),
                            error_payload: Some(payload.clone()),
                            binding: arm.binding,
                            body: body.clone(),
                        });
                        matched += 1;
                        false
                    } else {
                        true
                    }
                });
                if matched == 0 {
                    return Err(self.instance_error("Err arm has no remaining error members"));
                }
                continue;
            }
            if arm.tag == resin_hir::Case::Wildcard {
                if index + 1 != arms.len() || arm.binding.is_some() || tags.is_empty() {
                    return Err(self.instance_error(
                        "wildcard must be the final arm and cover a remaining variant",
                    ));
                }
                let body = self.term(&arm.body)?;
                completed.extend(tags.drain(..).map(|tag| concrete::MatchArm {
                    error_payload: None,
                    tag,
                    binding: None,
                    body: body.clone(),
                }));
                continue;
            }
            let arm = self.arm(arm)?;
            let Some(index) = tags.iter().position(|tag| *tag == arm.tag) else {
                return Err(
                    self.instance_error("unknown or duplicate match variant after substitution")
                );
            };
            tags.remove(index);
            completed.push(arm);
        }
        if !tags.is_empty() || completed.is_empty() {
            return Err(self.instance_error("match must cover every concrete variant exactly once"));
        }
        Ok(concrete::TermKind::Match {
            value,
            arms: completed,
        })
    }

    fn receiver(&self, source: resin_hir::ReceiverConversion) -> concrete::ReceiverConversion {
        match source {
            resin_hir::ReceiverConversion::Value => concrete::ReceiverConversion::Value,
            resin_hir::ReceiverConversion::ReadOnly => concrete::ReceiverConversion::ReadOnly,
            resin_hir::ReceiverConversion::Borrow => concrete::ReceiverConversion::Borrow,
            resin_hir::ReceiverConversion::Load => concrete::ReceiverConversion::Load,
        }
    }

    fn error(&self, kind: crate::ErrorKind) -> Error {
        self.instances.lower_error(
            super::LowerError {
                span: self.span,
                kind,
            },
            Some(self.current),
            self.location.clone(),
        )
    }

    fn typing_error(&self, error: TypeError) -> Error {
        self.error(crate::ErrorKind::Type { kind: error.kind })
    }

    fn instance_error(&self, message: impl Into<std::sync::Arc<str>>) -> Error {
        self.error(crate::ErrorKind::InvalidInstance {
            message: message.into(),
        })
    }

    fn method(
        &mut self,
        lookup: &resin_hir::MethodLookup,
    ) -> Result<super::substitute::ResolvedMethod, Error> {
        self.substitution
            .method(lookup, self.instances)
            .map_err(|mut error| {
                error.span = self.span;
                self.instances
                    .lower_error(error, Some(self.current), self.location.clone())
            })
    }

    fn dependent_method(
        &mut self,
        lookup: &resin_hir::MethodLookup,
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        if !lookup.associated {
            return Err(self.error(crate::ErrorKind::InvalidHir {
                message: "dependent method references must be associated functions".into(),
            }));
        }
        let method = self.method(lookup)?;
        let params = method
            .params
            .iter()
            .map(|ty| self.ty(ty))
            .collect::<Result<Vec<_>, _>>()?;
        let signature = Ty::Function {
            params,
            result: Box::new(self.ty(&method.result)?),
        };
        let expected = self.ty(expected)?;
        self.instances
            .typer()
            .same(&expected, &signature)
            .map_err(|error| self.typing_error(error))?;
        Ok(concrete::TermKind::Function {
            function: match method.target {
                super::substitute::MethodTarget::Source {
                    function,
                    arguments,
                } => self.request(function, arguments)?,
                super::substitute::MethodTarget::Primitive { .. }
                | super::substitute::MethodTarget::Intrinsic { .. } => {
                    return Err(
                        self.instance_error("primitive operators cannot be referenced as methods")
                    );
                }
            },
        })
    }

    fn operation_call(
        &mut self,
        lookup: &resin_hir::OperationLookup,
        arguments: &[resin_hir::Term],
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        let operation = self
            .substitution
            .operation(lookup, self.instances)
            .map_err(|error| self.error(error.kind))?;
        let params = operation
            .params
            .iter()
            .map(|ty| self.ty(ty))
            .collect::<Result<Vec<_>, _>>()?;
        let result = self.ty(&operation.result)?;
        let expected = self.ty(expected)?;
        self.require_assignable(&result, &expected)?;
        let args = self.call_arguments(arguments, &params)?;
        let (function, arguments) = match operation.target {
            super::substitute::MethodTarget::Intrinsic { op } => {
                return Ok(concrete::TermKind::Intrinsic {
                    op,
                    type_args: vec![],
                    args: concrete::Arguments {
                        params,
                        values: args,
                    },
                });
            }
            super::substitute::MethodTarget::Source {
                function,
                arguments,
            } => (function, arguments),
            super::substitute::MethodTarget::Primitive { symbol } => {
                return self.completed_builtin(&symbol, args, &expected);
            }
        };
        let function = concrete::Term {
            span: self.span,
            ty: Ty::Function {
                params,
                result: Box::new(result),
            },
            kind: concrete::TermKind::Function {
                function: self.request(function, arguments)?,
            },
        };
        Ok(concrete::TermKind::Call {
            func: Box::new(function),
            args,
        })
    }

    fn dependent_method_call(
        &mut self,
        lookup: &resin_hir::MethodLookup,
        receiver: Option<&resin_hir::Term>,
        arguments: &[resin_hir::Term],
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        if lookup.associated == receiver.is_some() {
            return Err(self.error(crate::ErrorKind::InvalidHir {
                message: "dependent method call has an inconsistent receiver".into(),
            }));
        }
        if let Some(receiver) = receiver {
            let owner = self.ty(&lookup.receiver)?;
            let source = self.ty(&receiver.ty)?;
            self.instances
                .typer()
                .same(&owner, &source)
                .map_err(|error| self.typing_error(error))?;
        }
        let method = self.method(lookup)?;
        let params = method
            .params
            .iter()
            .map(|ty| self.ty(ty))
            .collect::<Result<Vec<_>, _>>()?;
        let result = self.ty(&method.result)?;
        let expected = self.ty(expected)?;
        self.require_assignable(&result, &expected)?;
        let receiver = receiver
            .map(|receiver| {
                if matches!(
                    self.argument(&method.params[0])?,
                    resin_hir::Type::Reference { .. }
                ) {
                    Ok(Box::new(concrete::Term {
                        span: receiver.span,
                        ty: params[0].clone(),
                        kind: self.reference_use(receiver, &method.params[0], Access::Value)?,
                    }))
                } else {
                    self.method_receiver(receiver, &params[0])
                }
            })
            .transpose()?;
        let offset = usize::from(receiver.is_some());
        let mut args: Vec<_> = receiver.into_iter().map(|receiver| *receiver).collect();
        args.extend(self.call_arguments(arguments, &params[offset..])?);
        let (function, arguments) = match method.target {
            super::substitute::MethodTarget::Intrinsic { .. } => {
                return Err(
                    self.instance_error("intrinsic operations require free-function lookup")
                );
            }
            super::substitute::MethodTarget::Source {
                function,
                arguments,
            } => (function, arguments),
            super::substitute::MethodTarget::Primitive { symbol } => {
                return self.completed_builtin(&symbol, args, &expected);
            }
        };
        let function = concrete::Term {
            span: self.span,
            ty: Ty::Function {
                params,
                result: Box::new(result),
            },
            kind: concrete::TermKind::Function {
                function: self.request(function, arguments)?,
            },
        };
        Ok(concrete::TermKind::Call {
            func: Box::new(function),
            args,
        })
    }

    fn require_assignable(&self, from: &Ty, to: &Ty) -> Result<(), Error> {
        if from == to || from.widens_to(to) {
            return Ok(());
        }
        self.instances
            .typer()
            .same(to, from)
            .map_err(|error| self.typing_error(error))
    }

    fn call(
        &mut self,
        function: &resin_hir::Term,
        arguments: &[resin_hir::Term],
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        let function = self.boxed(function)?;
        let Ty::Function { params, result } = &function.ty else {
            return Err(self.instance_error("a call requires a function value"));
        };
        let args = self.call_arguments(arguments, params)?;
        let expected = self.ty(expected)?;
        self.require_assignable(result, &expected)?;
        Ok(concrete::TermKind::Call {
            func: function,
            args,
        })
    }

    fn call_arguments(
        &mut self,
        arguments: &[resin_hir::Term],
        params: &[Ty],
    ) -> Result<Vec<concrete::Term>, Error> {
        if arguments.len() != params.len() {
            return Err(self.instance_error(format!(
                "expected {} arguments, found {}",
                params.len(),
                arguments.len()
            )));
        }
        arguments
            .iter()
            .zip(params)
            .map(|(source, param)| {
                let value = self.term(source)?;
                self.require_assignable(&value.ty, param)?;
                Ok(value)
            })
            .collect()
    }

    fn record(
        &mut self,
        fields: &[(Ident, resin_hir::Term)],
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        let expected = self.ty(expected)?;
        let Ty::Record { fields: expected } = expected else {
            return Err(self.instance_error("record arguments require a record parameter type"));
        };
        if fields.len() != expected.len() {
            return Err(self.instance_error("record arguments do not match the parameter fields"));
        }
        let mut seen = vec![false; expected.len()];
        let mut completed = Vec::with_capacity(fields.len());
        for (name, source) in fields {
            let Some((index, field)) = expected
                .iter()
                .enumerate()
                .find(|(_, field)| field.name == name.val)
            else {
                return Err(
                    self.instance_error(format!("record parameter has no field {}", name.val))
                );
            };
            if seen[index] {
                return Err(self.instance_error(format!("duplicate record argument {}", name.val)));
            }
            seen[index] = true;
            let value = self.term(source)?;
            self.require_assignable(&value.ty, &field.ty)?;
            completed.push(concrete::RecordInitializer { index, value });
        }
        Ok(concrete::TermKind::Record { fields: completed })
    }

    fn array(
        &mut self,
        elements: &[resin_hir::Term],
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        let expected = self.ty(expected)?;
        let Ty::Array { element, length } = expected else {
            return Err(self.instance_error("array arguments require an array parameter type"));
        };
        if elements.len() != length {
            return Err(self.instance_error("array arguments do not match the parameter length"));
        }
        let elements = elements
            .iter()
            .map(|source| {
                let value = self.term(source)?;
                self.require_assignable(&value.ty, &element)?;
                Ok(value)
            })
            .collect::<Result<_, Error>>()?;
        Ok(concrete::TermKind::Array { elems: elements })
    }

    fn method_receiver(
        &mut self,
        source: &resin_hir::Term,
        to: &Ty,
    ) -> Result<Box<concrete::Term>, Error> {
        use concrete::ReceiverConversion;
        let from = self.ty(&source.ty)?;
        let conversion = if &from == to {
            ReceiverConversion::Value
        } else if matches!((&from, to), (Ty::Reference { mutable: true, referent: a }, Ty::Reference { mutable: false, referent: b }) if a == b)
        {
            ReceiverConversion::ReadOnly
        } else if matches!(to, Ty::Reference { referent, .. } if **referent == from) {
            ReceiverConversion::Borrow
        } else if matches!(&from, Ty::Reference { referent, .. } if referent.as_ref() == to) {
            ReceiverConversion::Load
        } else {
            return Err(self.instance_error("method receiver does not match the first parameter"));
        };
        let argument = if conversion == ReceiverConversion::Borrow {
            self.reference_place(source, matches!(to, Ty::Reference { mutable: true, .. }))?
        } else {
            self.boxed(source)?
        };
        Ok(Box::new(concrete::Term {
            span: source.span,
            ty: to.clone(),
            kind: concrete::TermKind::Adapt {
                conversion,
                arg: argument,
            },
        }))
    }

    fn numeric(
        &mut self,
        text: &str,
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        let ty = self.ty(expected)?;
        let ty = self
            .instances
            .typer()
            .body(&ty)
            .map_err(|error| self.typing_error(error))?;
        let value = resin_types::literal::parse(text, &ty)
            .map_err(|message| self.instance_error(message))?;
        Ok(concrete::TermKind::Constant { value })
    }

    fn layout(&mut self, of: &resin_hir::Type, size: bool) -> Result<concrete::TermKind, Error> {
        let ty = self.ty(of)?;
        let layout = resin_types::layout::layout(self.instances.typer().definitions(), &ty)
            .map_err(|error| self.instance_error(error.to_string()))?;
        let value = if size { layout.size } else { layout.align } as u64;
        Ok(concrete::TermKind::Constant {
            value: Value::UInt64 { value },
        })
    }

    fn conversion(
        &mut self,
        arg: &resin_hir::Term,
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        let from = self.ty(&arg.ty)?;
        let to = self.ty(expected)?;
        let conversion = self
            .instances
            .typer()
            .explicit_conversion(&from, &to)
            .map_err(|error| self.typing_error(error))?;
        Ok(concrete::TermKind::Convert {
            conversion,
            arg: self.boxed(arg)?,
        })
    }

    fn reference_place(
        &mut self,
        source: &resin_hir::Term,
        mutable: bool,
    ) -> Result<Box<concrete::Term>, Error> {
        let place = self.place(source)?;
        if !reference_place(&place) {
            return Err(self.instance_error(
                "reference binding requires an initialized place; bind the temporary to a local first",
            ));
        }
        if mutable && !mutable_place(&place) {
            return Err(self.instance_error("writable access requires RefMut or a mutable place; declare local storage with `let mut`"));
        }
        Ok(place)
    }

    fn reference_use(
        &mut self,
        source: &resin_hir::Term,
        expected: &resin_hir::Type,
        access: Access,
    ) -> Result<concrete::TermKind, Error> {
        let from = self.argument(&source.ty)?;
        let target = self.argument(expected)?;
        if let resin_hir::Type::Reference { referent, mutable } = &target {
            if let resin_hir::Type::Reference {
                referent: source_type,
                mutable: source_mutable,
            } = &from
            {
                if source_type != referent {
                    return Err(self.instance_error("reference referent types must match exactly"));
                }
                if *mutable && !source_mutable {
                    return Err(self.instance_error("cannot obtain RefMut from a read-only Ref"));
                }
                let source = self.boxed(source)?;
                return Ok(if source_mutable == mutable {
                    source.kind
                } else {
                    concrete::TermKind::Adapt {
                        conversion: concrete::ReceiverConversion::ReadOnly,
                        arg: source,
                    }
                });
            }
            if &from != referent.as_ref() {
                return Err(self.instance_error("reference referent types must match exactly"));
            }
            let place = self.reference_place(source, *mutable)?;
            return Ok(concrete::TermKind::Borrow { place });
        }
        let value = if let resin_hir::Type::Reference { referent, .. } = from {
            if access == Access::Value
                && !self
                    .ty(&referent)?
                    .copies_implicitly(self.instances.typer().definitions())
            {
                return Err(self.instance_error("cannot move a value through a reference or pointer; replace its contents instead"));
            }
            concrete::Term {
                span: source.span,
                ty: self.ty(&referent)?,
                kind: concrete::TermKind::Deref {
                    pointer: self.boxed(source)?,
                },
            }
        } else {
            // Identity uses preserve place access, including opaque managed fields
            // addressed from a shader. Do not introduce a value read here.
            if from == target {
                return Ok(self.complete_term(source, access)?.kind);
            }
            self.term(source)?
        };
        let target = self.ty(&target)?;
        self.require_assignable(&value.ty, &target)?;
        if value.ty == target {
            Ok(value.kind)
        } else {
            Ok(concrete::TermKind::Convert {
                conversion: ExplicitConversion::Widen,
                arg: Box::new(value),
            })
        }
    }

    // Generic operation results acquire their final reference contract here.
    // Preserve the same source capability after resolving dependent signatures.
    fn require_addressable(&mut self, source: &resin_hir::Term) -> Result<(), Error> {
        if matches!(
            self.argument(&source.ty)?,
            resin_hir::Type::Reference { .. }
        ) {
            return Err(self.instance_error(
                "cannot take the address of a Ref; accept or return a Ptr when an address is required",
            ));
        }
        match &source.kind {
            resin_hir::TermKind::Local { .. } => Err(self.instance_error(
                "cannot take the address of a local value; borrow it with Ref or use explicitly allocated storage",
            )),
            resin_hir::TermKind::Use { arg } => self.require_addressable(arg),
            resin_hir::TermKind::Field { base, .. } => {
                let ty = self.argument(&base.ty)?;
                let ty = match &ty {
                    resin_hir::Type::Reference { referent, .. } => referent.as_ref(),
                    ty => ty,
                };
                if matches!(ty, resin_hir::Type::Pointer { .. }) {
                    Ok(())
                } else {
                    self.require_addressable(base)
                }
            }
            _ => Ok(()),
        }
    }

    fn field(
        &mut self,
        base: &resin_hir::Term,
        name: &str,
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        let ty = self.ty(&base.ty)?;
        let expected = self.ty(expected)?;
        let access = super::substitute::member(self.instances.typer(), &ty, name)
            .map_err(|error| self.error(error.kind))?;
        self.instances
            .typer()
            .same(&expected, &access.ty)
            .map_err(|error| self.typing_error(error))?;
        let base = self.place(base)?;
        if !reference_place(&base)
            && !access
                .ty
                .copies_implicitly(self.instances.typer().definitions())
            && let Ty::Defined { definition } = &base.ty
            && self.instances.typer().definitions()[definition.index()]
                .drop_hook()
                .is_some()
        {
            return Err(self.instance_error("cannot move a field out of a type with a drop hook"));
        }
        Ok(concrete::TermKind::Field { base, access })
    }

    // A dependent receiver may specialize to a pointer or a type with a drop
    // hook. Resolve those boundaries before storage lowering sees a Move.
    fn require_owned_move(&self, place: &concrete::Term) -> Result<(), Error> {
        match &place.kind {
            concrete::TermKind::Local { .. } if !matches!(place.ty, Ty::Reference { .. }) => Ok(()),
            concrete::TermKind::Field { base, access }
                if !matches!(base.ty, Ty::Pointer { .. } | Ty::Reference { .. })
                    && !access.steps.iter().any(|step| matches!(step, Conv::Deref)) =>
            {
                if let Ty::Defined { definition } = &base.ty
                    && self.instances.typer().definitions()[definition.index()]
                        .drop_hook()
                        .is_some()
                {
                    return Err(
                        self.instance_error("cannot move a field out of a type with a drop hook")
                    );
                }
                self.require_owned_move(base)
            }
            _ => Err(self.instance_error(
                "cannot move a value through a reference or pointer; replace its contents instead",
            )),
        }
    }

    fn builtin(
        &mut self,
        name: &std::sync::Arc<str>,
        args: &[resin_hir::Term],
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        let expected = self.ty(expected)?;
        let args = args
            .iter()
            .map(|arg| self.term(arg))
            .collect::<Result<Vec<_>, _>>()?;
        self.completed_builtin(name, args, &expected)
    }

    fn completed_builtin(
        &mut self,
        name: &std::sync::Arc<str>,
        args: Vec<concrete::Term>,
        expected: &Ty,
    ) -> Result<concrete::TermKind, Error> {
        let params = args.iter().map(|arg| arg.ty.clone()).collect::<Vec<_>>();
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
            .same(expected, &signature.result)
            .map_err(|error| {
                self.instances.lower_error(
                    super::LowerError::typing(self.span, error),
                    Some(self.current),
                    self.location.clone(),
                )
            })?;
        Ok(concrete::TermKind::Builtin {
            name: name.clone(),
            args,
        })
    }

    fn intrinsic(
        &mut self,
        op: Intrinsic,
        parameters: &[resin_hir::Type],
        args: &resin_hir::Arguments,
    ) -> Result<concrete::TermKind, Error> {
        if matches!(
            op,
            Intrinsic::GpuPointerProjection
                | Intrinsic::GpuSequenceProjection
                | Intrinsic::GpuPipelineType
        ) {
            return Err(self.instance_error("GPU projection and pipeline type declarations are contracts for dispatch and draw; they cannot be called directly"));
        }
        if matches!(op, Intrinsic::TraceRay | Intrinsic::RayHitInfo)
            && self.instances.profile(self.current) != crate::Profile::Shader
        {
            return Err(self.instance_error("ray operations require shader execution"));
        }
        let type_args = parameters
            .iter()
            .map(|ty| self.ty(ty))
            .collect::<Result<Vec<_>, _>>()?;
        if op == Intrinsic::OwnerAllocate
            && type_args.first().is_some_and(|element| {
                !element.copies_implicitly(self.instances.typer().definitions())
            })
        {
            return Err(self.instance_error("repeated allocation requires an implicitly copyable element; use single-value allocation to transfer ownership"));
        }
        if matches!(
            op,
            Intrinsic::GpuViewRange
                | Intrinsic::GpuViewLoad
                | Intrinsic::GpuViewStore
                | Intrinsic::GpuViewReplace
                | Intrinsic::GpuViewCopyTo
                | Intrinsic::GpuViewCopyFrom
        ) {
            let [element] = type_args.as_slice() else {
                return Err(self.instance_error("GPU access requires exactly one element type"));
            };
            if !element.gpu_element(self.instances.typer().definitions()) {
                return Err(self.instance_error(format!(
                    "GPU element {} must have plain shared storage",
                    resin_types::format_type(element, self.instances.typer().definitions())
                )));
            }
        }
        let args = self.arguments(args)?;
        let math = match op {
            Intrinsic::Repr => Some("repr"),
            Intrinsic::Sqrt => Some("sqrt"),
            Intrinsic::Sin => Some("sin"),
            Intrinsic::Cos => Some("cos"),
            _ => None,
        };
        if let Some(name) = math {
            if self.instances.profile(self.current) == crate::Profile::Shader {
                resin_types::shader::builtin_instance(self.instances.typer(), name, &args.params)
                    .map_err(|message| self.profile_error(message))?;
            } else {
                self.instances
                    .typer()
                    .builtin_instance(name, &args.params)
                    .map_err(|error| self.typing_error(error))?;
            }
            return Ok(concrete::TermKind::Builtin {
                name: name.into(),
                args: args.values,
            });
        }
        let builtin = match op {
            Intrinsic::FormatBytes => Some("format_bytes"),
            Intrinsic::StringFromBytes => Some("string_from_bytes"),
            _ => None,
        };
        if let Some(name) = builtin {
            self.instances
                .typer()
                .builtin_instance(name, &args.params)
                .map_err(|error| self.typing_error(error))?;
        }
        if op == Intrinsic::PointerBytes
            && !matches!(args.params.first(), Some(Ty::Pointer { pointee }) if pointee.is_numeric())
        {
            return Err(self.instance_error("byte views require numeric elements"));
        }
        Ok(concrete::TermKind::Intrinsic {
            op,
            type_args,
            args,
        })
    }

    fn kind(
        &mut self,
        source: &resin_hir::TermKind,
        expected: &resin_hir::Type,
    ) -> Result<concrete::TermKind, Error> {
        Ok(match source {
            resin_hir::TermKind::OperationCall { lookup, args } => {
                self.operation_call(lookup, args, expected)?
            }
            resin_hir::TermKind::RequireCopy { requirements, body } => {
                for requirement in requirements {
                    self.span = requirement.span;
                    if matches!(
                        self.argument(&requirement.target)?,
                        resin_hir::Type::Reference { .. }
                    ) {
                        continue;
                    }
                    let ty = self.ty(&requirement.source)?;
                    if !ty.copies_implicitly(self.instances.typer().definitions()) {
                        let name =
                            resin_types::format_type(&ty, self.instances.typer().definitions());
                        return Err(self.instance_error(format!("this use requires `{name}` to be implicitly copyable; the type is move-only")));
                    }
                }
                self.term(body)?.kind
            }
            resin_hir::TermKind::ReadOwned { place } => {
                let place = self.place(place)?;
                if place
                    .ty
                    .copies_implicitly(self.instances.typer().definitions())
                {
                    place.kind
                } else {
                    self.require_owned_move(&place)?;
                    concrete::TermKind::Move { place }
                }
            }
            resin_hir::TermKind::Read { place } => {
                let source = self.argument(&place.ty)?;
                let value = self.term(place)?;
                if !matches!(source, resin_hir::Type::Reference { .. })
                    && reference_place(&value)
                    && !value
                        .ty
                        .copies_implicitly(self.instances.typer().definitions())
                {
                    return Err(self.instance_error("cannot move a value through a reference or pointer; replace its contents instead"));
                }
                // A dependent result can become a fresh owned value instead of
                // a reference. Such a value transfers directly without copying.
                // Reference values preserve their address; reference_use checks
                // copyability only when a consumer requests the referent value.
                value.kind
            }
            resin_hir::TermKind::Move { place } => concrete::TermKind::Move {
                place: self.place(place)?,
            },
            resin_hir::TermKind::Constant { value } => concrete::TermKind::Constant {
                value: self.constant(value)?,
            },
            resin_hir::TermKind::Numeric { text } => self.numeric(text, expected)?,
            resin_hir::TermKind::Layout { of, size } => self.layout(of, *size)?,
            resin_hir::TermKind::SizeOf { of } => {
                let ty = self.ty(of)?;
                let layout = resin_types::layout::value(self.instances.typer().definitions(), &ty)
                    .map_err(|error| self.instance_error(error.to_string()))?;
                concrete::TermKind::Constant {
                    value: Value::UInt64 {
                        value: layout.size as u64,
                    },
                }
            }
            resin_hir::TermKind::Local {
                binding,
                name,
                mutable,
            } => concrete::TermKind::Local {
                binding: *binding,
                name: name.clone(),
                mutable: *mutable,
            },
            resin_hir::TermKind::Function {
                function,
                type_args,
            } => {
                let arguments = type_args
                    .iter()
                    .map(|ty| self.argument(ty))
                    .collect::<Result<_, _>>()?;
                concrete::TermKind::Function {
                    function: self.request(*function, arguments)?,
                }
            }
            resin_hir::TermKind::DependentMethod { lookup } => {
                self.dependent_method(lookup, expected)?
            }
            resin_hir::TermKind::DependentMethodCall {
                lookup,
                receiver,
                args,
            } => self.dependent_method_call(lookup, receiver.as_deref(), args, expected)?,
            resin_hir::TermKind::Unwrap { value } => concrete::TermKind::Unwrap {
                value: self.boxed(value)?,
            },
            resin_hir::TermKind::Break => concrete::TermKind::Break,
            resin_hir::TermKind::Continue => concrete::TermKind::Continue,
            resin_hir::TermKind::Return { value } => concrete::TermKind::Return {
                value: self.boxed(value)?,
            },
            resin_hir::TermKind::Try { value } => concrete::TermKind::Try {
                value: self.boxed(value)?,
            },
            resin_hir::TermKind::Match { value, arms } => self.match_expression(value, arms)?,
            resin_hir::TermKind::If { cond, then, els } => concrete::TermKind::If {
                cond: self.boxed(cond)?,
                then: self.boxed(then)?,
                els: self.boxed(els)?,
            },
            resin_hir::TermKind::ParallelMap {
                input,
                element,
                body,
                ..
            } => {
                self.require_host_parallel()?;
                concrete::TermKind::ParallelMap {
                    input: self.boxed(input)?,
                    element: self.parallel_parameter(element)?,
                    body: self.boxed(body)?,
                }
            }
            resin_hir::TermKind::ParallelReduce {
                input,
                identity,
                left,
                right,
                body,
                ..
            } => {
                self.require_host_parallel()?;
                concrete::TermKind::ParallelReduce {
                    input: self.boxed(input)?,
                    identity: self.boxed(identity)?,
                    left: self.parallel_parameter(left)?,
                    right: self.parallel_parameter(right)?,
                    body: self.boxed(body)?,
                }
            }
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
            resin_hir::TermKind::Record { fields } => self.record(fields, expected)?,
            resin_hir::TermKind::Array { elems } => self.array(elems, expected)?,
            resin_hir::TermKind::Builtin { name, args } => self.builtin(name, args, expected)?,
            resin_hir::TermKind::Call { func, args } => self.call(func, args, expected)?,
            resin_hir::TermKind::Intrinsic {
                op,
                type_args,
                args,
            } => self.intrinsic(*op, type_args, args)?,
            resin_hir::TermKind::Adapt { conversion, arg } => concrete::TermKind::Adapt {
                conversion: self.receiver(*conversion),
                arg: if *conversion == resin_hir::ReceiverConversion::Borrow {
                    {
                        let mutable = matches!(
                            self.argument(expected)?,
                            resin_hir::Type::Reference { mutable: true, .. }
                        );
                        self.reference_place(arg, mutable)?
                    }
                } else {
                    self.boxed(arg)?
                },
            },
            resin_hir::TermKind::Use { arg } => {
                return self.reference_use(arg, expected, Access::Value);
            }
            resin_hir::TermKind::Convert { arg } => self.conversion(arg, expected)?,
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
            } => {
                let args = self.arguments(args)?;
                let pipeline = resin_types::gpu_pipeline_contract(
                    self.instances.typer().definitions(),
                    args.values[1]
                        .ty
                        .deref_target()
                        .unwrap_or(&args.values[1].ty),
                )
                .map_err(|message| self.instance_error(message))?;
                let projection = if pipeline.root == Ty::None {
                    None
                } else {
                    Some(
                        resin_types::gpu_projection_plan(
                            self.instances.typer().definitions(),
                            &args.values[2].ty,
                            &pipeline.root,
                        )
                        .map_err(|message| self.instance_error(message))?,
                    )
                };
                concrete::TermKind::GpuPipelineDispatch {
                    projection,
                    context: self.host_bridge(*context)?,
                    allocator: allocator.map(|id| self.host_bridge(id)).transpose()?,
                    record: self.host_bridge(*record)?,
                    args,
                }
            }
            resin_hir::TermKind::Absurd { arg } => concrete::TermKind::Absurd {
                arg: self.boxed(arg)?,
            },
            resin_hir::TermKind::Assign { place, value } => {
                let place = self.place(place)?;
                if !matches!(place.kind, concrete::TermKind::Local { .. }) && !mutable_place(&place)
                {
                    return Err(self.instance_error(
                        "cannot assign through a read-only Ref; use RefMut for writable access",
                    ));
                }
                concrete::TermKind::Assign {
                    place,
                    value: self.boxed(value)?,
                }
            }
            resin_hir::TermKind::Address { place } => {
                self.require_addressable(place)?;
                concrete::TermKind::Address {
                    place: self.place(place)?,
                }
            }
            resin_hir::TermKind::Deref { pointer } => concrete::TermKind::Deref {
                pointer: self.boxed(pointer)?,
            },
            resin_hir::TermKind::Field { base, name } => self.field(base, name, expected)?,
        })
    }
}

// Dependent calls learn whether an argument is a reference during specialization.
// Check the completed expression before storage lowering can spill a temporary.
fn reference_place(term: &concrete::Term) -> bool {
    match &term.kind {
        concrete::TermKind::Local { .. } | concrete::TermKind::Deref { .. } => true,
        concrete::TermKind::Field { base, .. } => {
            matches!(base.ty, Ty::Pointer { .. }) || reference_place(base)
        }
        _ => false,
    }
}

fn mutable_place(term: &concrete::Term) -> bool {
    if let Ty::Reference { mutable, .. } = &term.ty {
        return *mutable;
    }
    match &term.kind {
        concrete::TermKind::Local { mutable, .. } => *mutable,
        concrete::TermKind::Deref { pointer } => mutable_access(&pointer.ty).unwrap_or(false),
        concrete::TermKind::Field { base, .. } => {
            mutable_access(&base.ty).unwrap_or_else(|| mutable_place(base))
        }
        _ => false,
    }
}

fn mutable_access(ty: &Ty) -> Option<bool> {
    match ty {
        Ty::Pointer { pointee } => Some(mutable_access(pointee).unwrap_or(true)),
        Ty::Reference { referent, mutable } => Some(mutable_access(referent).unwrap_or(*mutable)),
        _ => None,
    }
}
