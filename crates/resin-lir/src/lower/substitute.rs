//! Normalize type identities, then materialize only the nominal layouts an operation uses.
//! Neither operation performs inference or numeric defaulting.
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

const TYPE_DEPTH_LIMIT: usize = 256;
const TYPE_SIZE_LIMIT: usize = 65536;
const METHOD_DEPTH_LIMIT: usize = 32;

struct Normalization {
    remaining: usize,
    methods: BTreeSet<resin_hir::MethodLookup>,
    operations: BTreeSet<resin_hir::OperationLookup>,
}

impl Default for Normalization {
    fn default() -> Self {
        Self {
            remaining: TYPE_SIZE_LIMIT,
            methods: BTreeSet::new(),
            operations: BTreeSet::new(),
        }
    }
}

#[derive(Default)]
pub(super) struct Substitution {
    arguments: BTreeMap<resin_hir::TypeParameterId, resin_hir::Type>,
}

pub(super) struct ResolvedMethod {
    pub(super) target: MethodTarget,
    pub(super) params: Vec<resin_hir::Type>,
    pub(super) result: resin_hir::Type,
}

pub(super) enum MethodTarget {
    Source {
        function: FunctionId,
        arguments: Vec<resin_hir::Type>,
    },
    Primitive {
        symbol: std::sync::Arc<str>,
    },
}

impl Substitution {
    pub(super) fn new(
        parameters: &[resin_hir::TypeParameter],
        arguments: &[resin_hir::Type],
    ) -> Result<Self, super::LowerError> {
        let mut result = Self::default();
        for (parameter, argument) in parameters.iter().zip(arguments) {
            if result
                .arguments
                .insert(parameter.id, argument.clone())
                .is_some()
            {
                return Err(super::LowerError::invalid_hir(
                    parameter.name.span,
                    "duplicate type parameter identity",
                ));
            }
        }
        Ok(result)
    }

    pub(super) fn ty(
        &self,
        source: &resin_hir::Type,
        instances: &mut super::instances::Instances<'_>,
    ) -> Result<Ty, super::LowerError> {
        materialize(&self.normalize(source, instances)?, instances)
    }

    pub(super) fn normalize(
        &self,
        source: &resin_hir::Type,
        instances: &mut super::instances::Instances<'_>,
    ) -> Result<resin_hir::Type, super::LowerError> {
        self.normalize_at(source, 0, &mut Normalization::default(), instances)
    }

    pub(super) fn operation(
        &self,
        lookup: &resin_hir::OperationLookup,
        instances: &mut super::instances::Instances<'_>,
    ) -> Result<ResolvedMethod, super::LowerError> {
        self.operation_at(lookup, 0, &mut Normalization::default(), instances)
    }

    fn operation_at(
        &self,
        lookup: &resin_hir::OperationLookup,
        depth: usize,
        state: &mut Normalization,
        instances: &mut super::instances::Instances<'_>,
    ) -> Result<ResolvedMethod, super::LowerError> {
        consume_node(depth, &mut state.remaining)?;
        let args = lookup
            .arguments
            .iter()
            .map(|arg| self.normalize_at(arg, depth + 1, state, instances))
            .collect::<Result<Vec<_>, _>>()?;
        let explicit = lookup
            .type_args
            .as_ref()
            .map(|args| {
                args.iter()
                    .map(|arg| self.normalize_at(arg, depth + 1, state, instances))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        let query = resin_hir::OperationLookup {
            primitive: lookup.primitive.clone(),
            name: lookup.name.clone(),
            candidates: lookup.candidates.clone(),
            type_args: explicit.clone(),
            arguments: args.clone(),
        };
        if !state.operations.insert(query.clone()) {
            return Err(operation_error("cyclic operation signature"));
        }
        if state.operations.len() > METHOD_DEPTH_LIMIT {
            return Err(super::LowerError {
                span: Span { start: 0, end: 0 },
                kind: crate::ErrorKind::TypeExpansionLimit {
                    limit: METHOD_DEPTH_LIMIT,
                },
            });
        }
        let result = (|| {
            let mut matches = Vec::new();
            for function in &lookup.candidates {
                let signature = instances.signature(*function)?;
                if signature.params.len() != args.len() {
                    continue;
                }
                if explicit
                    .as_ref()
                    .is_some_and(|args| args.len() != signature.type_params.len())
                {
                    continue;
                }
                let mut substitution = Self::default();
                if let Some(explicit) = &explicit {
                    substitution = Self::new(&signature.type_params, explicit)?;
                }
                let bound = signature.params.iter().zip(&args).all(|(parameter, arg)| {
                    match_parameter(
                        &parameter.annotation.ty,
                        arg,
                        &mut substitution.arguments,
                        0,
                    )
                });
                if !bound {
                    continue;
                }
                let Some(arguments) = signature
                    .type_params
                    .iter()
                    .map(|parameter| substitution.arguments.get(&parameter.id).cloned())
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                let candidate = (|| {
                    let params = signature
                        .params
                        .iter()
                        .map(|parameter| {
                            substitution.normalize_at(
                                &parameter.annotation.ty,
                                depth + 1,
                                state,
                                instances,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    for (param, arg) in params.iter().zip(&args) {
                        let (target, source) = match param {
                            resin_hir::Type::Reference { referent } => {
                                (referent.as_ref(), value_type(arg))
                            }
                            _ => (param, value_type(arg)),
                        };
                        let source = materialize(source, instances)?;
                        let target = materialize(target, instances)?;
                        if !source.widens_to(&target) {
                            return Ok(None);
                        }
                    }
                    let result = substitution.normalize_at(
                        &signature.result.ty,
                        depth + 1,
                        state,
                        instances,
                    )?;
                    Ok(Some(ResolvedMethod {
                        target: MethodTarget::Source {
                            function: *function,
                            arguments,
                        },
                        params,
                        result,
                    }))
                })();
                match candidate {
                    Ok(Some(candidate)) => matches.push(candidate),
                    Ok(None) => {}
                    Err(super::LowerError {
                        kind: crate::ErrorKind::InvalidInstance { .. },
                        ..
                    }) => {}
                    Err(error) => return Err(error),
                }
            }
            if let Some(symbol) = &lookup.primitive {
                let types = args
                    .iter()
                    .map(|arg| materialize(value_type(arg), instances))
                    .collect::<Result<Vec<_>, _>>()?;
                if let Ok(call) = instances.typer().type_builtin_call(symbol, &types) {
                    let result = if call.result == Ty::Bool {
                        resin_hir::Type::Bool
                    } else {
                        value_type(&args[0]).clone()
                    };
                    matches.push(ResolvedMethod {
                        target: MethodTarget::Primitive {
                            symbol: symbol.clone(),
                        },
                        params: args.iter().map(|arg| value_type(arg).clone()).collect(),
                        result,
                    });
                }
            }
            if matches.len() == 1 {
                return Ok(matches.pop().unwrap());
            }
            Err(operation_error(format!(
                "{} overload of `{}` for the substituted arguments",
                if matches.is_empty() {
                    "no matching"
                } else {
                    "ambiguous"
                },
                lookup.name
            )))
        })();
        state.operations.remove(&query);
        result
    }

    pub(super) fn method(
        &self,
        lookup: &resin_hir::MethodLookup,
        instances: &mut super::instances::Instances<'_>,
    ) -> Result<ResolvedMethod, super::LowerError> {
        self.method_at(lookup, 0, &mut Normalization::default(), instances)
    }

    fn method_at(
        &self,
        lookup: &resin_hir::MethodLookup,
        depth: usize,
        state: &mut Normalization,
        instances: &mut super::instances::Instances<'_>,
    ) -> Result<ResolvedMethod, super::LowerError> {
        consume_node(depth, &mut state.remaining)?;
        if let resin_hir::MethodName::Operator { symbol, arity } = &lookup.name
            && (resin_hir::MethodName::operator(symbol, *arity).is_none()
                || !lookup.associated
                || !lookup.type_args.is_empty())
        {
            return Err(super::LowerError::invalid_hir(
                Span { start: 0, end: 0 },
                "operator lookup requires a supported symbol and arity, all operands, and no method type arguments",
            ));
        }
        let receiver = self.normalize_at(&lookup.receiver, depth + 1, state, instances)?;
        if let resin_hir::MethodName::Operator { symbol, arity } = &lookup.name
            && !matches!(receiver, resin_hir::Type::Defined { .. })
        {
            let ty = materialize(&receiver, instances)?;
            let call = instances
                .typer()
                .type_builtin_call(symbol, &vec![ty; *arity])
                .map_err(|error| super::LowerError::typing(Span { start: 0, end: 0 }, error))?;
            // Primitive arithmetic preserves its operand type. Comparisons and
            // logical not produce bool; neither relationship needs inference.
            let result = if call.result == Ty::Bool {
                resin_hir::Type::Bool
            } else {
                receiver.clone()
            };
            return Ok(ResolvedMethod {
                target: MethodTarget::Primitive {
                    symbol: symbol.clone(),
                },
                params: vec![receiver; *arity],
                result,
            });
        }
        let (function, signature, mut arguments) = instances.method(&receiver, &lookup.name)?;
        let count = signature
            .type_params
            .len()
            .checked_sub(arguments.len())
            .ok_or_else(|| {
                super::LowerError::invalid_hir(
                    Span { start: 0, end: 0 },
                    "method signature omits its owner's type parameters",
                )
            })?;
        if lookup.type_args.len() != count {
            return Err(super::LowerError {
                span: Span { start: 0, end: 0 },
                kind: crate::ErrorKind::InvalidInstance {
                    message: format!(
                        "method {} expects {count} explicit type arguments, found {}; annotate the dependent method application",
                        lookup.name, lookup.type_args.len()
                    )
                    .into(),
                },
            });
        }
        for argument in &lookup.type_args {
            arguments.push(self.normalize_at(argument, depth + 1, state, instances)?);
        }
        if !lookup.associated && signature.params.is_empty() {
            return Err(super::LowerError {
                span: Span { start: 0, end: 0 },
                kind: crate::ErrorKind::InvalidInstance {
                    message: format!("method {} has no receiver parameter", lookup.name).into(),
                },
            });
        }
        let query = resin_hir::MethodLookup {
            receiver,
            name: lookup.name.clone(),
            type_args: arguments[arguments.len() - count..].to_vec(),
            associated: lookup.associated,
        };
        if state.methods.contains(&query) {
            return Err(super::LowerError {
                span: Span { start: 0, end: 0 },
                kind: crate::ErrorKind::InvalidInstance {
                    message: format!(
                        "cyclic dependent method signature for {}; annotate its result type",
                        lookup.name
                    )
                    .into(),
                },
            });
        }
        if state.methods.len() == METHOD_DEPTH_LIMIT {
            return Err(super::LowerError {
                span: Span { start: 0, end: 0 },
                kind: crate::ErrorKind::TypeExpansionLimit {
                    limit: METHOD_DEPTH_LIMIT,
                },
            });
        }
        state.methods.insert(query.clone());
        // Method lookup selects a declaration and substitutes completed arguments.
        // It never creates variables, deduces arguments, or chooses literal defaults.
        let completed = (|| {
            let substitution = Self::new(&signature.type_params, &arguments)?;
            let params = signature
                .params
                .iter()
                .map(|parameter| {
                    substitution.normalize_at(&parameter.annotation.ty, depth + 1, state, instances)
                })
                .collect::<Result<_, _>>()?;
            let result =
                substitution.normalize_at(&signature.result.ty, depth + 1, state, instances)?;
            Ok(ResolvedMethod {
                target: MethodTarget::Source {
                    function,
                    arguments,
                },
                params,
                result,
            })
        })();
        state.methods.remove(&query);
        completed
    }

    fn normalize_at(
        &self,
        source: &resin_hir::Type,
        depth: usize,
        state: &mut Normalization,
        instances: &mut super::instances::Instances<'_>,
    ) -> Result<resin_hir::Type, super::LowerError> {
        consume_node(depth, &mut state.remaining)?;
        Ok(match source {
            resin_hir::Type::Parameter { parameter } => {
                let argument = self.arguments.get(parameter).ok_or_else(|| {
                    super::LowerError::invalid_hir(
                        Span { start: 0, end: 0 },
                        "unbound type parameter; annotate the application",
                    )
                })?;
                // Charge substituted structure before cloning: recursive requests may double it.
                state.remaining += 1;
                check_size(argument, depth, &mut state.remaining)?;
                argument.clone()
            }
            resin_hir::Type::Member { base, name } => {
                let base = self.normalize_at(base, depth + 1, state, instances)?;
                let base = materialize(&base, instances)?;
                let result = expression(&member(instances.typer(), &base, name)?.ty, instances);
                check_size(&result, depth, &mut state.remaining)?;
                result
            }
            resin_hir::Type::Operation { lookup } => {
                let operation = self.operation_at(lookup, depth + 1, state, instances)?;
                resin_hir::Type::Function {
                    params: operation.params,
                    result: Box::new(operation.result),
                }
            }
            resin_hir::Type::Method { lookup } => {
                let method = self.method_at(lookup, depth + 1, state, instances)?;
                resin_hir::Type::Function {
                    params: method.params[usize::from(!lookup.associated)..].to_vec(),
                    result: Box::new(method.result),
                }
            }
            resin_hir::Type::FunctionParameter { function, .. }
            | resin_hir::Type::FunctionResult { function } => {
                let function = self.normalize_at(function, depth + 1, state, instances)?;
                let resin_hir::Type::Function { params, result } = function else {
                    return Err(super::LowerError {
                        span: Span { start: 0, end: 0 },
                        kind: crate::ErrorKind::InvalidInstance {
                            message: "a function type projection requires a function".into(),
                        },
                    });
                };
                if let resin_hir::Type::FunctionParameter { index, .. } = source {
                    params
                        .into_iter()
                        .nth(*index)
                        .ok_or_else(|| super::LowerError {
                            span: Span { start: 0, end: 0 },
                            kind: crate::ErrorKind::InvalidInstance {
                                message: format!("function has no parameter {index}").into(),
                            },
                        })?
                } else {
                    *result
                }
            }
            resin_hir::Type::Defined {
                definition,
                arguments,
            } => {
                instances.nominal_arity(*definition, arguments.len())?;
                resin_hir::Type::Defined {
                    definition: *definition,
                    arguments: arguments
                        .iter()
                        .map(|argument| self.normalize_at(argument, depth + 1, state, instances))
                        .collect::<Result<_, _>>()?,
                }
            }
            resin_hir::Type::Type => resin_hir::Type::Type,
            resin_hir::Type::Unit => resin_hir::Type::Unit,
            resin_hir::Type::None => resin_hir::Type::None,
            resin_hir::Type::Bool => resin_hir::Type::Bool,
            resin_hir::Type::Int8 => resin_hir::Type::Int8,
            resin_hir::Type::Int16 => resin_hir::Type::Int16,
            resin_hir::Type::Int32 => resin_hir::Type::Int32,
            resin_hir::Type::Int64 => resin_hir::Type::Int64,
            resin_hir::Type::UInt8 => resin_hir::Type::UInt8,
            resin_hir::Type::UInt16 => resin_hir::Type::UInt16,
            resin_hir::Type::UInt32 => resin_hir::Type::UInt32,
            resin_hir::Type::UInt64 => resin_hir::Type::UInt64,
            resin_hir::Type::Float32 => resin_hir::Type::Float32,
            resin_hir::Type::Float64 => resin_hir::Type::Float64,
            resin_hir::Type::Str => resin_hir::Type::Str,
            resin_hir::Type::GpuView => resin_hir::Type::GpuView,
            resin_hir::Type::GpuPipelineContract => resin_hir::Type::GpuPipelineContract,
            resin_hir::Type::GpuArguments => resin_hir::Type::GpuArguments,
            resin_hir::Type::Foreign { name } => resin_hir::Type::Foreign { name: name.clone() },
            resin_hir::Type::Reference { referent } => resin_hir::Type::Reference {
                referent: Box::new(self.normalize_at(referent, depth + 1, state, instances)?),
            },
            resin_hir::Type::Value { of } => {
                match self.normalize_at(of, depth + 1, state, instances)? {
                    resin_hir::Type::Reference { referent } => *referent,
                    value => value,
                }
            }
            resin_hir::Type::Pointer { pointee } => resin_hir::Type::Pointer {
                pointee: Box::new(self.normalize_at(pointee, depth + 1, state, instances)?),
            },
            resin_hir::Type::StrongOwner => resin_hir::Type::StrongOwner,
            resin_hir::Type::WeakOwner => resin_hir::Type::WeakOwner,
            resin_hir::Type::Function { params, result } => resin_hir::Type::Function {
                params: params
                    .iter()
                    .map(|ty| self.normalize_at(ty, depth + 1, state, instances))
                    .collect::<Result<_, _>>()?,
                result: Box::new(self.normalize_at(result, depth + 1, state, instances)?),
            },
            resin_hir::Type::Error { payload } => resin_hir::Type::Error {
                payload: Box::new(self.normalize_at(payload, depth + 1, state, instances)?),
            },
            resin_hir::Type::Array { element, length } => resin_hir::Type::Array {
                element: Box::new(self.normalize_at(element, depth + 1, state, instances)?),
                length: *length,
            },
            resin_hir::Type::Record { fields } => resin_hir::Type::Record {
                fields: fields
                    .iter()
                    .map(|field| {
                        Ok(resin_hir::RecordField {
                            name: field.name.clone(),
                            ty: self.normalize_at(&field.ty, depth + 1, state, instances)?,
                        })
                    })
                    .collect::<Result<_, super::LowerError>>()?,
            },
            resin_hir::Type::Union { variants } => {
                let mut members = vec![];
                for variant in variants {
                    match self.normalize_at(variant, depth + 1, state, instances)? {
                        resin_hir::Type::Union { variants } => members.extend(variants),
                        ty => members.push(ty),
                    }
                }
                members.sort();
                members.dedup();
                if members.len() == 1 {
                    members.pop().unwrap()
                } else {
                    resin_hir::Type::Union { variants: members }
                }
            }
        })
    }
}

fn materialize(
    source: &resin_hir::Type,
    instances: &mut super::instances::Instances<'_>,
) -> Result<Ty, super::LowerError> {
    Ok(match source {
        resin_hir::Type::Parameter { .. }
        | resin_hir::Type::Member { .. }
        | resin_hir::Type::Method { .. }
        | resin_hir::Type::Operation { .. }
        | resin_hir::Type::FunctionParameter { .. }
        | resin_hir::Type::FunctionResult { .. }
        | resin_hir::Type::Value { .. } => {
            unreachable!("normalized type")
        }
        resin_hir::Type::Defined {
            definition,
            arguments,
        } => Ty::Defined {
            definition: instances.nominal(*definition, arguments.clone())?,
        },
        resin_hir::Type::Type => Ty::Type,
        resin_hir::Type::Unit => Ty::Unit,
        resin_hir::Type::None => Ty::None,
        resin_hir::Type::Bool => Ty::Bool,
        resin_hir::Type::Int8 => Ty::Int8,
        resin_hir::Type::Int16 => Ty::Int16,
        resin_hir::Type::Int32 => Ty::Int32,
        resin_hir::Type::Int64 => Ty::Int64,
        resin_hir::Type::UInt8 => Ty::UInt8,
        resin_hir::Type::UInt16 => Ty::UInt16,
        resin_hir::Type::UInt32 => Ty::UInt32,
        resin_hir::Type::UInt64 => Ty::UInt64,
        resin_hir::Type::Float32 => Ty::Float32,
        resin_hir::Type::Float64 => Ty::Float64,
        resin_hir::Type::Str => Ty::Str,
        resin_hir::Type::GpuView => Ty::GpuView,
        resin_hir::Type::GpuPipelineContract => Ty::GpuPipelineContract,
        resin_hir::Type::GpuArguments => Ty::GpuArguments,
        resin_hir::Type::Foreign { name } => Ty::Foreign { name: name.clone() },
        resin_hir::Type::Reference { referent: pointee } | resin_hir::Type::Pointer { pointee } => {
            Ty::Pointer {
                pointee: Box::new(materialize(pointee, instances)?),
            }
        }
        resin_hir::Type::StrongOwner => Ty::StrongOwner,
        resin_hir::Type::WeakOwner => Ty::WeakOwner,
        resin_hir::Type::Function { params, result } => Ty::Function {
            params: params
                .iter()
                .map(|ty| materialize(ty, instances))
                .collect::<Result<_, _>>()?,
            result: Box::new(materialize(result, instances)?),
        },
        resin_hir::Type::Error { payload } => Ty::Error {
            payload: Box::new(materialize(payload, instances)?),
        },
        resin_hir::Type::Array { element, length } => Ty::Array {
            element: Box::new(materialize(element, instances)?),
            length: *length,
        },
        resin_hir::Type::Record { fields } => Ty::Record {
            fields: fields
                .iter()
                .map(|field| {
                    Ok(RecordField {
                        name: field.name.clone(),
                        ty: materialize(&field.ty, instances)?,
                    })
                })
                .collect::<Result<_, super::LowerError>>()?,
        },
        resin_hir::Type::Union { variants } => Ty::union_of(
            variants
                .iter()
                .map(|variant| materialize(variant, instances))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    })
}

// Restore nominal origins when a determining member lookup supplies an argument.
fn expression(source: &Ty, instances: &super::instances::Instances<'_>) -> resin_hir::Type {
    match source {
        Ty::Defined { definition } => instances.nominal_origin(*definition),
        Ty::Type => resin_hir::Type::Type,
        Ty::Unit => resin_hir::Type::Unit,
        Ty::None => resin_hir::Type::None,
        Ty::Bool => resin_hir::Type::Bool,
        Ty::Int8 => resin_hir::Type::Int8,
        Ty::Int16 => resin_hir::Type::Int16,
        Ty::Int32 => resin_hir::Type::Int32,
        Ty::Int64 => resin_hir::Type::Int64,
        Ty::UInt8 => resin_hir::Type::UInt8,
        Ty::UInt16 => resin_hir::Type::UInt16,
        Ty::UInt32 => resin_hir::Type::UInt32,
        Ty::UInt64 => resin_hir::Type::UInt64,
        Ty::Float32 => resin_hir::Type::Float32,
        Ty::Float64 => resin_hir::Type::Float64,
        Ty::Str => resin_hir::Type::Str,
        Ty::GpuView => resin_hir::Type::GpuView,
        Ty::GpuPipelineContract => resin_hir::Type::GpuPipelineContract,
        Ty::GpuArguments => resin_hir::Type::GpuArguments,
        Ty::Foreign { name } => resin_hir::Type::Foreign { name: name.clone() },
        Ty::Pointer { pointee } => resin_hir::Type::Pointer {
            pointee: Box::new(expression(pointee, instances)),
        },
        Ty::StrongOwner => resin_hir::Type::StrongOwner,
        Ty::WeakOwner => resin_hir::Type::WeakOwner,
        Ty::Function { params, result } => resin_hir::Type::Function {
            params: params.iter().map(|ty| expression(ty, instances)).collect(),
            result: Box::new(expression(result, instances)),
        },
        Ty::Error { payload } => resin_hir::Type::Error {
            payload: Box::new(expression(payload, instances)),
        },
        Ty::Array { element, length } => resin_hir::Type::Array {
            element: Box::new(expression(element, instances)),
            length: *length,
        },
        Ty::Record { fields } => resin_hir::Type::Record {
            fields: fields
                .iter()
                .map(|field| resin_hir::RecordField {
                    name: field.name.clone(),
                    ty: expression(&field.ty, instances),
                })
                .collect(),
        },
        Ty::Union { variants } => {
            // Concrete IDs follow layout discovery; identity keys follow source
            // origins. Restore canonical ordering when crossing back to keys.
            let mut variants: Vec<_> = variants
                .iter()
                .map(|variant| expression(variant, instances))
                .collect();
            variants.sort();
            variants.dedup();
            if variants.len() == 1 {
                variants.pop().unwrap()
            } else {
                resin_hir::Type::Union { variants }
            }
        }
    }
}

pub(super) fn member(
    typer: &TyperContext,
    base: &Ty,
    name: &str,
) -> Result<FieldAccess, super::LowerError> {
    let mut base = base;
    while let Some(pointee) = base.deref_target() {
        base = pointee;
    }
    typer
        .type_field(base, name)
        .map_err(|error| super::LowerError::typing(Span { start: 0, end: 0 }, error))
}

fn expansion_limit() -> super::LowerError {
    super::LowerError {
        span: Span { start: 0, end: 0 },
        kind: crate::ErrorKind::TypeExpansionLimit {
            limit: TYPE_DEPTH_LIMIT,
        },
    }
}

fn check_size(
    ty: &resin_hir::Type,
    depth: usize,
    remaining: &mut usize,
) -> Result<(), super::LowerError> {
    consume_node(depth, remaining)?;
    match ty {
        resin_hir::Type::Defined { arguments, .. } => {
            for argument in arguments {
                check_size(argument, depth + 1, remaining)?;
            }
        }
        resin_hir::Type::Parameter { .. }
        | resin_hir::Type::Member { .. }
        | resin_hir::Type::Method { .. }
        | resin_hir::Type::Operation { .. }
        | resin_hir::Type::FunctionParameter { .. }
        | resin_hir::Type::FunctionResult { .. }
        | resin_hir::Type::Value { .. } => {
            unreachable!("normalized argument")
        }
        resin_hir::Type::Reference { referent: pointee } | resin_hir::Type::Pointer { pointee } => {
            check_size(pointee, depth + 1, remaining)?
        }
        resin_hir::Type::Function { params, result } => {
            for param in params {
                check_size(param, depth + 1, remaining)?;
            }
            check_size(result, depth + 1, remaining)?;
        }
        resin_hir::Type::Error { payload } => check_size(payload, depth + 1, remaining)?,
        resin_hir::Type::Array { element, .. } => check_size(element, depth + 1, remaining)?,
        resin_hir::Type::Record { fields } => {
            for field in fields {
                check_size(&field.ty, depth + 1, remaining)?;
            }
        }
        resin_hir::Type::Union { variants } => {
            for variant in variants {
                check_size(variant, depth + 1, remaining)?;
            }
        }
        resin_hir::Type::Type
        | resin_hir::Type::Unit
        | resin_hir::Type::None
        | resin_hir::Type::Bool
        | resin_hir::Type::Int8
        | resin_hir::Type::Int16
        | resin_hir::Type::Int32
        | resin_hir::Type::Int64
        | resin_hir::Type::UInt8
        | resin_hir::Type::UInt16
        | resin_hir::Type::UInt32
        | resin_hir::Type::UInt64
        | resin_hir::Type::Float32
        | resin_hir::Type::Float64
        | resin_hir::Type::Str
        | resin_hir::Type::StrongOwner
        | resin_hir::Type::WeakOwner
        | resin_hir::Type::GpuView
        | resin_hir::Type::GpuPipelineContract
        | resin_hir::Type::GpuArguments
        | resin_hir::Type::Foreign { .. } => {}
    }
    Ok(())
}

fn consume_node(depth: usize, remaining: &mut usize) -> Result<(), super::LowerError> {
    if depth > TYPE_DEPTH_LIMIT {
        return Err(expansion_limit());
    }
    *remaining = remaining.checked_sub(1).ok_or(super::LowerError {
        span: Span { start: 0, end: 0 },
        kind: crate::ErrorKind::TypeSizeLimit {
            limit: TYPE_SIZE_LIMIT,
        },
    })?;
    Ok(())
}

fn operation_error(message: impl Into<std::sync::Arc<str>>) -> super::LowerError {
    super::LowerError {
        span: Span { start: 0, end: 0 },
        kind: crate::ErrorKind::InvalidInstance {
            message: message.into(),
        },
    }
}

fn value_type(ty: &resin_hir::Type) -> &resin_hir::Type {
    if let resin_hir::Type::Reference { referent } = ty {
        referent
    } else {
        ty
    }
}

// Candidate signatures are patterns over already determined argument types.
// This never invents weak variables, defaults a literal, or probes a body.
fn match_parameter(
    pattern: &resin_hir::Type,
    value: &resin_hir::Type,
    arguments: &mut BTreeMap<resin_hir::TypeParameterId, resin_hir::Type>,
    depth: usize,
) -> bool {
    use resin_hir::Type;
    if depth >= TYPE_DEPTH_LIMIT {
        return false;
    }
    let value = value_type(value);
    match pattern {
        Type::Parameter { parameter } => match arguments.get(parameter) {
            Some(previous) => previous == value,
            None => {
                arguments.insert(*parameter, value.clone());
                true
            }
        },
        Type::Reference { referent } => match_parameter(referent, value, arguments, depth + 1),
        Type::Defined {
            definition,
            arguments: params,
        } => {
            matches!(value, Type::Defined { definition: actual, arguments: values } if actual == definition && params.len() == values.len() && params.iter().zip(values).all(|(p,v)| match_parameter(p,v,arguments,depth+1)))
        }
        Type::Pointer { pointee } => {
            matches!(value, Type::Pointer { pointee: actual } if match_parameter(pointee, actual, arguments, depth+1))
        }
        Type::Array { element, length } => {
            matches!(value, Type::Array { element: actual, length: count } if length == count && match_parameter(element, actual, arguments, depth+1))
        }
        Type::Record { fields } => {
            matches!(value, Type::Record { fields: actual } if fields.len()==actual.len() && fields.iter().zip(actual).all(|(p,v)| p.name==v.name && match_parameter(&p.ty,&v.ty,arguments,depth+1)))
        }
        Type::Function { params, result } => {
            matches!(value, Type::Function { params: actual, result: output } if params.len()==actual.len() && params.iter().zip(actual).all(|(p,v)|match_parameter(p,v,arguments,depth+1)) && match_parameter(result,output,arguments,depth+1))
        }
        Type::Error { payload } => {
            matches!(value, Type::Error { payload: actual } if match_parameter(payload, actual, arguments, depth+1))
        }
        Type::Member { .. }
        | Type::FunctionParameter { .. }
        | Type::FunctionResult { .. }
        | Type::Operation { .. }
        | Type::Method { .. }
        | Type::Value { .. }
        | Type::Union { .. } => true,
        _ => pattern == value,
    }
}
