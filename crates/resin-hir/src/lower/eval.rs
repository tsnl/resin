//! Evaluate source types and literals, retaining inference holes during annotation decoding.
use super::{
    context::Context,
    infer::{Head, Solver, Type, VariableId},
    scope::ContextView,
};
use crate::{GenerateError, GenerateErrorKind};
use resin_ast::TypeKind;
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::collections::HashSet;

pub(super) struct Decoded {
    pub ty: Type,
    pub holes: Vec<(Span, VariableId)>,
}

pub(super) struct Decoder<'a> {
    pub solver: &'a mut Solver,
    pub holes: Vec<(Span, VariableId)>,
    pub resolve: &'a mut dyn FnMut(&Ident) -> Result<Type, GenerateError>,
}

impl Decoder<'_> {
    pub fn decode(mut self, ann: &resin_ast::Type, infer: bool) -> Result<Decoded, GenerateError> {
        let ty = self.ty(ann, infer)?;
        Ok(Decoded {
            ty,
            holes: self.holes,
        })
    }

    fn ty(&mut self, ann: &resin_ast::Type, infer: bool) -> Result<Type, GenerateError> {
        Ok(match &ann.val {
            TypeKind::Unit => Ty::Unit.into(),
            TypeKind::Hole => {
                return Err(GenerateError {
                    span: ann.span,
                    kind: GenerateErrorKind::IncompleteSyntax,
                });
            }
            TypeKind::Infer => {
                if !infer {
                    return Err(GenerateError::inference(
                        ann.span,
                        "type holes are only allowed in local annotations and function results",
                    ));
                }
                let variable = self.solver.fresh_variable();
                self.holes.push((ann.span, variable));
                variable.ty()
            }
            TypeKind::Atom { name } => builtin_ty(&name.val)
                .map(|ty| Ok(ty.into()))
                .unwrap_or_else(|| (self.resolve)(name))?,
            TypeKind::App { head, args } => {
                let [arg] = args.as_slice() else {
                    return Err(GenerateError::inference(
                        ann.span,
                        "type application has the wrong number of arguments",
                    ));
                };
                let arg = self.ty(arg, infer)?;
                let head = match head.val.as_ref() {
                    "Ptr" => Head::Pointer,
                    "GpuPtr" => Head::GpuPointer,
                    "GpuSpan" => Head::GpuSpan,
                    "Arc" => Head::Arc,
                    "Weak" => Head::Weak,
                    "Span" => Head::Span,
                    _ => {
                        return Err(GenerateError {
                            span: head.span,
                            kind: GenerateErrorKind::UnknownTypeFormer {
                                name: head.val.clone(),
                            },
                        });
                    }
                };
                Type::Node(head, vec![arg])
            }
            TypeKind::GpuPipeline { head, root, owner } => {
                let kind = match head.val.as_ref() {
                    "GpuComputePipeline" => Head::GpuComputePipeline,
                    "GpuGraphicsPipeline" => Head::GpuGraphicsPipeline,
                    _ => unreachable!("pipeline type former"),
                };
                Type::Node(kind, vec![self.ty(root, infer)?, self.ty(owner, infer)?])
            }
            TypeKind::Func { from, to } => {
                Type::function(self.ty(from, infer)?, self.ty(to, infer)?)
            }
            TypeKind::Record { fields } => {
                let fields = fields
                    .iter()
                    .map(|(name, ty)| Ok((name.val.clone(), self.ty(ty, infer)?)))
                    .collect::<Result<Vec<_>, GenerateError>>()?;
                let mut names = HashSet::new();
                for (name, _) in &fields {
                    if !names.insert(name) {
                        return Err(GenerateError::typing(
                            ann.span,
                            TypeError {
                                kind: TypeErrorKind::DuplicateField { name: name.clone() },
                            },
                        ));
                    }
                }
                Type::record(fields)
            }
            TypeKind::Result { value, error } => {
                let value = self.ty(value, infer)?;
                let error = self.ty(error, infer)?;
                self.solver.errors(&error, ann.span)?;
                Type::result(value, error)
            }
            TypeKind::Union { left, right } => {
                let left = self.ty(left, false)?;
                let right = self.ty(right, false)?;
                Ty::union_of([
                    self.solver.require(&left, ann.span)?,
                    self.solver.require(&right, ann.span)?,
                ])
                .into()
            }
        })
    }
}

fn builtin_ty(name: &str) -> Option<Ty> {
    Some(match name {
        "Never" => Ty::union([]),
        "None" => Ty::None,
        "GpuArguments" => Ty::GpuArguments,
        "bool" => Ty::Bool,
        "str" => Ty::Str,
        "sbyte" => Ty::Int8,
        "short" => Ty::Int16,
        "int" => Ty::Int32,
        "long" => Ty::Int64,
        "ubyte" => Ty::UInt8,
        "ushort" => Ty::UInt16,
        "uint" => Ty::UInt32,
        "ulong" => Ty::UInt64,
        "float32" => Ty::Float32,
        "float64" => Ty::Float64,
        _ => return None,
    })
}

pub(crate) struct Evaluator<'a> {
    pub(crate) scopes: &'a ContextView,
    pub(crate) typer: &'a Context,
}

pub(super) fn number(
    typer: &TyperContext,
    span: Span,
    text: &str,
    expected: Option<&Ty>,
) -> Result<(Value, Ty), GenerateError> {
    let ty = numeric_type(typer, span, text, expected)?;
    let value = resin_types::literal::parse(text, &ty).map_err(|message| GenerateError {
        span,
        kind: GenerateErrorKind::InvalidLiteral {
            message: message.into(),
        },
    })?;
    Ok((value, ty))
}

fn numeric_type(
    typer: &TyperContext,
    span: Span,
    text: &str,
    expected: Option<&Ty>,
) -> Result<Ty, GenerateError> {
    if let (_, Some(ty)) = resin_types::literal::split(text) {
        return Ok(ty);
    }
    if let Some(expected) = expected {
        let shape = typer
            .body(expected)
            .map_err(|err| GenerateError::typing(span, err))?;
        if shape.is_numeric() {
            return Ok(shape);
        }
    }
    Ok(typer.type_num(text))
}

impl Evaluator<'_> {
    pub(crate) fn ty(&self, ty: &resin_ast::Type) -> Result<Ty, GenerateError> {
        let mut solver = crate::lower::infer::Solver::default();
        let inferred = Decoder {
            solver: &mut solver,
            holes: Vec::new(),
            resolve: &mut |name| {
                if name.val.as_ref() == "String" {
                    return Ok(self
                        .typer
                        .string_type()
                        .cloned()
                        .expect("builtin String")
                        .into());
                }
                self.scopes.resolve_type(name)
            },
        }
        .decode(ty, false)?;
        solver.require(&inferred.ty, ty.span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::scope::Scopes;
    use resin_ast::TypeKind;

    #[test]
    fn evaluation_only_needs_scopes_and_types() {
        let typer = Context::new();
        let mut scopes = Scopes::new();
        scopes
            .define_alias(
                &Ident::new("Byte".into(), Span { start: 0, end: 4 }),
                Ty::Int8,
            )
            .unwrap();
        let evaluator = Evaluator {
            scopes: scopes.view(),
            typer: &typer,
        };
        let span = Span { start: 4, end: 8 };
        let named = resin_ast::Type::new(
            TypeKind::Atom {
                name: Ident::new("Byte".into(), span),
            },
            span,
        );
        let ty = evaluator.ty(&named).unwrap();
        assert_eq!(ty, Ty::Int8);
        assert_eq!(
            number(&typer, span, "-128", Some(&ty)).unwrap(),
            (Value::Int8 { value: -128 }, Ty::Int8),
        );
        let error = number(&typer, span, "128", Some(&ty)).unwrap_err();
        assert_eq!(error.span, span);
        assert!(matches!(
            error.kind,
            GenerateErrorKind::InvalidLiteral { .. }
        ));
    }

    #[test]
    fn numeric_values_keep_the_selected_type() {
        let typer = Context::new();
        let span = Span { start: 0, end: 0 };
        for (text, expected, value, ty) in [
            ("0x1_e", None, Value::Int64 { value: 30 }, Ty::Int64),
            ("1e3", None, Value::Float64 { value: 1000.0 }, Ty::Float64),
            (
                "1.5",
                Some(Ty::Float32),
                Value::Float32 { value: 1.5 },
                Ty::Float32,
            ),
            (
                "255",
                Some(Ty::UInt8),
                Value::UInt8 { value: 255 },
                Ty::UInt8,
            ),
        ] {
            assert_eq!(
                number(&typer, span, text, expected.as_ref()).unwrap(),
                (value, ty),
            );
        }
    }
}
