//! Decode annotations with injected name lookup and inference state.
use std::collections::HashSet;

use super::{GenerateError, GenerateErrorKind};
use crate::ast::{self, Ident, Span, TypeKind};
use crate::ir::typecheck::infer::{
    solver::{Solver, VariableId},
    types::{Head, Type},
};
use crate::ir::{Ty, TypeError, TypeErrorKind};

type Result<T> = std::result::Result<T, GenerateError>;

pub(super) struct Decoded {
    pub ty: Type,
    pub holes: Vec<(Span, VariableId)>,
}

pub(super) struct Decoder<'a> {
    pub solver: &'a mut Solver,
    pub holes: Vec<(Span, VariableId)>,
    pub resolve: &'a mut dyn FnMut(&Ident) -> Result<Type>,
}

impl Decoder<'_> {
    pub fn decode(mut self, ann: &ast::Type, infer: bool) -> Result<Decoded> {
        let ty = self.ty(ann, infer)?;
        Ok(Decoded {
            ty,
            holes: self.holes,
        })
    }

    fn ty(&mut self, ann: &ast::Type, infer: bool) -> Result<Type> {
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
            TypeKind::App { head, arg } => {
                let arg = self.ty(arg, infer)?;
                let head = match head.val.as_ref() {
                    "Ptr" => Head::Pointer,
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
            TypeKind::Func { from, to } => {
                Type::function(self.ty(from, infer)?, self.ty(to, infer)?)
            }
            TypeKind::Record { fields } => {
                let fields = fields
                    .iter()
                    .map(|(name, ty)| Ok((name.val.clone(), self.ty(ty, infer)?)))
                    .collect::<Result<Vec<_>>>()?;
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
        "bool" => Ty::Bool,
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
