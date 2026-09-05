use std::collections::HashSet;

use crate::ir::{Ty, TypeId};

use super::{TypeError, TypeErrorKind, TyperContext};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv {
    Unwrap { definition: TypeId },
    Wrap { definition: TypeId },
    Deref,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Converted {
    pub ty: Ty,
    pub steps: Vec<Conv>,
}

impl TyperContext {
    pub fn same(&self, expected: &Ty, found: &Ty) -> Result<(), TypeError> {
        if expected == found {
            Ok(())
        } else {
            Err(TypeError::new(TypeErrorKind::TypeMismatch {
                expected: expected.clone(),
                found: found.clone(),
            }))
        }
    }

    /// `T (e)`: identity, or exactly one wrap or unwrap.
    pub fn ascribe(&self, from: &Ty, to: &Ty) -> Result<Vec<Conv>, TypeError> {
        if from == to {
            return Ok(Vec::new());
        }
        if let Ty::Defined { definition } = to
            && from == &self.body(to)?
        {
            return Ok(vec![Conv::Wrap {
                definition: *definition,
            }]);
        }
        if let Ty::Defined { definition } = from
            && to == &self.body(from)?
        {
            return Ok(vec![Conv::Unwrap {
                definition: *definition,
            }]);
        }
        Err(TypeError::new(TypeErrorKind::TypeMismatch {
            expected: to.clone(),
            found: from.clone(),
        }))
    }

    pub fn as_bool(&self, ty: &Ty) -> Result<Converted, TypeError> {
        self.convert(
            ty,
            false,
            |ty| matches!(ty, Ty::Bool),
            TypeErrorKind::ExpectedBoolean { found: ty.clone() },
        )
    }

    pub fn as_pointer(&self, ty: &Ty) -> Result<Converted, TypeError> {
        self.convert(
            ty,
            false,
            |ty| matches!(ty, Ty::Pointer { .. }),
            TypeErrorKind::ExpectedPointer { found: ty.clone() },
        )
    }

    pub fn as_function(&self, ty: &Ty) -> Result<Converted, TypeError> {
        self.convert(
            ty,
            false,
            |ty| matches!(ty, Ty::Function { .. }),
            TypeErrorKind::ExpectedFunction { found: ty.clone() },
        )
    }

    pub fn as_record(&self, ty: &Ty) -> Result<Converted, TypeError> {
        self.convert(
            ty,
            true,
            |ty| matches!(ty, Ty::Record { .. }),
            TypeErrorKind::ExpectedRecord { found: ty.clone() },
        )
    }

    fn convert(
        &self,
        start: &Ty,
        allow_deref: bool,
        matches: impl Fn(&Ty) -> bool,
        error: TypeErrorKind,
    ) -> Result<Converted, TypeError> {
        let mut current = start.clone();
        let mut steps = Vec::new();
        let mut visited = HashSet::new();
        loop {
            if matches(&current) {
                return Ok(Converted { ty: current, steps });
            }
            if !visited.insert(current.clone()) {
                break;
            }
            current = match current {
                Ty::Defined { definition } => {
                    steps.push(Conv::Unwrap { definition });
                    self.definition_body(definition)?.clone()
                }
                Ty::Pointer { pointee } if allow_deref => {
                    steps.push(Conv::Deref);
                    *pointee
                }
                _ => break,
            };
        }
        Err(TypeError::new(error))
    }
}
