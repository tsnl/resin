use crate::ir::{Ty, TypeId};

use super::{TypeError, TypeErrorKind, TyperContext};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv {
    Unwrap { definition: TypeId },
    Wrap { definition: TypeId },
    Deref,
    SpanRecord,
    MakeSpan,
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
        if to.span_record().as_ref() == Some(from) {
            return Ok(vec![Conv::MakeSpan]);
        }
        if from.span_record().as_ref() == Some(to) {
            return Ok(vec![Conv::SpanRecord]);
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

    pub fn as_bool(&self, ty: &Ty) -> Result<(), TypeError> {
        if ty == &Ty::Bool {
            Ok(())
        } else {
            Err(TypeError::new(TypeErrorKind::ExpectedBoolean {
                found: ty.clone(),
            }))
        }
    }

    pub fn as_record(&self, ty: &Ty) -> Result<Converted, TypeError> {
        let mut current = ty.clone();
        let mut steps = Vec::new();
        while let Ty::Pointer { pointee } | Ty::Arc { pointee } = current {
            steps.push(Conv::Deref);
            current = *pointee;
        }
        if let Ty::Defined { definition } = current {
            steps.push(Conv::Unwrap { definition });
            current = self.definition_body(definition)?.clone();
        }
        if matches!(current, Ty::Record { .. } | Ty::Span { .. }) {
            Ok(Converted { ty: current, steps })
        } else {
            Err(TypeError::new(TypeErrorKind::ExpectedRecord {
                found: ty.clone(),
            }))
        }
    }
}
