use crate::ir::types::definitions::{self, DefinitionError};
use crate::ir::{Ty, TypeDef, TypeId};

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
pub enum ExplicitConversion {
    Ascribe(Vec<Conv>),
    Widen,
    NumericCast,
    PointerCast,
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

    /// Identity, a Span representation change, or exactly one nominal wrap or unwrap.
    pub fn ascribe(&self, from: &Ty, to: &Ty) -> Result<Vec<Conv>, TypeError> {
        ascription(self.definitions(), from, to)?.ok_or_else(|| {
            TypeError::new(TypeErrorKind::TypeMismatch {
                expected: to.clone(),
                found: from.clone(),
            })
        })
    }

    /// Select the operation for an explicit source conversion. Unlike IR
    /// ascription, this also permits widening and numeric or pointer casts.
    pub fn explicit_conversion(&self, from: &Ty, to: &Ty) -> Result<ExplicitConversion, TypeError> {
        if from != to {
            if from.widens_to(to) {
                return Ok(ExplicitConversion::Widen);
            }
            if from.is_numeric() && to.is_numeric() {
                return Ok(ExplicitConversion::NumericCast);
            }
            if from.pointer_cast(to) {
                return Ok(ExplicitConversion::PointerCast);
            }
        }
        self.ascribe(from, to).map(ExplicitConversion::Ascribe)
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
        while let Some(pointee) = current.deref_target() {
            steps.push(Conv::Deref);
            current = pointee.clone();
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

/// Shared rule for `Instr::Ascribe`; `None` means the types are incompatible.
/// Borrow the definition table so verification can use the same nominal rules.
pub(crate) fn ascription(
    table: &[TypeDef],
    from: &Ty,
    to: &Ty,
) -> Result<Option<Vec<Conv>>, DefinitionError> {
    let step = if from == to {
        return Ok(Some(Vec::new()));
    } else if to.span_record().as_ref() == Some(from) {
        Conv::MakeSpan
    } else if from.span_record().as_ref() == Some(to) {
        Conv::SpanRecord
    } else if let Ty::Defined { definition } = to
        && from == definitions::body(table, *definition)?
    {
        Conv::Wrap {
            definition: *definition,
        }
    } else if let Ty::Defined { definition } = from
        && to == definitions::body(table, *definition)?
    {
        Conv::Unwrap {
            definition: *definition,
        }
    } else {
        return Ok(None);
    };
    Ok(Some(vec![step]))
}
