use crate::ir::types::definitions;
use crate::ir::{Function, Ty, TypeDef, TypeId, TypeTable};

use super::error::Location;
use super::{VerifyError, VerifyErrorKind};

pub(super) fn check_definitions(table: &TypeTable) -> Result<(), VerifyError> {
    for index in 0..table.len() {
        let id = TypeId::from_index(index);
        let location = Location::type_definition(id);
        if table.id(&table[index].ty(id)) != Some(id)
            || matches!(&table[index], TypeDef::Structural(Ty::Defined { .. }))
        {
            return Err(
                location.error(VerifyErrorKind::InvalidTypeDefinition { definition: index })
            );
        }
        if let TypeDef::Structural(ty) = &table[index] {
            check_type(table, ty, location)?;
        } else {
            let body = definition_body(table, id, location)?;
            check_type(table, body, location)?;
            definitions::check_layout(table, id, body)
                .map_err(|error| location.error(error.into()))?;
        }
    }
    Ok(())
}

pub(super) fn check_type(
    table: &[TypeDef],
    ty: &Ty,
    location: Location,
) -> Result<(), VerifyError> {
    definitions::check_references(table, ty).map_err(|error| location.error(error.into()))
}

pub(super) fn check_value(
    table: &[TypeDef],
    ty: &Ty,
    location: Location,
) -> Result<(), VerifyError> {
    fn visit(
        table: &[TypeDef],
        ty: &Ty,
        location: Location,
        seen: &mut Vec<TypeId>,
    ) -> Result<(), VerifyError> {
        match ty {
            Ty::Union { variants } => {
                for member in variants {
                    visit(table, member, location, seen)?;
                }
            }
            Ty::Result { value, error } => {
                visit(table, value, location, seen)?;
                visit(table, error, location, seen)?;
            }
            Ty::Foreign { .. } => {
                return Err(location.error(VerifyErrorKind::OpaqueValue { ty: ty.clone() }));
            }
            Ty::Defined { definition } if !seen.contains(definition) => {
                seen.push(*definition);
                visit(
                    table,
                    definition_body(table, *definition, location)?,
                    location,
                    seen,
                )?;
            }
            Ty::Arc { pointee: value } | Ty::Weak { pointee: value } => {
                visit(table, value, location, seen)?
            }
            Ty::Array { element, .. } => visit(table, element, location, seen)?,
            Ty::Record { fields } => {
                for field in fields {
                    visit(table, &field.ty, location, seen)?;
                }
            }
            Ty::Function { param, result } => {
                visit(table, param, location, seen)?;
                visit(table, result, location, seen)?;
            }
            _ => {}
        }
        Ok(())
    }
    check_type(table, ty, location)?;
    visit(table, ty, location, &mut Vec::new())
}

pub(super) fn shape(table: &[TypeDef], mut ty: Ty, location: Location) -> Result<Ty, VerifyError> {
    let mut visited = Vec::new();
    while let Ty::Defined { definition } = ty {
        if visited.contains(&definition) {
            return Err(
                location.error(VerifyErrorKind::RecursiveTypeWithoutIndirection { definition })
            );
        }
        visited.push(definition);
        ty = definition_body(table, definition, location)?.clone();
    }
    Ok(ty)
}

pub(super) fn is_integer(
    table: &[TypeDef],
    ty: &Ty,
    location: Location,
) -> Result<bool, VerifyError> {
    Ok(shape(table, ty.clone(), location)?.is_integer())
}

pub(super) fn function_type(function: &Function, location: Location) -> Result<Ty, VerifyError> {
    function
        .ty()
        .ok_or_else(|| location.error(VerifyErrorKind::InvalidLocal { local: 0 }))
}

pub(super) fn expect_types(
    expected: &[Ty],
    found: &[Ty],
    location: Location,
) -> Result<(), VerifyError> {
    if expected.len() != found.len() {
        return Err(location.error(VerifyErrorKind::ArgumentCount {
            expected: expected.len(),
            found: found.len(),
        }));
    }
    for (expected, found) in expected.iter().cloned().zip(found.iter().cloned()) {
        expect_type(expected, found, location)?;
    }
    Ok(())
}

fn definition_body(
    table: &[TypeDef],
    definition: TypeId,
    location: Location,
) -> Result<&Ty, VerifyError> {
    definitions::body(table, definition).map_err(|error| location.error(error.into()))
}

pub(super) fn ascribe(
    table: &[TypeDef],
    expected: &Ty,
    found: Ty,
    location: Location,
) -> Result<(), VerifyError> {
    if crate::ir::typecheck::ascription(table, &found, expected)
        .map_err(|error| location.error(error.into()))?
        .is_some()
    {
        return Ok(());
    }
    Err(location.error(VerifyErrorKind::TypeMismatch {
        expected: expected.clone(),
        found,
    }))
}

pub(super) fn expect_type(expected: Ty, found: Ty, location: Location) -> Result<(), VerifyError> {
    if expected == found {
        Ok(())
    } else {
        Err(location.error(VerifyErrorKind::TypeMismatch { expected, found }))
    }
}
