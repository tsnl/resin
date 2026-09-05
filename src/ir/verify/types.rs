use crate::ir::types::definitions;
use crate::ir::{Function, Ty, TypeDef, TypeId};

use super::error::Location;
use super::{VerifyError, VerifyErrorKind};

pub(super) fn check_definitions(table: &[TypeDef]) -> Result<(), VerifyError> {
    for index in 0..table.len() {
        let id = TypeId::from_index(index);
        let location = Location::type_definition(id);
        let body = definition_body(table, id, location)?;
        check_type(table, body, location)?;
        definitions::check_layout(table, id, body).map_err(|error| location.error(error.into()))?;
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
    function.ty().ok_or_else(|| {
        location.error(VerifyErrorKind::InvalidLocal {
            local: function.param.index(),
        })
    })
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
    if expected == &found {
        return Ok(());
    }
    if let Ty::Defined { definition } = expected
        && &found == definition_body(table, *definition, location)?
    {
        return Ok(());
    }
    if let Ty::Defined { definition } = &found
        && expected == definition_body(table, *definition, location)?
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
