use super::{Ty, TypeDef, TypeId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DefinitionError {
    Invalid(TypeId),
    Incomplete(TypeId),
    Recursive(TypeId),
}

pub(crate) fn get(definitions: &[TypeDef], id: TypeId) -> Result<&TypeDef, DefinitionError> {
    definitions
        .get(id.index())
        .ok_or(DefinitionError::Invalid(id))
}

pub(crate) fn body(definitions: &[TypeDef], id: TypeId) -> Result<&Ty, DefinitionError> {
    get(definitions, id)?
        .body()
        .ok_or(DefinitionError::Incomplete(id))
}

pub(crate) fn check_references(definitions: &[TypeDef], ty: &Ty) -> Result<(), DefinitionError> {
    match ty {
        Ty::Defined { definition } => {
            get(definitions, *definition)?;
        }
        Ty::Pointer { pointee } => check_references(definitions, pointee)?,
        Ty::Span { element } | Ty::Array { element, .. } => check_references(definitions, element)?,
        Ty::Record { fields } => {
            for field in fields {
                check_references(definitions, &field.ty)?;
            }
        }
        Ty::Function { param, result } => {
            check_references(definitions, param)?;
            check_references(definitions, result)?;
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn check_layout(
    definitions: &[TypeDef],
    id: TypeId,
    ty: &Ty,
) -> Result<(), DefinitionError> {
    check_inline(definitions, ty, &mut vec![id])
}

fn check_inline(
    definitions: &[TypeDef],
    ty: &Ty,
    active: &mut Vec<TypeId>,
) -> Result<(), DefinitionError> {
    match ty {
        Ty::Defined { definition } => {
            if active.contains(definition) {
                return Err(DefinitionError::Recursive(*definition));
            }
            let body = body(definitions, *definition)?;
            active.push(*definition);
            check_inline(definitions, body, active)?;
            active.pop();
        }
        Ty::Array { element, .. } => check_inline(definitions, element, active)?,
        Ty::Record { fields } => {
            for field in fields {
                check_inline(definitions, &field.ty, active)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_and_unfinished_definitions_are_distinct() {
        let id = TypeId::from_index(0);
        assert_eq!(body(&[], id), Err(DefinitionError::Invalid(id)));
        let table = [TypeDef {
            name: "Pending".into(),
            body: None,
        }];
        assert!(get(&table, id).is_ok());
        assert_eq!(body(&table, id), Err(DefinitionError::Incomplete(id)));
    }

    #[test]
    fn indirection_breaks_layout_cycles_but_not_reference_checks() {
        let id = TypeId::from_index(0);
        let named = Ty::Defined { definition: id };
        for indirect in [
            Ty::Pointer {
                pointee: Box::new(named.clone()),
            },
            Ty::Span {
                element: Box::new(named.clone()),
            },
            Ty::Function {
                param: Box::new(named.clone()),
                result: Box::new(named.clone()),
            },
        ] {
            assert_eq!(
                check_references(&[], &indirect),
                Err(DefinitionError::Invalid(id))
            );
            let table = [TypeDef::new("Recursive", indirect.clone())];
            check_references(&table, &indirect).unwrap();
            check_layout(&table, id, &indirect).unwrap();
            assert_eq!(
                check_layout(&table, id, &named),
                Err(DefinitionError::Recursive(id))
            );
        }
    }
}
