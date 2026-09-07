use crate::ir::{Module, Ty};

use super::Error;

pub(super) struct Layout {
    pub size: usize,
    pub align: usize,
    pub offsets: Vec<usize>,
}

pub(super) fn layout(module: &Module, ty: &Ty) -> Result<Layout, Error> {
    let scalar = match ty {
        Ty::Int32 | Ty::UInt32 | Ty::Float32 => Some(4),
        Ty::UInt64 | Ty::Pointer { .. } => Some(8),
        _ => None,
    };
    if let Some(size) = scalar {
        return Ok(Layout {
            size,
            align: size,
            offsets: Vec::new(),
        });
    }
    if let Ty::Span { .. } = ty {
        return Ok(Layout {
            size: 16,
            align: 8,
            offsets: vec![0, 8],
        });
    }
    if let Ty::Array { element, length } = ty {
        if *length == 0 {
            return Err(Error(
                "empty arrays have no shared host/device layout".into(),
            ));
        }
        let element = layout(module, element)?;
        return Ok(Layout {
            size: element.size.checked_mul(*length).ok_or_else(overflow)?,
            align: element.align,
            offsets: vec![],
        });
    }
    if let Ty::Defined { definition } = ty {
        return layout(module, module.types[definition.index()].body().unwrap());
    }
    if let Ty::Record { fields } = ty
        && !fields.is_empty()
    {
        let mut size = 0;
        let mut align = 1;
        let mut offsets = Vec::new();
        for field in fields {
            let field = layout(module, &field.ty)?;
            size = round_up(size, field.align)?;
            offsets.push(size);
            size = size.checked_add(field.size).ok_or_else(overflow)?;
            align = align.max(field.align);
        }
        return Ok(Layout {
            size: round_up(size, align)?,
            align,
            offsets,
        });
    }
    Err(Error(format!(
        "type {ty:?} has no shared host/device layout"
    )))
}

fn round_up(size: usize, align: usize) -> Result<usize, Error> {
    size.checked_add(align - 1)
        .map(|n| n & !(align - 1))
        .ok_or_else(overflow)
}

fn overflow() -> Error {
    Error("shared host/device layout is too large".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{RecordField, TypeDef, TypeId};

    fn record(types: Vec<Ty>) -> Ty {
        Ty::Record {
            fields: types
                .into_iter()
                .enumerate()
                .map(|(i, ty)| RecordField {
                    name: format!("f{i}").into(),
                    ty,
                })
                .collect(),
        }
    }

    #[test]
    fn scalar_records_match_c_and_std430_padding() {
        let module = Module::default();
        let inner = record(vec![Ty::UInt32, Ty::UInt64, Ty::Float32]);
        let outer = record(vec![Ty::UInt32, inner, Ty::Float32]);
        let l = layout(&module, &outer).unwrap();
        assert_eq!((l.size, l.align, l.offsets), (40, 8, vec![0, 8, 32]));
    }

    #[test]
    fn recursive_pointers_do_not_recurse_through_memory_layout() {
        let node = Ty::Defined {
            definition: TypeId::from_index(0),
        };
        let module = Module {
            types: vec![TypeDef::new(
                "Node",
                record(vec![
                    Ty::UInt32,
                    Ty::Pointer {
                        pointee: Box::new(node.clone()),
                    },
                ]),
            )],
            ..Module::default()
        };
        let l = layout(&module, &node).unwrap();
        assert_eq!((l.size, l.align, l.offsets), (16, 8, vec![0, 8]));
    }

    #[test]
    fn incompatible_storage_types_are_rejected() {
        for ty in [
            Ty::Bool,
            Ty::Unit,
            Ty::UInt8,
            Ty::Int64,
            record(vec![]),
            record(vec![Ty::Bool]),
        ] {
            assert!(layout(&Module::default(), &ty).is_err());
        }
    }
}
