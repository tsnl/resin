//! Type representations, canonical tables, literals, and storage layout.
use crate::layout;
use crate::prelude::*;
use std::sync::Arc;

//
// Definition invariants
//

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DefinitionError {
    InvalidUnion,
    NonRecord(TypeId),
    Invalid(TypeId),
    Incomplete(TypeId),
    Recursive(TypeId),
}

pub(super) fn get(definitions: &[TypeDef], id: TypeId) -> Result<&TypeDef, DefinitionError> {
    definitions
        .get(id.index())
        .ok_or(DefinitionError::Invalid(id))
}

pub(super) fn body(definitions: &[TypeDef], id: TypeId) -> Result<&Ty, DefinitionError> {
    let definition = get(definitions, id)?;
    if definition.name().is_none() {
        return Err(DefinitionError::NonRecord(id));
    }
    let body = definition.body().ok_or(DefinitionError::Incomplete(id))?;
    check_record(id, body)?;
    Ok(body)
}

pub(super) fn check_record(id: TypeId, ty: &Ty) -> Result<(), DefinitionError> {
    if matches!(ty, Ty::Record { .. }) {
        Ok(())
    } else {
        Err(DefinitionError::NonRecord(id))
    }
}

pub(super) fn check_references(definitions: &[TypeDef], ty: &Ty) -> Result<(), DefinitionError> {
    match ty {
        Ty::Union { variants } => {
            if variants.len() == 1 || variants.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(DefinitionError::InvalidUnion);
            }
            for variant in variants {
                if matches!(variant, Ty::Union { .. }) {
                    return Err(DefinitionError::InvalidUnion);
                }
                check_references(definitions, variant)?;
            }
        }
        Ty::Result { value, error } => {
            if error.variants().is_none() {
                return Err(DefinitionError::InvalidUnion);
            }
            check_references(definitions, value)?;
            check_references(definitions, error)?;
        }
        Ty::Defined { definition } => {
            if get(definitions, *definition)?.name().is_none() {
                return Err(DefinitionError::NonRecord(*definition));
            }
        }
        Ty::Pointer { pointee }
        | Ty::GpuPointer { pointee }
        | Ty::Arc { pointee }
        | Ty::Weak { pointee } => check_references(definitions, pointee)?,
        Ty::Span { element } | Ty::GpuSpan { element } | Ty::Array { element, .. } => {
            check_references(definitions, element)?
        }
        Ty::GpuComputePipeline { root, owner } | Ty::GpuGraphicsPipeline { root, owner } => {
            check_references(definitions, root)?;
            check_references(definitions, owner)?;
        }
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

pub(super) fn check_layout(
    definitions: &[TypeDef],
    id: TypeId,
    ty: &Ty,
) -> Result<(), DefinitionError> {
    check_record(id, ty)?;
    check_inline(definitions, ty, &mut vec![id])
}

fn check_inline(
    definitions: &[TypeDef],
    ty: &Ty,
    active: &mut Vec<TypeId>,
) -> Result<(), DefinitionError> {
    match ty {
        Ty::Union { variants } => {
            for variant in variants {
                check_inline(definitions, variant, active)?;
            }
        }
        Ty::Result { value, error } => {
            check_inline(definitions, value, active)?;
            check_inline(definitions, error, active)?;
        }
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
mod definition_tests {
    use super::*;

    #[test]
    fn invalid_and_unfinished_definitions_are_distinct() {
        let id = TypeId::from_index(0);
        assert_eq!(body(&[], id), Err(DefinitionError::Invalid(id)));
        let table = [TypeDef::Nominal {
            name: "Pending".into(),
            body: None,
            drop: None,
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
            let record = |ty| Ty::Record {
                fields: vec![crate::RecordField {
                    name: "value".into(),
                    ty,
                }],
            };
            let indirect = record(indirect);
            let table = [TypeDef::new("Recursive", indirect.clone())];
            check_references(&table, &indirect).unwrap();
            check_layout(&table, id, &indirect).unwrap();
            assert_eq!(
                check_layout(&table, id, &record(named.clone())),
                Err(DefinitionError::Recursive(id))
            );
        }
    }
}

//
// Structural type operations
//

pub(super) fn needs_drop(ty: &Ty, definitions: &[TypeDef]) -> bool {
    match ty {
        Ty::Arc { .. }
        | Ty::Weak { .. }
        | Ty::GpuPointer { .. }
        | Ty::GpuSpan { .. }
        | Ty::GpuArguments
        | Ty::GpuComputePipeline { .. }
        | Ty::GpuGraphicsPipeline { .. } => true,
        Ty::Defined { definition } => {
            let d = &definitions[definition.index()];
            d.drop_hook().is_some() || d.body().is_some_and(|t| t.needs_drop(definitions))
        }
        Ty::Record { fields } => fields.iter().any(|f| f.ty.needs_drop(definitions)),
        Ty::Array { element, .. } => element.needs_drop(definitions),
        Ty::Result { value, error } => {
            value.needs_drop(definitions) || error.needs_drop(definitions)
        }
        Ty::Union { variants } => variants.iter().any(|member| member.needs_drop(definitions)),
        _ => false,
    }
}

pub(super) fn gpu_element(ty: &Ty, definitions: &[TypeDef]) -> bool {
    let plain = match ty {
        Ty::UInt8 | Ty::Int32 | Ty::UInt32 | Ty::Int64 | Ty::UInt64 | Ty::Float32 => true,
        Ty::Array { element, .. } => gpu_element(element, definitions),
        Ty::Record { fields } => fields
            .iter()
            .all(|field| gpu_element(&field.ty, definitions)),
        Ty::Defined { definition } => definitions.get(definition.index()).is_some_and(|def| {
            def.drop_hook().is_none()
                && def
                    .body()
                    .is_some_and(|body| gpu_element(body, definitions))
        }),
        _ => false,
    };
    plain && storage_layout(definitions, ty).is_ok()
}

pub(super) fn gpu_projection(ty: &Ty, definitions: &[TypeDef]) -> Option<Ty> {
    Some(match ty {
        Ty::Pointer { pointee } if pointee.gpu_element(definitions) => Ty::GpuPointer {
            pointee: pointee.clone(),
        },
        Ty::Span { element } if element.gpu_element(definitions) => Ty::GpuSpan {
            element: element.clone(),
        },
        Ty::Array { element, length } if *length > 0 => Ty::Array {
            element: Box::new(gpu_projection(element, definitions)?),
            length: *length,
        },
        Ty::Record { fields } if !fields.is_empty() => Ty::Record {
            fields: fields
                .iter()
                .map(|field| {
                    Some(RecordField {
                        name: field.name.clone(),
                        ty: gpu_projection(&field.ty, definitions)?,
                    })
                })
                .collect::<Option<_>>()?,
        },
        Ty::Defined { definition } => {
            let definition = definitions.get(definition.index())?;
            if definition.drop_hook().is_some() {
                return None;
            }
            gpu_projection(definition.body()?, definitions)?
        }
        ty if ty.gpu_element(definitions) => ty.clone(),
        _ => return None,
    })
}

pub(super) fn payloads(ty: &Ty) -> Option<Vec<(Case, Ty)>> {
    match ty {
        Ty::Result { value, error } => Some(vec![
            (Case::Ok, *value.clone()),
            (Case::Err, *error.clone()),
        ]),
        Ty::Union { variants } => Some(
            variants
                .iter()
                .map(|ty| (Case::Type(ty.clone()), ty.clone()))
                .collect(),
        ),
        _ => None,
    }
}

pub(super) fn variants(ty: &Ty) -> Option<Vec<TypeId>> {
    ty.members()
        .into_iter()
        .map(|ty| match ty {
            Ty::Defined { definition } => Some(definition),
            _ => None,
        })
        .collect()
}

pub(super) fn union_of(variants: impl IntoIterator<Item = Ty>) -> Ty {
    let mut variants: Vec<_> = variants.into_iter().flat_map(|ty| ty.members()).collect();
    variants.sort();
    variants.dedup();
    match variants.as_slice() {
        [ty] => ty.clone(),
        _ => Ty::Union { variants },
    }
}

pub(super) fn widens_to(ty: &Ty, to: &Ty) -> bool {
    if ty == to {
        return true;
    }
    match (ty, to) {
        (
            Ty::Result {
                value: av,
                error: ae,
            },
            Ty::Result {
                value: bv,
                error: be,
            },
        ) => av == bv && ae.widens_to(be),
        _ => ty.members().iter().all(|ty| to.members().contains(ty)),
    }
}

pub(super) fn view_record(ty: &Ty) -> Option<Ty> {
    let pointer = match ty {
        Ty::Span { element } => Ty::Pointer {
            pointee: element.clone(),
        },
        Ty::GpuSpan { element } => Ty::GpuPointer {
            pointee: element.clone(),
        },
        Ty::Str => Ty::Pointer {
            pointee: Box::new(Ty::UInt8),
        },
        _ => return None,
    };
    Some(Ty::Record {
        fields: vec![
            RecordField {
                name: "data".into(),
                ty: pointer,
            },
            RecordField {
                name: "length".into(),
                ty: Ty::UInt64,
            },
        ],
    })
}

pub(super) fn parameter(types: &[Ty]) -> Ty {
    match types {
        [] => Ty::Unit,
        [ty] => ty.clone(),
        _ => Ty::Record {
            fields: types
                .iter()
                .enumerate()
                .map(|(i, ty)| RecordField {
                    name: format!("_{i}").into(),
                    ty: ty.clone(),
                })
                .collect(),
        },
    }
}

//
// Canonical type tables
//

pub(super) fn intern(table: &mut TypeTable, ty: &Ty) -> TypeId {
    if let Some(id) = table.id(ty) {
        if !matches!(ty, Ty::Defined { .. }) {
            table.components(ty);
        }
        return id;
    }
    assert!(!matches!(ty, Ty::Defined { .. }), "unreserved nominal type");
    let id = TypeId::from_index(table.len());
    table.definitions.push(TypeDef::Structural(ty.clone()));
    table.ids.insert(ty.clone(), id);
    table.components(ty);
    id
}

impl TypeTable {
    pub(super) fn reserve(&mut self, name: Arc<str>) -> TypeId {
        let id = TypeId::from_index(self.len());
        self.definitions.push(TypeDef::Nominal {
            name,
            body: None,
            drop: None,
        });
        self.ids.insert(Ty::Defined { definition: id }, id);
        id
    }

    pub(super) fn define(&mut self, id: TypeId, body: Ty) {
        let TypeDef::Nominal { body: slot, .. } = &mut self.definitions[id.index()] else {
            unreachable!("only nominal definitions have pending bodies")
        };
        *slot = Some(body);
    }

    pub(super) fn set_drop(&mut self, id: TypeId, function: FunctionId) {
        let TypeDef::Nominal { drop, .. } = &mut self.definitions[id.index()] else {
            unreachable!("only nominal definitions have destruction hooks")
        };
        *drop = Some(function);
    }

    pub(super) fn pop_nominal(&mut self) {
        let id = TypeId::from_index(self.len() - 1);
        assert!(matches!(
            self.definitions.pop(),
            Some(TypeDef::Nominal { .. })
        ));
        self.ids.remove(&Ty::Defined { definition: id });
    }
}

impl TypeTable {
    fn components(&mut self, ty: &Ty) {
        match ty {
            Ty::Str => {
                self.intern(&Ty::Pointer {
                    pointee: Box::new(Ty::UInt8),
                });
                self.intern(&Ty::UInt64);
            }
            Ty::Union { variants } => {
                for member in variants {
                    self.intern(member);
                }
                self.intern(&Ty::UInt32);
            }
            Ty::Result { value, error } => {
                self.intern(value);
                self.intern(error);
                self.intern(&Ty::UInt32);
            }
            Ty::Pointer { pointee }
            | Ty::GpuPointer { pointee }
            | Ty::Arc { pointee }
            | Ty::Weak { pointee } => {
                self.intern(pointee);
            }
            Ty::GpuSpan { element } => {
                self.intern(&Ty::GpuPointer {
                    pointee: element.clone(),
                });
                self.intern(&Ty::UInt64);
            }
            Ty::GpuComputePipeline { root, owner } | Ty::GpuGraphicsPipeline { root, owner } => {
                self.intern(root);
                self.intern(owner);
            }
            Ty::Span { element } | Ty::Array { element, .. } => {
                self.intern(element);
            }
            Ty::Record { fields } => {
                for field in fields {
                    self.intern(&field.ty);
                }
            }
            Ty::Function { param, result } => {
                self.intern(param);
                self.intern(result);
            }
            _ => {}
        }
    }
}

//
// Type rendering
//

pub(super) fn format_type(ty: &Ty, definitions: &[TypeDef]) -> String {
    match ty {
        Ty::Union { variants } => {
            if variants.is_empty() {
                return "Never".into();
            }
            variants
                .iter()
                .map(|member| format_type(member, definitions))
                .collect::<Vec<_>>()
                .join(" | ")
        }
        Ty::Result { value, error } => format!(
            "Result<{}, {}>",
            format_type(value, definitions),
            format_type(error, definitions)
        ),
        Ty::Type => "type".into(),
        Ty::Unit => "()".into(),
        Ty::None => "None".into(),
        Ty::Bool => "bool".into(),
        Ty::Int8 => "sbyte".into(),
        Ty::Int16 => "short".into(),
        Ty::Int32 => "int".into(),
        Ty::Int64 => "long".into(),
        Ty::UInt8 => "ubyte".into(),
        Ty::UInt16 => "ushort".into(),
        Ty::UInt32 => "uint".into(),
        Ty::UInt64 => "ulong".into(),
        Ty::Float32 => "float32".into(),
        Ty::Float64 => "float64".into(),
        Ty::Str => "str".into(),
        Ty::Foreign { name } => name.to_string(),
        Ty::Defined { definition } => definitions
            .get(definition.index())
            .and_then(|d| d.name())
            .map(ToString::to_string)
            .unwrap_or_else(|| "?".into()),
        Ty::Pointer { pointee } => format!("Ptr<{}>", format_type(pointee, definitions)),
        Ty::GpuPointer { pointee } => format!("GpuPtr<{}>", format_type(pointee, definitions)),
        Ty::GpuSpan { element } => format!("GpuSpan<{}>", format_type(element, definitions)),
        Ty::GpuArguments => "GpuArguments".into(),
        Ty::GpuComputePipeline { root, owner } => format!(
            "GpuComputePipeline<{}, {}>",
            format_type(root, definitions),
            format_type(owner, definitions)
        ),
        Ty::GpuGraphicsPipeline { root, owner } => format!(
            "GpuGraphicsPipeline<{}, {}>",
            format_type(root, definitions),
            format_type(owner, definitions)
        ),
        Ty::Arc { pointee } => format!("Arc<{}>", format_type(pointee, definitions)),
        Ty::Weak { pointee } => format!("Weak<{}>", format_type(pointee, definitions)),
        Ty::Span { element } => format!("Span<{}>", format_type(element, definitions)),
        Ty::Array { element, length } => {
            format!("[{}; {length}]", format_type(element, definitions))
        }
        Ty::Record { fields } => format!(
            "{{ {} }}",
            fields
                .iter()
                .map(|f| format!("{}: {}", f.name, format_type(&f.ty, definitions)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Ty::Function { param, result } => format!(
            "({}) -> {}",
            format_type(param, definitions),
            format_type(result, definitions)
        ),
    }
}

//
// Host/device storage layout
//

pub(super) fn storage_layout(
    definitions: &[TypeDef],
    ty: &Ty,
) -> Result<layout::Layout, layout::Error> {
    let scalar = match ty {
        Ty::UInt8 => Some(1),
        Ty::Int32 | Ty::UInt32 | Ty::Float32 => Some(4),
        Ty::Int64 | Ty::UInt64 | Ty::Pointer { .. } | Ty::Arc { .. } | Ty::Weak { .. } => Some(8),
        _ => None,
    };
    if let Some(size) = scalar {
        return Ok(layout::Layout {
            size,
            align: size,
            offsets: Vec::new(),
        });
    }
    if let Ty::Span { .. } = ty {
        return Ok(layout::Layout {
            size: 16,
            align: 8,
            offsets: vec![0, 8],
        });
    }
    if let Ty::Array { element, length } = ty {
        if *length == 0 {
            return Err(layout::Error(
                "empty arrays have no shared host/device layout".into(),
            ));
        }
        let element = storage_layout(definitions, element)?;
        return Ok(layout::Layout {
            size: element.size.checked_mul(*length).ok_or_else(overflow)?,
            align: element.align,
            offsets: vec![],
        });
    }
    if let Ty::Defined { definition } = ty {
        return storage_layout(definitions, definitions[definition.index()].body().unwrap());
    }
    if let Ty::Record { fields } = ty
        && !fields.is_empty()
    {
        let mut size = 0;
        let mut align = 1;
        let mut offsets = Vec::new();
        for field in fields {
            let field = storage_layout(definitions, &field.ty)?;
            size = round_up(size, field.align)?;
            offsets.push(size);
            size = size.checked_add(field.size).ok_or_else(overflow)?;
            align = align.max(field.align);
        }
        return Ok(layout::Layout {
            size: round_up(size, align)?,
            align,
            offsets,
        });
    }
    Err(layout::Error(format!(
        "type {ty:?} has no shared host/device layout"
    )))
}

fn round_up(size: usize, align: usize) -> Result<usize, layout::Error> {
    size.checked_add(align - 1)
        .map(|n| n & !(align - 1))
        .ok_or_else(overflow)
}

fn overflow() -> layout::Error {
    layout::Error("shared host/device layout is too large".into())
}

#[cfg(test)]
mod layout_tests {
    use crate::layout::*;
    use crate::prelude::*;
    use crate::{RecordField, TypeDef, TypeId};

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
        let table = Vec::new();
        let inner = record(vec![Ty::UInt32, Ty::UInt64, Ty::Float32]);
        let outer = record(vec![Ty::UInt32, inner, Ty::Float32]);
        let l = layout(&table, &outer).unwrap();
        assert_eq!((l.size, l.align, l.offsets), (40, 8, vec![0, 8, 32]));
    }

    #[test]
    fn recursive_pointers_do_not_recurse_through_memory_layout() {
        let node = Ty::Defined {
            definition: TypeId::from_index(0),
        };
        let table = vec![TypeDef::new(
            "Node",
            record(vec![
                Ty::UInt32,
                Ty::Pointer {
                    pointee: Box::new(node.clone()),
                },
            ]),
        )];
        let l = layout(&table, &node).unwrap();
        assert_eq!((l.size, l.align, l.offsets), (16, 8, vec![0, 8]));
    }

    #[test]
    fn incompatible_storage_types_are_rejected() {
        for ty in [
            Ty::Str,
            Ty::Bool,
            Ty::Unit,
            Ty::Float64,
            record(vec![]),
            record(vec![Ty::Bool]),
        ] {
            assert!(layout(&[], &ty).is_err());
        }
    }

    #[test]
    fn byte_arrays_pack_with_exact_nested_strides() {
        let bytes = Ty::Array {
            element: Box::new(Ty::UInt8),
            length: 3,
        };
        let rows = Ty::Array {
            element: Box::new(bytes.clone()),
            length: 2,
        };
        let l = layout(&[], &bytes).unwrap();
        assert_eq!((l.size, l.align), (3, 1));
        let l = layout(&[], &rows).unwrap();
        assert_eq!((l.size, l.align), (6, 1));
        let l = layout(&[], &record(vec![Ty::UInt8, rows, Ty::UInt32])).unwrap();
        assert_eq!((l.size, l.align, l.offsets), (12, 4, vec![0, 1, 8]));
    }

    #[test]
    fn packed_byte_arrays_reject_empty_or_overflowing_shared_storage() {
        let empty = Ty::Array {
            element: Box::new(Ty::UInt8),
            length: 0,
        };
        let huge = Ty::Array {
            element: Box::new(Ty::UInt8),
            length: usize::MAX,
        };
        let overflow = Ty::Array {
            element: Box::new(huge),
            length: 2,
        };
        assert!(layout(&[], &empty).is_err());
        assert!(layout(&[], &overflow).is_err());
    }
}

//
// Numeric literal spelling
//

pub(super) fn split_literal(text: &str) -> (&str, Option<Ty>) {
    let hex = is_hex(text);
    let Some(width) = text.as_bytes().last().map(u8::to_ascii_lowercase) else {
        return (text, None);
    };
    let mut end = text.len() - 1;
    let unsigned = end > 0 && text.as_bytes()[end - 1].eq_ignore_ascii_case(&b'u');
    if unsigned {
        end -= 1;
    }
    // Hex b/B is a digit unless an underscore or unsigned qualifier separates it.
    // Hex d/D/f/F always remain digits; hexadecimal floats are unsupported.
    let separated = end > 0 && text.as_bytes()[end - 1] == b'_';
    let ty = match (width, unsigned) {
        (b'b', false) if !hex || separated => Ty::Int8,
        (b'b', true) => Ty::UInt8,
        (b'h', false) => Ty::Int16,
        (b'h', true) => Ty::UInt16,
        (b'i', false) => Ty::Int32,
        (b'i', true) => Ty::UInt32,
        (b'l', false) => Ty::Int64,
        (b'l', true) => Ty::UInt64,
        (b'f', false) if !hex => Ty::Float32,
        (b'd', false) if !hex => Ty::Float64,
        _ => return (text, None),
    };
    (text[..end].trim_end_matches('_'), Some(ty))
}

pub(super) fn format_literal(text: &str) -> String {
    let (digits, ty) = split_literal(text);
    let suffix = match ty {
        Some(Ty::Int8) => "b",
        Some(Ty::UInt8) => "ub",
        Some(Ty::Int16) => "h",
        Some(Ty::UInt16) => "uh",
        Some(Ty::Int32) => "i",
        Some(Ty::UInt32) => "ui",
        Some(Ty::Int64) => "l",
        Some(Ty::UInt64) => "ul",
        Some(Ty::Float32) => "f",
        Some(Ty::Float64) => "d",
        _ => return text.into(),
    };
    format!("{digits}_{suffix}")
}

pub(super) fn unsuffixed_literal_type(text: &str) -> Ty {
    if text.contains('.') || (!is_hex(text) && text.contains(['e', 'E'])) {
        Ty::Float64
    } else {
        Ty::Int64
    }
}

fn is_hex(text: &str) -> bool {
    let magnitude = text.strip_prefix('-').unwrap_or(text);
    magnitude.starts_with("0x") || magnitude.starts_with("0X")
}

#[cfg(test)]
mod literal_tests {
    use crate::literal::*;
    use crate::prelude::*;

    #[test]
    fn hex_digits_and_signs_do_not_change_suffix_or_exponent_rules() {
        for (text, body, suffix, default) in [
            ("-0xdead", "-0xdead", None, Ty::Int64),
            ("0XAB", "0XAB", None, Ty::Int64),
            ("-0xFF_ul", "-0xFF", Some(Ty::UInt64), Ty::Int64),
            ("-12_b", "-12", Some(Ty::Int8), Ty::Int64),
            ("12_ub", "12", Some(Ty::UInt8), Ty::Int64),
            ("-1e2", "-1e2", None, Ty::Float64),
            ("1E2_f", "1E2", Some(Ty::Float32), Ty::Float64),
            ("-1.5_d", "-1.5", Some(Ty::Float64), Ty::Float64),
        ] {
            assert_eq!(split(text), (body, suffix), "{text}");
            assert_eq!(unsuffixed_type(body), default, "{text}");
        }
    }
}
