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
        Ty::Defined { definition } => {
            if get(definitions, *definition)?.name().is_none() {
                return Err(DefinitionError::NonRecord(*definition));
            }
        }
        Ty::Error { payload } => check_references(definitions, payload)?,
        Ty::Pointer { pointee } | Ty::Reference { referent: pointee } => {
            check_references(definitions, pointee)?
        }
        Ty::Array { element, .. } => check_references(definitions, element)?,
        Ty::Record { fields } => {
            for field in fields {
                check_references(definitions, &field.ty)?;
            }
        }
        Ty::Function { params, result } => {
            for param in params {
                check_references(definitions, param)?;
            }
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
        Ty::Defined { definition } => {
            if active.contains(definition) {
                return Err(DefinitionError::Recursive(*definition));
            }
            let body = body(definitions, *definition)?;
            active.push(*definition);
            check_inline(definitions, body, active)?;
            active.pop();
        }
        Ty::Error { payload } => check_inline(definitions, payload, active)?,
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
            gpu_projection: None,
            gpu_pipeline: None,
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
            Ty::pointer_length(named.clone()),
            Ty::Function {
                params: vec![named.clone()],
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
        Ty::StrongOwner
        | Ty::WeakOwner
        | Ty::GpuView
        | Ty::GpuPipelineContract
        | Ty::GpuArguments => true,
        Ty::Defined { definition } => {
            let d = &definitions[definition.index()];
            d.drop_hook().is_some() || d.body().is_some_and(|t| t.needs_drop(definitions))
        }
        Ty::Record { fields } => fields.iter().any(|f| f.ty.needs_drop(definitions)),
        Ty::Array { element, .. } => element.needs_drop(definitions),
        Ty::Error { payload } => payload.needs_drop(definitions),
        Ty::Union { variants } => variants.iter().any(|member| member.needs_drop(definitions)),
        _ => false,
    }
}

pub(super) fn gpu_element(ty: &Ty, definitions: &[TypeDef]) -> bool {
    let plain = match ty {
        Ty::UInt8 | Ty::Int32 | Ty::UInt32 | Ty::Int64 | Ty::UInt64 | Ty::Float32 => true,
        Ty::Array { element, .. } => gpu_element(element, definitions),
        Ty::Error { payload } => gpu_element(payload, definitions),
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

pub(super) fn payloads(ty: &Ty) -> Option<Vec<(Case, Ty)>> {
    match ty {
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
        (Ty::Error { payload: source }, Ty::Error { payload: target }) => source.widens_to(target),
        _ => ty.members().iter().all(|source| to.members().iter().any(|target|
            source == target || matches!((source, target), (Ty::Error { payload: a }, Ty::Error { payload: b }) if a.widens_to(b))
        )),
    }
}

pub(super) fn view_record(ty: &Ty) -> Option<Ty> {
    matches!(ty, Ty::Str).then(|| Ty::pointer_length(Ty::UInt8))
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
            gpu_projection: None,
            gpu_pipeline: None,
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
            Ty::Error { payload } => {
                self.intern(payload);
            }
            Ty::Pointer { pointee } | Ty::Reference { referent: pointee } => {
                self.intern(pointee);
            }
            Ty::Array { element, .. } => {
                self.intern(element);
            }
            Ty::Record { fields } => {
                for field in fields {
                    self.intern(&field.ty);
                }
            }
            Ty::Function { params, result } => {
                for param in params {
                    self.intern(param);
                }
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
        Ty::Error { payload } => format!("Err<{}>", format_type(payload, definitions)),
        Ty::Type => "type".into(),
        Ty::Unit => "()".into(),
        Ty::None => "None".into(),
        Ty::Bool => "bool".into(),
        Ty::Int8 => "i8".into(),
        Ty::Int16 => "i16".into(),
        Ty::Int32 => "i32".into(),
        Ty::Int64 => "i64".into(),
        Ty::UInt8 => "u8".into(),
        Ty::UInt16 => "u16".into(),
        Ty::UInt32 => "u32".into(),
        Ty::UInt64 => "u64".into(),
        Ty::Float32 => "f32".into(),
        Ty::Float64 => "f64".into(),
        Ty::Str => "str".into(),
        Ty::Foreign { name } => name.to_string(),
        Ty::Defined { definition } => definitions
            .get(definition.index())
            .and_then(|d| d.name())
            .map(ToString::to_string)
            .unwrap_or_else(|| "?".into()),
        Ty::Reference { referent } => format!("Ref<{}>", format_type(referent, definitions)),
        Ty::Pointer { pointee } => format!("Ptr<{}>", format_type(pointee, definitions)),
        Ty::GpuView => "GpuView".into(),
        Ty::GpuPipelineContract => "GpuPipelineContract".into(),
        Ty::GpuArguments => "GpuArguments".into(),
        Ty::StrongOwner => "StrongOwner".into(),
        Ty::WeakOwner => "WeakOwner".into(),
        Ty::Array { element, length } => {
            format!("[{}; {length}]", format_type(element, definitions))
        }
        Ty::Record { fields }
            if !fields.is_empty()
                && fields
                    .iter()
                    .enumerate()
                    .all(|(i, field)| field.name.as_ref() == format!("_{i}")) =>
        {
            let values = fields
                .iter()
                .map(|field| format_type(&field.ty, definitions))
                .collect::<Vec<_>>()
                .join(", ");
            format!("({values}{})", if fields.len() == 1 { "," } else { "" })
        }
        Ty::Record { fields } => format!(
            "{{ {} }}",
            fields
                .iter()
                .map(|f| format!("{}: {}", f.name, format_type(&f.ty, definitions)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Ty::Function { params, result } => format!(
            "({}) -> {}",
            params
                .iter()
                .map(|ty| format_type(ty, definitions))
                .collect::<Vec<_>>()
                .join(", "),
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
        Ty::Int64 | Ty::UInt64 | Ty::Pointer { .. } | Ty::StrongOwner | Ty::WeakOwner => Some(8),
        _ => None,
    };
    if let Some(size) = scalar {
        return Ok(layout::Layout {
            size,
            align: size,
            offsets: Vec::new(),
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
    if let Ty::Error { payload } = ty {
        return storage_layout(definitions, payload);
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

/// Native value representations on all supported (64-bit) targets. Shared GPU
/// storage has a narrower contract and continues to use `storage_layout`.
pub(super) fn value_layout(
    definitions: &[TypeDef],
    ty: &Ty,
    depth: usize,
) -> Result<layout::Layout, layout::Error> {
    if depth > 128 {
        return Err(layout::Error(
            "value layout is recursive or too deep".into(),
        ));
    }
    let scalar = match ty {
        Ty::Unit | Ty::None | Ty::Bool | Ty::Int8 | Ty::UInt8 => Some(1),
        Ty::Int16 | Ty::UInt16 => Some(2),
        Ty::Type | Ty::Int32 | Ty::UInt32 | Ty::Float32 => Some(4),
        Ty::Int64
        | Ty::UInt64
        | Ty::Float64
        | Ty::Pointer { .. }
        | Ty::Reference { .. }
        | Ty::Function { .. }
        | Ty::StrongOwner
        | Ty::WeakOwner
        | Ty::GpuArguments => Some(8),
        _ => None,
    };
    if let Some(size) = scalar {
        return Ok(layout::Layout {
            size,
            align: size,
            offsets: vec![],
        });
    }
    let child = |ty| value_layout(definitions, ty, depth + 1);
    match ty {
        Ty::Defined { definition } => child(
            definitions
                .get(definition.index())
                .and_then(TypeDef::body)
                .ok_or_else(|| {
                    layout::Error("value layout requires a completed type definition".into())
                })?,
        ),
        Ty::Record { fields } => value_record(
            fields
                .iter()
                .map(|field| child(&field.ty))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Ty::Str => value_record(vec![child(&Ty::UInt64)?, child(&Ty::UInt64)?]),
        Ty::GpuView => value_record(vec![
            child(&Ty::UInt64)?,
            child(&Ty::UInt64)?,
            child(&Ty::UInt32)?,
        ]),
        Ty::GpuPipelineContract => value_record(vec![
            child(&Ty::UInt64)?,
            child(&Ty::UInt32)?,
            child(&Ty::UInt32)?,
            child(&Ty::UInt32)?,
        ]),
        Ty::Array { element, length } => {
            let element = child(element)?;
            Ok(layout::Layout {
                size: element
                    .size
                    .checked_mul((*length).max(1))
                    .ok_or_else(value_overflow)?,
                align: element.align,
                offsets: vec![],
            })
        }
        Ty::Union { variants } => {
            value_tagged(variants.iter().map(child).collect::<Result<Vec<_>, _>>()?)
        }
        Ty::Error { payload } => child(payload),
        _ => Err(layout::Error(format!(
            "type {ty:?} has no known value layout"
        ))),
    }
}

fn value_record(fields: Vec<layout::Layout>) -> Result<layout::Layout, layout::Error> {
    let mut size = 0_usize;
    let mut align = 1;
    let mut offsets = Vec::new();
    for field in fields {
        size = round_up(size, field.align)?;
        offsets.push(size);
        size = size.checked_add(field.size).ok_or_else(value_overflow)?;
        align = align.max(field.align);
    }
    Ok(layout::Layout {
        size: round_up(size.max(1), align)?,
        align,
        offsets,
    })
}

fn value_tagged(payloads: Vec<layout::Layout>) -> Result<layout::Layout, layout::Error> {
    let tag = layout::Layout {
        size: 4,
        align: 4,
        offsets: vec![],
    };
    if payloads.is_empty() {
        return Ok(tag);
    }
    let align = payloads.iter().map(|payload| payload.align).max().unwrap();
    let size = payloads.iter().map(|payload| payload.size).max().unwrap();
    value_record(vec![
        tag,
        layout::Layout {
            size: round_up(size, align)?,
            align,
            offsets: vec![],
        },
    ])
}

fn value_overflow() -> layout::Error {
    layout::Error("value layout is too large".into())
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

pub(super) fn default_literal_type(text: &str) -> Ty {
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
    fn hex_digits_and_signs_do_not_change_exponent_rules() {
        for (text, expected) in [
            ("-0xdead", Ty::Int64),
            ("0XAB", Ty::Int64),
            ("0x7f_b", Ty::Int64),
            ("-12", Ty::Int64),
            ("-1e2", Ty::Float64),
            ("1E2", Ty::Float64),
            ("-1.5", Ty::Float64),
        ] {
            assert_eq!(default_type(text), expected, "{text}");
        }
    }
}

//
// Concrete numeric values
//

pub(super) fn parse_number(text: &str, ty: &Ty) -> Result<Value, String> {
    let compact: String = text.chars().filter(|c| *c != '_').collect();
    match ty {
        Ty::Float32 => {
            if is_hex_literal(&compact) {
                return Err("hexadecimal float literals are not supported".into());
            }
            let value: f32 = compact
                .parse()
                .map_err(|err| format!("invalid float literal: {err}"))?;
            if !value.is_finite() {
                return Err("float literal out of range for f32".into());
            }
            Ok(Value::Float32 { value })
        }
        Ty::Float64 => Ok(Value::Float64 {
            value: parse_float(&compact)?,
        }),
        Ty::Int8 => Ok(Value::Int8 {
            value: parse_signed(&compact)?,
        }),
        Ty::Int16 => Ok(Value::Int16 {
            value: parse_signed(&compact)?,
        }),
        Ty::Int32 => Ok(Value::Int32 {
            value: parse_signed(&compact)?,
        }),
        Ty::Int64 => Ok(Value::Int64 {
            value: parse_signed(&compact)?,
        }),
        Ty::UInt8 => Ok(Value::UInt8 {
            value: parse_unsigned(&compact)?,
        }),
        Ty::UInt16 => Ok(Value::UInt16 {
            value: parse_unsigned(&compact)?,
        }),
        Ty::UInt32 => Ok(Value::UInt32 {
            value: parse_unsigned(&compact)?,
        }),
        Ty::UInt64 => Ok(Value::UInt64 {
            value: parse_unsigned(&compact)?,
        }),
        other => Err(format!("cannot use numeric literal as {other:?}")),
    }
}

fn parse_float(text: &str) -> Result<f64, String> {
    if is_hex_literal(text) {
        return Err("hexadecimal float literals are not supported".into());
    }
    let value: f64 = text
        .parse()
        .map_err(|err| format!("invalid float literal: {err}"))?;
    if !value.is_finite() {
        return Err("float literal out of range for f64".into());
    }
    Ok(value)
}

fn parse_signed<T: TryFrom<i128>>(text: &str) -> Result<T, String>
where
    T::Error: std::fmt::Display,
{
    let (negative, magnitude) = text.strip_prefix('-').map_or((false, text), |s| (true, s));
    let value = if let Some(hex) = magnitude
        .strip_prefix("0x")
        .or_else(|| magnitude.strip_prefix("0X"))
    {
        i128::from_str_radix(hex, 16).map_err(|err| format!("invalid hex literal: {err}"))?
    } else {
        magnitude
            .parse::<i128>()
            .map_err(|err| format!("invalid integer literal: {err}"))?
    };
    let value = if negative { -value } else { value };
    T::try_from(value).map_err(|err| format!("integer literal out of range: {err}"))
}

fn parse_unsigned<T: TryFrom<u128>>(text: &str) -> Result<T, String>
where
    T::Error: std::fmt::Display,
{
    let value = if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u128::from_str_radix(hex, 16).map_err(|err| format!("invalid hex literal: {err}"))?
    } else {
        text.parse()
            .map_err(|err| format!("invalid integer literal: {err}"))?
    };
    T::try_from(value).map_err(|err| format!("integer literal out of range: {err}"))
}

fn is_hex_literal(value: &str) -> bool {
    let value = value.strip_prefix('-').unwrap_or(value);
    value.len() >= 2
        && value.as_bytes()[0] == b'0'
        && (value.as_bytes()[1] == b'x' || value.as_bytes()[1] == b'X')
}

pub(super) fn gpu_projection_plan(
    definitions: &[TypeDef],
    source: &Ty,
    target: &Ty,
) -> Result<crate::GpuProjectionPlan, String> {
    let operation = projection_operation(definitions, source, target)?;
    Ok(crate::GpuProjectionPlan {
        source: source.clone(),
        target: target.clone(),
        operation,
    })
}

fn projection_operation(
    definitions: &[TypeDef],
    source: &Ty,
    target: &Ty,
) -> Result<crate::GpuProjectionOperation, String> {
    use crate::{GpuProjectionKind, GpuProjectionOperation};
    let invalid = || {
        format!(
            "cannot project {} into shader {}",
            format_type(source, definitions),
            format_type(target, definitions)
        )
    };
    if let Ty::Defined { definition } = source
        && let Some(projection) = get(definitions, *definition)
            .map_err(|_| invalid())?
            .gpu_projection()
    {
        if projection.target != *target
            || get(definitions, *definition)
                .map_err(|_| invalid())?
                .drop_hook()
                .is_some()
        {
            return Err(invalid());
        }
        return match projection.kind {
            GpuProjectionKind::Pointer => {
                let element =
                    projection_pointer(definitions, source, target).ok_or_else(invalid)?;
                Ok(GpuProjectionOperation::Pointer {
                    element: element.clone(),
                })
            }
            GpuProjectionKind::Sequence => {
                let Ty::Record {
                    fields: source_fields,
                } = projection_shape(definitions, source).ok_or_else(invalid)?
                else {
                    return Err(invalid());
                };
                let Ty::Record {
                    fields: target_fields,
                } = projection_shape(definitions, target).ok_or_else(invalid)?
                else {
                    return Err(invalid());
                };
                if source_fields.len() != 2
                    || target_fields.len() != 2
                    || source_fields[1].ty != Ty::UInt64
                    || target_fields[1].ty != Ty::UInt64
                {
                    return Err(invalid());
                }
                let element =
                    projection_pointer(definitions, &source_fields[0].ty, &target_fields[0].ty)
                        .ok_or_else(invalid)?;
                Ok(GpuProjectionOperation::Sequence {
                    element: element.clone(),
                })
            }
        };
    }
    if source == target && target.gpu_element(definitions) {
        return Ok(GpuProjectionOperation::Copy);
    }
    if let Ty::Defined { definition } = target
        && get(definitions, *definition)
            .map_err(|_| invalid())?
            .drop_hook()
            .is_some()
    {
        return Err(invalid());
    }
    match (
        projection_shape(definitions, source),
        projection_shape(definitions, target),
    ) {
        (Some(Ty::Record { fields: input }), Some(Ty::Record { fields: output }))
            if !output.is_empty() && input.len() == output.len() =>
        {
            let fields = input
                .iter()
                .zip(output)
                .map(|(input, output)| {
                    if input.name != output.name {
                        return Err(invalid());
                    }
                    gpu_projection_plan(definitions, &input.ty, &output.ty)
                })
                .collect::<Result<_, _>>()?;
            Ok(GpuProjectionOperation::Record { fields })
        }
        (
            Some(Ty::Array {
                element: input,
                length: input_length,
            }),
            Some(Ty::Array {
                element: output,
                length,
            }),
        ) if *length > 0 && input_length == length => Ok(GpuProjectionOperation::Array {
            element: Box::new(gpu_projection_plan(definitions, input, output)?),
            length: *length,
        }),
        _ => Err(invalid()),
    }
}

fn projection_shape<'a>(definitions: &'a [TypeDef], ty: &'a Ty) -> Option<&'a Ty> {
    match ty {
        Ty::Defined { definition } => get(definitions, *definition).ok()?.body(),
        _ => Some(ty),
    }
}

fn projection_pointer<'a>(
    definitions: &'a [TypeDef],
    source: &'a Ty,
    target: &'a Ty,
) -> Option<&'a Ty> {
    let Ty::Defined { definition } = source else {
        return None;
    };
    let definition = get(definitions, *definition).ok()?;
    let projection = definition.gpu_projection()?;
    if projection.kind != crate::GpuProjectionKind::Pointer
        || projection.target != *target
        || definition.drop_hook().is_some()
    {
        return None;
    }
    let Ty::Record { fields } = definition.body()? else {
        return None;
    };
    let [field] = fields.as_slice() else {
        return None;
    };
    if field.ty != Ty::GpuView {
        return None;
    }
    let Ty::Pointer { pointee } = target else {
        return None;
    };
    pointee.gpu_element(definitions).then_some(pointee)
}
