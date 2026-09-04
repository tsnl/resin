//! Bottom-up typing rules.
//!
//! This module does not walk the AST, resolve names, or emit IR. Each rule
//! takes child types and returns the parent type, plus any conversions the
//! operation requires.
//!
//! Types are nominal: `T = U` mints a new `T`. Seeing through a name is not a
//! property of the type; it is a property of the operation:
//!
//! - [`Typer::same`] — arguments and assignment: types must already match
//! - [`Typer::ascribe`] — `T (e)`: one wrap or unwrap
//! - [`Typer::as_bool`], [`Typer::as_pointer`], [`Typer::as_function`] — `if`,
//!   `.*`, and calling `h`: unwrap names until that constructor
//! - [`Typer::as_record`] — `.field`: unwrap names and autoderef pointers
//!
//! The `type_*` rules are thin wrappers around those. The generator emits the
//! returned steps. Conversion is always the operation's choice, never a
//! canonical form of the type.

use std::{collections::HashSet, fmt, sync::Arc};

use super::{RecordField, Ty, TypeDef, TypeId};

#[derive(Debug, Clone, Copy)]
pub struct Typer<'types> {
    definitions: &'types [TypeDef],
}

/// One step toward the type an operation needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv {
    Unwrap { definition: TypeId },
    Wrap { definition: TypeId },
    Deref,
}

/// A type reached by applying `steps` to a starting type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Converted {
    pub ty: Ty,
    pub steps: Vec<Conv>,
}

/// Field type, record index, and conversions that turn the base into a record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldAccess {
    pub ty: Ty,
    pub index: usize,
    pub steps: Vec<Conv>,
}

/// Monomorphic signature of a privileged builtin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinCall {
    pub params: Vec<Ty>,
    pub result: Ty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeError {
    pub kind: TypeErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeErrorKind {
    EmptyArrayNeedsElementType,
    TypeMismatch { expected: Ty, found: Ty },
    ExpectedBoolean { found: Ty },
    ExpectedPointer { found: Ty },
    ExpectedRecord { found: Ty },
    ExpectedFunction { found: Ty },
    ArgumentCount { expected: usize, found: usize },
    InvalidBuiltinArgumentCount { name: Arc<str>, found: usize },
    UnknownBuiltin { name: Arc<str> },
    UnknownField { name: Arc<str> },
    DuplicateField { name: Arc<str> },
    InvalidTypeDefinition { definition: TypeId },
    RecursiveTypeWithoutIndirection { definition: TypeId },
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type error: {:?}", self.kind)
    }
}

impl std::error::Error for TypeError {}

impl TypeError {
    const fn new(kind: TypeErrorKind) -> Self {
        Self { kind }
    }
}

impl<'types> Typer<'types> {
    pub const fn new(definitions: &'types [TypeDef]) -> Self {
        Self { definitions }
    }

    /// Defining RHS of a nominal type, or `ty` itself.
    pub fn body(&self, ty: &Ty) -> Result<Ty, TypeError> {
        match ty {
            Ty::Defined { definition } => self
                .definitions
                .get(definition.index())
                .map(|def| def.body.clone())
                .ok_or_else(|| {
                    TypeError::new(TypeErrorKind::InvalidTypeDefinition {
                        definition: *definition,
                    })
                }),
            other => Ok(other.clone()),
        }
    }

    /// No conversion: the types must already be equal.
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
        if let Ty::Defined { definition } = to {
            if from == &self.body(to)? {
                return Ok(vec![Conv::Wrap {
                    definition: *definition,
                }]);
            }
        }
        if let Ty::Defined { definition } = from {
            if to == &self.body(from)? {
                return Ok(vec![Conv::Unwrap {
                    definition: *definition,
                }]);
            }
        }
        Err(TypeError::new(TypeErrorKind::TypeMismatch {
            expected: to.clone(),
            found: from.clone(),
        }))
    }

    pub fn as_bool(&self, ty: &Ty) -> Result<Converted, TypeError> {
        self.peel(
            ty,
            false,
            |ty| matches!(ty, Ty::Bool),
            TypeErrorKind::ExpectedBoolean { found: ty.clone() },
        )
    }

    pub fn as_pointer(&self, ty: &Ty) -> Result<Converted, TypeError> {
        self.peel(
            ty,
            false,
            |ty| matches!(ty, Ty::Pointer { .. }),
            TypeErrorKind::ExpectedPointer { found: ty.clone() },
        )
    }

    pub fn as_function(&self, ty: &Ty) -> Result<Converted, TypeError> {
        self.peel(
            ty,
            false,
            |ty| matches!(ty, Ty::Function { .. }),
            TypeErrorKind::ExpectedFunction { found: ty.clone() },
        )
    }

    pub fn as_record(&self, ty: &Ty) -> Result<Converted, TypeError> {
        self.peel(
            ty,
            true,
            |ty| matches!(ty, Ty::Record { .. }),
            TypeErrorKind::ExpectedRecord { found: ty.clone() },
        )
    }

    fn peel(
        &self,
        start: &Ty,
        deref: bool,
        found: impl Fn(&Ty) -> bool,
        fail: TypeErrorKind,
    ) -> Result<Converted, TypeError> {
        let mut current = start.clone();
        let mut steps = Vec::new();
        let mut visited = HashSet::new();
        loop {
            if found(&current) {
                return Ok(Converted { ty: current, steps });
            }
            if !visited.insert(current.clone()) {
                return Err(TypeError::new(fail.clone()));
            }
            if let Ty::Defined { definition } = current {
                steps.push(Conv::Unwrap { definition });
                current = self.body(&Ty::Defined { definition })?;
                continue;
            }
            if deref {
                if let Ty::Pointer { pointee } = current {
                    steps.push(Conv::Deref);
                    current = *pointee;
                    continue;
                }
            }
            return Err(TypeError::new(fail));
        }
    }

    pub fn type_num(&self, value: &str) -> Ty {
        if value.contains('.') || (!is_hex_literal(value) && value.contains(['e', 'E'])) {
            Ty::Float64
        } else {
            Ty::Int32
        }
    }

    pub fn type_var(&self, binding: &Ty) -> Ty {
        binding.clone()
    }

    pub fn type_lambda(&self, params: &[Ty], body: &Ty) -> Ty {
        Ty::Function {
            params: params.to_vec(),
            result: Box::new(body.clone()),
        }
    }

    pub fn type_if(&self, condition: &Ty, then: &Ty, els: &Ty) -> Result<Ty, TypeError> {
        self.as_bool(condition)?;
        self.same(then, els)?;
        Ok(then.clone())
    }

    pub fn type_array(&self, elements: &[Ty]) -> Result<Ty, TypeError> {
        let Some(element) = elements.first() else {
            return Err(TypeError::new(TypeErrorKind::EmptyArrayNeedsElementType));
        };
        self.type_array_of(element, elements)
    }

    pub fn type_array_of(&self, element: &Ty, elements: &[Ty]) -> Result<Ty, TypeError> {
        for found in elements {
            self.same(element, found)?;
        }
        Ok(Ty::Array {
            element: Box::new(element.clone()),
            length: elements.len(),
        })
    }

    pub fn type_record(&self, fields: &[RecordField]) -> Result<Ty, TypeError> {
        let mut names = HashSet::with_capacity(fields.len());
        for field in fields {
            if !names.insert(field.name.clone()) {
                return Err(TypeError::new(TypeErrorKind::DuplicateField {
                    name: field.name.clone(),
                }));
            }
        }
        Ok(Ty::Record {
            fields: fields.to_vec(),
        })
    }

    pub fn type_block(&self, tail: &Ty) -> Ty {
        tail.clone()
    }

    pub const fn type_type(&self, _value: &Ty) -> Ty {
        Ty::Type
    }

    pub const fn type_unit(&self) -> Ty {
        Ty::Unit
    }

    pub fn type_call(&self, callee: &Ty, args: &[Ty]) -> Result<Ty, TypeError> {
        let converted = self.as_function(callee)?;
        let Ty::Function { params, result } = converted.ty else {
            return Err(TypeError::new(TypeErrorKind::ExpectedFunction {
                found: callee.clone(),
            }));
        };
        if params.len() != args.len() {
            return Err(TypeError::new(TypeErrorKind::ArgumentCount {
                expected: params.len(),
                found: args.len(),
            }));
        }
        for (expected, found) in params.iter().zip(args) {
            self.same(expected, found)?;
        }
        Ok(*result)
    }

    pub fn type_assign(&self, place: &Ty, value: &Ty) -> Result<Ty, TypeError> {
        self.same(place, value)?;
        Ok(value.clone())
    }

    pub fn type_deref(&self, pointer: &Ty) -> Result<Ty, TypeError> {
        let converted = self.as_pointer(pointer)?;
        let Ty::Pointer { pointee } = converted.ty else {
            return Err(TypeError::new(TypeErrorKind::ExpectedPointer {
                found: pointer.clone(),
            }));
        };
        Ok(*pointee)
    }

    pub fn type_field(&self, base: &Ty, name: &str) -> Result<FieldAccess, TypeError> {
        let converted = self.as_record(base)?;
        let Ty::Record { fields } = converted.ty else {
            return Err(TypeError::new(TypeErrorKind::ExpectedRecord {
                found: base.clone(),
            }));
        };
        fields
            .into_iter()
            .enumerate()
            .find(|(_, field)| field.name.as_ref() == name)
            .map(|(index, field)| FieldAccess {
                ty: field.ty,
                index,
                steps: converted.steps,
            })
            .ok_or_else(|| {
                TypeError::new(TypeErrorKind::UnknownField {
                    name: Arc::from(name),
                })
            })
    }

    pub fn type_ascription(&self, expected: &Ty, value: &Ty) -> Result<Ty, TypeError> {
        self.ascribe(value, expected)?;
        Ok(expected.clone())
    }

    pub fn type_builtin_call(&self, name: &str, args: &[Ty]) -> Result<BuiltinCall, TypeError> {
        match name {
            "+" | "-" if args.len() == 1 => Ok(BuiltinCall {
                params: args.to_vec(),
                result: args[0].clone(),
            }),
            "~" if args.len() == 1 => Ok(BuiltinCall {
                params: args.to_vec(),
                result: args[0].clone(),
            }),
            "!" if args.len() == 1 => {
                self.as_bool(&args[0])?;
                Ok(BuiltinCall {
                    params: args.to_vec(),
                    result: Ty::Bool,
                })
            }
            "+" | "-" | "*" | "/" | "%" | "<<" | ">>" | "&" | "|" | "^" if args.len() == 2 => {
                self.same(&args[0], &args[1])?;
                Ok(BuiltinCall {
                    params: args.to_vec(),
                    result: args[0].clone(),
                })
            }
            "==" | "!=" | "<" | "<=" | ">" | ">=" if args.len() == 2 => {
                self.same(&args[0], &args[1])?;
                Ok(BuiltinCall {
                    params: args.to_vec(),
                    result: Ty::Bool,
                })
            }
            "&&" | "||" if args.len() == 2 => {
                self.as_bool(&args[0])?;
                self.as_bool(&args[1])?;
                Ok(BuiltinCall {
                    params: args.to_vec(),
                    result: Ty::Bool,
                })
            }
            "+" | "-" | "~" | "!" | "*" | "/" | "%" | "<<" | ">>" | "&" | "|" | "^" | "=="
            | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||" => {
                Err(TypeError::new(TypeErrorKind::InvalidBuiltinArgumentCount {
                    name: Arc::from(name),
                    found: args.len(),
                }))
            }
            _ => Err(TypeError::new(TypeErrorKind::UnknownBuiltin {
                name: Arc::from(name),
            })),
        }
    }
}

fn is_hex_literal(value: &str) -> bool {
    value.len() >= 2
        && (value.as_bytes()[1] == b'x' || value.as_bytes()[1] == b'X')
        && value.as_bytes()[0] == b'0'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typer() -> Typer<'static> {
        Typer::new(&[])
    }

    #[test]
    fn child_types_compose_into_a_function_call() {
        let typer = typer();
        let lambda = typer.type_lambda(&[Ty::Int32], &Ty::Float64);

        assert_eq!(typer.type_call(&lambda, &[Ty::Int32]).unwrap(), Ty::Float64);
    }

    #[test]
    fn type_terms_inhabit_the_type_universe() {
        assert_eq!(typer().type_type(&Ty::Int32), Ty::Type);
    }

    #[test]
    fn builtin_arithmetic_has_no_type_trait_constraint() {
        let typer = typer();
        let unusual_operand = Ty::Record {
            fields: vec![RecordField {
                name: "value".into(),
                ty: Ty::Int32,
            }],
        };

        let call = typer
            .type_builtin_call("+", &[unusual_operand.clone(), unusual_operand.clone()])
            .unwrap();

        assert_eq!(
            call.params,
            vec![unusual_operand.clone(), unusual_operand.clone()]
        );
        assert_eq!(call.result, unusual_operand);
    }

    #[test]
    fn nominal_records_expose_fields_without_losing_identity() {
        let definitions = [TypeDef {
            name: "Node".into(),
            body: Ty::Record {
                fields: vec![RecordField {
                    name: "value".into(),
                    ty: Ty::Int32,
                }],
            },
        }];
        let typer = Typer::new(&definitions);
        let node = Ty::Defined {
            definition: TypeId::from_index(0),
        };

        assert_eq!(
            typer.type_field(&node, "value").unwrap(),
            FieldAccess {
                ty: Ty::Int32,
                index: 0,
                steps: vec![Conv::Unwrap {
                    definition: TypeId::from_index(0),
                }],
            }
        );
        assert!(matches!(
            typer.type_assign(&node, &definitions[0].body),
            Err(TypeError {
                kind: TypeErrorKind::TypeMismatch { .. }
            })
        ));
    }

    #[test]
    fn empty_arrays_need_an_injected_element_type() {
        let typer = typer();

        assert_eq!(
            typer.type_array(&[]).unwrap_err().kind,
            TypeErrorKind::EmptyArrayNeedsElementType
        );
        assert_eq!(
            typer.type_array_of(&Ty::UInt8, &[]).unwrap(),
            Ty::Array {
                element: Box::new(Ty::UInt8),
                length: 0,
            }
        );
    }

    #[test]
    fn hex_literals_are_integers() {
        assert_eq!(typer().type_num("0x1e"), Ty::Int32);
        assert_eq!(typer().type_num("0X10"), Ty::Int32);
        assert_eq!(typer().type_num("1e3"), Ty::Float64);
    }

    #[test]
    fn ascription_moves_one_nominal_layer() {
        let definitions = [
            TypeDef {
                name: "Meters".into(),
                body: Ty::Int32,
            },
            TypeDef {
                name: "Distance".into(),
                body: Ty::Defined {
                    definition: TypeId::from_index(0),
                },
            },
        ];
        let typer = Typer::new(&definitions);
        let meters = Ty::Defined {
            definition: TypeId::from_index(0),
        };
        let distance = Ty::Defined {
            definition: TypeId::from_index(1),
        };

        assert_eq!(typer.type_ascription(&meters, &Ty::Int32).unwrap(), meters);
        assert_eq!(
            typer.type_ascription(&Ty::Int32, &meters).unwrap(),
            Ty::Int32
        );
        assert_eq!(typer.type_ascription(&distance, &meters).unwrap(), distance);
        assert_eq!(typer.type_ascription(&meters, &distance).unwrap(), meters);
        assert!(matches!(
            typer.type_ascription(&distance, &Ty::Int32),
            Err(TypeError {
                kind: TypeErrorKind::TypeMismatch { .. }
            })
        ));
        assert!(matches!(
            typer.type_ascription(&Ty::Int32, &distance),
            Err(TypeError {
                kind: TypeErrorKind::TypeMismatch { .. }
            })
        ));
    }

    #[test]
    fn pointer_dereference_is_explicit() {
        let typer = typer();
        let pointer = Ty::Pointer {
            pointee: Box::new(Ty::Int64),
        };

        assert_eq!(typer.type_deref(&pointer).unwrap(), Ty::Int64);
        assert!(matches!(
            typer.type_field(&pointer, "value"),
            Err(TypeError {
                kind: TypeErrorKind::ExpectedRecord { .. }
            })
        ));
    }

    #[test]
    fn field_access_autoderefs_pointers() {
        let record = Ty::Record {
            fields: vec![RecordField {
                name: "x".into(),
                ty: Ty::Int32,
            }],
        };
        let pointer = Ty::Pointer {
            pointee: Box::new(record),
        };
        let access = typer().type_field(&pointer, "x").unwrap();
        assert_eq!(access.ty, Ty::Int32);
        assert_eq!(access.steps, vec![Conv::Deref]);
    }

    #[test]
    fn convert_does_not_unwrap_function_arguments() {
        let definitions = [TypeDef {
            name: "Meters".into(),
            body: Ty::Int32,
        }];
        let typer = Typer::new(&definitions);
        let meters = Ty::Defined {
            definition: TypeId::from_index(0),
        };
        let callee = Ty::Function {
            params: vec![Ty::Int32],
            result: Box::new(Ty::Int32),
        };
        assert!(matches!(
            typer.type_call(&callee, &[meters]),
            Err(TypeError {
                kind: TypeErrorKind::TypeMismatch { .. }
            })
        ));
    }
}
