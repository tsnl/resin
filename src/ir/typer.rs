//! Emitter-independent, bottom-up type inference.
//!
//! Each rule accepts the types already inferred for an AST node's children and
//! returns the type of the enclosing node. This module deliberately does not
//! walk the AST, manage lexical scopes, evaluate terms, or emit instructions.
//! Those concerns can be composed around the same rules by an interpreter or
//! any backend-specific emitter.

use std::{collections::HashSet, fmt, sync::Arc};

use super::{RecordField, Ty, TypeDef, TypeId};

/// Stateless typing rules plus the nominal definitions needed to inspect type
/// representations.
#[derive(Debug, Clone, Copy)]
pub struct Typer<'types> {
    definitions: &'types [TypeDef],
}

/// The result of typing a field access.
///
/// The type belongs to the AST expression; the index is auxiliary information
/// an interpreter or emitter can use without repeating name resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldAccess {
    pub ty: Ty,
    pub index: usize,
}

/// The monomorphic signature selected for a privileged builtin call.
///
/// Type checking only enforces relationships intrinsic to the operator's
/// polymorphic signature. Whether a backend can synthesize this particular
/// specialization is intentionally a backend concern.
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

impl<'types> Typer<'types> {
    pub const fn new(definitions: &'types [TypeDef]) -> Self {
        Self { definitions }
    }

    /// Infer the default type of a source numeric literal.
    pub fn type_num(&self, value: &str) -> Ty {
        if value.contains('.') || value.contains('e') || value.contains('E') {
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
        let condition_shape = self.representation(condition)?;
        if condition_shape != Ty::Bool {
            return Err(TypeError::new(TypeErrorKind::ExpectedBoolean {
                found: condition.clone(),
            }));
        }
        self.expect_same(then, els)?;
        Ok(then.clone())
    }

    /// Infer a homogeneous array from its element types.
    ///
    /// Empty arrays require an element type from syntax or an expected-type
    /// context and should use [`Self::type_array_of`].
    pub fn type_array(&self, elements: &[Ty]) -> Result<Ty, TypeError> {
        let Some(element) = elements.first() else {
            return Err(TypeError::new(TypeErrorKind::EmptyArrayNeedsElementType));
        };
        self.type_array_of(element, elements)
    }

    /// Check an array against an explicitly supplied or contextually inferred
    /// element type. This also types an empty array.
    pub fn type_array_of(&self, element: &Ty, elements: &[Ty]) -> Result<Ty, TypeError> {
        for found in elements {
            self.expect_same(element, found)?;
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
        let callee_shape = self.representation(callee)?;
        let Ty::Function { params, result } = callee_shape else {
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
            self.expect_same(expected, found)?;
        }
        Ok(*result)
    }

    /// Type a C-like assignment expression, whose result is the assigned value.
    ///
    /// Addressability and mutability are properties of how the caller produced
    /// `place`; they remain outside this purely type-level rule.
    pub fn type_assign(&self, place: &Ty, value: &Ty) -> Result<Ty, TypeError> {
        self.expect_same(place, value)?;
        Ok(value.clone())
    }

    pub fn type_deref(&self, pointer: &Ty) -> Result<Ty, TypeError> {
        let pointer_shape = self.representation(pointer)?;
        let Ty::Pointer { pointee } = pointer_shape else {
            return Err(TypeError::new(TypeErrorKind::ExpectedPointer {
                found: pointer.clone(),
            }));
        };
        Ok(*pointee)
    }

    pub fn type_field(&self, base: &Ty, name: &str) -> Result<FieldAccess, TypeError> {
        let base_shape = self.representation(base)?;
        let Ty::Record { fields } = base_shape else {
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
            })
            .ok_or_else(|| {
                TypeError::new(TypeErrorKind::UnknownField {
                    name: Arc::from(name),
                })
            })
    }

    /// Check a type ascription or other expression-level expected type.
    pub fn type_ascription(&self, expected: &Ty, value: &Ty) -> Result<Ty, TypeError> {
        self.expect_same(expected, value)?;
        Ok(expected.clone())
    }

    /// Type one of Resin's privileged polymorphic operators.
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
                self.expect_boolean(&args[0])?;
                Ok(BuiltinCall {
                    params: args.to_vec(),
                    result: Ty::Bool,
                })
            }
            "+" | "-" | "*" | "/" | "%" | "<<" | ">>" | "&" | "|" | "^" if args.len() == 2 => {
                self.expect_same(&args[0], &args[1])?;
                Ok(BuiltinCall {
                    params: args.to_vec(),
                    result: args[0].clone(),
                })
            }
            "==" | "!=" | "<" | "<=" | ">" | ">=" if args.len() == 2 => {
                self.expect_same(&args[0], &args[1])?;
                Ok(BuiltinCall {
                    params: args.to_vec(),
                    result: Ty::Bool,
                })
            }
            "&&" | "||" if args.len() == 2 => {
                self.expect_boolean(&args[0])?;
                self.expect_boolean(&args[1])?;
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

    /// Reveal only enough nominal definitions to determine a type's outer
    /// representation. Nominal identity is never erased for equality checks.
    pub fn representation(&self, ty: &Ty) -> Result<Ty, TypeError> {
        let mut current = ty.clone();
        let mut visited = HashSet::new();
        while let Ty::Defined { definition } = current {
            if !visited.insert(definition) {
                return Err(TypeError::new(
                    TypeErrorKind::RecursiveTypeWithoutIndirection { definition },
                ));
            }
            current = self
                .definitions
                .get(definition.index())
                .ok_or_else(|| TypeError::new(TypeErrorKind::InvalidTypeDefinition { definition }))?
                .body
                .clone();
        }
        Ok(current)
    }

    fn expect_boolean(&self, found: &Ty) -> Result<(), TypeError> {
        if self.representation(found)? == Ty::Bool {
            Ok(())
        } else {
            Err(TypeError::new(TypeErrorKind::ExpectedBoolean {
                found: found.clone(),
            }))
        }
    }

    fn expect_same(&self, expected: &Ty, found: &Ty) -> Result<(), TypeError> {
        if expected == found {
            Ok(())
        } else {
            Err(TypeError::new(TypeErrorKind::TypeMismatch {
                expected: expected.clone(),
                found: found.clone(),
            }))
        }
    }
}

impl TypeError {
    const fn new(kind: TypeErrorKind) -> Self {
        Self { kind }
    }
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
}
