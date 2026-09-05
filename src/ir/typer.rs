//! Bottom-up typing rules over an owned nominal type table.
//!
//! Types stay nominal; each operation chooses whether to unwrap or dereference.

use std::{collections::HashSet, fmt, sync::Arc};

use super::{RecordField, Ty, TypeDef, TypeId};

/// Type IDs are local to this context's definition table.
#[derive(Debug, Clone, Default)]
pub struct TyperContext {
    definitions: Vec<TypeDef>,
}

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

impl TyperContext {
    pub const fn new() -> Self {
        Self {
            definitions: Vec::new(),
        }
    }

    pub fn from_definitions(definitions: Vec<TypeDef>) -> Self {
        Self { definitions }
    }

    pub fn create_type(
        &mut self,
        name: impl Into<Arc<str>>,
        body: Ty,
    ) -> Result<TypeId, TypeError> {
        let definition = self.reserve_type(name);
        if let Err(err) = self.define_type(definition, body) {
            self.definitions.pop();
            return Err(err);
        }
        Ok(definition)
    }

    /// Reserve a recursive identity; its body is unavailable until defined.
    pub fn reserve_type(&mut self, name: impl Into<Arc<str>>) -> TypeId {
        let definition = TypeId::from_index(self.definitions.len());
        self.definitions.push(TypeDef {
            name: name.into(),
            body: None,
        });
        definition
    }

    /// Define once. Invalid bodies leave the reservation unfinished and retryable.
    pub fn define_type(&mut self, definition: TypeId, body: Ty) -> Result<(), TypeError> {
        if self.definition(definition)?.body().is_some() {
            return Err(TypeError::new(TypeErrorKind::TypeAlreadyDefined {
                definition,
            }));
        }
        self.validate_type(&body)?;
        self.check_finite_ty(&body, &mut vec![definition])?;
        self.definitions[definition.index()].body = Some(body);
        Ok(())
    }

    pub fn definition(&self, definition: TypeId) -> Result<&TypeDef, TypeError> {
        self.definitions
            .get(definition.index())
            .ok_or_else(|| TypeError::new(TypeErrorKind::InvalidTypeDefinition { definition }))
    }

    /// Defining RHS of a nominal type, or `ty` itself.
    pub fn body(&self, ty: &Ty) -> Result<Ty, TypeError> {
        match ty {
            Ty::Defined { definition } => Ok(self.definition_body(*definition)?.clone()),
            other => Ok(other.clone()),
        }
    }

    pub fn definitions(&self) -> &[TypeDef] {
        &self.definitions
    }

    pub fn into_definitions(self) -> Result<Vec<TypeDef>, TypeError> {
        for index in 0..self.definitions.len() {
            self.definition_body(TypeId::from_index(index))?;
        }
        Ok(self.definitions)
    }

    pub const fn type_unit(&self) -> Ty {
        Ty::Unit
    }

    pub fn type_num(&self, value: &str) -> Ty {
        if value.contains('.') || (!is_hex_literal(value) && value.contains(['e', 'E'])) {
            Ty::Float64
        } else {
            Ty::Int32
        }
    }

    pub const fn type_type(&self, _value: &Ty) -> Ty {
        Ty::Type
    }

    pub fn type_var(&self, binding: &Ty) -> Ty {
        binding.clone()
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

    pub fn type_lambda(&self, param: &Ty, body: &Ty) -> Ty {
        Ty::Function {
            param: Box::new(param.clone()),
            result: Box::new(body.clone()),
        }
    }

    pub fn type_call(&self, callee: &Ty, arg: &Ty) -> Result<Ty, TypeError> {
        let converted = self.as_function(callee)?;
        let Ty::Function { param, result } = converted.ty else {
            return Err(TypeError::new(TypeErrorKind::ExpectedFunction {
                found: callee.clone(),
            }));
        };
        self.same(&param, arg)?;
        Ok(*result)
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

    pub fn type_block(&self, tail: &Ty) -> Ty {
        tail.clone()
    }

    pub fn type_if(&self, condition: &Ty, then: &Ty, els: &Ty) -> Result<Ty, TypeError> {
        self.as_bool(condition)?;
        self.same(then, els)?;
        Ok(then.clone())
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
            if deref && let Ty::Pointer { pointee } = current {
                steps.push(Conv::Deref);
                current = *pointee;
                continue;
            }
            return Err(TypeError::new(fail));
        }
    }

    fn definition_body(&self, definition: TypeId) -> Result<&Ty, TypeError> {
        self.definition(definition)?
            .body()
            .ok_or_else(|| TypeError::new(TypeErrorKind::IncompleteTypeDefinition { definition }))
    }

    /// Check nominal references, including those behind indirection.
    fn validate_type(&self, ty: &Ty) -> Result<(), TypeError> {
        match ty {
            Ty::Defined { definition } => {
                self.definition(*definition)?;
            }
            Ty::Pointer { pointee } => self.validate_type(pointee)?,
            Ty::Span { element } | Ty::Array { element, .. } => self.validate_type(element)?,
            Ty::Record { fields } => {
                for field in fields {
                    self.validate_type(&field.ty)?;
                }
            }
            Ty::Function { param, result } => {
                self.validate_type(param)?;
                self.validate_type(result)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn check_finite_ty(&self, ty: &Ty, active: &mut Vec<TypeId>) -> Result<(), TypeError> {
        match ty {
            Ty::Defined { definition } => {
                if active.contains(definition) {
                    return Err(TypeError::new(
                        TypeErrorKind::RecursiveTypeWithoutIndirection {
                            definition: *definition,
                        },
                    ));
                }
                let body = self.definition_body(*definition)?;
                active.push(*definition);
                self.check_finite_ty(body, active)?;
                active.pop();
            }
            Ty::Array { element, .. } => self.check_finite_ty(element, active)?,
            Ty::Record { fields } => {
                for field in fields {
                    self.check_finite_ty(&field.ty, active)?;
                }
            }
            // Indirection breaks layout cycles: pointers, spans, and closures have fixed size.
            _ => {}
        }
        Ok(())
    }
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
    InvalidBuiltinArgumentCount { name: Arc<str>, found: usize },
    UnknownBuiltin { name: Arc<str> },
    UnknownField { name: Arc<str> },
    DuplicateField { name: Arc<str> },
    InvalidTypeDefinition { definition: TypeId },
    IncompleteTypeDefinition { definition: TypeId },
    TypeAlreadyDefined { definition: TypeId },
    RecursiveTypeWithoutIndirection { definition: TypeId },
}

impl TypeError {
    const fn new(kind: TypeErrorKind) -> Self {
        Self { kind }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type error: {:?}", self.kind)
    }
}

impl std::error::Error for TypeError {}

fn is_hex_literal(value: &str) -> bool {
    let value = value.strip_prefix('-').unwrap_or(value);
    value.len() >= 2
        && (value.as_bytes()[1] == b'x' || value.as_bytes()[1] == b'X')
        && value.as_bytes()[0] == b'0'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typer() -> TyperContext {
        TyperContext::new()
    }

    #[test]
    fn context_creates_distinct_nominal_identities_without_a_module() {
        let mut context = TyperContext::default();
        assert!(context.definitions().is_empty());
        let first = context.create_type("Meters", Ty::Int32).unwrap();
        let second = context.create_type("Meters", Ty::Int32).unwrap();
        assert_ne!(first, second);
        assert_eq!(first.index(), 0);
        assert_eq!(second.index(), 1);
        assert_eq!(context.definitions().len(), 2);
        assert_eq!(context.definition(first).unwrap().body(), Some(&Ty::Int32));
        assert!(
            context
                .same(
                    &Ty::Defined { definition: first },
                    &Ty::Defined { definition: second },
                )
                .is_err()
        );
    }

    #[test]
    fn unfinished_definitions_are_not_unit_and_cannot_be_exported() {
        let mut context = TyperContext::new();
        let definition = context.reserve_type("Pending");
        let ty = Ty::Defined { definition };
        let incomplete = TypeErrorKind::IncompleteTypeDefinition { definition };
        assert!(context.definition(definition).unwrap().body().is_none());
        assert_eq!(context.body(&ty).unwrap_err().kind, incomplete);
        assert_eq!(
            context.ascribe(&Ty::Unit, &ty).unwrap_err().kind,
            incomplete
        );
        assert_eq!(context.as_bool(&ty).unwrap_err().kind, incomplete);
        assert_eq!(context.as_record(&ty).unwrap_err().kind, incomplete);
        assert_eq!(
            context.clone().into_definitions().unwrap_err().kind,
            incomplete
        );

        context.define_type(definition, Ty::Unit).unwrap();
        assert_eq!(context.body(&ty).unwrap(), Ty::Unit);
        assert_eq!(
            context.into_definitions().unwrap()[0].body(),
            Some(&Ty::Unit)
        );
    }

    #[test]
    fn completed_definitions_cannot_be_redefined_even_after_export_and_import() {
        let mut context = TyperContext::new();
        let definition = context.create_type("Value", Ty::Int32).unwrap();
        for body in [Ty::Int32, Ty::Float64] {
            assert_eq!(
                context.define_type(definition, body).unwrap_err().kind,
                TypeErrorKind::TypeAlreadyDefined { definition }
            );
        }
        let mut context = TyperContext::from_definitions(context.into_definitions().unwrap());
        assert_eq!(
            context.define_type(definition, Ty::Unit).unwrap_err().kind,
            TypeErrorKind::TypeAlreadyDefined { definition }
        );
        assert_eq!(
            context.definition(definition).unwrap().body(),
            Some(&Ty::Int32)
        );
    }

    #[test]
    fn inline_dependencies_must_be_complete_but_pointer_dependencies_can_be_pending() {
        let mut context = TyperContext::new();
        let first = context.reserve_type("First");
        let second = context.reserve_type("Second");
        let second_ty = Ty::Defined { definition: second };
        assert_eq!(
            context
                .define_type(first, second_ty.clone())
                .unwrap_err()
                .kind,
            TypeErrorKind::IncompleteTypeDefinition { definition: second }
        );
        assert!(context.definition(first).unwrap().body().is_none());
        context
            .define_type(
                second,
                Ty::Pointer {
                    pointee: Box::new(Ty::Defined { definition: first }),
                },
            )
            .unwrap();
        context.define_type(first, second_ty).unwrap();
        let module = crate::ir::Module {
            types: context.into_definitions().unwrap(),
            ..Default::default()
        };
        crate::ir::verify(&module).unwrap();
    }

    #[test]
    fn invalid_references_do_not_initialize_a_reserved_body() {
        let mut context = TyperContext::new();
        let definition = context.reserve_type("Value");
        let missing = TypeId::from_index(99);
        assert_eq!(
            context
                .define_type(
                    definition,
                    Ty::Pointer {
                        pointee: Box::new(Ty::Defined {
                            definition: missing
                        }),
                    }
                )
                .unwrap_err()
                .kind,
            TypeErrorKind::InvalidTypeDefinition {
                definition: missing
            }
        );
        assert!(context.definition(definition).unwrap().body().is_none());
        context.define_type(definition, Ty::Int32).unwrap();
    }

    #[test]
    fn recursive_definitions_can_be_reserved_and_completed_in_the_context() {
        let mut context = TyperContext::new();
        let definition = context.reserve_type("Node");
        let node = Ty::Defined { definition };
        let pointer = Ty::Pointer {
            pointee: Box::new(node.clone()),
        };
        let body = Ty::Record {
            fields: vec![
                RecordField {
                    name: "value".into(),
                    ty: Ty::Int32,
                },
                RecordField {
                    name: "next".into(),
                    ty: pointer.clone(),
                },
            ],
        };
        context.define_type(definition, body.clone()).unwrap();
        assert_eq!(context.body(&node).unwrap(), body);
        assert_eq!(context.type_field(&node, "next").unwrap().ty, pointer);

        let span_id = context.reserve_type("Span");
        let span = Ty::Defined {
            definition: span_id,
        };
        context
            .define_type(
                span_id,
                Ty::Span {
                    element: Box::new(span),
                },
            )
            .unwrap();

        let function_id = context.reserve_type("Function");
        let function = Ty::Defined {
            definition: function_id,
        };
        context
            .define_type(
                function_id,
                Ty::Function {
                    param: Box::new(function.clone()),
                    result: Box::new(function),
                },
            )
            .unwrap();
        assert_eq!(context.into_definitions().unwrap().len(), 3);
    }

    #[test]
    fn invalid_recursive_layouts_leave_the_reservation_retryable() {
        let mut context = TyperContext::new();
        let definition = context.reserve_type("Value");
        let value = Ty::Defined { definition };
        for body in [
            value.clone(),
            Ty::Array {
                element: Box::new(value.clone()),
                length: 1,
            },
            Ty::Record {
                fields: vec![RecordField {
                    name: "self".into(),
                    ty: value.clone(),
                }],
            },
        ] {
            assert_eq!(
                context.define_type(definition, body).unwrap_err().kind,
                TypeErrorKind::RecursiveTypeWithoutIndirection { definition }
            );
            assert!(context.definition(definition).unwrap().body().is_none());
        }

        context.define_type(definition, Ty::Int32).unwrap();
        assert_eq!(context.body(&value).unwrap(), Ty::Int32);
    }

    #[test]
    fn bad_references_do_not_leave_a_partially_created_definition() {
        let mut context = TyperContext::new();
        let missing = TypeId::from_index(99);
        let invalid = Ty::Defined {
            definition: missing,
        };
        for body in [
            invalid.clone(),
            Ty::Pointer {
                pointee: Box::new(invalid.clone()),
            },
            Ty::Span {
                element: Box::new(invalid.clone()),
            },
            Ty::Function {
                param: Box::new(Ty::Unit),
                result: Box::new(invalid),
            },
        ] {
            assert_eq!(
                context.create_type("Invalid", body).unwrap_err().kind,
                TypeErrorKind::InvalidTypeDefinition {
                    definition: missing
                }
            );
            assert!(context.definitions().is_empty());
        }
        assert_eq!(
            context.define_type(missing, Ty::Int32).unwrap_err().kind,
            TypeErrorKind::InvalidTypeDefinition {
                definition: missing
            }
        );
        assert_eq!(context.create_type("Valid", Ty::Int32).unwrap().index(), 0);
    }

    #[test]
    fn definition_tables_move_between_checking_passes_without_changing_ids() {
        let mut context = TyperContext::new();
        let definition = context.create_type("Meters", Ty::Int32).unwrap();
        let table = context.into_definitions().unwrap();
        let allocation = table.as_ptr();
        let mut context = TyperContext::from_definitions(table);
        assert_eq!(context.definitions().as_ptr(), allocation);
        let next = context.create_type("Seconds", Ty::Float64).unwrap();
        assert_eq!(next.index(), 1);
        let meters = Ty::Defined { definition };
        assert_eq!(
            context.type_ascription(&meters, &Ty::Int32).unwrap(),
            meters
        );
        assert_eq!(
            context.definition(definition).unwrap().name.as_ref(),
            "Meters"
        );
    }

    #[test]
    fn child_types_compose_into_a_function_call() {
        let typer = typer();
        let lambda = typer.type_lambda(&Ty::Int32, &Ty::Float64);

        assert_eq!(typer.type_call(&lambda, &Ty::Int32).unwrap(), Ty::Float64);
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
        let mut typer = TyperContext::new();
        let body = Ty::Record {
            fields: vec![RecordField {
                name: "value".into(),
                ty: Ty::Int32,
            }],
        };
        let definition = typer.create_type("Node", body.clone()).unwrap();
        let node = Ty::Defined { definition };

        assert_eq!(
            typer.type_field(&node, "value").unwrap(),
            FieldAccess {
                ty: Ty::Int32,
                index: 0,
                steps: vec![Conv::Unwrap { definition }],
            }
        );
        assert!(matches!(
            typer.type_assign(&node, &body),
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
        let mut typer = TyperContext::new();
        let meters = Ty::Defined {
            definition: typer.create_type("Meters", Ty::Int32).unwrap(),
        };
        let distance = Ty::Defined {
            definition: typer.create_type("Distance", meters.clone()).unwrap(),
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
        let mut typer = TyperContext::new();
        let meters = Ty::Defined {
            definition: typer.create_type("Meters", Ty::Int32).unwrap(),
        };
        let callee = Ty::Function {
            param: Box::new(Ty::Int32),
            result: Box::new(Ty::Int32),
        };
        assert!(matches!(
            typer.type_call(&callee, &meters),
            Err(TypeError {
                kind: TypeErrorKind::TypeMismatch { .. }
            })
        ));
    }
}
