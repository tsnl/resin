//! Schemes survive in HIR; requests and concrete function identities belong to LIR.
use resin_hir::{
    Annotation, Constant, Function, Module, Parameter, Signature, Statement, Term, TermKind, Type,
    TypeParameter, TypeParameterId,
};
use resin_lir::{ErrorKind, Instr, LoweringOptions};
use resin_source::prelude::*;
use resin_types::{FunctionId, Ty, TypeId};
use std::num::NonZeroUsize;

const SPAN: Span = Span { start: 0, end: 1 };
const T: TypeParameterId = TypeParameterId::from_index(0);

fn annotation(ty: Type) -> Annotation {
    Annotation { ty, span: SPAN }
}
fn term(ty: Type, kind: TermKind) -> Term {
    Term {
        span: SPAN,
        ty,
        kind,
    }
}
fn unit() -> Term {
    term(
        Type::Unit,
        TermKind::Constant {
            value: Constant::Unit,
        },
    )
}
fn function(name: &str, body: Term) -> Function {
    Function {
        location: Some(SourceLocation {
            source: Source::new(format!("test://{name}"), name),
            span: SPAN,
        }),
        name: name.into(),
        signature: Signature {
            type_params: vec![],
            params: vec![],
            result: annotation(body.ty.clone()),
        },
        foreign_header: None,
        body: Some(body),
    }
}
fn template(name: &str, body: Term) -> Function {
    let mut function = function(name, body);
    function.signature.type_params.push(TypeParameter {
        id: T,
        name: Ident {
            val: "T".into(),
            span: SPAN,
        },
    });
    function
}
fn reference(function: usize, argument: Type) -> Term {
    term(
        Type::Function {
            param: Box::new(Type::Unit),
            result: Box::new(Type::Unit),
        },
        TermKind::Function {
            function: FunctionId::from_index(function),
            type_args: vec![argument],
        },
    )
}
fn block(terms: impl IntoIterator<Item = Term>) -> Term {
    term(
        Type::Unit,
        TermKind::Block {
            stmts: terms
                .into_iter()
                .map(|term| Statement::Expr { term })
                .collect(),
            tail: Box::new(unit()),
        },
    )
}
fn program(arguments: Vec<Type>) -> Module {
    Module {
        functions: vec![
            template("mark", unit()),
            function(
                "main",
                block(arguments.into_iter().map(|ty| reference(0, ty))),
            ),
        ],
        entries: [("main".into(), FunctionId::from_index(1))].into(),
        ..Default::default()
    }
}
fn options(limit: usize) -> LoweringOptions {
    LoweringOptions {
        max_monomorphs_per_function: NonZeroUsize::new(limit).unwrap(),
    }
}

#[test]
fn instances_are_memoized_and_the_exact_allowance_is_admitted() {
    let hir = program(vec![Type::Int32, Type::Bool, Type::Int32, Type::Bool]);
    let lir = resin_lir::analyze_with_options(&hir, &options(2)).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(hir.functions.len(), 2);
    assert_eq!(lir.functions.len(), 3);
    assert_eq!(lir.entries["main"].index(), 0);
    let calls: Vec<_> = lir.functions[0]
        .blocks
        .iter()
        .flat_map(|b| &b.instrs)
        .filter_map(|i| match i {
            Instr::Function { function } => Some(function.index()),
            _ => None,
        })
        .collect();
    assert_eq!(calls, [1, 2, 1, 2]);
    let error = resin_lir::analyze_with_options(&hir, &options(1))
        .unwrap_err()
        .remove(0);
    assert_eq!(
        error.kind,
        ErrorKind::MonomorphLimit {
            function: "mark".into(),
            limit: 1,
            arguments: vec![Ty::Bool]
        }
    );
    assert_eq!(error.source.unwrap().name(), "test://main");
    assert_eq!(error.applications[0].function.as_ref(), "main");
}

#[test]
fn substitution_normalizes_unions_before_memoization() {
    let hir = program(vec![
        Type::Union {
            variants: vec![Type::Int32, Type::Int32],
        },
        Type::Int32,
    ]);
    let lir = resin_lir::analyze_with_options(&hir, &options(1)).unwrap();
    assert_eq!(lir.functions.len(), 2);
}

#[test]
fn growing_recursive_requests_fail_with_a_bounded_application_trace() {
    let mut hir = program(vec![Type::Int32]);
    hir.functions[0].body = Some(block([reference(
        0,
        Type::Pointer {
            pointee: Box::new(Type::Parameter { parameter: T }),
        },
    )]));
    let errors = resin_lir::analyze_with_options(&hir, &options(40)).unwrap_err();
    assert_eq!(errors.len(), 1);
    let error = &errors[0];
    assert!(matches!(
        error.kind,
        ErrorKind::MonomorphLimit { limit: 40, .. }
    ));
    assert_eq!(error.source.as_ref().unwrap().name(), "test://mark");
    assert_eq!(error.applications.len(), 32);
    assert!(
        error
            .applications
            .iter()
            .all(|note| note.function.as_ref() == "mark")
    );
}

#[test]
fn recursive_requests_reuse_pending_identities() {
    let mut hir = program(vec![Type::Int32]);
    hir.functions[0].body = Some(block([reference(0, Type::Parameter { parameter: T })]));
    let lir = resin_lir::analyze_with_options(&hir, &options(1)).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.functions.len(), 2);
    assert!(
        lir.functions[1]
            .blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .any(|i| *i
                == Instr::Function {
                    function: FunctionId::from_index(1)
                })
    );
}

#[test]
fn identity_signatures_and_bodies_are_concrete_without_changing_hir() {
    let ty = Type::Parameter { parameter: T };
    let name = Ident {
        val: "value".into(),
        span: SPAN,
    };
    let mut identity = template(
        "identity",
        term(
            ty.clone(),
            TermKind::Local {
                binding: 0,
                name: name.clone(),
            },
        ),
    );
    identity.signature.params = vec![Parameter {
        binding: Some(0),
        name,
        annotation: annotation(ty),
    }];
    let reference = term(
        Type::Function {
            param: Box::new(Type::Int32),
            result: Box::new(Type::Int32),
        },
        TermKind::Function {
            function: FunctionId::from_index(0),
            type_args: vec![Type::Int32],
        },
    );
    let hir = Module {
        functions: vec![identity, function("main", block([reference]))],
        ..Default::default()
    };
    let lir = resin_lir::generate(&hir).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.functions[1].locals[0].ty, Ty::Int32);
    assert_eq!(lir.functions[1].result, Ty::Int32);
    assert_eq!(
        hir.functions[0].signature.result.ty,
        Type::Parameter { parameter: T }
    );
}

#[test]
fn unused_families_do_not_constrain_supported_concrete_operations() {
    let ty = Type::Parameter { parameter: T };
    let name = Ident {
        val: "value".into(),
        span: SPAN,
    };
    let value = term(
        ty.clone(),
        TermKind::Local {
            binding: 0,
            name: name.clone(),
        },
    );
    let mut add = template(
        "add",
        term(
            ty.clone(),
            TermKind::Builtin {
                name: "+".into(),
                args: vec![value.clone(), value],
            },
        ),
    );
    add.signature.params.push(Parameter {
        binding: Some(0),
        name,
        annotation: annotation(ty),
    });
    let mut hir = Module {
        functions: vec![add, function("main", unit())],
        ..Default::default()
    };
    assert_eq!(resin_lir::generate(&hir).unwrap().functions.len(), 1);
    hir.functions[1].body = Some(block([term(
        Type::Function {
            param: Box::new(Type::Bool),
            result: Box::new(Type::Bool),
        },
        TermKind::Function {
            function: FunctionId::from_index(0),
            type_args: vec![Type::Bool],
        },
    )]));
    let error = resin_lir::generate(&hir).unwrap_err();
    assert!(matches!(error.kind, ErrorKind::Type { .. }));
    assert_eq!(error.applications[0].function.as_ref(), "add");
}

#[test]
fn implicit_drop_references_use_concrete_function_identities() {
    let mut hir = program(vec![]);
    hir.types.push(resin_hir::TypeDefinition {
        name: "Owner".into(),
        body: Type::Record { fields: vec![] },
        methods: Default::default(),
        drop: Some(FunctionId::from_index(2)),
    });
    let mut drop = function("drop", unit());
    drop.signature.params.push(Parameter {
        binding: Some(0),
        name: Ident {
            val: "self".into(),
            span: SPAN,
        },
        annotation: annotation(Type::Pointer {
            pointee: Box::new(Type::Defined {
                definition: TypeId::from_index(0),
            }),
        }),
    });
    hir.functions.push(drop);
    let lir = resin_lir::generate(&hir).unwrap();
    resin_lir::verify(&lir).unwrap();
    assert_eq!(lir.types[0].drop_hook(), Some(FunctionId::from_index(1)));
}

#[test]
fn type_expansion_has_a_separate_guard_from_the_function_allowance() {
    let mut hir = program(vec![Type::Int32]);
    hir.functions[0].body = Some(block([reference(
        0,
        Type::Pointer {
            pointee: Box::new(Type::Parameter { parameter: T }),
        },
    )]));
    let error = resin_lir::generate(&hir).unwrap_err();
    assert!(matches!(error.kind, ErrorKind::TypeExpansionLimit { .. }));
}

#[test]
fn exponentially_growing_arguments_hit_the_type_size_guard() {
    let mut hir = program(vec![Type::Int32]);
    let field = |name: &str| resin_hir::RecordField {
        name: name.into(),
        ty: Type::Parameter { parameter: T },
    };
    hir.functions[0].body = Some(block([reference(
        0,
        Type::Record {
            fields: vec![field("left"), field("right")],
        },
    )]));
    let error = resin_lir::generate(&hir).unwrap_err();
    assert!(matches!(error.kind, ErrorKind::TypeSizeLimit { .. }));
}

#[test]
fn failed_requests_are_memoized_without_publishing_an_incomplete_module() {
    let mut hir = program(vec![Type::Int32, Type::Int32]);
    hir.functions[0].body = Some(term(
        Type::Parameter {
            parameter: TypeParameterId::from_index(99),
        },
        TermKind::Constant {
            value: Constant::Unit,
        },
    ));
    let errors = resin_lir::analyze_with_options(&hir, &options(1)).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::InvalidHir { .. }));
}
