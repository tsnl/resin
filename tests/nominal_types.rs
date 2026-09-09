use resin_ast::{StmtKind, TypeKind, format_source};
use resin_lir::VerifyErrorKind;
use resin_lir::VerifyLocation;
use resin_lir::verify;
use resin_lir::{BasicBlock, BlockId, Function, Instr, Local, Module, Terminator, format_module};
use resin_types::prelude::*;

fn parse(src: &str) -> resin_ast::SourceFile {
    resin_ast::generate(&resin_cst::Document::reparse(src.to_string(), None))
        .unwrap_or_else(|err| panic!("{err}"))
}

fn linked_list_type() -> TypeDef {
    let list = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    TypeDef::new(
        "List",
        Ty::Record {
            fields: vec![
                RecordField {
                    name: "value".into(),
                    ty: Ty::Int32,
                },
                RecordField {
                    name: "next".into(),
                    ty: Ty::Pointer {
                        pointee: Box::new(list),
                    },
                },
            ],
        },
    )
}

#[test]
fn uppercase_definitions_remain_distinct_in_the_ast() {
    let file = parse("struct List { value: int, next: Ptr<List> };");
    let StmtKind::Struct { name, body: init } = &file.stmts[0].val else {
        panic!("expected a type definition, got {:?}", file.stmts[0].val);
    };
    assert_eq!(name.val.as_ref(), "List");
    assert!(matches!(init.val, TypeKind::Record { .. }));
    assert!(format_source(&file).contains("(struct"));
}

#[test]
fn a_linked_list_is_finite_through_its_next_pointer() {
    let list = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    let function = Function {
        foreign: None,
        name: None,
        result: Ty::Int32,
        locals: vec![Local {
            name: None,
            ty: list,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs: vec![
                Instr::LocalAddress {
                    local: LocalId::from_index(0),
                },
                Instr::AccessStatic { index: 0 },
                Instr::Load,
            ],
            terminator: Terminator::Return,
        }],
    };

    verify(&Module {
        shaders: Default::default(),
        origins: Default::default(),
        entries: Default::default(),
        types: vec![linked_list_type()].into(),
        functions: vec![function],
    })
    .unwrap();
}

#[test]
fn an_inline_recursive_type_is_rejected() {
    let recursive = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    let module = Module {
        shaders: Default::default(),
        origins: Default::default(),
        entries: Default::default(),
        types: vec![TypeDef::new(
            "Bad",
            Ty::Record {
                fields: vec![RecordField {
                    name: "next".into(),
                    ty: recursive,
                }],
            },
        )]
        .into(),
        functions: vec![],
    };

    let error = verify(&module).unwrap_err();
    assert!(matches!(
        error.kind,
        VerifyErrorKind::RecursiveTypeWithoutIndirection { .. }
    ));
}

#[test]
fn incomplete_definitions_are_rejected_by_the_verifier_and_printed_explicitly() {
    let mut context = TyperContext::new();
    let definition = context.reserve_type("Pending");
    let module = Module {
        shaders: Default::default(),
        origins: Default::default(),
        types: context.definitions().to_vec().into(),
        ..Default::default()
    };
    let err = verify(&module).unwrap_err();
    assert_eq!(
        err.kind,
        VerifyErrorKind::IncompleteTypeDefinition { definition }
    );
    assert_eq!(err.location, VerifyLocation::TypeDefinition { definition });
    assert!(format_module(&module).contains("incomplete"));
}

#[test]
fn nominal_types_do_not_equal_their_representations() {
    let meters = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    let function = Function {
        foreign: None,
        name: None,
        result: meters,
        locals: vec![Local {
            name: None,
            ty: Ty::Unit,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs: vec![Instr::Push {
                value: Value::Int32 { value: 1 },
            }],
            terminator: Terminator::Return,
        }],
    };
    let module = Module {
        shaders: Default::default(),
        origins: Default::default(),
        entries: Default::default(),
        types: vec![TypeDef::new(
            "Meters",
            Ty::Record {
                fields: vec![RecordField {
                    name: "value".into(),
                    ty: Ty::Int32,
                }],
            },
        )]
        .into(),
        functions: vec![function],
    };

    let error = verify(&module).unwrap_err();
    assert!(matches!(
        error.kind,
        VerifyErrorKind::InvalidReturnStack { .. }
    ));
}
