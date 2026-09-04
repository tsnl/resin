use resin::{
    ast::{StmtKind, TypeKind, generate::AstGen, sexpfmt},
    ir::{
        BasicBlock, BlockId, Function, Instr, Local, LocalId, Module, RecordField, Terminator, Ty,
        TypeDef, TypeId, Value, VerifyErrorKind, verify,
    },
};
use tree_sitter::Parser;

fn parse(src: &str) -> resin::ast::SourceFile {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .expect("failed to load Resin grammar");
    let tree = parser.parse(src, None).expect("parser returned no tree");
    assert!(
        !tree.root_node().has_error(),
        "parse error: {}",
        tree.root_node().to_sexp()
    );
    AstGen::new(src).gen_source_file(tree.root_node())
}

fn linked_list_type() -> TypeDef {
    let list = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    TypeDef {
        name: "List".into(),
        body: Ty::Record {
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
    }
}

#[test]
fn uppercase_definitions_remain_distinct_in_the_ast() {
    let file = parse("List = { value: int, next: Ptr (List) };");
    let StmtKind::DefineType { name, init } = &file.stmts[0].val else {
        panic!("expected a type definition, got {:?}", file.stmts[0].val);
    };
    assert_eq!(name.val.as_ref(), "List");
    assert!(matches!(init.val, TypeKind::Record { .. }));
    assert!(sexpfmt::format_source(&file).contains("(define-type"));
}

#[test]
fn a_linked_list_is_finite_through_its_next_pointer() {
    let list = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    let function = Function {
        nonlocals: vec![],
        params: vec![LocalId::from_index(0)],
        result: Ty::Int32,
        locals: vec![Local { ty: list }],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
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
        types: vec![linked_list_type()],
        globals: vec![],
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
        types: vec![TypeDef {
            name: "Bad".into(),
            body: Ty::Record {
                fields: vec![RecordField {
                    name: "next".into(),
                    ty: recursive,
                }],
            },
        }],
        globals: vec![],
        functions: vec![],
    };

    let error = verify(&module).unwrap_err();
    assert!(matches!(
        error.kind,
        VerifyErrorKind::RecursiveTypeWithoutIndirection { .. }
    ));
}

#[test]
fn nominal_types_do_not_equal_their_representations() {
    let meters = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    let function = Function {
        nonlocals: vec![],
        params: vec![],
        result: meters,
        locals: vec![],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            instrs: vec![Instr::Push {
                value: Value::Int32 { value: 1 },
            }],
            terminator: Terminator::Return,
        }],
    };
    let module = Module {
        types: vec![TypeDef {
            name: "Meters".into(),
            body: Ty::Int32,
        }],
        globals: vec![],
        functions: vec![function],
    };

    let error = verify(&module).unwrap_err();
    assert!(matches!(
        error.kind,
        VerifyErrorKind::InvalidReturnStack { .. }
    ));
}
