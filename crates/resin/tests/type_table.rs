mod support;

use resin::{
    c, glsl,
    lir::{self, Ty, TypeDef, TypeId, TypeTable},
};

#[test]
fn nominal_and_structural_types_share_one_index_space() {
    let module = support::module(
        r#"
        struct First { value: int };
        struct Second { value: int };
        type Alias = First;
        type Record = { value: int };
        type RecordAlias = Record;
        type Optional = int | None;
        type Flattened = None | Optional | int;
        struct Node { next: Ptr<Node>, value: Optional };
        def nominal(x: Alias) -> First = { x };
        def record(x: RecordAlias) -> Record = { x };
        def optional(x: Flattened) -> Optional = { x };
        "#,
    );
    for (index, ty) in module.types.types().enumerate() {
        let id = module.types.id(&ty).unwrap();
        assert_eq!(id.index(), index);
        assert_eq!(id.tag() as usize, index);
        assert_eq!(module.types[index].ty(id), ty);
    }
    let first = module.functions[0].result.clone();
    let second = Ty::Defined {
        definition: TypeId::from_index(2),
    };
    let record = module.functions[1].result.clone();
    assert_ne!(module.types.id(&first), module.types.id(&second));
    assert_ne!(module.types.id(&first), module.types.id(&record));
    assert_eq!(module.types[1].body(), module.types[2].body());
    assert_eq!(
        module.types.id(module.types[1].body().unwrap()),
        module.types.id(&record)
    );
    assert_eq!(
        module.functions[2].result,
        Ty::union_of([Ty::Int32, Ty::None])
    );
    assert!(module.types.id(&Ty::None).is_some());
    let mut reinterned = module.types.clone();
    for ty in module.types.types() {
        assert_eq!(reinterned.intern(&ty), module.types.id(&ty).unwrap());
    }
    assert_eq!(reinterned, module.types);
}

#[test]
fn both_emitters_use_payload_table_indices_as_union_tags() {
    let mut module = support::module(
        r#"
        export { main, kernel };
        struct HostOnly { unrelated: float64 };
        def choose(i: uint) -> uint | None = { if (i == 0_ui) { None } else { i } };
        def main() -> int = { int(choose(42_ui)!) };
        @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := choose(i)!; };
        "#,
    );
    let optional = module.functions[0].result.clone();
    let optional_id = module.types.id(&optional).unwrap().index();
    let none_id = module.types.id(&Ty::None).unwrap().index();
    let uint_id = module.types.id(&Ty::UInt32).unwrap().index();
    // A type literal is another consumer of the very same ID, including when
    // direct IR clients add it after source lowering has completed the table.
    let type_function = module.functions.len();
    module.functions.push(lir::Function {
        name: Some("type_value".into()),
        foreign: None,
        result: Ty::Type,
        locals: vec![lir::Local {
            name: None,
            ty: Ty::Unit,
        }],
        entry: lir::BlockId::from_index(0),
        blocks: vec![lir::BasicBlock {
            name: None,
            instrs: vec![lir::Instr::Push {
                value: lir::Value::Type {
                    ty: optional.clone(),
                },
            }],
            terminator: lir::Terminator::Return,
        }],
    });
    let host = c::emit(&module, "main").unwrap();
    let shader = glsl::emit(&module, "kernel", glsl::Stage::Compute).unwrap();
    for source in [&host, &shader] {
        assert!(
            source.contains(&format!("struct r_t{optional_id} {{")),
            "{source}"
        );
        assert!(source.contains(&format!(".tag == {none_id}u")), "{source}");
        assert!(source.contains(&format!(" v{uint_id};")), "{source}");
    }
    assert!(host.contains(&format!(".tag = {uint_id}u")), "{host}");
    assert!(
        shader.contains(&format!("r_t{optional_id}({uint_id}u,")),
        "{shader}"
    );
    let type_body = host
        .rsplit_once(&format!("r_fn{type_function}("))
        .unwrap()
        .1;
    assert!(
        type_body.contains(&format!(" r_v0_0 = {optional_id};")),
        "{type_body}"
    );
}

#[test]
fn verifier_rejects_duplicate_definitions_and_nested_union_payloads() {
    for definitions in [
        vec![
            TypeDef::Structural(Ty::Int32),
            TypeDef::Structural(Ty::Int32),
        ],
        vec![
            TypeDef::new("Nominal", Ty::Record { fields: vec![] }),
            TypeDef::Structural(Ty::Defined {
                definition: TypeId::from_index(0),
            }),
        ],
        vec![TypeDef::Structural(Ty::Union {
            variants: vec![Ty::None, Ty::union_of([Ty::Int32, Ty::Bool])],
        })],
    ] {
        let module = lir::Module {
            types: TypeTable::from(definitions),
            ..Default::default()
        };
        assert!(resin::lir_verifier::verify(&module).is_err());
    }
}
