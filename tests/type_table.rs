use resin_types::prelude::*;
mod support;

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
    module.functions.push(resin_lir::Function {
        name: Some("type_value".into()),
        foreign: None,
        result: Ty::Type,
        locals: vec![resin_lir::Local {
            name: None,
            ty: Ty::Unit,
        }],
        entry: resin_lir::BlockId::from_index(0),
        blocks: vec![resin_lir::BasicBlock {
            name: None,
            instrs: vec![resin_lir::Instr::Push {
                value: Value::Type {
                    ty: optional.clone(),
                },
            }],
            terminator: resin_lir::Terminator::Return,
        }],
    });
    let host_project = support::project::Project::new(&module, Some("main")).unwrap();
    let shader_project = support::project::Project::new(&module, None).unwrap();
    let host = std::fs::read_to_string(host_project.generated.c_source().unwrap()).unwrap();
    let shader = std::fs::read(shader_project.generated.shaders()[0].unoptimized_spirv()).unwrap();
    assert!(
        host.contains(&format!("struct r_t{optional_id} {{")),
        "{host}"
    );
    assert!(host.contains(&format!(".tag == {none_id}u")), "{host}");
    assert!(host.contains(&format!(" v{uint_id};")), "{host}");
    assert!(host.contains(&format!(".tag = {uint_id}u")), "{host}");
    use support::shaders::instructions;
    let uint_type = instructions(&shader, 21)
        .find(|args| args[1..] == [32, 0])
        .unwrap()[0];
    let constants: std::collections::HashMap<_, _> = instructions(&shader, 43)
        .filter(|args| args[0] == uint_type)
        .map(|args| (args[1], args[2]))
        .collect();
    // OpIEqual tests the canonical None tag; OpCompositeConstruct puts the
    // canonical uint tag in the aggregate's first field.
    assert!(instructions(&shader, 170).any(|args| {
        args[2..]
            .iter()
            .any(|id| constants.get(id) == Some(&(none_id as u32)))
    }));
    assert!(
        instructions(&shader, 80)
            .any(|args| { constants.get(&args[2]) == Some(&(uint_id as u32)) })
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
        let module = resin_lir::Module {
            types: TypeTable::from(definitions),
            ..Default::default()
        };
        assert!(resin_lir::verify(&module).is_err());
    }
}
