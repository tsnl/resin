use resin_lir::{
    BasicBlock, BlockId, Function, Instr, Local, Module, Profile, Terminator, VerifyErrorKind,
};
use resin_types::{LocalId, RecordField, Ty};

fn projection(ty: Ty, path: Vec<usize>) -> Module {
    Module {
        functions: vec![Function {
            name: None,
            profile: Profile::Host,
            foreign: None,
            result: Ty::Unit,
            parameter_count: 1,
            locals: vec![Local { name: None, ty }],
            entry: BlockId::from_index(0),
            blocks: vec![BasicBlock {
                name: None,
                instrs: vec![Instr::TakeField {
                    local: LocalId::from_index(0),
                    path,
                }],
                terminator: Terminator::Return,
            }],
        }],
        ..Default::default()
    }
}

#[test]
fn owned_field_paths_only_traverse_inline_record_fields() {
    let record = Ty::Record {
        fields: vec![RecordField {
            name: "field".into(),
            ty: Ty::Unit,
        }],
    };
    resin_lir::verify(&projection(record.clone(), vec![0])).unwrap();
    for (ty, path) in [
        (record.clone(), vec![]),
        (record.clone(), vec![1]),
        (
            Ty::Pointer {
                pointee: Box::new(record.clone()),
            },
            vec![0],
        ),
        (
            Ty::Array {
                element: Box::new(record),
                length: 1,
            },
            vec![0, 0],
        ),
    ] {
        let error = resin_lir::verify(&projection(ty, path)).unwrap_err();
        assert!(
            matches!(error.kind, VerifyErrorKind::InvalidOwnedField { .. }),
            "{error}"
        );
    }
}
