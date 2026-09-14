use resin_lir::{BasicBlock, BlockId, Function, Instr, Local, Module, Terminator, VerifyErrorKind};
use resin_types::prelude::*;

fn owner() -> Ty {
    Ty::StrongOwner
}
fn result(value: Ty) -> Ty {
    Ty::Result {
        value: Box::new(value),
        error: Box::new(Ty::union([])),
    }
}
fn pipeline() -> Ty {
    Ty::GpuComputePipeline {
        root: Box::new(Ty::UInt32),
        owner: Box::new(owner()),
    }
}
fn id(index: usize) -> FunctionId {
    FunctionId::from_index(index)
}

// These functions are never executed. Recursive bodies give the bridge signatures
// valid LIR without requiring a runtime or manufacturing opaque native values.
fn declaration(index: usize, parameters: Vec<Ty>, result: Ty) -> Function {
    let count = parameters.len();
    let mut instrs = vec![Instr::Function {
        function: id(index),
    }];
    instrs.extend((0..count).map(|index| Instr::TakeLocal {
        local: LocalId::from_index(index),
    }));
    instrs.push(Instr::Call { arguments: count });
    Function {
        name: None,
        profile: resin_lir::Profile::Host,
        foreign: None,
        result,
        parameter_count: count,
        locals: parameters
            .into_iter()
            .map(|ty| Local { name: None, ty })
            .collect(),
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs,
            terminator: Terminator::Return,
        }],
    }
}

fn fixture() -> Module {
    let bytes = Ty::byte_span();
    let pointer = Ty::GpuPointer {
        pointee: Box::new(Ty::UInt8),
    };
    let mut module = Module {
        functions: vec![
            declaration(0, vec![owner()], result(pipeline())),
            declaration(1, vec![owner(), bytes], result(owner())),
            declaration(
                2,
                vec![
                    Ty::UInt64,
                    Ty::Pointer {
                        pointee: Box::new(Ty::UInt32),
                    },
                ],
                Ty::Unit,
            ),
            declaration(3, vec![owner()], owner()),
            declaration(
                4,
                vec![owner(), Ty::UInt64, Ty::UInt64, Ty::Int32],
                result(pointer),
            ),
            declaration(
                5,
                vec![
                    Ty::Unit,
                    owner(),
                    Ty::GpuArguments,
                    Ty::UInt32,
                    Ty::UInt32,
                    Ty::UInt32,
                ],
                result(Ty::Unit),
            ),
        ],
        ..Default::default()
    };
    module.functions[2].profile = resin_lir::Profile::Shader;
    module.functions[2].blocks[0].instrs = vec![Instr::Push { value: Value::Unit }];
    module.shaders.insert(
        id(2),
        ShaderEntry {
            stage: "compute".into(),
            embedded: true,
        },
    );
    module.functions[0].blocks[0].instrs = vec![
        Instr::TakeLocal {
            local: LocalId::from_index(0),
        },
        Instr::GpuComputePipeline {
            factory: id(1),
            shader: id(2),
        },
    ];
    module
}

fn recording() -> Module {
    let mut module = fixture();
    let caller = &mut module.functions[0];
    caller.locals[0].ty = pipeline();
    caller.result = result(Ty::Unit);
    caller.blocks[0].instrs = vec![
        Instr::Push { value: Value::Unit },
        Instr::TakeLocal {
            local: LocalId::from_index(0),
        },
        Instr::Push {
            value: Value::UInt32 { value: 10 },
        },
        Instr::Push {
            value: Value::UInt32 { value: 1 },
        },
        Instr::Push {
            value: Value::UInt32 { value: 1 },
        },
        Instr::Push {
            value: Value::UInt32 { value: 1 },
        },
        Instr::GpuDispatch {
            context: id(3),
            allocator: id(4),
            record: id(5),
        },
    ];
    module
}

#[test]
fn creation_derives_root_from_an_embedded_declaration() {
    let module = fixture();
    resin_lir::verify(&module).unwrap();
    let mut unembedded = module.clone();
    unembedded.shaders.get_mut(&id(2)).unwrap().embedded = false;
    assert_eq!(
        resin_lir::verify(&unembedded).unwrap_err().kind,
        VerifyErrorKind::InvalidGpuOperation
    );
    let mut undeclared = module.clone();
    undeclared.shaders.clear();
    assert_eq!(
        resin_lir::verify(&undeclared).unwrap_err().kind,
        VerifyErrorKind::InvalidGpuOperation
    );
    let mut wrong_root = module;
    wrong_root.functions[0].result = result(Ty::GpuComputePipeline {
        root: Box::new(Ty::Int32),
        owner: Box::new(owner()),
    });
    assert!(resin_lir::verify(&wrong_root).is_err());
}

#[test]
fn recording_checks_arguments_context_and_allocator_together() {
    let module = recording();
    resin_lir::verify(&module).unwrap();
    let mut bad_root = module.clone();
    bad_root.functions[0].blocks[0].instrs[2] = Instr::Push {
        value: Value::Int32 { value: 10 },
    };
    assert!(resin_lir::verify(&bad_root).is_err());
    let mut bad_context = module.clone();
    bad_context.functions[3].locals[0].ty = Ty::StrongOwner;
    assert!(resin_lir::verify(&bad_context).is_err());
    let mut bad_allocator = module;
    bad_allocator.functions[0].blocks[0].instrs[6] = Instr::GpuDispatch {
        context: id(3),
        allocator: id(500),
        record: id(5),
    };
    assert_eq!(
        resin_lir::verify(&bad_allocator).unwrap_err().kind,
        VerifyErrorKind::InvalidFunction { function: 500 }
    );
}

#[test]
fn pipeline_owner_cannot_be_forged_by_ascription_or_reused_for_another_stage() {
    let mut forged = fixture();
    forged.functions[0].result = pipeline();
    forged.functions[0].blocks[0].instrs[1] = Instr::Ascribe { ty: pipeline() };
    assert!(resin_lir::verify(&forged).is_err());
    let mut graphics_dispatch = recording();
    graphics_dispatch.functions[0].locals[0].ty = Ty::GpuGraphicsPipeline {
        root: Box::new(Ty::UInt32),
        owner: Box::new(owner()),
    };
    assert_eq!(
        resin_lir::verify(&graphics_dispatch).unwrap_err().kind,
        VerifyErrorKind::InvalidGpuOperation
    );
}

#[test]
fn native_pipeline_bridges_require_host_instances() {
    let mut module = fixture();
    module.functions[1].profile = resin_lir::Profile::Shader;
    assert!(matches!(
        resin_lir::verify(&module).unwrap_err().kind,
        VerifyErrorKind::InvalidProfile {
            expected: resin_lir::Profile::Host,
            found: resin_lir::Profile::Shader
        }
    ));
}
