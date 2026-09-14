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
    Ty::Defined {
        definition: TypeId::from_index(0),
    }
}
fn pipeline_definition(kind: resin_types::GpuPipelineKind, root: Ty) -> TypeDef {
    TypeDef::Nominal {
        name: "Pipeline".into(),
        body: Some(Ty::Record {
            fields: vec![RecordField {
                name: "token".into(),
                ty: Ty::GpuPipelineContract,
            }],
        }),
        drop: None,
        gpu_projection: None,
        gpu_pipeline: Some(resin_types::GpuPipeline {
            kind,
            root,
            owner: owner(),
        }),
    }
}
fn projection() -> resin_types::GpuProjectionPlan {
    resin_types::GpuProjectionPlan {
        source: Ty::UInt32,
        target: Ty::UInt32,
        operation: resin_types::GpuProjectionOperation::Copy,
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
    let pointer = Ty::GpuView;
    let mut module = Module {
        types: vec![pipeline_definition(
            resin_types::GpuPipelineKind::Compute,
            Ty::UInt32,
        )]
        .into(),
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
            pipeline: pipeline(),
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
            projection: projection(),
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
    wrong_root.types = vec![pipeline_definition(
        resin_types::GpuPipelineKind::Compute,
        Ty::Int32,
    )]
    .into();
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
    bad_context.functions[3].locals[0].ty = Ty::Unit;
    assert!(resin_lir::verify(&bad_context).is_err());
    let mut bad_allocator = module;
    bad_allocator.functions[0].blocks[0].instrs[6] = Instr::GpuDispatch {
        projection: projection(),
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
    graphics_dispatch.types = vec![pipeline_definition(
        resin_types::GpuPipelineKind::Graphics,
        Ty::UInt32,
    )]
    .into();
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

#[test]
fn recording_rejects_forged_projection_plan_and_pipeline_layout() {
    let mut module = recording();
    if let Instr::GpuDispatch { projection, .. } = &mut module.functions[0].blocks[0].instrs[6] {
        projection.target = Ty::Float32;
    }
    assert!(resin_lir::verify(&module).is_err());
    let mut module = fixture();
    let mut definition = pipeline_definition(resin_types::GpuPipelineKind::Compute, Ty::UInt32);
    if let TypeDef::Nominal {
        body: Some(Ty::Record { fields }),
        ..
    } = &mut definition
    {
        fields.push(RecordField {
            name: "extra".into(),
            ty: Ty::UInt32,
        });
    }
    module.types = vec![definition].into();
    assert!(resin_lir::verify(&module).is_err());
}
