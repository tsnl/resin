//! Public backend operations consume verified LIR and return owned binary artifacts.
use resin_codegen::{NativeInputs, NativeObject, NativeOptimization};
use resin_executor::{Cancellation, Execution};
use resin_lir::{BasicBlock, BlockId, Function, Instr, Local, Module, Terminator, VerifiedModule};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::{num::NonZeroUsize, sync::Arc, time::Duration};

async fn native(
    checked: &Arc<VerifiedModule>,
    entry: &str,
    inputs: NativeInputs,
) -> Result<NativeObject, resin_codegen::GenerationError> {
    resin_codegen::generate_native(
        checked.clone(),
        entry.into(),
        NativeOptimization::Speed,
        Arc::new(inputs),
        &Execution::default(),
        &Cancellation::new(),
    )
    .await
}

async fn shader(
    checked: &Arc<VerifiedModule>,
    function: usize,
) -> Result<Arc<[u8]>, resin_codegen::GenerationError> {
    resin_codegen::generate_spirv(
        checked.clone(),
        FunctionId::from_index(function),
        &Execution::default(),
        &Cancellation::new(),
    )
    .await
}

fn constant_function(parameters: Vec<Ty>, result: Ty, value: Value) -> Function {
    Function {
        profile: resin_lir::Profile::Host,
        name: None,
        foreign: None,
        result,
        parameter_count: parameters.len(),
        locals: parameters
            .into_iter()
            .map(|ty| Local { name: None, ty })
            .collect(),
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs: vec![Instr::Push { value }],
            terminator: Terminator::Return,
        }],
    }
}

fn module() -> Module {
    let mut module = Module {
        functions: vec![
            constant_function(vec![], Ty::Int32, Value::Int32 { value: 42 }),
            constant_function(
                vec![
                    Ty::UInt64,
                    Ty::Pointer {
                        pointee: Box::new(Ty::UInt64),
                    },
                ],
                Ty::Unit,
                Value::Unit,
            ),
        ],
        entries: [("main".into(), FunctionId::from_index(0))].into(),
        shaders: [(
            FunctionId::from_index(1),
            ShaderEntry {
                stage: "compute".into(),
                embedded: false,
            },
        )]
        .into(),
        ..Default::default()
    };
    module.functions[1].profile = resin_lir::Profile::Shader;
    module
}

fn bad_shader() -> Module {
    let mut module = module();
    module.functions[1].locals.extend([
        Local {
            name: None,
            ty: Ty::UInt64,
        },
        Local {
            name: None,
            ty: Ty::Pointer {
                pointee: Box::new(Ty::UInt64),
            },
        },
    ]);
    module.functions[1].blocks[0].instrs.splice(
        0..0,
        [
            Instr::LocalAddress {
                local: LocalId::from_index(2),
            },
            Instr::SetLocal {
                local: LocalId::from_index(3),
            },
        ],
    );
    module
}

#[tokio::test]
async fn artifacts_outlive_verified_inputs_and_opaque_source_names_are_not_paths() {
    let mut module = module();
    module.origins.functions.insert(
        FunctionId::from_index(0),
        SourceLocation {
            source: Source::new("editor://buffer/λ", "main"),
            span: Span { start: 0, end: 4 },
        },
    );
    let checked = Arc::new(VerifiedModule::new(module).unwrap());
    let weak = Arc::downgrade(&checked);
    let object = native(&checked, "main", NativeInputs::default())
        .await
        .unwrap();
    let binary = shader(&checked, 1).await.unwrap();
    let before = object.bytes().to_vec();
    drop(checked);
    assert!(
        weak.upgrade().is_none(),
        "completed artifacts must release their compiler inputs"
    );
    assert_eq!(object.bytes(), before);
    assert!(!object.bytes().is_empty());
    assert_eq!(&binary[..4], &[3, 2, 35, 7]);
}

#[tokio::test]
async fn shader_only_clients_select_each_declared_function_without_a_host_entry() {
    let mut module = module();
    module.entries.clear();
    module.functions.push(module.functions[1].clone());
    module.shaders.insert(
        FunctionId::from_index(2),
        ShaderEntry {
            stage: "compute".into(),
            embedded: false,
        },
    );
    let checked = Arc::new(VerifiedModule::new(module).unwrap());
    let (first, second) = tokio::join!(shader(&checked, 1), shader(&checked, 2));
    for bytes in [first.unwrap(), second.unwrap()] {
        assert_eq!(&bytes[..4], &[3, 2, 35, 7]);
        assert_eq!(bytes.len() % 4, 0);
    }
}

#[tokio::test]
async fn host_generation_does_not_lower_an_unrequested_shader() {
    let checked = Arc::new(VerifiedModule::new(bad_shader()).unwrap());
    assert!(
        !native(&checked, "main", NativeInputs::default())
            .await
            .unwrap()
            .bytes()
            .is_empty()
    );
    assert!(
        shader(&checked, 1)
            .await
            .unwrap_err()
            .to_string()
            .contains("shader-local addresses")
    );
}

#[tokio::test]
async fn later_failures_preserve_completed_objects_and_shader_bytes() {
    let checked = Arc::new(VerifiedModule::new(module()).unwrap());
    let object = native(&checked, "main", NativeInputs::default())
        .await
        .unwrap();
    let binary = shader(&checked, 1).await.unwrap();
    let before_object = object.bytes().to_vec();
    let before_shader = binary.to_vec();
    assert!(
        native(&checked, "missing", NativeInputs::default())
            .await
            .is_err()
    );
    assert!(
        shader(&Arc::new(VerifiedModule::new(bad_shader()).unwrap()), 1)
            .await
            .is_err()
    );
    assert_eq!(object.bytes(), before_object);
    assert_eq!(&*binary, before_shader);
}

#[tokio::test]
async fn embedded_shaders_require_explicit_complete_binaries_and_keep_exact_bytes() {
    let mut module = module();
    module
        .shaders
        .get_mut(&FunctionId::from_index(1))
        .unwrap()
        .embedded = true;
    let checked = Arc::new(VerifiedModule::new(module).unwrap());
    assert!(
        native(&checked, "main", NativeInputs::default())
            .await
            .unwrap_err()
            .to_string()
            .contains("missing shader binary")
    );
    let binary = shader(&checked, 1).await.unwrap();
    let inputs = NativeInputs {
        shaders: [(FunctionId::from_index(1), binary.clone())].into(),
        ..Default::default()
    };
    let object = native(&checked, "main", inputs).await.unwrap();
    assert!(
        object
            .bytes()
            .windows(binary.len())
            .any(|bytes| bytes == &*binary)
    );
    for (function, bytes) in [(1, vec![1, 2, 3]), (99, vec![1, 2, 3, 4])] {
        let inputs = NativeInputs {
            shaders: [(FunctionId::from_index(function), bytes.into())].into(),
            ..Default::default()
        };
        assert!(native(&checked, "main", inputs).await.is_err());
    }
}

#[tokio::test]
async fn code_generation_rejects_absent_entries_and_wrong_profiles() {
    let mut module = module();
    let checked = Arc::new(VerifiedModule::new(module.clone()).unwrap());
    assert!(
        native(&checked, "absent", NativeInputs::default())
            .await
            .is_err()
    );
    assert!(
        shader(&checked, 0)
            .await
            .unwrap_err()
            .to_string()
            .contains("not a shader")
    );
    module
        .entries
        .insert("kernel".into(), FunctionId::from_index(1));
    assert!(
        VerifiedModule::new(module).is_err(),
        "the verifier rejects shader-profile functions in the host export map"
    );
}

#[tokio::test]
async fn independent_parallel_generations_and_retained_clones_own_their_bytes() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<NativeObject>();
    let checked = Arc::new(VerifiedModule::new(module()).unwrap());
    let (first, second) = tokio::join!(
        native(&checked, "main", NativeInputs::default()),
        native(&checked, "main", NativeInputs::default())
    );
    let (first, second) = (first.unwrap(), second.unwrap());
    assert_eq!(first.bytes(), second.bytes());
    let retained = first.clone();
    let bytes = first.shared_bytes();
    let weak = Arc::downgrade(&bytes);
    assert!(Arc::ptr_eq(&bytes, &retained.shared_bytes()));
    drop(first);
    drop(bytes);
    drop(second);
    assert!(weak.upgrade().is_some());
    assert!(!retained.bytes().is_empty());
    drop(retained);
    assert!(weak.upgrade().is_none());
}

#[tokio::test]
async fn queued_cancellation_returns_no_native_or_shader_artifact() {
    let execution = Execution::new(NonZeroUsize::new(1).unwrap());
    let checked = Arc::new(VerifiedModule::new(module()).unwrap());
    let cancellation = Cancellation::new();
    let permit = execution.acquire(&Cancellation::new()).await.unwrap();
    let mut native = Box::pin(resin_codegen::generate_native(
        checked.clone(),
        "main".into(),
        NativeOptimization::None,
        Arc::new(NativeInputs::default()),
        &execution,
        &cancellation,
    ));
    let mut shader = Box::pin(resin_codegen::generate_spirv(
        checked,
        FunctionId::from_index(1),
        &execution,
        &cancellation,
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(5), &mut native)
            .await
            .is_err()
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(5), &mut shader)
            .await
            .is_err()
    );
    cancellation.cancel();
    assert!(matches!(
        native.await,
        Err(resin_codegen::GenerationError::Execution {
            error: resin_executor::Error::Cancelled
        })
    ));
    assert!(matches!(
        shader.await,
        Err(resin_codegen::GenerationError::Execution {
            error: resin_executor::Error::Cancelled
        })
    ));
    drop(permit);
    execution.wait_idle().await;
}

#[tokio::test]
async fn explicit_headers_remain_source_dependencies_without_codegen_filesystem_reads() {
    let mut module = module();
    let header = resin_lir::ForeignHeader {
        source: SourceId::new("module.resin"),
        spelling: "absent/header.h".into(),
    };
    module.foreign_headers.insert(header.clone());
    let checked = Arc::new(VerifiedModule::new(module).unwrap());
    native(&checked, "main", NativeInputs::default())
        .await
        .unwrap();
    assert!(checked.view().module().foreign_headers.contains(&header));
    // Acquiring/validating this dependency is the application's native-input pass.
    // Root extern_preamble tests still require deleting an empty header to fail.
}

#[tokio::test]
async fn source_scoped_foreign_bindings_keep_distinct_adapter_symbols() {
    let mut module = module();
    module.shaders.clear();
    module.functions.truncate(1);
    module.functions[0].blocks[0].instrs = vec![
        Instr::Function {
            function: FunctionId::from_index(1),
        },
        Instr::Call { arguments: 0 },
        Instr::Function {
            function: FunctionId::from_index(2),
        },
        Instr::Call { arguments: 0 },
        Instr::CallBuiltin {
            name: "+".into(),
            params: vec![Ty::Int32, Ty::Int32],
            result: Ty::Int32,
        },
    ];
    for side in ["left", "right"] {
        let header = resin_lir::ForeignHeader {
            source: SourceId::new(format!("{side}.resin")),
            spelling: "same.h".into(),
        };
        module.foreign_headers.insert(header.clone());
        let mut function = constant_function(vec![], Ty::Int32, Value::Int32 { value: 0 });
        function.name = Some(format!("{side}_call").into());
        function.foreign = Some(resin_lir::Foreign {
            header,
            params: vec![],
        });
        function.blocks.clear();
        module.functions.push(function);
    }
    let checked = Arc::new(VerifiedModule::new(module).unwrap());
    assert!(
        native(&checked, "main", NativeInputs::default())
            .await
            .is_err()
    );
    let inputs = NativeInputs {
        foreign: [
            (FunctionId::from_index(1), "left_bound_adapter".into()),
            (FunctionId::from_index(2), "right_bound_adapter".into()),
        ]
        .into(),
        ..Default::default()
    };
    let object = native(&checked, "main", inputs).await.unwrap();
    for symbol in [
        b"left_bound_adapter".as_slice(),
        b"right_bound_adapter".as_slice(),
    ] {
        assert!(
            object
                .bytes()
                .windows(symbol.len())
                .any(|bytes| bytes == symbol)
        );
    }
}
