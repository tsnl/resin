//! Backend clients supply verified LIR and receive source files plus a build graph.
use resin_lir::{BasicBlock, BlockId, Function, Instr, Local, Module, Terminator, VerifiedModule};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::{fs, path::Path, sync::Arc};
use tempfile::TempDir;

async fn generate(
    checked: &Arc<VerifiedModule>,
    entry: Option<&str>,
    parent: &Path,
) -> Result<resin_codegen::GeneratedProject, resin_codegen::GenerationError> {
    resin_codegen::generate(
        checked.clone(),
        entry.map(str::to_owned),
        Arc::new(resin_codegen::NativeHeaders::default()),
        parent,
        &resin_executor::Execution::default(),
        &resin_executor::Cancellation::new(),
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
    let pointer = Ty::Pointer {
        pointee: Box::new(Ty::UInt64),
    };
    let mut module = Module {
        functions: vec![
            constant_function(vec![], Ty::Int32, Value::Int32 { value: 42 }),
            constant_function(vec![Ty::UInt64, pointer], Ty::Unit, Value::Unit),
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

fn embedded_module() -> Module {
    let mut module = module();
    let shader = FunctionId::from_index(1);
    module.shaders.get_mut(&shader).unwrap().embedded = true;
    module
}

#[tokio::test]
async fn declared_headers_are_emitted_without_foreign_function_references() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut module = module();
    module.foreign_headers.insert(resin_lir::ForeignHeader {
        source: SourceId::new("native.resin"),
        spelling: "standalone/header.h".into(),
    });
    assert!(
        module
            .functions
            .iter()
            .all(|function| function.foreign.is_none())
    );
    let checked = Arc::new(VerifiedModule::new(module).unwrap());
    let project = generate(&checked, Some("main"), directory.path())
        .await
        .unwrap();
    let source = fs::read_to_string(project.c_source().unwrap()).unwrap();
    assert!(
        source.contains("#include <standalone/header.h>"),
        "{source}"
    );
}

#[tokio::test]
async fn generated_project_outlives_its_verified_input_and_retains_opaque_names() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let project = {
        let mut module = embedded_module();
        module.origins.functions.insert(
            FunctionId::from_index(0),
            SourceLocation {
                source: Source::new("editor://buffer/λ", "main"),
                span: Span { start: 0, end: 4 },
            },
        );
        let checked = Arc::new(VerifiedModule::new(module).unwrap());
        generate(&checked, Some("main"), directory.path())
            .await
            .unwrap()
    };
    assert_eq!(project.directory().parent(), Some(directory.path()));
    assert_eq!(project.name(), "editor://buffer/λ");
    assert_eq!(project.entry(), Some("main"));
    let source = fs::read_to_string(project.c_source().unwrap()).unwrap();
    assert!(source.contains("int main("));
    let shader = &project.shaders()[0];
    assert_eq!(shader.function(), FunctionId::from_index(1));
    assert_eq!(shader.stage(), Stage::Compute);
    assert!(source.contains(&format!(
        "#include \"{}\"",
        shader.header().file_name().unwrap().to_str().unwrap()
    )));
    assert_eq!(
        &fs::read(shader.unoptimized_spirv()).unwrap()[..4],
        &[3, 2, 35, 7]
    );
    assert!(project.build_file().is_file());
    assert!(!project.directory().join("main.i").exists());
    for output in [shader.spirv(), shader.header(), project.program().unwrap()] {
        assert!(!output.exists(), "generation must not invoke native tools");
    }
}

#[tokio::test]
async fn shader_only_generation_batches_declared_functions_without_a_host_entry() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut module = module();
    module.entries.clear();
    module.functions.push(module.functions[1].clone());
    module.shaders.insert(
        FunctionId::from_index(2),
        module.shaders[&FunctionId::from_index(1)].clone(),
    );
    let checked = Arc::new(VerifiedModule::new(module).unwrap());
    let project = generate(&checked, None, directory.path()).await.unwrap();
    assert!(project.c_source().is_none());
    assert!(project.program().is_none());
    assert!(project.entry().is_none());
    assert_eq!(project.name(), "shaders");
    assert!(!project.directory().join("native-inputs.json").exists());
    assert!(
        !fs::read_to_string(project.build_file())
            .unwrap()
            .contains("native-inputs.state")
    );
    assert_eq!(project.shaders().len(), 2);
    let first = &project.shaders()[0];
    let second = &project.shaders()[1];
    assert_ne!(first.unoptimized_spirv(), second.unoptimized_spirv());
    assert_ne!(first.symbol(), second.symbol());
    for shader in project.shaders() {
        assert_eq!(
            &fs::read(shader.unoptimized_spirv()).unwrap()[..4],
            &[3, 2, 35, 7]
        );
    }
}

#[tokio::test]
async fn host_generation_does_not_emit_shader_instances() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let checked = Arc::new(VerifiedModule::new(module()).unwrap());
    let project = generate(&checked, Some("main"), directory.path())
        .await
        .unwrap();
    assert!(project.shaders().is_empty());
    assert!(
        !fs::read_to_string(project.c_source().unwrap())
            .unwrap()
            .contains("r_fn1(")
    );
    let shaders = generate(&checked, None, directory.path()).await.unwrap();
    assert_ne!(shaders.directory(), project.directory());
}

#[tokio::test]
async fn lowering_failure_leaves_existing_outputs_untouched() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let checked = Arc::new(VerifiedModule::new(embedded_module()).unwrap());
    let project = generate(&checked, Some("main"), directory.path())
        .await
        .unwrap();
    let before_c = fs::read(project.c_source().unwrap()).unwrap();
    let before_spirv = fs::read(project.shaders()[0].unoptimized_spirv()).unwrap();
    let error = generate(&checked, Some("missing"), directory.path())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("not exported"));
    let mut bad = embedded_module();
    // The language permits pointers, but this backend representation cannot store a
    // shader-local address in a physical pointer value. This remains a target-lowering error.
    bad.functions[1].locals.extend([
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
    bad.functions[1].blocks[0].instrs.splice(
        0..0,
        [
            Instr::LocalAddress {
                local: LocalId::from_index(3),
            },
            Instr::LocalAddress {
                local: LocalId::from_index(2),
            },
            Instr::Store,
            Instr::Discard,
        ],
    );
    let checked = Arc::new(VerifiedModule::new(bad).unwrap());
    let error = generate(&checked, Some("main"), directory.path())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("shader-local addresses"));
    assert_eq!(fs::read(project.c_source().unwrap()).unwrap(), before_c);
    assert_eq!(
        fs::read(project.shaders()[0].unoptimized_spirv()).unwrap(),
        before_spirv
    );
    assert_eq!(
        fs::read_dir(directory.path()).unwrap().count(),
        1,
        "failed generations must leave no child directory"
    );
}

#[tokio::test]
async fn build_graph_orders_shader_optimization_embedding_and_c_compilation() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let checked = Arc::new(VerifiedModule::new(embedded_module()).unwrap());
    let project = generate(&checked, Some("main"), directory.path())
        .await
        .unwrap();
    let graph = fs::read_to_string(project.build_file()).unwrap();
    assert!(graph.contains("include toolchain.ninja"));
    assert!(
        graph.contains("shader_1.spv: optimize_shader shader_1.unoptimized.spv | toolchain.state")
    );
    assert!(graph.contains("shader_1.h: embed_shader shader_1.spv | toolchain.state"));
    assert!(graph.contains(
        "compile_preprocessed_program main.i | toolchain.state $runtime_library native-inputs.state shader_1.h"
    ));
    let inputs: serde_json::Value =
        serde_json::from_slice(&fs::read(project.directory().join("native-inputs.json")).unwrap())
            .unwrap();
    assert_eq!(
        inputs,
        serde_json::json!({
            "translation_units": [{ "source": "main.c", "preprocessed": "main.i" }],
            "c_flags": [],
            "restrict_header_paths": false,
            "preprocessing_flags": [],
            "generated_prerequisites": ["shader_1.h"],
        })
    );
    assert!(
        !graph.contains("command ="),
        "native commands belong to the toolchain"
    );
}

#[tokio::test]
async fn code_generation_cannot_select_an_absent_entry_or_profile() {
    let mut module = module();
    module.shaders.clear();
    module.functions.truncate(1);
    let checked = Arc::new(VerifiedModule::new(module).unwrap());
    let parent = TempDir::new().unwrap();
    for entry in [None, Some("missing")] {
        assert!(generate(&checked, entry, parent.path()).await.is_err());
        assert_eq!(fs::read_dir(parent.path()).unwrap().count(), 0);
    }
}

#[tokio::test]
async fn parallel_generations_and_retained_clones_own_independent_files() {
    fn send_and_sync<T: Send + Sync>() {}
    send_and_sync::<VerifiedModule>();
    send_and_sync::<resin_codegen::GeneratedProject>();

    let parent = TempDir::new().unwrap();
    let checked = Arc::new(VerifiedModule::new(embedded_module()).unwrap());
    let execution = resin_executor::Execution::new(2.try_into().unwrap());
    let cancellation = resin_executor::Cancellation::new();
    let build = || {
        resin_codegen::generate(
            checked.clone(),
            Some("main".into()),
            Arc::new(resin_codegen::NativeHeaders::default()),
            parent.path(),
            &execution,
            &cancellation,
        )
    };
    let (first, second) = tokio::join!(build(), build());
    let first = first.unwrap();
    let second = second.unwrap();
    let first_directory = first.directory().to_path_buf();
    let second_directory = second.directory().to_path_buf();
    assert_ne!(first_directory, second_directory);
    let original_c = fs::read(first.c_source().unwrap()).unwrap();
    assert_eq!(original_c, fs::read(second.c_source().unwrap()).unwrap());
    assert_eq!(
        fs::read(first.build_file()).unwrap(),
        fs::read(second.build_file()).unwrap()
    );

    let retained = first.clone();
    drop(first);
    drop(second);
    assert!(!second_directory.exists());
    let next = build().await.unwrap();
    assert_eq!(fs::read(retained.c_source().unwrap()).unwrap(), original_c);
    assert!(retained.shaders()[0].unoptimized_spirv().is_file());
    drop(retained);
    assert!(!first_directory.exists());
    assert!(next.build_file().is_file());
}

#[tokio::test]
async fn cancellation_while_queued_creates_no_generation_directory() {
    use std::{future::Future, task::Poll};
    let parent = TempDir::new().unwrap();
    let execution = resin_executor::Execution::new(1.try_into().unwrap());
    let cancellation = resin_executor::Cancellation::new();
    let permit = execution.acquire(&cancellation).await.unwrap();
    let mut future = Box::pin(resin_codegen::generate(
        Arc::new(VerifiedModule::new(module()).unwrap()),
        Some("main".into()),
        Arc::new(resin_codegen::NativeHeaders::default()),
        parent.path(),
        &execution,
        &cancellation,
    ));
    let pending = std::future::poll_fn(|cx| Poll::Ready(future.as_mut().poll(cx))).await;
    assert!(pending.is_pending());
    cancellation.cancel();
    assert!(matches!(
        future.await,
        Err(resin_codegen::GenerationError::Execution {
            error: resin_executor::Error::Cancelled
        })
    ));
    drop(permit);
    execution.wait_idle().await;
    assert_eq!(fs::read_dir(parent.path()).unwrap().count(), 0);
}

#[tokio::test]
async fn supplied_native_files_and_source_scoped_bindings_are_owned_and_complete() {
    use resin_codegen::{NativeHeaders, NativeInclude};
    let mut module = module();
    let left = resin_lir::ForeignHeader {
        source: SourceId::new("left.resin"),
        spelling: "same.h".into(),
    };
    let right = resin_lir::ForeignHeader {
        source: SourceId::new("right.resin"),
        spelling: "same.h".into(),
    };
    module.foreign_headers.extend([left.clone(), right.clone()]);
    let checked = Arc::new(VerifiedModule::new(module).unwrap());
    let headers = Arc::new(NativeHeaders {
        files: [
            (
                "native/left/same.h".into(),
                Arc::<[u8]>::from(b"/* left */\n".as_slice()),
            ),
            (
                "native/right/same.h".into(),
                Arc::<[u8]>::from(b"/* right */\n".as_slice()),
            ),
        ]
        .into(),
        bindings: [
            (
                left,
                NativeInclude::Staged {
                    path: "native/left/same.h".into(),
                },
            ),
            (
                right,
                NativeInclude::Staged {
                    path: "native/right/same.h".into(),
                },
            ),
        ]
        .into(),
        include_directories: vec!["native/left".into(), "native/right".into()],
        runtime: None,
    });
    let parent = TempDir::new().unwrap();
    let generated = resin_codegen::generate(
        checked,
        Some("main".into()),
        headers.clone(),
        parent.path(),
        &resin_executor::Execution::default(),
        &resin_executor::Cancellation::new(),
    )
    .await
    .unwrap();
    drop(headers);
    let c = fs::read_to_string(generated.c_source().unwrap()).unwrap();
    assert!(c.contains("#include \"native/left/same.h\""));
    assert!(c.contains("#include \"native/right/same.h\""));
    assert_eq!(
        fs::read(generated.directory().join("native/left/same.h")).unwrap(),
        b"/* left */\n"
    );
    let metadata: serde_json::Value = serde_json::from_slice(
        &fs::read(generated.directory().join("native-inputs.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        metadata["preprocessing_flags"],
        serde_json::json!(["-Inative/left", "-Inative/right"])
    );
}

#[tokio::test]
async fn unsafe_native_paths_fail_before_publishing_a_generation() {
    use resin_codegen::NativeHeaders;
    for path in [
        "../escape.h",
        "main.c",
        "native/../escape.h",
        "native/C:/escape.h",
        "native/NUL.h",
        "native/folder/CoM9.data",
    ] {
        let parent = TempDir::new().unwrap();
        let headers = Arc::new(NativeHeaders {
            files: [(path.into(), Arc::<[u8]>::from(b"".as_slice()))].into(),
            ..Default::default()
        });
        let result = resin_codegen::generate(
            Arc::new(VerifiedModule::new(module()).unwrap()),
            Some("main".into()),
            headers,
            parent.path(),
            &resin_executor::Execution::default(),
            &resin_executor::Cancellation::new(),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(fs::read_dir(parent.path()).unwrap().count(), 0);
    }
}

#[tokio::test]
async fn native_file_and_directory_case_aliases_are_rejected() {
    for paths in [
        ["native/left.h", "native/LEFT.h"],
        ["native/A/x.h", "native/a/y.h"],
    ] {
        let parent = TempDir::new().unwrap();
        let headers = Arc::new(resin_codegen::NativeHeaders {
            files: paths
                .into_iter()
                .map(|path| (path.into(), Arc::<[u8]>::from(b"".as_slice())))
                .collect(),
            ..Default::default()
        });
        let result = resin_codegen::generate(
            Arc::new(VerifiedModule::new(module()).unwrap()),
            Some("main".into()),
            headers,
            parent.path(),
            &resin_executor::Execution::default(),
            &resin_executor::Cancellation::new(),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(fs::read_dir(parent.path()).unwrap().count(), 0);
    }
}
