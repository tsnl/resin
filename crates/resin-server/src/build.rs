//! Build requests explicitly select each immutable pass and then run native tools.
use crate::{
    OwnedArtifact, Server,
    caches::{ForeignKey, LinkKey, NativeKey, ShaderKey, ToolBytesKey},
    http::failure,
    inputs::{self, internal},
    publication,
};
use resin_executor::Cancellation;
use resin_protocol::*;
use std::{
    num::NonZeroUsize,
    sync::{Arc, atomic::Ordering},
};

pub(crate) async fn run(
    server: &Server,
    request: BuildRequest,
    cancellation: &Cancellation,
) -> Result<OwnedArtifact, Failure> {
    if request.contract.target != server.config.target
        || request.contract.entry.profile != EntryProfile::Host
    {
        return Err(failure(
            ErrorCode::UnsupportedTarget,
            "the server does not support this target or entry profile",
        ));
    }
    let max_monomorphs_per_function =
        usize::try_from(request.contract.options.max_instances_per_function)
            .ok()
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| {
                failure(
                    ErrorCode::InvalidRequest,
                    "instance allowance must be a positive platform-sized integer",
                )
            })?;
    let frozen = inputs::capture(server, request.inputs, cancellation).await?;
    let documents = publication::select(
        &server.caches.ast,
        frozen.graph.sources().cloned().collect(),
        |source| {
            let syntax = frozen.syntax[&source].clone();
            async move {
                server.caches.ast_builds.fetch_add(1, Ordering::Relaxed);
                let parsed =
                    resin_ast::build_ast(syntax.clone(), &server.execution, cancellation).await?;
                Ok::<_, resin_executor::Error>(Arc::new(resin_ast::ModuleDocument {
                    source,
                    syntax,
                    file: Arc::new(parsed.file),
                    errors: parsed.errors,
                }))
            }
        },
        &server.execution,
        cancellation,
    )
    .await
    .map_err(internal)?;
    let graph = frozen.graph;
    let mut program =
        resin_ast::build_program(graph.clone(), documents, &server.execution, cancellation)
            .await
            .map_err(internal)?;
    let hir = if frozen.acquisition.is_empty() {
        let program = Arc::new(program);
        publication::select(
            &server.caches.hir,
            vec![graph.clone()],
            |_| {
                let program = program.clone();
                async move {
                    server.caches.hir_builds.fetch_add(1, Ordering::Relaxed);
                    resin_hir::Hir::build(program, &server.execution, cancellation)
                        .await
                        .map(Arc::new)
                }
            },
            &server.execution,
            cancellation,
        )
        .await
        .map_err(internal)?
        .remove(&graph)
        .expect("selected HIR")
    } else {
        inputs::acquisition_diagnostics(&mut program, frozen.acquisition);
        Arc::new(
            resin_hir::Hir::build(Arc::new(program), &server.execution, cancellation)
                .await
                .map_err(internal)?,
        )
    };
    if hir.hir().is_err() {
        return server
            .execution
            .run(cancellation, move |_| {
                let diagnostics = crate::analyze::diagnostics(&hir);
                let managed_sources = crate::analyze::managed_sources(&hir, &diagnostics, &[]);
                Failure {
                    code: ErrorCode::CompilationFailed,
                    message: "source analysis failed".into(),
                    diagnostics,
                    managed_sources,
                }
            })
            .await
            .map_err(internal)
            .and_then(Err);
    }
    let module = hir.hir().map_err(internal)?.clone();
    let entry = resin_lir::Entry::exported(
        &module,
        request.contract.entry.export.clone(),
        resin_lir::Profile::Host,
    )
    .map_err(|error| failure(ErrorCode::CompilationFailed, error.to_string()))?;
    let key = resin_lir::LirKey::new(
        Arc::new(graph),
        [entry],
        resin_lir::LoweringOptions {
            max_monomorphs_per_function,
        },
    );
    let verified = publication::select(
        &server.caches.verified,
        vec![key.clone()],
        |key| {
            let module = module.clone();
            async move {
                server
                    .caches
                    .verified_builds
                    .fetch_add(1, Ordering::Relaxed);
                let lir = resin_lir::build_lir(
                    module,
                    key.entries().to_vec(),
                    key.options().clone(),
                    &server.execution,
                    cancellation,
                )
                .await
                .map_err(internal)?;
                resin_lir::VerifiedModule::build(lir, &server.execution, cancellation)
                    .await
                    .map(Arc::new)
                    .map_err(internal)
            }
        },
        &server.execution,
        cancellation,
    )
    .await
    .map_err(internal)?
    .remove(&key)
    .expect("selected verified LIR");
    let executable = build_native(
        server,
        verified,
        key,
        frozen.headers,
        &request.contract,
        cancellation,
    )
    .await?;
    let path = executable.path().to_owned();
    let (length, digest) = server
        .execution
        .run(cancellation, move |cancellation| {
            use std::io::Read;
            let mut file = std::fs::File::open(path)?;
            let mut digest = blake3::Hasher::new();
            let mut length = 0;
            let mut buffer = [0; 65536];
            loop {
                cancellation.check().map_err(std::io::Error::other)?;
                let read = file.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                length += read as u64;
                digest.update(&buffer[..read]);
            }
            Ok::<_, std::io::Error>((length, digest.finalize().to_hex().to_string()))
        })
        .await
        .map_err(internal)?
        .map_err(internal)?;
    Ok(OwnedArtifact {
        metadata: BuildMetadata {
            revision: request.revision,
            input: frozen.input,
            artifact: ArtifactMetadata {
                name: format!("program{}", std::env::consts::EXE_SUFFIX),
                kind: ArtifactKind::Executable,
                target: request.contract.target,
                length,
                blake3: digest,
                executable: true,
            },
        },
        executable,
    })
}

async fn build_native(
    server: &Server,
    verified: Arc<resin_lir::VerifiedModule>,
    lir: resin_lir::LirKey,
    headers: Arc<crate::headers::NativeHeaders>,
    contract: &BuildContract,
    cancellation: &Cancellation,
) -> Result<resin_toolchain::Executable, Failure> {
    // Validate every shader before invoking any optimizer, compiler or linker.
    let shader_keys = verified
        .view()
        .module()
        .shaders
        .iter()
        .filter(|(_, shader)| shader.embedded)
        .map(|(&function, _)| ShaderKey {
            lir: lir.clone(),
            function,
        })
        .collect();
    let shaders = publication::select(
        &server.caches.shaders,
        shader_keys,
        |key| {
            let verified = verified.clone();
            async move {
                server.caches.shader_builds.fetch_add(1, Ordering::Relaxed);
                resin_codegen::generate_spirv(
                    verified,
                    key.function,
                    &server.execution,
                    cancellation,
                )
                .await
                .map(Arc::new)
            }
        },
        &server.execution,
        cancellation,
    )
    .await
    .map_err(compilation)?;
    let shader_tools = if shaders.is_empty() {
        String::new()
    } else {
        server
            .config
            .tools
            .fingerprint(
                resin_toolchain::NativeOperation::Shader,
                &server.execution,
                cancellation,
            )
            .await
            .map_err(compilation)?
    };
    let optimized = publication::select(
        &server.caches.optimized,
        shaders
            .values()
            .map(|bytes| ToolBytesKey {
                bytes: bytes.as_ref().clone(),
                tools: shader_tools.clone(),
            })
            .collect(),
        |key| async move {
            server
                .caches
                .shader_optimizations
                .fetch_add(1, Ordering::Relaxed);
            server
                .config
                .tools
                .optimize_shader(
                    key.bytes,
                    &server.config.temporary,
                    &server.execution,
                    cancellation,
                )
                .await
                .map(Arc::new)
        },
        &server.execution,
        cancellation,
    )
    .await
    .map_err(compilation)?;
    let crate::foreign::Prepared {
        inputs: foreign_inputs,
        bindings: foreign,
    } = crate::foreign::prepare(verified.view().module(), &headers)?;
    let foreign_object =
        if foreign_inputs.functions.is_empty() && foreign_inputs.includes.is_empty() {
            None
        } else {
            let key = ForeignKey {
                inputs: foreign_inputs,
                tools: server
                    .config
                    .tools
                    .fingerprint(
                        resin_toolchain::NativeOperation::Foreign,
                        &server.execution,
                        cancellation,
                    )
                    .await
                    .map_err(compilation)?,
            };
            Some(
                publication::select(
                    &server.caches.foreign,
                    vec![key.clone()],
                    |key| async move {
                        server.caches.foreign_builds.fetch_add(1, Ordering::Relaxed);
                        server
                            .config
                            .tools
                            .compile_foreign(
                                key.inputs,
                                &server.config.temporary,
                                &server.execution,
                                cancellation,
                            )
                            .await
                            .map(Arc::new)
                    },
                    &server.execution,
                    cancellation,
                )
                .await
                .map_err(compilation)?
                .remove(&key)
                .expect("selected foreign object"),
            )
        };
    let inputs = Arc::new(resin_codegen::NativeInputs {
        foreign,
        shaders: shaders
            .into_iter()
            .map(|(key, bytes)| {
                let optimized = optimized[&ToolBytesKey {
                    bytes: bytes.as_ref().clone(),
                    tools: shader_tools.clone(),
                }]
                    .as_ref()
                    .clone();
                (key.function, optimized)
            })
            .collect(),
        runtime: Default::default(),
    });
    let key = NativeKey {
        lir,
        entry: contract.entry.export.clone(),
        inputs,
        optimization: match contract.profile {
            BuildProfile::Debug => resin_codegen::NativeOptimization::None,
            BuildProfile::Release => resin_codegen::NativeOptimization::Speed,
        },
        target: contract.target.clone(),
    };
    let object = publication::select(
        &server.caches.native,
        vec![key.clone()],
        |key| {
            let verified = verified.clone();
            async move {
                server
                    .caches
                    .native_object_builds
                    .fetch_add(1, Ordering::Relaxed);
                resin_codegen::generate_native(
                    verified,
                    key.entry,
                    key.optimization,
                    key.inputs,
                    &server.execution,
                    cancellation,
                )
                .await
                .map(Arc::new)
            }
        },
        &server.execution,
        cancellation,
    )
    .await
    .map_err(compilation)?
    .remove(&key)
    .expect("selected native object");
    let mut objects = vec![object.shared_bytes()];
    if let Some(foreign) = foreign_object {
        objects.push(foreign.bytes());
    }
    let tools = server
        .config
        .tools
        .fingerprint(
            resin_toolchain::NativeOperation::Link { runtime: true },
            &server.execution,
            cancellation,
        )
        .await
        .map_err(compilation)?;
    link(
        server,
        LinkKey {
            generation: 0,
            objects,
            tools,
        },
        cancellation,
    )
    .await
}

async fn link(
    server: &Server,
    mut key: LinkKey,
    cancellation: &Cancellation,
) -> Result<resin_toolchain::Executable, Failure> {
    // External links for the same immutable inputs share one in-flight operation.
    // Waiting retains no executor permit; unrelated object keys proceed independently.
    let coordination = link_coordination(server, &key)?;
    let _linking = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(compilation(resin_executor::Error::Cancelled)),
        guard = coordination.lock() => guard,
    };
    // Recheck the cache after acquiring the key, including its current owned file.
    // An administrator may remove temporary outputs. Rebuild a missing generation
    // from retained object bytes, without mutating any existing executable handle.
    let previous = server
        .caches
        .executables
        .load_full()
        .iter()
        .filter(|(old, _)| old.objects == key.objects && old.tools == key.tools)
        .max_by_key(|(old, _)| old.generation)
        .map(|(old, output)| (old.generation, output.clone()));
    if let Some((generation, output)) = previous {
        key.generation = generation;
        if tokio::fs::metadata(output.path()).await.is_err() {
            key.generation = generation
                .checked_add(1)
                .ok_or_else(|| compilation("executable generation overflow"))?;
        }
    }
    let executable = publication::select(
        &server.caches.executables,
        vec![key.clone()],
        |key| async move {
            server
                .caches
                .executable_builds
                .fetch_add(1, Ordering::Relaxed);
            server
                .config
                .tools
                .link_native(
                    resin_toolchain::NativeLink {
                        objects: key.objects,
                        runtime: true,
                    },
                    &server.config.temporary,
                    &server.execution,
                    cancellation,
                )
                .await
                .map(Arc::new)
        },
        &server.execution,
        cancellation,
    )
    .await
    .map_err(compilation)?
    .remove(&key)
    .expect("selected executable");
    Ok(executable.as_ref().clone())
}

fn link_coordination(
    server: &Server,
    key: &LinkKey,
) -> Result<Arc<tokio::sync::Mutex<()>>, Failure> {
    let mut links = server.links.lock().map_err(internal)?;
    links.retain(|_, waiting| waiting.strong_count() != 0);
    if let Some(waiting) = links.get(key).and_then(std::sync::Weak::upgrade) {
        return Ok(waiting);
    }
    let waiting = Arc::new(tokio::sync::Mutex::new(()));
    links.insert(key.clone(), Arc::downgrade(&waiting));
    Ok(waiting)
}

fn compilation(error: impl std::fmt::Display) -> Failure {
    failure(ErrorCode::CompilationFailed, error.to_string())
}
