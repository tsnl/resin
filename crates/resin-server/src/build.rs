//! Build requests explicitly select each immutable pass and then run native tools.
use crate::{
    OwnedArtifact, Server,
    caches::GenerationKey,
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
            let hir = hir.clone();
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
                .map_err(|error| lir_failure(error, &hir))?;
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
    .map_err(publication_failure)?
    .remove(&key)
    .expect("selected verified LIR");
    let generation_key = GenerationKey {
        lir: key,
        host_entry: Some(request.contract.entry.export.clone()),
        headers: frozen.headers,
        target: server.config.target.clone(),
        managed_snapshot: server.managed.snapshot().into(),
    };
    let generated = publication::select(
        &server.caches.generated,
        vec![generation_key.clone()],
        |key| {
            let verified = verified.clone();
            let hir = hir.clone();
            async move {
                server
                    .caches
                    .generated_builds
                    .fetch_add(1, Ordering::Relaxed);
                resin_codegen::generate(
                    verified,
                    key.host_entry,
                    key.headers,
                    &server.config.temporary,
                    &server.execution,
                    cancellation,
                )
                .await
                .map(Arc::new)
                .map_err(|error| generation_failure(error, &hir))
            }
        },
        &server.execution,
        cancellation,
    )
    .await
    .map_err(publication_failure)?
    .remove(&generation_key)
    .expect("selected generated project");
    let profile = match request.contract.profile {
        BuildProfile::Debug => resin_toolchain::CProfile::Debug,
        BuildProfile::Release => resin_toolchain::CProfile::Release,
    };
    let built = server
        .config
        .tools
        .build(
            generated.directory(),
            generated.name(),
            &request.contract.entry.export,
            profile,
            &server.execution,
            cancellation,
        )
        .await
        .map_err(|error| failure(ErrorCode::CompilationFailed, error.to_string()))?;
    let executable = built
        .executable(
            generated
                .program()
                .expect("host output")
                .file_name()
                .expect("program filename"),
        )
        .map_err(internal)?;
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

fn publication_failure(error: resin_cache::UpdateError<Failure>) -> Failure {
    match error {
        resin_cache::UpdateError::Build { error } => error,
        resin_cache::UpdateError::Cancelled => {
            failure(ErrorCode::Cancelled, "compilation cancelled")
        }
    }
}

fn lir_failure(error: resin_lir::BuildError, hir: &resin_hir::Hir) -> Failure {
    let resin_lir::BuildError::Diagnostics { errors } = error else {
        return internal(error);
    };
    let internal_error = errors
        .iter()
        .any(|error| matches!(error.kind, resin_lir::ErrorKind::InvalidHir { .. }));
    let diagnostics = errors
        .into_iter()
        .map(|error| {
            let code = match error.kind {
                resin_lir::ErrorKind::UnsupportedProfile { .. } => "unsupported-profile",
                resin_lir::ErrorKind::MonomorphLimit { .. } => "specialization-limit",
                resin_lir::ErrorKind::TypeExpansionLimit { .. }
                | resin_lir::ErrorKind::TypeSizeLimit { .. } => "type-expansion-limit",
                resin_lir::ErrorKind::InvalidHir { .. } => "invalid-hir",
                _ => "invalid-specialization",
            };
            let mut diagnostic = Diagnostic {
                code: Some(code.into()),
                severity: Severity::Error,
                message: error.kind.to_string(),
                span: error.source.map(|source| {
                    crate::analyze::location(&resin_source::SourceLocation {
                        source,
                        span: error.span,
                    })
                }),
                related: Vec::new(),
                notes: Vec::new(),
                help: None,
            };
            for application in error.applications {
                let message = format!(
                    "while specializing {}<{}> for {:?}",
                    application.function,
                    application
                        .arguments
                        .iter()
                        .map(AsRef::as_ref)
                        .collect::<Vec<&str>>()
                        .join(", "),
                    application.profile
                );
                if let Some(location) = application.location {
                    diagnostic.related.push(RelatedDiagnostic {
                        message,
                        span: crate::analyze::location(&location),
                    });
                } else {
                    diagnostic.notes.push(message);
                }
            }
            diagnostic
        })
        .collect();
    diagnostic_failure(
        if internal_error {
            ErrorCode::Internal
        } else {
            ErrorCode::CompilationFailed
        },
        diagnostics,
        hir,
    )
}

fn generation_failure(error: resin_codegen::GenerationError, hir: &resin_hir::Hir) -> Failure {
    let resin_codegen::GenerationError::Codegen { error } = error else {
        return internal(error);
    };
    let (code, diagnostic_code) = match error.kind() {
        resin_codegen::ErrorKind::UnsupportedTarget => {
            (ErrorCode::CompilationFailed, "unsupported-target-feature")
        }
        resin_codegen::ErrorKind::InvalidProgram => (ErrorCode::CompilationFailed, "invalid-entry"),
        resin_codegen::ErrorKind::InvalidLir => (ErrorCode::Internal, "invalid-target-input"),
        resin_codegen::ErrorKind::Io => return internal(error),
    };
    diagnostic_failure(
        code,
        vec![Diagnostic {
            code: Some(diagnostic_code.into()),
            severity: Severity::Error,
            message: error.message().into(),
            span: error.location().map(crate::analyze::location),
            related: Vec::new(),
            notes: Vec::new(),
            help: None,
        }],
        hir,
    )
}

fn diagnostic_failure(
    code: ErrorCode,
    diagnostics: Vec<Diagnostic>,
    hir: &resin_hir::Hir,
) -> Failure {
    let managed_sources = crate::analyze::managed_sources(hir, &diagnostics, &[]);
    Failure {
        code,
        message: "compilation failed".into(),
        diagnostics,
        managed_sources,
    }
}
