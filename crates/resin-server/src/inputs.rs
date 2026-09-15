//! Canonical immutable source graphs from uploads and already-frozen managed inputs.
use crate::{Server, headers, http::failure, publication};
use resin_executor::Cancellation;
use resin_protocol::{ErrorCode, Failure, InputHandle, InputSelection, Inputs};
use resin_source::{Source, SourceError, SourceGraph};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, atomic::Ordering},
};

pub(crate) struct Frozen {
    pub input: InputHandle,
    pub graph: SourceGraph,
    pub syntax: BTreeMap<Source, Arc<resin_cst::Document>>,
    pub acquisition: Vec<SourceError>,
    pub headers: Arc<resin_codegen::NativeHeaders>,
}

pub(crate) async fn capture(
    server: &Server,
    selection: InputSelection,
    cancellation: &Cancellation,
) -> Result<Frozen, Failure> {
    let inputs = select(server, selection, cancellation).await?;
    if inputs.managed_snapshot != server.managed.snapshot() {
        return Err(failure(
            ErrorCode::ManagedSnapshotUnavailable,
            "the selected managed snapshot is unavailable; refresh capabilities and resubmit",
        ));
    }
    let (inputs, uploaded) = server
        .execution
        .run(cancellation, move |_| validate(inputs))
        .await
        .map_err(internal)??;
    let entry = uploaded
        .get(&inputs.entry)
        .or_else(|| server.managed.sources.get(&inputs.entry))
        .ok_or_else(|| {
            failure(
                ErrorCode::InvalidRequest,
                "entry source is absent from supplied inputs and managed sources",
            )
        })?
        .clone();
    let imports: BTreeMap<_, _> = inputs
        .imports
        .iter()
        .map(|binding| {
            (
                (binding.importer.clone(), binding.reference.clone()),
                binding.target.clone(),
            )
        })
        .collect();
    let mut selected = BTreeMap::from([(entry.name().to_owned(), entry.clone())]);
    let mut pending = vec![entry.clone()];
    let mut syntax = BTreeMap::new();
    let mut bindings = Vec::new();
    let mut acquisition = Vec::new();
    let mut used_imports = BTreeSet::new();
    while !pending.is_empty() {
        let values = publication::select(
            &server.caches.sources,
            selected.values().cloned().collect(),
            |source| async {
                server.caches.source_builds.fetch_add(1, Ordering::Relaxed);
                Ok::<_, resin_executor::Error>(Arc::new(source))
            },
            &server.execution,
            cancellation,
        )
        .await
        .map_err(internal)?;
        for source in selected.values_mut() {
            *source = values[source].as_ref().clone();
        }
        syntax = publication::select(
            &server.caches.syntax,
            selected.values().cloned().collect(),
            |source| {
                let retained = syntax.get(&source).cloned();
                async move {
                    if let Some(retained) = retained {
                        return Ok(retained);
                    }
                    server.caches.syntax_builds.fetch_add(1, Ordering::Relaxed);
                    resin_cst::build_cst(
                        source.text().to_owned(),
                        None,
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
        .map_err(internal)?;
        let documents: Vec<_> = pending
            .into_iter()
            .map(|source| {
                let source = selected[source.name()].clone();
                let document = syntax[&source].clone();
                (source, document)
            })
            .collect();
        let preambles = server
            .execution
            .run(cancellation, move |_| {
                documents
                    .into_iter()
                    .map(|(source, document)| (source, document.preamble()))
                    .collect::<Vec<_>>()
            })
            .await
            .map_err(internal)?;
        pending = Vec::new();
        for (source, preamble) in preambles {
            for import in preamble.imports {
                let target = if import.val.starts_with("$/") || source.name().starts_with("$/") {
                    let name = managed_reference(source.name(), &import.val);
                    name.and_then(|name| server.managed.sources.get(&name).cloned())
                } else {
                    let key = (source.name().to_owned(), import.val.to_string());
                    used_imports.insert(key.clone());
                    imports
                        .get(&key)
                        .and_then(|name| uploaded.get(name))
                        .cloned()
                };
                match target {
                    Some(target) => {
                        bindings.push(resin_source::ImportBinding {
                            source: source.id(),
                            reference: import.val,
                            target: target.id(),
                        });
                        if !selected.contains_key(target.name()) {
                            selected.insert(target.name().to_owned(), target.clone());
                            pending.push(target);
                        }
                    }
                    None => acquisition.push(SourceError::new(
                        source.clone(),
                        Some(import.span),
                        format!("unresolved source import: {}", import.val),
                    )),
                }
            }
        }
    }
    // Supplied edges must describe actual declarations. Unreachable uploads do not
    // become implicit roots, nor may they inject bindings into managed modules.
    if imports.keys().any(|key| !used_imports.contains(key)) {
        return Err(failure(
            ErrorCode::InvalidRequest,
            "an uploaded import binding is not a reachable user declaration",
        ));
    }
    for diagnostic in &inputs.acquisition_diagnostics {
        if let Some(span) = &diagnostic.span {
            if let Some(source) = selected.get(&span.source) {
                acquisition.push(SourceError::new(
                    source.clone(),
                    Some(resin_source::Span {
                        start: span.start as usize,
                        end: span.end as usize,
                    }),
                    diagnostic.message.clone(),
                ));
            }
        } else {
            acquisition.push(SourceError::new(
                entry.clone(),
                None,
                diagnostic.message.clone(),
            ));
        }
    }
    let graph = SourceGraph::new(
        selected[&inputs.entry].clone(),
        selected.values().cloned(),
        bindings,
    )
    .map_err(internal)?;
    let managed = server.managed.clone();
    let header_inputs = inputs.headers.clone();
    let header_sources = selected.clone();
    let header_syntax = syntax.clone();
    let (headers, header_errors) = server
        .execution
        .run(cancellation, move |_| {
            headers::prepare(&header_inputs, &header_sources, &header_syntax, &managed)
        })
        .await
        .map_err(internal)??;
    acquisition.extend(header_errors);
    let id = crate::token();
    publication::select(
        &server.caches.inputs,
        vec![id.clone()],
        |_| {
            let inputs = inputs.clone();
            async move { Ok::<_, resin_executor::Error>(Arc::new(inputs)) }
        },
        &server.execution,
        cancellation,
    )
    .await
    .map_err(internal)?;
    Ok(Frozen {
        input: InputHandle {
            instance: server.instance.clone(),
            id,
        },
        graph,
        syntax,
        acquisition,
        headers: Arc::new(headers),
    })
}

async fn select(
    server: &Server,
    selection: InputSelection,
    cancellation: &Cancellation,
) -> Result<Inputs, Failure> {
    match selection {
        InputSelection::Full { inputs } => Ok(inputs),
        InputSelection::Delta {
            base,
            entry,
            replacements,
            deleted,
            imports,
            headers,
            acquisition_diagnostics,
            managed_snapshot,
        } => {
            if base.instance != server.instance {
                return Err(unavailable());
            }
            let original = server
                .caches
                .inputs
                .load_full()
                .get(&base.id)
                .cloned()
                .ok_or_else(unavailable)?;
            let selected = publication::select(
                &server.caches.inputs,
                vec![base.id.clone()],
                |_| {
                    let original = original.clone();
                    async move { Ok::<_, resin_executor::Error>(original) }
                },
                &server.execution,
                cancellation,
            )
            .await
            .map_err(internal)?;
            let base = selected[&base.id].clone();
            server
                .execution
                .run(cancellation, move |_| {
                    let mut sources: BTreeMap<_, _> = base
                        .sources
                        .iter()
                        .map(|source| (source.name.clone(), source.clone()))
                        .collect();
                    let mut changed = BTreeSet::new();
                    for name in deleted {
                        if !changed.insert(name.clone()) {
                            return Err(failure(
                                ErrorCode::InvalidRequest,
                                "duplicate delta source change",
                            ));
                        }
                        sources.remove(&name);
                    }
                    for source in replacements {
                        if !changed.insert(source.name.clone()) {
                            return Err(failure(
                                ErrorCode::InvalidRequest,
                                "duplicate delta source change",
                            ));
                        }
                        sources.insert(source.name.clone(), source);
                    }
                    Ok(Inputs {
                        entry,
                        sources: sources.into_values().collect(),
                        imports,
                        headers,
                        acquisition_diagnostics,
                        managed_snapshot,
                    })
                })
                .await
                .map_err(internal)?
        }
    }
}

fn validate(mut inputs: Inputs) -> Result<(Inputs, BTreeMap<String, Source>), Failure> {
    let mut sources = BTreeMap::new();
    for file in &inputs.sources {
        if file.name.is_empty()
            || file.name.starts_with(['/', '$'])
            || file.name.chars().any(|c| c.is_control() || c == '\\')
        {
            return Err(failure(
                ErrorCode::InvalidRequest,
                "user source names must be nonempty logical names outside $/",
            ));
        }
        if sources
            .insert(
                file.name.clone(),
                Source::new(file.name.as_str(), file.text.as_str()),
            )
            .is_some()
        {
            return Err(failure(
                ErrorCode::InvalidRequest,
                "duplicate uploaded source name",
            ));
        }
    }
    let mut edges = BTreeSet::new();
    for edge in &inputs.imports {
        if !sources.contains_key(&edge.importer)
            || !sources.contains_key(&edge.target)
            || edge.reference.starts_with("$/")
            || !edges.insert((edge.importer.clone(), edge.reference.clone()))
        {
            return Err(failure(
                ErrorCode::InvalidRequest,
                "invalid or duplicate uploaded source binding",
            ));
        }
    }
    for diagnostic in &inputs.acquisition_diagnostics {
        for span in diagnostic
            .span
            .iter()
            .chain(diagnostic.related.iter().map(|note| &note.span))
        {
            let source = sources.get(&span.source).ok_or_else(|| {
                failure(
                    ErrorCode::InvalidRequest,
                    "diagnostic source is not uploaded",
                )
            })?;
            validate_span(source, span.start, span.end)?;
        }
    }
    inputs.sources.sort_by(|a, b| a.name.cmp(&b.name));
    inputs.imports.sort_by(|a, b| {
        (&a.importer, &a.reference, &a.target).cmp(&(&b.importer, &b.reference, &b.target))
    });
    Ok((inputs, sources))
}

pub(crate) fn validate_span(source: &Source, start: u64, end: u64) -> Result<(), Failure> {
    if start > end
        || end > source.text().len() as u64
        || !source.text().is_char_boundary(start as usize)
        || !source.text().is_char_boundary(end as usize)
    {
        return Err(failure(
            ErrorCode::InvalidRequest,
            "source position is outside the captured UTF-8 text",
        ));
    }
    Ok(())
}
fn managed_reference(importer: &str, reference: &str) -> Option<String> {
    let path = if let Some(path) = reference.strip_prefix("$/") {
        path.to_owned()
    } else {
        format!(
            "{}/{}",
            importer
                .strip_prefix("$/")?
                .rsplit_once('/')
                .map_or("", |(parent, _)| parent),
            reference
        )
    };
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            part => parts.push(part),
        }
    }
    Some(format!("$/{}", parts.join("/")))
}
fn unavailable() -> Failure {
    failure(
        ErrorCode::InputUnavailable,
        "prior inputs were evicted or the server restarted; submit the complete captured inputs",
    )
}
pub(crate) fn internal(error: impl std::fmt::Display) -> Failure {
    failure(ErrorCode::Internal, error.to_string())
}

pub(crate) fn acquisition_diagnostics(
    program: &mut resin_ast::BuiltProgram,
    errors: Vec<SourceError>,
) {
    for error in errors {
        if let Some(existing) = program.diagnostics.iter_mut().find(|existing| {
            existing.source == error.source
                && existing.span == error.span
                && existing.diagnostic.starts_with("unresolved source import:")
        }) {
            *existing = error;
        } else {
            program.diagnostics.push(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use resin_protocol::{
        AnalyzeRequest, BuildContract, BuildOptions, BuildProfile, BuildRequest, EntryProfile,
        EntryTarget, HeaderBinding, HeaderBundle, HeaderInputs, HeaderTarget, SourceFile,
    };
    use tempfile::TempDir;

    async fn server(directory: &TempDir, input_capacity: usize) -> Server {
        let library = directory.path().join("library");
        std::fs::create_dir_all(&library).unwrap();
        let mut environment = resin_toolchain::Environment::capture().unwrap();
        environment.directory = directory.path().to_owned();
        environment.temporary = directory.path().to_owned();
        Server::new(crate::Config {
            library_root: library,
            runtime_include: resin_runtime::INCLUDE_DIR.into(),
            dependencies: vec![],
            storage: directory.path().join("storage"),
            temporary: directory.path().into(),
            tools: environment.toolchain(None, None),
            target: crate::host_target(),
            host_backend: crate::HostBackend::C,
            capacities: crate::Capacities {
                inputs: input_capacity,
                ..Default::default()
            },
        })
        .await
        .unwrap()
    }
    fn inputs(server: &Server, text: &str) -> Inputs {
        Inputs {
            entry: "main.resin".into(),
            sources: vec![SourceFile {
                name: "main.resin".into(),
                text: text.into(),
            }],
            imports: vec![],
            headers: HeaderInputs::default(),
            acquisition_diagnostics: vec![],
            managed_snapshot: server.managed.snapshot().into(),
        }
    }
    async fn analyze(server: &Server, inputs: InputSelection) -> resin_protocol::AnalyzeResponse {
        crate::analyze::run(
            server,
            AnalyzeRequest {
                request: crate::token(),
                revision: 1,
                inputs,
                queries: vec![],
            },
            &Cancellation::new(),
        )
        .await
        .unwrap()
    }
    fn request(server: &Server, inputs: Inputs) -> BuildRequest {
        BuildRequest {
            request: crate::token(),
            revision: 4,
            inputs: InputSelection::Full { inputs },
            contract: BuildContract {
                target: server.config.target.clone(),
                entry: EntryTarget {
                    export: "main".into(),
                    profile: EntryProfile::Host,
                },
                profile: BuildProfile::Debug,
                options: BuildOptions::default(),
            },
        }
    }
    fn bundle(files: &[(&str, &str)]) -> HeaderBundle {
        let files: BTreeMap<_, _> = files.iter().copied().collect();
        let mut digest = blake3::Hasher::new();
        digest.update(b"resin-header-bundle-v1\0");
        for (path, bytes) in &files {
            for bytes in [path.as_bytes(), bytes.as_bytes()] {
                digest.update(&(bytes.len() as u64).to_le_bytes());
                digest.update(bytes);
            }
        }
        HeaderBundle {
            id: digest.finalize().to_hex().to_string(),
            files: files
                .into_iter()
                .map(|(path, text)| resin_protocol::BundleFile {
                    path: path.into(),
                    contents_base64: base64::engine::general_purpose::STANDARD.encode(text),
                })
                .collect(),
        }
    }

    #[tokio::test]
    async fn repeated_full_and_delta_submissions_share_passes_and_missing_handles_recover() {
        let directory = TempDir::new().unwrap();
        let server = server(&directory, 1).await;
        let full = inputs(&server, "export { main }; def main() -> int = { 7 };");
        let first = analyze(
            &server,
            InputSelection::Full {
                inputs: full.clone(),
            },
        )
        .await;
        assert!(first.diagnostics.is_empty());
        let counts = server.counters();
        let second = analyze(
            &server,
            InputSelection::Full {
                inputs: full.clone(),
            },
        )
        .await;
        assert_eq!(counts, server.counters());
        let delta = |base| InputSelection::Delta {
            base,
            entry: full.entry.clone(),
            replacements: vec![],
            deleted: vec![],
            imports: vec![],
            headers: full.headers.clone(),
            acquisition_diagnostics: vec![],
            managed_snapshot: full.managed_snapshot.clone(),
        };
        let error = capture(&server, delta(first.input), &Cancellation::new())
            .await
            .err()
            .unwrap();
        assert_eq!(error.code, ErrorCode::InputUnavailable);
        let third = analyze(&server, delta(second.input)).await;
        assert!(third.diagnostics.is_empty());
        assert_eq!(counts, server.counters());
        let built = crate::build::run(&server, request(&server, full), &Cancellation::new())
            .await
            .unwrap();
        assert_eq!(counts.source_builds, server.counters().source_builds);
        assert_eq!(counts.syntax_builds, server.counters().syntax_builds);
        assert_eq!(counts.ast_builds, server.counters().ast_builds);
        assert_eq!(counts.hir_builds, server.counters().hir_builds);
        assert_eq!(
            tokio::process::Command::new(built.executable.path())
                .status()
                .await
                .unwrap()
                .code(),
            Some(7)
        );
    }

    #[tokio::test]
    async fn user_sources_never_fall_back_to_coincident_server_files() {
        let directory = TempDir::new().unwrap();
        let server = server(&directory, 4).await;
        let path = directory.path().join("coincidental.resin");
        std::fs::write(&path, "export { answer }; def answer() -> int = { 1 };").unwrap();
        let text = format!(
            "import {{ {:?} }}; def main() -> int = {{ answer() }};",
            path.to_str().unwrap()
        );
        let response = analyze(
            &server,
            InputSelection::Full {
                inputs: inputs(&server, &text),
            },
        )
        .await;
        assert!(
            response
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("unresolved source import"))
        );
        assert_eq!(server.counters().source_builds, 1);
    }

    #[tokio::test]
    async fn source_scoped_headers_preserve_duplicate_basenames_and_nested_binary_safe_files() {
        let directory = TempDir::new().unwrap();
        let server = server(&directory, 8).await;
        let mut full = inputs(
            &server,
            "export { main }; import { \"left/module.resin\", \"right/module.resin\" }; def main() -> int = { left() + right() };",
        );
        for (side, number) in [("left", 7), ("right", 8)] {
            let name = format!("{side}/module.resin");
            full.sources.push(SourceFile { name: name.clone(), text: format!("export {{ {side} }}; extern {{ \"same.h\": {{ def native_{side}() -> int; }} }}; def {side}() -> int = {{ native_{side}() }};") });
            full.imports.push(resin_protocol::ImportBinding {
                importer: "main.resin".into(),
                reference: name.clone(),
                target: name.clone(),
            });
            let header = format!(
                "#include \"nested/{side}.inc\"\nstatic inline int native_{side}(void) {{ return {side}_value; }}\n"
            );
            let nested = format!("enum {{ {side}_value = {number} }};\n");
            let bundle = bundle(&[
                ("same.h", &header),
                (&format!("nested/{side}.inc"), &nested),
            ]);
            full.headers.bindings.push(HeaderBinding {
                source: name,
                spelling: "same.h".into(),
                target: HeaderTarget::Uploaded {
                    bundle: bundle.id.clone(),
                    path: "same.h".into(),
                },
            });
            full.headers.bundles.push(bundle);
        }
        let frozen = capture(
            &server,
            InputSelection::Full {
                inputs: full.clone(),
            },
            &Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(frozen.headers.bindings.len(), 2);
        assert_ne!(
            frozen.headers.bindings.values().next(),
            frozen.headers.bindings.values().next_back()
        );
        let artifact = crate::build::run(
            &server,
            request(&server, full.clone()),
            &Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(
            tokio::process::Command::new(artifact.executable.path())
                .status()
                .await
                .unwrap()
                .code(),
            Some(15)
        );
        let counts = server.counters();
        // Even an unused file addition changes whole-bundle generation identity.
        let original = full.headers.bundles[0].clone();
        let mut files: Vec<_> = original
            .files
            .iter()
            .map(|file| {
                (
                    file.path.clone(),
                    String::from_utf8(
                        base64::engine::general_purpose::STANDARD
                            .decode(&file.contents_base64)
                            .unwrap(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        files.push(("unused.data".into(), "unused\n".into()));
        let replacement = bundle(
            &files
                .iter()
                .map(|(path, text)| (path.as_str(), text.as_str()))
                .collect::<Vec<_>>(),
        );
        for binding in &mut full.headers.bindings {
            if let HeaderTarget::Uploaded { bundle, .. } = &mut binding.target
                && bundle == &original.id
            {
                *bundle = replacement.id.clone();
            }
        }
        full.headers.bundles[0] = replacement;
        let next = crate::build::run(&server, request(&server, full), &Cancellation::new())
            .await
            .unwrap();
        assert_eq!(server.counters().hir_builds, counts.hir_builds);
        assert_eq!(
            server.counters().generated_builds,
            counts.generated_builds + 1
        );
        assert_eq!(
            tokio::process::Command::new(next.executable.path())
                .status()
                .await
                .unwrap()
                .code(),
            Some(15)
        );
    }

    #[tokio::test]
    async fn uploaded_runtime_basename_does_not_replace_the_compiler_abi() {
        let directory = TempDir::new().unwrap();
        let server = server(&directory, 4).await;
        let mut full = inputs(&server, "export { main }; def main() -> int = { 9 };");
        let shadow = bundle(&[(
            "resin_runtime.h",
            "#error user include root replaced compiler runtime\n",
        )]);
        full.headers
            .include_roots
            .push(resin_protocol::IncludeRoot::Uploaded {
                bundle: shadow.id.clone(),
                directory: "".into(),
            });
        full.headers.bundles.push(shadow);
        let artifact = crate::build::run(&server, request(&server, full), &Cancellation::new())
            .await
            .unwrap();
        assert_eq!(
            tokio::process::Command::new(artifact.executable.path())
                .status()
                .await
                .unwrap()
                .code(),
            Some(9)
        );
    }

    #[tokio::test]
    async fn malformed_bundles_and_conflicting_or_forged_bindings_are_rejected() {
        let directory = TempDir::new().unwrap();
        let server = server(&directory, 4).await;
        for path in [
            "../outside.h",
            "/absolute.h",
            "C:/drive.h",
            "nested\\escape.h",
            "a./trailing.h",
            "NUL.h",
            "folder/CoM1.data",
            "conout$",
        ] {
            let mut full = inputs(&server, "def main() = {};");
            full.headers.bundles.push(bundle(&[(path, "")]));
            assert_eq!(
                capture(
                    &server,
                    InputSelection::Full { inputs: full },
                    &Cancellation::new()
                )
                .await
                .err()
                .unwrap()
                .code,
                ErrorCode::InvalidRequest
            );
        }
        for files in [
            [("A/x.h", ""), ("a/y.h", "")],
            [("same.h", ""), ("SAME.h", "")],
        ] {
            let mut full = inputs(&server, "def main() = {};");
            full.headers.bundles.push(bundle(&files));
            assert_eq!(
                capture(
                    &server,
                    InputSelection::Full { inputs: full },
                    &Cancellation::new()
                )
                .await
                .err()
                .unwrap()
                .code,
                ErrorCode::InvalidRequest
            );
        }
        let mut full = inputs(&server, "extern { \"empty.h\": {} }; def main() = {};");
        let header = bundle(&[("empty.h", "")]);
        full.headers.bundles.push(header.clone());
        full.headers.bindings.push(HeaderBinding {
            source: "main.resin".into(),
            spelling: "forged.h".into(),
            target: HeaderTarget::Uploaded {
                bundle: header.id,
                path: "empty.h".into(),
            },
        });
        assert_eq!(
            capture(
                &server,
                InputSelection::Full { inputs: full },
                &Cancellation::new()
            )
            .await
            .err()
            .unwrap()
            .code,
            ErrorCode::InvalidRequest
        );
    }
}
