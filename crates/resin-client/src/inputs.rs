//! Capture a complete user source selection; semantic analysis belongs to the server.
use crate::headers;
use futures::{StreamExt, TryStreamExt, stream};
use resin_executor::{Cancellation, Execution};
use resin_protocol::{Capabilities, Diagnostic, InputHandle, InputSelection, Inputs, Severity};
use resin_source::{Loader, Source, SourceId};
use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub(crate) struct CaptureOptions {
    pub include_roots: Vec<PathBuf>,
    pub capabilities: Arc<Capabilities>,
}

#[derive(Clone)]
pub(crate) struct CapturedInputs {
    pub inputs: Inputs,
    pub origins: BTreeMap<String, LocalSource>,
}

#[derive(Clone)]
pub(crate) struct LocalSource {
    pub source: Source,
    pub path: PathBuf,
}

struct AcquisitionError {
    source: SourceId,
    span: resin_source::Span,
    message: String,
}

/// Capture each selected physical source once, with supplied buffers taking precedence.
/// Local paths remain in `origins`; only canonical user names enter the wire graph.
pub(crate) async fn capture(
    entry: Source,
    loader: &mut Loader,
    options: &CaptureOptions,
    execution: &Execution,
    cancellation: &Cancellation,
) -> crate::Result<CapturedInputs> {
    cancellation.check()?;
    let entry_path = registered_path(loader, &entry)?;
    let root = entry_path
        .parent()
        .ok_or("source entry has no parent directory")?
        .to_path_buf();
    let mut selected = BTreeMap::from([(entry.id(), entry.clone())]);
    let mut paths = BTreeMap::from([(entry_path, entry.clone())]);
    let mut pending = vec![entry.clone()];
    let mut bindings = BTreeMap::new();
    let mut declarations = Vec::new();
    let mut errors = Vec::new();
    while !pending.is_empty() {
        let preambles = preambles(std::mem::take(&mut pending), execution, cancellation).await?;
        for (source, preamble) in preambles {
            declarations.extend(
                preamble
                    .headers
                    .into_iter()
                    .map(|header| (source.id(), header)),
            );
            for import in preamble.imports {
                if import.val.starts_with("$/") {
                    continue;
                }
                match imported(
                    &source,
                    &import.val,
                    loader,
                    &paths,
                    execution,
                    cancellation,
                )
                .await
                {
                    Ok(target) => {
                        bindings.insert((source.id(), import.val.to_string()), target.id());
                        if let std::collections::btree_map::Entry::Vacant(slot) =
                            selected.entry(target.id())
                        {
                            paths.insert(registered_path(loader, &target)?, target.clone());
                            slot.insert(target.clone());
                            pending.push(target);
                        }
                    }
                    Err(resin_source::LoadError::Io { error }) => errors.push(AcquisitionError {
                        source: source.id(),
                        span: import.span,
                        message: format!(
                            "could not load import {:?}: {}",
                            import.val,
                            io_reason(error.kind())
                        ),
                    }),
                    Err(error) => return Err(error.into()),
                }
            }
        }
    }
    let logical = loader
        .logical_user_sources(selected.values().cloned(), &root, execution, cancellation)
        .await?;
    let origins: BTreeMap<_, _> = selected
        .values()
        .map(|source| {
            Ok((
                logical[&source.id()].name().to_owned(),
                LocalSource {
                    source: source.clone(),
                    path: registered_path(loader, source)?,
                },
            ))
        })
        .collect::<crate::Result<_>>()?;
    if origins.len() != selected.len() || origins.keys().any(|name| name.starts_with("$/")) {
        return Err("user source identities are ambiguous or enter the managed namespace".into());
    }
    let declarations = declarations
        .into_iter()
        .map(|(source, header)| {
            let name = logical[&source].name().to_owned();
            headers::Declaration {
                source: name.clone(),
                path: origins[&name].path.clone(),
                spelling: header.val.to_string(),
                span: wire_span(name, header.span),
            }
        })
        .collect();
    let (headers, header_diagnostics) =
        headers::capture(declarations, options, execution, cancellation).await?;
    let mut acquisition_diagnostics: Vec<_> = errors
        .into_iter()
        .map(|error| Diagnostic {
            code: None,
            notes: Vec::new(),
            help: None,
            severity: Severity::Error,
            message: error.message,
            span: Some(wire_span(
                logical[&error.source].name().to_owned(),
                error.span,
            )),
            related: Vec::new(),
        })
        .collect();
    acquisition_diagnostics.extend(header_diagnostics);
    let mut imports: Vec<_> = bindings
        .into_iter()
        .map(
            |((importer, reference), target)| resin_protocol::ImportBinding {
                importer: logical[&importer].name().to_owned(),
                reference,
                target: logical[&target].name().to_owned(),
            },
        )
        .collect();
    imports.sort_by(|left, right| {
        (&left.importer, &left.reference, &left.target).cmp(&(
            &right.importer,
            &right.reference,
            &right.target,
        ))
    });
    let sources = origins
        .keys()
        .map(|name| resin_protocol::SourceFile {
            name: name.clone(),
            text: origins[name].source.text().to_owned(),
        })
        .collect();
    Ok(CapturedInputs {
        inputs: Inputs {
            entry: logical[&entry.id()].name().to_owned(),
            sources,
            imports,
            headers,
            acquisition_diagnostics,
            managed_snapshot: options.capabilities.managed_snapshot.clone(),
        },
        origins,
    })
}

async fn preambles(
    sources: Vec<Source>,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<Vec<(Source, resin_cst::Preamble)>, resin_executor::Error> {
    let mut parsed: Vec<_> = stream::iter(sources)
        .map(|source| async move {
            let document =
                resin_cst::build_cst(source.text().to_owned(), None, execution, cancellation)
                    .await?;
            execution
                .run(cancellation, move |_| (source, document.preamble()))
                .await
        })
        .buffer_unordered(execution.jobs())
        .try_collect()
        .await?;
    parsed.sort_by_key(|(source, _)| source.id());
    Ok(parsed)
}

async fn imported(
    source: &Source,
    reference: &str,
    loader: &mut Loader,
    selected: &BTreeMap<PathBuf, Source>,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<Source, resin_source::LoadError> {
    if reference.starts_with('$') {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "unknown import namespace").into());
    }
    let path = loader.path(source).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "source has no registered path")
    })?;
    let path = path.parent().unwrap_or(Path::new(".")).join(reference);
    let normalized = execution
        .run(cancellation, move |_| resin_source::normalize_path(&path))
        .await??;
    if let Some(source) = selected.get(&normalized) {
        return Ok(source.clone());
    }
    loader
        .load_import_async(source, reference, execution, cancellation)
        .await
}

fn registered_path(loader: &Loader, source: &Source) -> crate::Result<PathBuf> {
    loader
        .path(source)
        .map(Path::to_path_buf)
        .ok_or_else(|| "client source acquisition requires registered local file sources".into())
}

fn wire_span(source: String, span: resin_source::Span) -> resin_protocol::Span {
    resin_protocol::Span {
        source,
        start: span.start as u64,
        end: span.end as u64,
    }
}

pub(super) fn io_reason(kind: io::ErrorKind) -> &'static str {
    match kind {
        io::ErrorKind::NotFound => "file not found",
        io::ErrorKind::PermissionDenied => "permission denied",
        io::ErrorKind::InvalidInput => "invalid local reference",
        io::ErrorKind::InvalidData => "invalid file contents",
        _ => "local input could not be read",
    }
}

/// A delta changes only source membership; all other snapshot metadata is complete.
pub(crate) fn selection(
    inputs: &Inputs,
    previous: Option<(&InputHandle, &Inputs)>,
) -> InputSelection {
    let Some((handle, previous)) = previous else {
        return InputSelection::Full {
            inputs: inputs.clone(),
        };
    };
    let old: BTreeMap<_, _> = previous
        .sources
        .iter()
        .map(|source| (&source.name, &source.text))
        .collect();
    let current: BTreeMap<_, _> = inputs
        .sources
        .iter()
        .map(|source| (&source.name, source))
        .collect();
    InputSelection::Delta {
        base: handle.clone(),
        entry: inputs.entry.clone(),
        replacements: current
            .values()
            .filter(|source| {
                old.get(&source.name)
                    .is_none_or(|text| **text != source.text)
            })
            .map(|source| (*source).clone())
            .collect(),
        deleted: old
            .keys()
            .filter(|name| !current.contains_key(*name))
            .map(|name| (*name).clone())
            .collect(),
        imports: inputs.imports.clone(),
        headers: inputs.headers.clone(),
        acquisition_diagnostics: inputs.acquisition_diagnostics.clone(),
        managed_snapshot: inputs.managed_snapshot.clone(),
    }
}
