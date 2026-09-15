//! Capture source inputs before entering the filesystem-independent compiler.
use resin_cache::Cache;
use resin_executor::{Cancellation, Execution};
use resin_source::{ImportBinding, Loader, Source, SourceError, SourceGraph, SourceId};
use std::{collections::BTreeMap, path::Path, sync::Arc};

pub(super) struct Inputs {
    pub graph: SourceGraph,
    pub syntax: Cache<Source, resin_cst::Document>,
    pub diagnostics: Vec<SourceError>,
    /// Client presentation is deliberately outside the reusable compiler graph.
    pub origins: BTreeMap<SourceId, Source>,
}

pub(super) async fn capture(
    entry: Source,
    loader: &mut Loader,
    previous: &Cache<Source, resin_cst::Document>,
    execution: &Execution,
    cancellation: &Cancellation,
) -> super::Result<Inputs> {
    let root = loader
        .path(&entry)
        .and_then(Path::parent)
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let mut selected = BTreeMap::from([(entry.id(), entry.clone())]);
    let mut pending = vec![entry.clone()];
    let mut logical = BTreeMap::new();
    let mut origins = BTreeMap::new();
    let mut bindings = Vec::new();
    let mut diagnostics = Vec::new();
    let mut syntax = previous.clone();
    while !pending.is_empty() {
        logical.extend(
            loader
                .logical_sources(pending.clone(), &root, execution, cancellation)
                .await?,
        );
        for source in &pending {
            origins.insert(logical[&source.id()].id(), source.clone());
        }
        syntax = syntax
            .update(
                logical.values().cloned(),
                |source| async move {
                    resin_cst::build_cst(source.text().to_owned(), None, execution, cancellation)
                        .await
                        .map(Arc::new)
                },
                execution,
                cancellation,
            )
            .await?;
        let documents: Vec<_> = pending
            .into_iter()
            .map(|source| {
                let document = syntax
                    .get(&logical[&source.id()])
                    .expect("requested syntax retained")
                    .clone();
                (source, document)
            })
            .collect();
        let preambles = execution
            .run(cancellation, move |_| {
                documents
                    .into_iter()
                    .map(|(source, document)| (source, document.preamble()))
                    .collect::<Vec<_>>()
            })
            .await?;
        pending = Vec::new();
        for (source, preamble) in preambles {
            for import in preamble.imports {
                match loader
                    .load_import_async(&source, &import.val, execution, cancellation)
                    .await
                {
                    Ok(imported) => {
                        let target = select_source(imported, &mut selected, &mut pending);
                        bindings.push(ImportBinding {
                            source: source.id(),
                            reference: import.val,
                            target,
                        });
                    }
                    Err(resin_source::LoadError::Io { error }) => {
                        diagnostics.push(SourceError::new(
                            logical[&source.id()].clone(),
                            Some(import.span),
                            format!("could not load {}: {error}", import.val),
                        ))
                    }
                    Err(error) => return Err(error.into()),
                }
            }
        }
    }
    let bindings = bindings.into_iter().map(|binding| ImportBinding {
        source: logical[&binding.source].id(),
        reference: binding.reference,
        target: logical[&binding.target].id(),
    });
    let graph = SourceGraph::new(
        logical[&entry.id()].clone(),
        logical.values().cloned(),
        bindings,
    )?;
    Ok(Inputs {
        graph,
        syntax,
        diagnostics,
        origins,
    })
}

fn select_source(
    source: Source,
    selected: &mut BTreeMap<SourceId, Source>,
    pending: &mut Vec<Source>,
) -> SourceId {
    let id = source.id();
    // One request selects one version of each logical source, even if disk changes
    // while traversing a diamond-shaped import graph.
    if let std::collections::btree_map::Entry::Vacant(entry) = selected.entry(id.clone()) {
        entry.insert(source.clone());
        pending.push(source);
    }
    id
}

/// Replace the unresolved-binding diagnostic with the acquisition failure at that
/// import. Keep parser and semantic diagnostics at other locations intact.
pub(super) fn acquisition_diagnostics(
    program: &mut resin_ast::BuiltProgram,
    errors: Vec<SourceError>,
) {
    for mut error in errors {
        if let Some(existing) = program.diagnostics.iter_mut().find(|existing| {
            existing.source == error.source
                && existing.span == error.span
                && existing.diagnostic.starts_with("unresolved source import:")
        }) {
            let prefix = existing
                .message
                .strip_suffix(&existing.diagnostic)
                .unwrap_or("");
            error.message = format!("{prefix}{}", error.diagnostic).into();
            error.related.extend(existing.related.iter().cloned());
            *existing = error;
        } else {
            program.diagnostics.push(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn independent_checkouts_reuse_parsing_with_separate_local_origins() {
        let directory = tempfile::TempDir::new().unwrap();
        let execution = Execution::default();
        let cancellation = Cancellation::new();
        let mut previous = Cache::new(16);
        let mut snapshots = Vec::new();
        for name in ["left", "right"] {
            let root = directory.path().join(name);
            tokio::fs::create_dir(&root).await.unwrap();
            tokio::fs::write(
                root.join("main.resin"),
                "import { \"value.resin\" }; def main() -> int = { value() };",
            )
            .await
            .unwrap();
            tokio::fs::write(
                root.join("value.resin"),
                "export { value }; def value() -> int = { 42 };",
            )
            .await
            .unwrap();
            let mut loader = Loader::new(directory.path().join("builtin"));
            let source = loader
                .load_file_async(&root.join("main.resin"), &execution, &cancellation)
                .await
                .unwrap();
            let inputs = capture(source, &mut loader, &previous, &execution, &cancellation)
                .await
                .unwrap();
            assert!(inputs.diagnostics.is_empty());
            previous = inputs.syntax.clone();
            snapshots.push(inputs);
        }
        let (left, right) = (&snapshots[0], &snapshots[1]);
        assert_eq!(left.graph, right.graph);
        for source in left.graph.sources() {
            assert!(Arc::ptr_eq(
                left.syntax.get(source).unwrap(),
                right.syntax.get(source).unwrap()
            ));
            assert_ne!(left.origins[&source.id()], right.origins[&source.id()]);
            assert!(!source.name().starts_with('/'));
        }
    }
}
