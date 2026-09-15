//! Acquire the benchmark's immutable source graph before running compiler passes.
use resin_ast::ModuleDocument;
use resin_executor::{Cancellation, Execution};
use resin_source::{ImportBinding, Loader, Source, SourceGraph};
use std::{collections::BTreeMap, path::Path, sync::Arc};

pub(super) async fn capture(
    entry: Source,
    loader: &mut Loader,
    execution: &Execution,
    cancellation: &Cancellation,
) -> super::Result<resin_ast::BuiltProgram> {
    let root = loader
        .path(&entry)
        .and_then(Path::parent)
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let mut selected = BTreeMap::from([(entry.id(), entry.clone())]);
    let mut pending = vec![entry.clone()];
    let mut logical = BTreeMap::new();
    let mut documents = BTreeMap::new();
    let mut bindings = Vec::new();
    while let Some(source) = pending.pop() {
        logical.extend(
            loader
                .logical_sources([source.clone()], &root, execution, cancellation)
                .await?,
        );
        let canonical = logical[&source.id()].clone();
        let syntax =
            Arc::new(resin_cst::build_cst(canonical.text(), None, execution, cancellation).await?);
        let parsed = resin_ast::build_ast(syntax.clone(), execution, cancellation).await?;
        for import in &parsed.file.imports {
            let target = loader
                .load_import_async(&source, &import.val, execution, cancellation)
                .await?;
            bindings.push(ImportBinding {
                source: source.id(),
                reference: import.val.clone(),
                target: target.id(),
            });
            if let std::collections::btree_map::Entry::Vacant(slot) = selected.entry(target.id()) {
                slot.insert(target.clone());
                pending.push(target);
            }
        }
        documents.insert(
            canonical.clone(),
            Arc::new(ModuleDocument {
                source: canonical,
                syntax,
                file: Arc::new(parsed.file),
                errors: parsed.errors,
            }),
        );
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
    Ok(resin_ast::build_program(graph, documents, execution, cancellation).await?)
}
