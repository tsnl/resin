//! Test application: acquire inputs, then explicitly cache a completed semantic pass.
use resin_ast::ModuleDocument;
use resin_cache::Cache;
use resin_executor::{Cancellation, Execution};
use resin_hir::Hir;
use resin_source::{GraphError, ImportBinding, Loader, Source, SourceGraph};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub struct Snapshot {
    pub graph: SourceGraph,
    pub documents: BTreeMap<Source, Arc<ModuleDocument>>,
}

pub async fn acquire(entry: Source, loader: &mut Loader) -> Result<Snapshot, GraphError> {
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let mut pending = vec![entry.clone()];
    let mut seen = BTreeSet::new();
    let mut documents = BTreeMap::new();
    let mut bindings = Vec::new();
    while let Some(source) = pending.pop() {
        if !seen.insert(source.clone()) {
            continue;
        }
        let syntax = Arc::new(
            resin_cst::build_cst(source.text(), None, &execution, &cancellation)
                .await
                .unwrap(),
        );
        let parsed = resin_ast::build_ast(syntax.clone(), &execution, &cancellation)
            .await
            .unwrap();
        for import in &parsed.file.imports {
            // Missing test inputs deliberately remain unbound so AST assembly diagnoses them.
            if let Ok(target) = loader
                .load_import_async(&source, &import.val, &execution, &cancellation)
                .await
            {
                bindings.push(ImportBinding {
                    source: source.id(),
                    reference: import.val.clone(),
                    target: target.id(),
                });
                pending.push(target);
            }
        }
        documents.insert(
            source.clone(),
            Arc::new(ModuleDocument {
                source,
                syntax,
                file: Arc::new(parsed.file),
                errors: parsed.errors,
            }),
        );
    }
    let graph = SourceGraph::new(entry, seen, bindings)?;
    Ok(Snapshot { graph, documents })
}

pub async fn request(
    previous: &Cache<SourceGraph, Hir>,
    entry: Source,
    loader: &mut Loader,
) -> Result<(Cache<SourceGraph, Hir>, Arc<Hir>), GraphError> {
    let Snapshot { graph, documents } = acquire(entry, loader).await?;
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let next = previous
        .update(
            [graph.clone()],
            |graph| async {
                let ast =
                    resin_ast::build_program(graph, documents.clone(), &execution, &cancellation)
                        .await?;
                Hir::build(Arc::new(ast), &execution, &cancellation)
                    .await
                    .map(Arc::new)
            },
            &execution,
            &cancellation,
        )
        .await
        .unwrap();
    let result = next.get(&graph).unwrap().clone();
    Ok((next, result))
}
