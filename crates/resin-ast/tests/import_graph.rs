mod common;

use resin_ast::ModuleDocument;
use resin_executor::{Cancellation, Execution};
use resin_source::{ImportBinding, Source, SourceGraph};
use std::{collections::BTreeMap, sync::Arc};

fn document(source: &Source) -> Arc<ModuleDocument> {
    let syntax = common::syntax(source.text());
    let parsed = common::ast(syntax.clone());
    Arc::new(ModuleDocument {
        source: source.clone(),
        syntax,
        file: Arc::new(parsed.file),
        errors: parsed.errors,
    })
}

#[test]
fn deep_import_chains_do_not_consume_the_worker_stack() {
    let count = 4096;
    let sources = (0..count)
        .map(|index| {
            Source::new(
                format!("{index}.resin"),
                if index + 1 == count {
                    ""
                } else {
                    "import { \"next\" };"
                },
            )
        })
        .collect::<Vec<_>>();
    let imported = document(&sources[0]);
    let mut documents = BTreeMap::new();
    for source in &sources[..count - 1] {
        documents.insert(
            source.clone(),
            Arc::new(ModuleDocument {
                source: source.clone(),
                syntax: imported.syntax.clone(),
                file: imported.file.clone(),
                errors: vec![],
            }),
        );
    }
    documents.insert(sources[count - 1].clone(), document(&sources[count - 1]));
    let bindings = sources.windows(2).map(|pair| ImportBinding {
        source: pair[0].id(),
        reference: "next".into(),
        target: pair[1].id(),
    });
    let graph = SourceGraph::new(sources[0].clone(), sources.clone(), bindings).unwrap();
    let built = common::run(resin_ast::build_program(
        graph,
        documents,
        &Execution::default(),
        &Cancellation::new(),
    ))
    .unwrap();
    assert!(built.diagnostics.is_empty());
    assert_eq!(built.program.modules.len(), count);
    for (index, module) in built.program.modules.iter().enumerate().skip(1) {
        assert_eq!(module.imports[0].1, index - 1);
    }
    assert_eq!(built.program.modules.last().unwrap().source, sources[0]);
}

#[test]
fn nested_import_errors_retain_their_import_locations() {
    let sources = [
        Source::new("entry.resin", "import { \"next\" };"),
        Source::new("middle.resin", "import { \"next\" };"),
        Source::new("broken.resin", "import { \"missing\" };"),
    ];
    let graph = SourceGraph::new(
        sources[0].clone(),
        sources.clone(),
        sources.windows(2).map(|pair| ImportBinding {
            source: pair[0].id(),
            reference: "next".into(),
            target: pair[1].id(),
        }),
    )
    .unwrap();
    let documents = sources
        .iter()
        .map(|source| (source.clone(), document(source)))
        .collect();
    let built = common::run(resin_ast::build_program(
        graph,
        documents,
        &Execution::default(),
        &Cancellation::new(),
    ))
    .unwrap();
    let error = &built.diagnostics[0];
    assert_eq!(built.diagnostics.len(), 1);
    assert_eq!(error.source, sources[2]);
    assert_eq!(error.related.len(), 2);
    assert_eq!(error.related[0].location.source, sources[1]);
    assert_eq!(error.related[1].location.source, sources[0]);
}
