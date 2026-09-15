#![allow(dead_code)]

pub mod application;

use resin_executor::{Cancellation, Execution};
use resin_source::prelude::*;
use std::{future::Future, sync::Arc};

pub fn run<T>(work: impl Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(work)
}

pub fn syntax(text: &str) -> Arc<resin_cst::Document> {
    Arc::new(
        run(resin_cst::build_cst(
            text,
            None,
            &Execution::default(),
            &Cancellation::new(),
        ))
        .unwrap(),
    )
}

pub fn source_module(source: Source) -> resin_ast::SourceModule {
    let parsed = run(resin_ast::build_ast(
        syntax(source.text()),
        &Execution::default(),
        &Cancellation::new(),
    ))
    .unwrap();
    resin_ast::SourceModule {
        source,
        file: Arc::new(parsed.file),
        imports: vec![],
    }
}

pub fn check(program: resin_ast::Program) -> resin_hir::CheckedProgram {
    run(resin_hir::build_hir(
        Arc::new(program),
        &Execution::default(),
        &Cancellation::new(),
    ))
    .unwrap()
}

pub fn hir_module(text: &str) -> Result<resin_hir::Module, SourceError> {
    check(resin_ast::Program {
        modules: vec![source_module(Source::new("test.resin", text))],
    })
    .into_module()
}
