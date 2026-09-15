#![allow(dead_code)]

use resin_executor::{Cancellation, Execution};
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

pub fn ast(document: Arc<resin_cst::Document>) -> resin_ast::Parsed {
    run(resin_ast::build_ast(
        document,
        &Execution::default(),
        &Cancellation::new(),
    ))
    .unwrap()
}

pub fn parse(text: &str) -> resin_ast::Parsed {
    ast(syntax(text))
}
