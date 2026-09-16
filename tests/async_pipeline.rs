//! Compiler clients sequence immutable passes without a filesystem source loader.
use resin_executor::{Cancellation, Execution};
use resin_source::{ImportBinding, Source, SourceEdit, SourceGraph, SourceId, Span};
use std::{collections::BTreeMap, sync::Arc};

fn send_future<F: std::future::Future + Send>(future: F) -> F {
    future
}

#[tokio::test]
async fn explicit_snapshot_pipeline_retains_editor_facts_and_owned_codegen() {
    fn send_and_sync<T: Send + Sync>() {}
    send_and_sync::<Source>();
    send_and_sync::<SourceGraph>();
    send_and_sync::<resin_cst::Document>();
    send_and_sync::<resin_ast::Parsed>();
    send_and_sync::<resin_ast::ModuleDocument>();
    send_and_sync::<resin_ast::BuiltProgram>();
    send_and_sync::<resin_hir::Hir>();
    send_and_sync::<resin_hir::Module>();
    send_and_sync::<resin_lir::Module>();
    send_and_sync::<resin_lir::VerifiedModule>();
    send_and_sync::<resin_codegen::GeneratedProject>();

    let execution = Execution::new(2.try_into().unwrap());
    let cancellation = Cancellation::new();
    let main = Source::with_identity(
        SourceId::new("package/main.resin"),
        "main.resin",
        "export { main }; import { \"library.resin\" }; fn main() -> int  { answer() }",
    );
    let library = Source::with_identity(
        SourceId::new("package/library.resin"),
        "library.resin",
        "export { answer }; fn answer() -> int  { 42 }",
    );
    let binding = ImportBinding {
        source: main.id(),
        reference: "library.resin".into(),
        target: library.id(),
    };
    let graph = SourceGraph::new(main.clone(), [library.clone()], [binding.clone()]).unwrap();
    let (main_cst, library_cst) = tokio::join!(
        send_future(resin_cst::build_cst(
            main.text(),
            None,
            &execution,
            &cancellation
        )),
        resin_cst::build_cst(library.text(), None, &execution, &cancellation),
    );
    let main_cst = Arc::new(main_cst.unwrap());
    let library_cst = Arc::new(library_cst.unwrap());
    let (main_ast, library_ast) = tokio::join!(
        send_future(resin_ast::build_ast(
            main_cst.clone(),
            &execution,
            &cancellation
        )),
        resin_ast::build_ast(library_cst.clone(), &execution, &cancellation),
    );
    let mut documents = BTreeMap::new();
    for (source, syntax, parsed) in [
        (main.clone(), main_cst, main_ast.unwrap()),
        (library.clone(), library_cst, library_ast.unwrap()),
    ] {
        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        documents.insert(
            source.clone(),
            Arc::new(resin_ast::ModuleDocument {
                source,
                syntax,
                file: Arc::new(parsed.file),
                errors: parsed.errors,
            }),
        );
    }

    // A fresh client can reconstruct the exact graph without allocation identities
    // or any private server state. Existing per-file translations remain usable.
    let fresh_main = Source::with_identity(main.id(), main.name(), main.text());
    let fresh_library = Source::with_identity(library.id(), library.name(), library.text());
    let reconstructed = SourceGraph::new(
        fresh_main.clone(),
        [fresh_main.clone(), fresh_library.clone()],
        [binding],
    )
    .unwrap();
    assert_eq!(graph, reconstructed);
    let (built, rebuilt) = tokio::join!(
        send_future(resin_ast::build_program(
            graph,
            documents.clone(),
            &execution,
            &cancellation
        )),
        resin_ast::build_program(reconstructed, documents, &execution, &cancellation),
    );
    let built = Arc::new(built.unwrap());
    let rebuilt = Arc::new(rebuilt.unwrap());
    assert!(built.diagnostics.is_empty());
    assert!(rebuilt.diagnostics.is_empty());
    assert!(Arc::ptr_eq(
        &built.documents[&library],
        &rebuilt.documents[&fresh_library],
    ));
    let (hir, reconstructed_hir) = tokio::join!(
        send_future(resin_hir::Hir::build(built, &execution, &cancellation)),
        resin_hir::Hir::build(rebuilt, &execution, &cancellation),
    );
    let hir = hir.unwrap();
    let reconstructed_hir = reconstructed_hir.unwrap();
    assert!(hir.diagnostics().is_empty(), "{:?}", hir.diagnostics());
    let offset = main.text().find("answer()").unwrap();
    let original_hover = hir.hover(&main, offset).unwrap();
    let fresh_hover = hir.hover(&fresh_main, offset).unwrap();
    assert_eq!(original_hover.text, fresh_hover.text);
    assert_eq!(
        original_hover.text,
        reconstructed_hir.hover(&main, offset).unwrap().text
    );
    let definition = hir.definition(&fresh_main, offset).unwrap();
    assert_eq!(definition.source, fresh_library);
    assert_eq!(
        definition.span.start,
        library.text().find("fn answer").unwrap() + 3
    );
    let unrelated =
        Source::with_identity(SourceId::new("other/main.resin"), main.name(), main.text());
    assert!(hir.hover(&unrelated, offset).is_none());
    assert!(
        hir.hover(&main.with_text(format!("{} ", main.text())), offset)
            .is_none()
    );

    let module = hir.hir().unwrap().clone();
    let entry = resin_lir::Entry::exported(&module, "main", resin_lir::Profile::Host).unwrap();
    let lir = send_future(resin_lir::build_lir(
        module,
        vec![entry],
        Default::default(),
        &execution,
        &cancellation,
    ))
    .await
    .unwrap();
    let verified = Arc::new(
        send_future(resin_lir::VerifiedModule::build(
            lir,
            &execution,
            &cancellation,
        ))
        .await
        .unwrap(),
    );
    let parent = tempfile::tempdir().unwrap();
    let generated = send_future(resin_codegen::generate(
        verified.clone(),
        Some("main".into()),
        std::sync::Arc::new(resin_codegen::NativeHeaders::default()),
        parent.path(),
        &execution,
        &cancellation,
    ))
    .await
    .unwrap();
    drop(verified);
    drop(hir);
    let c = tokio::fs::read_to_string(generated.c_source().unwrap())
        .await
        .unwrap();
    assert!(c.contains("int main("), "{c}");
    assert!(generated.build_file().is_file());
    assert!(!generated.program().unwrap().exists());
    assert_eq!(
        reconstructed_hir.hover(&main, offset).unwrap().text,
        original_hover.text
    );
}

#[tokio::test]
async fn cold_and_incremental_successors_match_without_changing_recovery_inputs() {
    let execution = Execution::new(2.try_into().unwrap());
    let cancellation = Cancellation::new();
    let original = Source::new(
        "editor/main.resin",
        "export { main }; fn main() -> int  { 1 }",
    );
    let previous = resin_cst::build_cst(original.text(), None, &execution, &cancellation)
        .await
        .unwrap();
    let before_tree = previous.tree().root_node().to_sexp();
    let replacement = original.text().find("1 }").unwrap();
    let edited = Source::from_edits(
        original.id(),
        original.name(),
        Some(&original),
        &[SourceEdit {
            range: Span {
                start: replacement,
                end: replacement + 1,
            },
            text: "42".into(),
        }],
    )
    .unwrap();
    let cold_source = Source::with_identity(original.id(), original.name(), edited.text());
    assert_eq!(edited, cold_source);
    let incomplete = original.with_text("export { main }; fn main() -> int = { 42");
    let (complete_tree, incomplete_tree) = tokio::join!(
        resin_cst::build_cst(edited.text(), Some(&previous), &execution, &cancellation),
        resin_cst::build_cst(
            incomplete.text(),
            Some(&previous),
            &execution,
            &cancellation
        ),
    );
    for (source, incremental) in [
        (edited, complete_tree.unwrap()),
        (incomplete, incomplete_tree.unwrap()),
    ] {
        let cold = resin_cst::build_cst(source.text(), None, &execution, &cancellation)
            .await
            .unwrap();
        assert_eq!(incremental.source(), cold.source());
        assert_eq!(
            incremental.tree().root_node().to_sexp(),
            cold.tree().root_node().to_sexp()
        );
        assert_eq!(
            incremental
                .recovery()
                .map(|document| document.tree().root_node().to_sexp()),
            cold.recovery()
                .map(|document| document.tree().root_node().to_sexp()),
        );
        let incremental = Arc::new(incremental);
        let cold = Arc::new(cold);
        let (incremental_ast, cold_ast) = tokio::join!(
            resin_ast::build_ast(incremental, &execution, &cancellation),
            resin_ast::build_ast(cold, &execution, &cancellation),
        );
        let incremental_ast = incremental_ast.unwrap();
        let cold_ast = cold_ast.unwrap();
        assert_eq!(incremental_ast.errors, cold_ast.errors);
        assert_eq!(
            resin_ast::format_source(&incremental_ast.file),
            resin_ast::format_source(&cold_ast.file),
        );
    }
    assert_eq!(previous.source(), original.text());
    assert_eq!(previous.tree().root_node().to_sexp(), before_tree);
    assert!(!previous.tree().root_node().has_error());
}
