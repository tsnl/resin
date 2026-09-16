//! Exercise public phase APIs without retaining construction state between them.

#[allow(dead_code)]
mod support;

use resin_source::prelude::*;
fn hir(source: &str) -> resin_hir::Module {
    support::hir(source)
}

fn function<'a>(module: &'a resin_hir::Module, name: &str) -> &'a resin_hir::Function {
    module
        .functions
        .iter()
        .find(|function| function.name.as_ref() == name)
        .unwrap()
}

fn tail(function: &resin_hir::Function) -> &resin_hir::Term {
    let resin_hir::TermKind::Block { tail, .. } = &function.body.as_ref().unwrap().kind else {
        panic!("block")
    };
    tail
}

fn native(
    checked: std::sync::Arc<resin_lir::VerifiedModule>,
) -> Result<resin_codegen::NativeObject, resin_codegen::GenerationError> {
    support::frontend::block_on(resin_codegen::generate_native(
        checked,
        "main".into(),
        resin_codegen::NativeOptimization::None,
        std::sync::Arc::new(resin_codegen::NativeInputs::default()),
        support::frontend::execution(),
        &resin_executor::Cancellation::new(),
    ))
}

#[test]
fn hir_resolves_calls_and_preserves_type_dependent_operations_for_lir() {
    let module = hir(r#"
        struct Item { value: int,
            def read(item: Item) -> int = { item.value };
        };

        def read(item: Item) -> int = { item.read() };
        def both(a: bool, b: bool) -> bool = { a && b };
        def measure() -> ulong = { size_of(int) };
    "#);
    let read = function(&module, "read");
    let resin_hir::TermKind::Call { func, args } = &tail(read).kind else {
        panic!("ordinary call")
    };
    let resin_hir::TermKind::Function { function: id, .. } = func.kind else {
        panic!("resolved function")
    };
    assert_eq!(module.functions[id.index()].name.as_ref(), "Item.read");
    assert_eq!(args.len(), 1);
    assert!(matches!(
        tail(function(&module, "both")).kind,
        resin_hir::TermKind::If { .. }
    ));
    assert!(matches!(
        tail(function(&module, "measure")).kind,
        resin_hir::TermKind::Layout {
            of: resin_hir::Type::Int32,
            size: true
        }
    ));
    assert!(matches!(
        tail(function(&module, "Item.read")).kind,
        resin_hir::TermKind::Field { ref name, .. } if name.as_ref() == "value"
    ));
    let printed = resin_hir::format_module(&module);
    assert!(printed.contains("Item.read") && printed.contains("(if") && printed.contains("(call"));
    resin_lir::verify(
        &support::frontend::lower(&module, &[], &resin_lir::LoweringOptions::default()).unwrap(),
    )
    .unwrap();
}

#[test]
fn lir_lowering_needs_only_the_resolved_tree() {
    let mut module = hir(r#"
        export { main };
        def narrow(n: int) -> int = { n };
        def main() -> _ = {
            var item = 42;
            var pointer: Ptr<_>;
            pointer := &item;
            narrow(pointer.*)
        };
    "#);
    assert_eq!(
        function(&module, "main").signature.result.ty,
        resin_hir::Type::Int32
    );
    // Source text and AST have already been dropped. Origins are optional metadata.
    for function in &mut module.functions {
        function.location = None;
    }
    let first =
        support::frontend::lower(&module, &[], &resin_lir::LoweringOptions::default()).unwrap();
    let second =
        support::frontend::lower(&module, &[], &resin_lir::LoweringOptions::default()).unwrap();
    assert_eq!(first, second);
    drop(module);
    let checked = resin_lir::VerifiedModule::new(first).unwrap();
    assert!(native(std::sync::Arc::new(checked)).is_ok());
}

#[test]
fn generated_bytes_outlive_lir_and_its_verification_certificate() {
    let module = hir(r#"
        export { main, kernel };
        @compute_shader def kernel(i: ulong, output: Ptr<ulong>) = { output.* := i; };
        def main() -> int = { 42 };
    "#);
    let checked = resin_lir::VerifiedModule::new(
        support::frontend::lower(&module, &[], &resin_lir::LoweringOptions::default()).unwrap(),
    )
    .unwrap();
    let checked = std::sync::Arc::new(checked);
    let host = native(checked.clone()).unwrap();
    let function = *checked.view().module().shaders.keys().next().unwrap();
    let shaders = support::frontend::block_on(resin_codegen::generate_spirv(
        checked.clone(),
        function,
        support::frontend::execution(),
        &resin_executor::Cancellation::new(),
    ))
    .unwrap();
    drop(checked);
    drop(module);
    assert!(!host.bytes().is_empty());
    assert_eq!(&shaders[..4], &[3, 2, 35, 7]);
}

#[test]
fn mutating_lir_discards_the_certificate_and_requires_reverification() {
    let checked = resin_lir::VerifiedModule::new(
        support::frontend::lower(
            &hir("def f() = {}; "),
            &[],
            &resin_lir::LoweringOptions::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut module = checked.into_module();
    module.functions[0].parameter_count = module.functions[0].locals.len() + 1;
    assert!(resin_lir::VerifiedModule::new(module).is_err());
}

#[test]
fn a_later_phase_error_preserves_earlier_compilation_products() {
    let source = Source::new(
        "phase-error.resin",
        "export { f }; def f() -> bool = { (1 == 1) + (1 == 1) };",
    );
    let mut loader = resin_source::Loader::new(Default::default());
    let output = support::frontend::analyze(source, &mut loader, None);
    assert!(output.program().is_ok());
    assert!(output.hir().is_ok());
    let errors = match support::pipeline::verified_lir(&output, "f", resin_lir::Profile::Host) {
        Ok(_) => panic!("expected unsupported builtin"),
        Err(errors) => errors,
    };
    assert!(
        errors
            .iter()
            .any(|error| error.diagnostic.contains("UnsupportedBuiltin"))
    );
}

#[test]
fn unsupported_concrete_operations_fail_during_lir_construction() {
    let source = "export { main }; def main() -> int = { var r = { x = 1 }; r + r; 0 };";
    let hir = hir(source);
    let error = support::frontend::lower(&hir, &[], &resin_lir::LoweringOptions::default())
        .unwrap_err()
        .remove(0);
    assert!(matches!(
        error.kind,
        resin_lir::ErrorKind::Type {
            kind: resin_types::TypeErrorKind::UnsupportedBuiltin { .. }
        }
    ));
    assert_eq!(&source[error.span.start..error.span.end], "r + r");
}
