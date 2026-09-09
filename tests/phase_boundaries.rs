//! Exercise public phase APIs without retaining construction state between them.
use resin_ast as ast;
use resin_codegen as codegen;
use resin_common::types::Ty;
use resin_cst as cst;
use resin_hir as hir;
use resin_lir as lir;
use resin_lir_verifier as lir_verifier;

fn hir(source: &str) -> hir::Module {
    let syntax = cst::Document::reparse(source.to_owned(), None);
    let file = ast::generate(&syntax).unwrap();
    hir::generate(&file).unwrap()
}

fn function<'a>(module: &'a hir::Module, name: &str) -> &'a hir::Function {
    module
        .functions
        .iter()
        .find(|function| function.name.as_ref() == name)
        .unwrap()
}

fn tail(function: &hir::Function) -> &hir::Term {
    let hir::TermKind::Block { tail, .. } = &function.body.as_ref().unwrap().kind else {
        panic!("block")
    };
    tail
}

#[test]
fn hir_resolves_calls_short_circuiting_and_layout_before_lir() {
    let module = hir(r#"
        struct Item { value: int };
        impl Item { def read(item: Item) -> int = { item.value }; }
        def read(item: Item) -> int = { item.read() };
        def both(a: bool, b: bool) -> bool = { a && b };
        def measure() -> ulong = { size_of(int) };
    "#);
    let read = function(&module, "read");
    let hir::TermKind::Call { func, arg } = &tail(read).kind else {
        panic!("ordinary call")
    };
    let hir::TermKind::Function { function: id } = func.kind else {
        panic!("resolved function")
    };
    assert_eq!(module.functions[id.index()].name.as_ref(), "Item.read");
    let hir::TermKind::Pack(args) = &arg.kind else {
        panic!("explicit arguments")
    };
    assert!(args.receiver.is_some());
    assert!(matches!(
        tail(function(&module, "both")).kind,
        hir::TermKind::If { .. }
    ));
    assert!(matches!(
        tail(function(&module, "measure")).kind,
        hir::TermKind::Constant(_)
    ));
    let printed = hir::format_module(&module);
    assert!(printed.contains("Item.read") && printed.contains("(if") && printed.contains("(pack"));
    lir_verifier::verify(&lir::generate(&module).unwrap()).unwrap();
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
    assert_eq!(function(&module, "main").signature.result.ty, Ty::Int32);
    // Source text and AST have already been dropped. Origins are optional metadata.
    module.origins = hir::SourceMap::default();
    let first = lir::generate(&module).unwrap();
    let second = lir::generate(&module).unwrap();
    assert_eq!(first, second);
    drop(module);
    let checked = lir_verifier::VerifiedModule::new(first).unwrap();
    assert!(codegen::generate_c(checked.view(), "main", &[]).is_ok());
}

#[test]
fn target_trees_outlive_lir_and_its_verification_certificate() {
    let module = hir(r#"
        export { main, kernel };
        @compute_shader def kernel(i: ulong, output: Ptr<ulong>) = { output.* := i; };
        def main() -> int = { 42 };
    "#);
    let checked = lir_verifier::VerifiedModule::new(lir::generate(&module).unwrap()).unwrap();
    let kernel = checked.view().module().entries["kernel"];
    let c = codegen::generate_c(checked.view(), "main", &[]).unwrap();
    let glsl = codegen::generate_glsl(checked.view(), kernel, codegen::Stage::Compute).unwrap();
    drop(checked);
    drop(module);
    assert!(codegen::print_c(&c).contains("int main("));
    assert!(codegen::print_glsl(&glsl).contains("void main()"));
}

#[test]
fn mutating_lir_discards_the_certificate_and_requires_reverification() {
    let checked =
        lir_verifier::VerifiedModule::new(lir::generate(&hir("def f() = {}; ")).unwrap()).unwrap();
    let mut module = checked.into_module();
    module.functions[0].locals.clear();
    assert!(lir_verifier::VerifiedModule::new(module).is_err());
}

#[test]
fn a_later_phase_error_preserves_the_earlier_snapshot_products() {
    let path = resin_compiler::normalize_path(std::path::Path::new("phase-error.resin")).unwrap();
    let mut sources = resin_compiler::Sources::default();
    sources
        .overlays
        .insert(path.clone(), "def f() -> int = { var n: int; n };".into());
    let snapshot =
        resin_compiler::Compilation::new(&path, &sources, &resin_compiler::stdlib_path());
    assert!(snapshot.program().is_ok());
    assert!(snapshot.hir().is_ok());
    assert!(snapshot.module().is_err());
    assert!(
        snapshot
            .diagnostics()
            .iter()
            .any(|d| d.message.to_lowercase().contains("uninitialized"))
    );
}
