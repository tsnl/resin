#[test]
fn template_syntax_cannot_silently_acquire_monomorphic_meaning() {
    for source in [
        "struct Owner { def unused<T>() -> int = { 42 }; };",
        "struct Owner { def f() -> int = { 42 }; }; def main() = { Owner.f::<int>(); };",
    ] {
        let document = resin_cst::Document::reparse(source.into(), None);
        let file = resin_ast::generate(&document).unwrap();
        let error = resin_hir::generate(&file).unwrap_err();
        assert!(error.to_string().contains("template"), "{source}: {error}");
    }
}
