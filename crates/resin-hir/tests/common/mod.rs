use resin_source::prelude::*;

pub fn hir_module(text: &str) -> Result<resin_hir::Module, SourceError> {
    let source = Source::new("test.resin", text);
    resin_hir::build_hir(&resin_ast::Program {
        modules: vec![resin_ast::SourceModule {
            file: resin_ast::build_ast(&resin_cst::build_cst(source.text(), None)).file,
            source,
            imports: vec![],
        }],
    })
    .into_module()
}
