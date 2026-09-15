pub mod pipeline;
pub mod project;
pub mod shaders;

pub fn parse(source: &str) -> resin_ast::SourceFile {
    let parsed = resin_ast::build_ast(&resin_cst::build_cst(source, None));
    assert!(parsed.errors.is_empty(), "{source}\n{:?}", parsed.errors);
    parsed.file
}

pub fn program(source: &str) -> resin_ast::Program {
    let src = resin_source::Source::new("test.resin", source);
    resin_ast::Program {
        modules: vec![resin_ast::SourceModule {
            file: parse(source),
            source: src,
            imports: vec![],
        }],
    }
}

pub fn hir(source: &str) -> resin_hir::Module {
    resin_hir::build_hir(&program(source))
        .into_module()
        .unwrap_or_else(|error| panic!("{source}\n{error}"))
}

pub fn module(source: &str) -> resin_lir::Module {
    pipeline::source_module(source).unwrap_or_else(|error| panic!("{source}\n{error}"))
}

#[allow(dead_code)]
pub fn statements(source: &str) -> Vec<resin_ast::Stmt> {
    let mut file = parse(&format!("def main() -> () = {{ {source} }};"));
    let resin_ast::StmtKind::Function { body, .. } = file.stmts.remove(0).val else {
        unreachable!()
    };
    let resin_ast::TermKind::Block { stmts, .. } = body.val else {
        unreachable!()
    };
    stmts
}
