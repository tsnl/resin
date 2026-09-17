pub mod frontend;
pub mod integer_arithmetic;
pub mod pipeline;
pub mod project;
pub mod service;
pub mod shaders;
pub mod toolchain;

pub fn parse(source: &str) -> resin_ast::SourceFile {
    let parsed = frontend::ast(&frontend::cst(source, None));
    assert!(parsed.errors.is_empty(), "{source}\n{:?}", parsed.errors);
    parsed.file
}

#[allow(dead_code)]
pub fn program(source: &str) -> resin_ast::Program {
    let src = resin_source::Source::new("test.resin", source);
    resin_ast::Program {
        modules: vec![resin_ast::SourceModule {
            file: std::sync::Arc::new(parse(source)),
            source: src,
            imports: vec![],
        }],
    }
}

#[allow(dead_code)]
pub fn hir(source: &str) -> resin_hir::Module {
    frontend::check_hir(&program(source))
        .into_module()
        .unwrap_or_else(|error| panic!("{source}\n{error}"))
}

pub fn module(source: &str) -> resin_lir::Module {
    pipeline::source_module(source).unwrap_or_else(|error| panic!("{source}\n{error}"))
}

#[allow(dead_code)]
pub fn statements(source: &str) -> Vec<resin_ast::Stmt> {
    let mut file = parse(&format!("fn main() -> ()  {{ {source} }}"));
    let resin_ast::StmtKind::Function { body, .. } = file.stmts.remove(0).val else {
        unreachable!()
    };
    let resin_ast::TermKind::Block { stmts, .. } = body.val else {
        unreachable!()
    };
    stmts
}
