use resin::{ast::AstGen, ir};
use tree_sitter::Parser;

pub fn parse(source: &str) -> resin::ast::SourceFile {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .unwrap();
    let tree = parser.parse(source, None).unwrap();
    AstGen::new(source)
        .gen_source_file(tree.root_node())
        .unwrap()
}

pub fn module(source: &str) -> ir::Module {
    ir::generate(&parse(source)).unwrap_or_else(|error| panic!("{source}\n{error}"))
}

#[allow(dead_code)]
pub fn statements(source: &str) -> Vec<resin::ast::Stmt> {
    let mut file = parse(&format!("main() -> () = {{ {source} }};"));
    let resin::ast::StmtKind::Function { body, .. } = file.stmts.remove(0).val else {
        unreachable!()
    };
    let resin::ast::TermKind::Block { stmts, .. } = body.val else {
        unreachable!()
    };
    stmts
}
