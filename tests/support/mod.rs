use resin::{ast::AstGen, ir};
use tree_sitter::Parser;

pub fn module(source: &str) -> ir::Module {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .unwrap();
    let tree = parser.parse(source, None).unwrap();
    let ast = AstGen::new(source)
        .gen_source_file(tree.root_node())
        .unwrap();
    ir::generate(&ast).unwrap_or_else(|error| panic!("{source}\n{error}"))
}
