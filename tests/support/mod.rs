pub mod pipeline;

pub fn parse(source: &str) -> resin_ast::SourceFile {
    resin_ast::generate(&resin_cst::Document::reparse(source.to_string(), None)).unwrap()
}

pub fn module(source: &str) -> resin_lir::Module {
    pipeline::generate(&parse(source)).unwrap_or_else(|error| panic!("{source}\n{error}"))
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
