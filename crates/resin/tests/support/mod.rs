use resin::lir;

pub fn parse(source: &str) -> resin::ast::SourceFile {
    resin::ast::lower::generate(&resin::cst::Document::reparse(source.to_string(), None)).unwrap()
}

pub fn module(source: &str) -> lir::Module {
    resin::compiler::generate(&parse(source)).unwrap_or_else(|error| panic!("{source}\n{error}"))
}

#[allow(dead_code)]
pub fn statements(source: &str) -> Vec<resin::ast::Stmt> {
    let mut file = parse(&format!("def main() -> () = {{ {source} }};"));
    let resin::ast::StmtKind::Function { body, .. } = file.stmts.remove(0).val else {
        unreachable!()
    };
    let resin::ast::TermKind::Block { stmts, .. } = body.val else {
        unreachable!()
    };
    stmts
}
