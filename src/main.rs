use resin::*;

fn main() {
    let lexer = Lexer::new(Config::default());
    let src = std::fs::read_to_string("examples/eg001.resin").unwrap();
    let source = Source::new("eg001.resin", src, lexer.config());
    let tokens = lexer
        .lex(source)
        .unwrap_or_else(|e| panic!("Lex error: {e:?}"));

    eprintln!("Lexed {} tokens", tokens.len());

    let ts = TokenStream::new(tokens);
    match parse(ts) {
        Ok(file) => {
            eprintln!("Parsed {} top-level statements:", file.stmts.len());
            for stmt in &file.stmts {
                eprintln!("  {stmt:#?}");
            }
        }
        Err(e) => {
            eprintln!("Parse error:\n{e}");
        }
    }
}
