use resin::*;

fn main() {
    let mut config = Config::default();
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--compiler-debug") {
        config.debug_ast = true;
    }

    let lexer = Lexer::new(config.clone());
    let src = std::fs::read_to_string("examples/eg001.resin").unwrap();
    let source = Source::new("eg001.resin", src, lexer.config());
    let tokens = lexer
        .lex(source)
        .unwrap_or_else(|e| panic!("Lex error: {e:?}"));

    eprintln!("Lexed {} tokens", tokens.len());

    let ts = TokenStream::new(tokens);
    let file = match parse(ts) {
        Ok(file) => {
            let sexpr = ast_sexpr::print(&file);
            if config.debug_ast {
                std::fs::create_dir_all("resin-debug").unwrap();
                std::fs::write("resin-debug/ast.sexp", &sexpr).unwrap();
            }
            file
        }
        Err(e) => {
            eprintln!("Parse error:\n{e}");
            return;
        }
    };

    let top = match typer::check(&file) {
        Ok(top) => {
            eprintln!("Collected {} top-level bindings", top.bindings.len());
            top
        }
        Err(e) => {
            eprintln!("Declaration error:\n{e}");
            return;
        }
    };

    let program = match ir_gen::lower(&file, &top) {
        Ok(program) => {
            eprintln!("Lowered to {} functions", program.functions.len());
            program
        }
        Err(e) => {
            eprintln!("IR generation error:\n{e}");
            return;
        }
    };

    let ir_sexp = ir_sexpr::print(&program);
    if config.debug_ast {
        std::fs::create_dir_all("resin-debug").unwrap();
        std::fs::write("resin-debug/ir.sexp", &ir_sexp).unwrap();
    }
    println!("{ir_sexp}");
}
