use std::path::PathBuf;

use clap::Parser;
use resin::*;

#[derive(Parser)]
#[command(name = "resin", about = "The Resin compiler")]
struct Cli {
    /// Entry point file for compilation
    entry_point: String,

    /// Output directory for generated artifacts
    #[arg(short, long, default_value = "resin-build")]
    output: PathBuf,
}

fn main() {
    let cli = Cli::parse();

    let mut config = Config::default();
    config.output_dir = cli.output.display().to_string();

    let lexer = Lexer::new(config.clone());
    let src = std::fs::read_to_string(&cli.entry_point).unwrap();
    let source = Source::new(&cli.entry_point, src, lexer.config());
    let tokens = lexer
        .lex(source)
        .unwrap_or_else(|e| panic!("Lex error: {e:?}"));

    eprintln!("Lexed {} tokens", tokens.len());

    let ts = TokenStream::new(tokens);
    match parse(ts) {
        Ok(file) => {
            if config.debug_ast {
                let sexpr = ast_sexpr::print(&file);
                let debug_dir = cli.output.join("debug");
                std::fs::create_dir_all(&debug_dir).unwrap();
                std::fs::write(debug_dir.join("ast.sexp"), &sexpr).unwrap();
            }
            eprintln!("Parsed {} top-level statements", file.stmts.len());
        }
        Err(e) => {
            eprintln!("Parse error:\n{e}");
        }
    };
}
