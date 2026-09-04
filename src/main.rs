use resin::{ast_gen, ast_sexpfmt};

use std::path::PathBuf;

use clap::ValueEnum;
use tree_sitter::Parser;

#[derive(clap::Parser)]
#[command(name = "resin")]
struct Cli {
    /// Entry-point file.
    file: PathBuf,

    /// What to print.
    #[arg(long, value_enum, default_value_t = Output::Ast)]
    output: Output,
}

#[derive(Clone, ValueEnum)]
enum Output {
    /// Print the lowered AST.
    Ast,
    /// Print the tree-sitter parse tree (S-expressions).
    Cst,
    /// Print the source with parse errors highlighted.
    Check,
}

fn main() {
    let cli = <Cli as clap::Parser>::parse();
    let src = std::fs::read_to_string(&cli.file).unwrap_or_else(|e| {
        eprintln!("error: cannot read {}: {e}", cli.file.display());
        std::process::exit(1);
    });

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .expect("failed to load resin language");

    let tree = parser.parse(&src, None).unwrap();

    match cli.output {
        Output::Cst => println!("{}", tree.root_node().to_sexp()),
        Output::Check => {
            if tree.root_node().has_error() {
                eprintln!("parse error");
                std::process::exit(1);
            }
            println!("ok");
        }
        Output::Ast => {
            let g = ast_gen::AstGen::new(&src);
            let file = g.gen_source_file(tree.root_node());
            println!("{}", ast_sexpfmt::format_source(&file));
        }
    }
}
