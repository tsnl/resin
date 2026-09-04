use resin::{ast, ir};

use std::path::PathBuf;

use clap::ValueEnum;
use tree_sitter::Parser;

#[derive(clap::Parser)]
#[command(name = "resin")]
struct Cli {
    /// Entry-point file.
    file: PathBuf,

    /// What to print.
    #[arg(long, value_enum, default_value_t = Output::Ir)]
    output: Output,
}

#[derive(Clone, PartialEq, Eq, ValueEnum)]
enum Output {
    /// Print the typed stack IR.
    Ir,
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
    let tree = parser.parse(&src, None).expect("parser language is set");
    if cli.output == Output::Cst {
        println!("{}", tree.root_node().to_sexp());
        return;
    }

    let file = match ast::generate::AstGen::new(&src).gen_source_file(tree.root_node()) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };
    if cli.output == Output::Check {
        println!("ok");
        return;
    }
    if cli.output == Output::Ast {
        println!("{}", ast::print::format_source(&file));
        return;
    }

    match ir::generate(&file) {
        Ok(module) => println!("{}", ir::format_module(&module)),
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}
