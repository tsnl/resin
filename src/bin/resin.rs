use resin::{ast, backend, ir, toolchain};

use std::{ffi::OsString, path::PathBuf, process::Command};

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

    /// Destination file; required for executable output.
    #[arg(short = 'o', long = "out")]
    destination: Option<PathBuf>,

    /// C compiler executable (defaults to CC or cc; no shell parsing).
    #[arg(long)]
    cc: Option<OsString>,
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
    /// Emit a standalone C11 program.
    C,
    /// Compile an executable with the C compiler.
    Exe,
    /// Compile and run in a temporary directory.
    Run,
}

fn main() {
    let cli = <Cli as clap::Parser>::parse();
    match run(cli) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run(cli: Cli) -> Result<i32, Box<dyn std::error::Error>> {
    if let Some(output) = &cli.destination
        && output.exists()
        && std::fs::canonicalize(output)? == std::fs::canonicalize(&cli.file)?
    {
        return Err("output would overwrite the source file".into());
    }
    if cli.output == Output::Exe && cli.destination.is_none() {
        return Err("executable output requires -o PATH".into());
    }
    if cli.output == Output::Run && cli.destination.is_some() {
        return Err("--output run does not accept -o".into());
    }
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
        return print(&cli, tree.root_node().to_sexp());
    }

    let file = match ast::generate::AstGen::new(&src).gen_source_file(tree.root_node()) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };
    if cli.output == Output::Check {
        return print(&cli, "ok".into());
    }
    if cli.output == Output::Ast {
        return print(&cli, ast::print::format_source(&file));
    }

    let module = ir::generate(&file)?;
    if cli.output == Output::Ir {
        return print(&cli, ir::format_module(&module));
    }
    let source = backend::c::emit(&module)?;
    if cli.output == Output::C {
        return print(&cli, source);
    }
    let compiler = cli
        .cc
        .or_else(|| std::env::var_os("CC"))
        .unwrap_or_else(|| "cc".into());
    if let Some(output) = cli.destination {
        toolchain::compile_c(&source, &output, &compiler)?;
        return Ok(0);
    }
    let temp = toolchain::TempDir::new(&std::env::temp_dir())?;
    let executable = temp.path().join("program");
    toolchain::compile_c(&source, &executable, &compiler)?;
    Ok(Command::new(executable).status()?.code().unwrap_or(1))
}

fn print(cli: &Cli, text: String) -> Result<i32, Box<dyn std::error::Error>> {
    if let Some(output) = &cli.destination {
        toolchain::write_output(format!("{text}\n").as_bytes(), output)?;
    } else {
        println!("{text}");
    }
    Ok(0)
}
