use resin::{ast, backend, ir, toolchain};

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

use clap::ValueEnum;
use tree_sitter::Parser;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(clap::Parser)]
#[command(name = "resin")]
struct Cli {
    /// Entry-point file.
    file: PathBuf,

    /// What to emit or execute.
    #[arg(long, value_enum, default_value_t = Output::Run)]
    output: Output,

    /// Destination file, or directory for host executables. Run mode also keeps a copy here.
    #[arg(short = 'o', long = "out")]
    destination: Option<PathBuf>,

    /// C compiler executable (defaults to CC or cc; no shell parsing).
    #[arg(long)]
    cc: Option<OsString>,

    /// Shader stage for GLSL and SPIR-V output.
    #[arg(long, default_value = "compute")]
    stage: backend::glsl::Stage,

    /// GLSL/SPIR-V function name (defaults to kernel, vertex, or fragment).
    #[arg(long)]
    entry: Option<String>,

    /// Shader compiler executable (defaults to GLSLC or glslc).
    #[arg(long)]
    glslc: Option<OsString>,
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
    /// Emit C11 source using resin_runtime.h.
    C,
    /// Compile an executable without running it (requires -o).
    Exe,
    /// Compile under ./build and run; optionally copy the executable with -o.
    Run,
    /// Emit GLSL for one shader entry.
    Glsl,
    /// Compile one shader entry to SPIR-V.
    Spirv,
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

fn run(cli: Cli) -> Result<i32> {
    validate(&cli)?;
    let src = std::fs::read_to_string(&cli.file)
        .map_err(|e| format!("cannot read {}: {e}", cli.file.display()))?;
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_resin::LANGUAGE.into())?;
    let tree = parser.parse(&src, None).expect("parser language is set");
    if cli.output == Output::Cst {
        return print(&cli, tree.root_node().to_sexp());
    }
    let file = ast::load(&cli.file)?;
    match cli.output {
        Output::Check => return print(&cli, "ok".into()),
        Output::Ast => return print(&cli, ast::print::format_program(&file)),
        _ => {}
    }
    let module = ir::generate_program(&file)?;
    match cli.output {
        Output::Ir => print(&cli, ir::format_module(&module)),
        Output::C | Output::Exe | Output::Run => host(&cli, &module),
        Output::Glsl | Output::Spirv => shader(&cli, &module),
        _ => unreachable!(),
    }
}

fn validate(cli: &Cli) -> Result<()> {
    if let Some(output) = &cli.destination {
        protect_source(&cli.file, output)?;
    }
    if matches!(cli.output, Output::Exe | Output::Spirv) && cli.destination.is_none() {
        return Err("binary output requires -o PATH".into());
    }
    Ok(())
}

fn protect_source(source: &Path, output: &Path) -> Result<()> {
    if output.exists() && std::fs::canonicalize(output)? == std::fs::canonicalize(source)? {
        return Err("output would overwrite the source file".into());
    }
    Ok(())
}

fn host(cli: &Cli, module: &ir::Module) -> Result<i32> {
    let shaders = toolchain::build_shaders(module, &compiler(&cli.glslc, "GLSLC", "glslc"))?;
    let source = backend::c::emit_with_shaders(module, &shaders)?;
    if cli.output == Output::C {
        return print(cli, source);
    }
    let name = cli.file.file_stem().ok_or("source file needs a name")?;
    let output = host_destination(cli, name)?;
    let compiler = compiler(&cli.cc, "CC", "cc");
    let build = toolchain::build_c(&cli.file, &source, &compiler)?;
    let executable = build.executable();
    let code = if cli.output == Output::Run {
        Command::new(executable).status()?.code().unwrap_or(1)
    } else {
        0
    };
    if let Some(output) = output {
        toolchain::copy_output(executable, &output)?;
    }
    Ok(code)
}

fn host_destination(cli: &Cli, name: &std::ffi::OsStr) -> Result<Option<PathBuf>> {
    let Some(path) = &cli.destination else {
        return Ok(None);
    };
    let trailing_separator = path
        .as_os_str()
        .as_encoded_bytes()
        .last()
        .is_some_and(|&b| b == b'/' || b == std::path::MAIN_SEPARATOR as u8);
    let output = if path.is_dir() || trailing_separator {
        let mut filename = name.to_os_string();
        filename.push(std::env::consts::EXE_SUFFIX);
        path.join(filename)
    } else {
        path.clone()
    };
    protect_source(&cli.file, &output)?;
    Ok(Some(output))
}

fn shader(cli: &Cli, module: &ir::Module) -> Result<i32> {
    let entry = cli.entry.as_deref().unwrap_or(cli.stage.entry());
    let source = backend::glsl::emit(module, entry, cli.stage)?;
    if cli.output == Output::Glsl {
        return print(cli, source);
    }
    let compiler = compiler(&cli.glslc, "GLSLC", "glslc");
    let bytes = toolchain::compile_glsl(&source, cli.stage, &compiler)?;
    toolchain::write_output(&bytes, cli.destination.as_ref().unwrap())?;
    Ok(0)
}

fn compiler(option: &Option<OsString>, env: &str, fallback: &str) -> OsString {
    option
        .clone()
        .or_else(|| std::env::var_os(env))
        .unwrap_or_else(|| fallback.into())
}

fn print(cli: &Cli, text: String) -> Result<i32> {
    if let Some(output) = &cli.destination {
        toolchain::write_output(format!("{text}\n").as_bytes(), output)?;
    } else {
        println!("{text}");
    }
    Ok(0)
}
