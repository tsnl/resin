use resin::{ast, backend, ir, toolchain};

use std::{ffi::OsString, path::PathBuf, process::Command};

use clap::ValueEnum;
use tree_sitter::Parser;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(clap::Parser)]
#[command(name = "resin")]
struct Cli {
    /// Entry-point file.
    file: PathBuf,

    /// What to emit or execute.
    #[arg(long, value_enum, default_value_t = Output::Ir)]
    output: Output,

    /// Destination file; required for executable, SPIR-V, and image output.
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
    /// Emit a standalone C11 program.
    C,
    /// Compile an executable with the C compiler.
    Exe,
    /// Compile and run in a temporary directory.
    Run,
    /// Emit GLSL for one shader entry.
    Glsl,
    /// Compile one shader entry to SPIR-V.
    Spirv,
    /// Run the kernel shader and write a 256x256 PNG (requires --features gpu).
    Compute,
    /// Run vertex/fragment shaders and write a 256x256 PNG (requires --features gpu).
    Graphics,
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
    let file = ast::generate::AstGen::new(&src).gen_source_file(tree.root_node())?;
    match cli.output {
        Output::Check => return print(&cli, "ok".into()),
        Output::Ast => return print(&cli, ast::print::format_source(&file)),
        _ => {}
    }
    let module = ir::generate(&file)?;
    match cli.output {
        Output::Ir => print(&cli, ir::format_module(&module)),
        Output::C | Output::Exe | Output::Run => host(&cli, &module),
        Output::Glsl | Output::Spirv => shader(&cli, &module),
        Output::Compute | Output::Graphics => render(&cli, &module),
        _ => unreachable!(),
    }
}

fn validate(cli: &Cli) -> Result<()> {
    if let Some(output) = &cli.destination
        && output.exists()
        && std::fs::canonicalize(output)? == std::fs::canonicalize(&cli.file)?
    {
        return Err("output would overwrite the source file".into());
    }
    if matches!(
        cli.output,
        Output::Exe | Output::Spirv | Output::Compute | Output::Graphics
    ) && cli.destination.is_none()
    {
        return Err("binary output requires -o PATH".into());
    }
    if cli.output == Output::Run && cli.destination.is_some() {
        return Err("--output run does not accept -o".into());
    }
    Ok(())
}

fn host(cli: &Cli, module: &ir::Module) -> Result<i32> {
    let source = backend::c::emit(module)?;
    if cli.output == Output::C {
        return print(cli, source);
    }
    let compiler = compiler(&cli.cc, "CC", "cc");
    if let Some(output) = &cli.destination {
        toolchain::compile_c(&source, output, &compiler)?;
        return Ok(0);
    }
    let temp = toolchain::TempDir::new(&std::env::temp_dir())?;
    let executable = temp.path().join("program");
    toolchain::compile_c(&source, &executable, &compiler)?;
    Ok(Command::new(executable).status()?.code().unwrap_or(1))
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

#[cfg(feature = "gpu")]
fn render(cli: &Cli, module: &ir::Module) -> Result<i32> {
    use resin::gpu::{self, Pipeline};
    let pipeline = if cli.output == Output::Compute {
        Pipeline::Compute
    } else {
        Pipeline::Graphics
    };
    let compiler = compiler(&cli.glslc, "GLSLC", "glslc");
    let image = gpu::render(module, pipeline, &compiler)?;
    gpu::write_png(&image, cli.destination.as_ref().unwrap())?;
    Ok(0)
}

#[cfg(not(feature = "gpu"))]
fn render(_cli: &Cli, _module: &ir::Module) -> Result<i32> {
    Err("GPU execution requires building resin with --features gpu".into())
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
