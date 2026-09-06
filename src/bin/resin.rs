use resin::{ast, backend, compiler::Session, ir, toolchain};

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

use clap::{ValueEnum, builder::TypedValueParser};

#[path = "resin/format.rs"]
mod format;
#[path = "resin/source.rs"]
mod source;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(clap::Parser)]
#[command(
    name = "resin",
    after_help = "Formatting:\n  resin fmt PATH...          Format files/directories in place\n  resin fmt --check PATH...  Check formatting without writing\n  resin fmt --help           Show formatting options"
)]
struct Cli {
    /// Source file and exported entry function (defaults to main).
    #[arg(
        value_name = "FILE[:ENTRY]",
        value_parser = clap::builder::OsStringValueParser::new().try_map(|value| source::parse(&value))
    )]
    source: source::Source,

    /// What to emit or execute.
    #[arg(long, value_enum, default_value_t = Output::Run)]
    output: Output,

    /// Destination file, or directory for host executables. Executables use -O3 and are not run.
    #[arg(short = 'o', long = "out")]
    destination: Option<PathBuf>,

    /// C compiler executable (defaults to CC, then cc on Unix or clang on Windows MSVC).
    #[arg(long)]
    cc: Option<OsString>,

    /// Shader stage for GLSL and SPIR-V output.
    #[arg(long, default_value = "compute")]
    stage: backend::glsl::Stage,

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
    /// Compile under ./build and run, or copy without running when -o is supplied.
    Run,
    /// Emit GLSL for one shader entry.
    Glsl,
    /// Compile one shader entry to SPIR-V.
    Spirv,
}

fn main() {
    // Preserve FILE[:ENTRY] invocation while reserving `fmt` as a command.
    // A source literally named fmt can still be passed as ./fmt or after --.
    let result = if std::env::args_os().nth(1).is_some_and(|arg| arg == "fmt") {
        format::run(<format::Options as clap::Parser>::parse_from(
            std::env::args_os().skip(1),
        ))
    } else {
        run(<Cli as clap::Parser>::parse())
    };
    match result {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run(cli: Cli) -> Result<i32> {
    validate(&cli)?;
    let mut compiler = Session::default();
    let path = resin::analysis::normalize_path(&cli.source.path)?;
    let snapshot = compiler.analyze(&path)?;
    if cli.output == Output::Cst {
        let tree = snapshot
            .syntax_tree(&path)
            .ok_or_else(|| format!("cannot read {}", cli.source.path.display()))?;
        return print(&cli, tree.root_node().to_sexp());
    }
    let file = snapshot.program()?;
    match cli.output {
        Output::Check => return print(&cli, "ok".into()),
        Output::Ast => return print(&cli, ast::print::format_program(file)),
        _ => {}
    }
    let module = snapshot.module()?;
    match cli.output {
        Output::Ir => print(&cli, ir::format_module(module)),
        Output::C | Output::Exe | Output::Run => host(&cli, module),
        Output::Glsl | Output::Spirv => shader(&cli, module),
        _ => unreachable!(),
    }
}

fn validate(cli: &Cli) -> Result<()> {
    if let Some(output) = &cli.destination {
        protect_source(&cli.source.path, output)?;
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
    let source = backend::c::emit_with_shaders(module, &cli.source.entry, &shaders)?;
    if cli.output == Output::C {
        return print(cli, source);
    }
    let mut name = cli
        .source
        .path
        .file_stem()
        .ok_or("source file needs a name")?
        .to_os_string();
    if cli.source.entry != "main" {
        name.push(format!("-{}", cli.source.entry));
    }
    let output = host_destination(cli, &name)?;
    let compiler = compiler(&cli.cc, "CC", toolchain::DEFAULT_C_COMPILER);
    let profile = if output.is_some() {
        toolchain::CProfile::Release
    } else {
        toolchain::CProfile::Debug
    };
    let build = toolchain::build_c(
        &cli.source.path,
        &cli.source.entry,
        &source,
        &compiler,
        profile,
    )?;
    let executable = build.executable();
    if let Some(output) = output {
        toolchain::copy_output(executable, &output)?;
        Ok(0)
    } else {
        Ok(Command::new(executable).status()?.code().unwrap_or(1))
    }
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
    protect_source(&cli.source.path, &output)?;
    Ok(Some(output))
}

fn shader(cli: &Cli, module: &ir::Module) -> Result<i32> {
    let source = backend::glsl::emit(module, &cli.source.entry, cli.stage)?;
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
