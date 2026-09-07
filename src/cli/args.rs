//! CLI syntax and conversion into an explicit execution mode.
use super::{Result, inspect, source};
use crate::{
    backend,
    compiler::{Input, Request, Target},
};
use clap::{CommandFactory, ValueEnum};
use std::{ffi::OsString, path::PathBuf};

pub enum Mode {
    Interpreter(Request),
    Compiler(Request),
    Inspector {
        input: Input,
        output: inspect::Output,
        destination: Option<PathBuf>,
    },
    Formatter {
        paths: Vec<PathBuf>,
        check: bool,
    },
}

pub fn parse() -> Result<Mode> {
    <Cli as clap::Parser>::parse().mode()
}

#[derive(clap::Parser)]
#[command(name = "resin")]
struct Cli {
    /// One FILE[:ENTRY] to run/compile, or files/directories to format with --format.
    #[arg(required = true, value_name = "PATH")]
    paths: Vec<PathBuf>,

    /// Format files in place; search directories recursively for .resin files.
    #[arg(short = 'f', long, conflicts_with_all = ["output", "destination", "cc", "stage", "glslc"])]
    format: bool,

    /// With --format, check without writing; exit 1 on differences or file/syntax errors.
    #[arg(long, requires = "format")]
    check: bool,

    #[command(flatten)]
    compile: CompileOptions,
}

#[derive(clap::Args)]
struct CompileOptions {
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
    #[arg(long)]
    stage: Option<backend::glsl::Stage>,

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

impl Cli {
    fn mode(self) -> Result<Mode> {
        if self.format {
            return Ok(Mode::Formatter {
                paths: self.paths,
                check: self.check,
            });
        }
        let [path] = self.paths.as_slice() else {
            Self::command().error(clap::error::ErrorKind::WrongNumberOfValues,
                "running or compiling requires exactly one FILE[:ENTRY]; use --format for multiple paths").exit();
        };
        let input = source::parse(path.as_os_str()).unwrap_or_else(|error| {
            Self::command()
                .error(clap::error::ErrorKind::InvalidValue, error)
                .exit()
        });
        let options = self.compile;
        if matches!(options.output, Output::Exe | Output::Spirv) && options.destination.is_none() {
            return Err("binary output requires -o PATH".into());
        }
        let output = match options.output {
            Output::Cst => Some(inspect::Output::Cst),
            Output::Ast => Some(inspect::Output::Ast),
            Output::Ir => Some(inspect::Output::Ir),
            Output::Check => Some(inspect::Output::Check),
            _ => None,
        };
        if let Some(output) = output {
            return Ok(Mode::Inspector {
                input,
                output,
                destination: options.destination,
            });
        }
        let target = match options.output {
            Output::Run | Output::Exe => Target::Executable,
            Output::C => Target::C,
            Output::Glsl => Target::Glsl,
            Output::Spirv => Target::Spirv,
            _ => unreachable!("inspection modes were handled above"),
        };
        let interpret = options.output == Output::Run && options.destination.is_none();
        let request = Request {
            input,
            target,
            destination: options.destination,
            cc: options.cc,
            stage: options.stage,
            glslc: options.glslc,
        };
        Ok(if interpret {
            Mode::Interpreter(request)
        } else {
            Mode::Compiler(request)
        })
    }
}
