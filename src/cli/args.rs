//! CLI syntax and conversion into an explicit execution mode.
use super::{Environment, Result, source};
use crate::{
    compiler::{Options, Request},
    toolchain::CProfile,
};
use clap::CommandFactory;
use std::{ffi::OsString, path::PathBuf};

pub struct Invocation {
    pub mode: Mode,
    pub stdlib: PathBuf,
}

pub enum Mode {
    Interpreter {
        request: Box<Request>,
        args: Vec<OsString>,
    },
    Compiler(Box<Request>),
    Formatter {
        paths: Vec<PathBuf>,
        check: bool,
    },
}

pub fn parse(
    args: impl IntoIterator<Item = OsString>,
    environment: &Environment,
) -> Result<Invocation> {
    let mode = <Cli as clap::Parser>::parse_from(args).mode(environment)?;
    Ok(Invocation {
        mode,
        stdlib: environment.stdlib(),
    })
}

#[derive(clap::Parser)]
#[command(name = "resin")]
struct Cli {
    /// One FILE[:ENTRY] to run/compile, or files/directories to format with --format.
    #[arg(required_unless_present = "format", value_name = "PATH")]
    paths: Vec<PathBuf>,

    /// Arguments passed literally after --, or additional formatter paths.
    #[arg(last = true, value_name = "ARG", conflicts_with = "destination")]
    program_args: Vec<OsString>,

    /// Format files in place; search directories recursively for .resin files.
    #[arg(short = 'f', long, conflicts_with_all = ["destination", "cc", "glslc"])]
    format: bool,

    /// With --format, check without writing; exit 1 on differences or file/syntax errors.
    #[arg(long, requires = "format")]
    check: bool,

    #[command(flatten)]
    compile: CompileOptions,
}

#[derive(clap::Args)]
struct CompileOptions {
    /// Destination file, or directory for host executables. Executables use -O3 and are not run.
    #[arg(short = 'o', long = "out")]
    destination: Option<PathBuf>,

    /// C compiler executable (defaults to CC, then cc on Unix or clang on Windows MSVC).
    #[arg(long)]
    cc: Option<OsString>,

    /// Shader compiler executable (defaults to GLSLC or glslc).
    #[arg(long)]
    glslc: Option<OsString>,
}

impl Cli {
    fn mode(self, environment: &Environment) -> Result<Mode> {
        if self.format {
            if self.paths.is_empty() && self.program_args.is_empty() {
                Self::command()
                    .error(
                        clap::error::ErrorKind::MissingRequiredArgument,
                        "formatting requires at least one path",
                    )
                    .exit();
            }
            return Ok(Mode::Formatter {
                paths: self
                    .paths
                    .into_iter()
                    .chain(self.program_args.into_iter().map(PathBuf::from))
                    .map(|path| environment.directory.join(path))
                    .collect(),
                check: self.check,
            });
        }
        let [path] = self.paths.as_slice() else {
            Self::command().error(clap::error::ErrorKind::WrongNumberOfValues,
                "running or compiling requires exactly one FILE[:ENTRY]; use --format for multiple paths").exit();
        };
        let mut input = source::parse(path.as_os_str()).unwrap_or_else(|error| {
            Self::command()
                .error(clap::error::ErrorKind::InvalidValue, error)
                .exit()
        });
        input.path = environment.directory.join(input.path);
        let mut options = self.compile;
        options.destination = options
            .destination
            .map(|path| environment.directory.join(path));
        let interpret = options.destination.is_none();
        let profile = if interpret {
            CProfile::Debug
        } else {
            CProfile::Release
        };
        let request = Box::new(Request::new(
            input,
            options.destination,
            Options {
                profile,
                tools: environment.toolchain(options.cc.as_deref(), options.glslc.as_deref()),
            },
        )?);
        Ok(if interpret {
            Mode::Interpreter {
                request,
                args: self.program_args,
            }
        } else {
            Mode::Compiler(request)
        })
    }
}
