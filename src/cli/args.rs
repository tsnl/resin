//! CLI syntax and conversion into an explicit execution mode.
use super::{Environment, Result, source};
use clap::CommandFactory;
use resin_compiler::{Options, Request};
use resin_toolchain::CProfile;
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
    LanguageServer {
        directory: PathBuf,
    },
}

pub fn parse(
    args: impl IntoIterator<Item = OsString>,
    environment: &Environment,
) -> Result<Invocation> {
    let mode = <Cli as clap::Parser>::parse_from(args).mode(environment)?;
    Ok(Invocation {
        mode,
        stdlib: environment.path("RESIN_STDLIB", resin_compiler::stdlib_path()),
    })
}

#[derive(clap::Parser)]
#[command(name = "resin", version)]
struct Cli {
    /// A FILE[:ENTRY] to run/compile, paths to --format, or a directory for --lsp.
    #[arg(required_unless_present_any = ["format", "lsp"], value_name = "PATH")]
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

    /// Serve Language Server Protocol requests over stdin/stdout for this directory.
    #[arg(long, conflicts_with_all = ["format", "check", "destination", "cc", "glslc", "program_args"])]
    lsp: bool,

    #[command(flatten)]
    compile: CompileOptions,
}

#[derive(clap::Args)]
struct CompileOptions {
    /// Destination file, or directory for host executables. Executables use -O3 and are not run.
    #[arg(short = 'o', long = "output", visible_alias = "out")]
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
        if self.lsp {
            return self.language_server(environment);
        }
        if self.format {
            return self.formatter(environment);
        }
        self.build(environment)
    }

    fn language_server(&self, environment: &Environment) -> Result<Mode> {
        let directory = match self.paths.as_slice() {
            [] => environment.directory.clone(),
            [path] => environment.directory.join(path),
            _ => return Err("--lsp accepts one project directory".into()),
        };
        if !directory.is_dir() {
            return Err(format!("LSP project is not a directory: {}", directory.display()).into());
        }
        Ok(Mode::LanguageServer { directory })
    }

    fn formatter(self, environment: &Environment) -> Result<Mode> {
        if self.paths.is_empty() && self.program_args.is_empty() {
            Self::command()
                .error(
                    clap::error::ErrorKind::MissingRequiredArgument,
                    "formatting requires at least one path",
                )
                .exit();
        }
        let paths = self
            .paths
            .into_iter()
            .chain(self.program_args.into_iter().map(PathBuf::from));
        Ok(Mode::Formatter {
            paths: paths.map(|path| environment.directory.join(path)).collect(),
            check: self.check,
        })
    }

    fn build(self, environment: &Environment) -> Result<Mode> {
        let input = self.input(environment)?;
        let request = Box::new(self.compile.request(input, environment)?);
        Ok(if request.destination().is_none() {
            Mode::Interpreter {
                request,
                args: self.program_args,
            }
        } else {
            Mode::Compiler(request)
        })
    }

    fn input(&self, environment: &Environment) -> Result<resin_compiler::Input> {
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
        Ok(input)
    }
}

impl CompileOptions {
    fn request(self, input: resin_compiler::Input, environment: &Environment) -> Result<Request> {
        let destination = self
            .destination
            .map(|path| environment.directory.join(path));
        let profile = if destination.is_none() {
            CProfile::Debug
        } else {
            CProfile::Release
        };
        let tools = environment.toolchain(self.cc.as_deref(), self.glslc.as_deref());
        Ok(Request::new(
            input,
            destination,
            Options { profile, tools },
        )?)
    }
}
