//! CLI syntax and conversion into an explicit execution mode.
use super::request::{Options, Request};
use super::{Environment, Result, source};
use clap::CommandFactory;
use resin_protocol::BuildProfile;
use std::{ffi::OsString, path::PathBuf};

pub enum Mode {
    Interpreter {
        request: Box<Request>,
        args: Vec<OsString>,
    },
    Embed {
        input: PathBuf,
        output: PathBuf,
        symbol: String,
    },
    Documentation {
        input: PathBuf,
        output: Option<PathBuf>,
    },
    Formatter {
        paths: Vec<PathBuf>,
        check: bool,
    },
    LanguageServer {
        directory: PathBuf,
        include_roots: Vec<PathBuf>,
    },
}

pub fn parse(args: impl IntoIterator<Item = OsString>, environment: &Environment) -> Result<Mode> {
    <Cli as clap::Parser>::parse_from(args).mode(environment)
}

#[derive(clap::Parser)]
#[command(name = "resin", version)]
struct Cli {
    /// A FILE[:ENTRY] to run/compile, a file for --doc, paths to --format, or a directory for --lsp.
    #[arg(required_unless_present_any = ["format", "lsp", "embed"], value_name = "PATH")]
    paths: Vec<PathBuf>,

    /// Arguments passed literally after --, or additional formatter paths.
    #[arg(last = true, value_name = "ARG", conflicts_with = "destination")]
    program_args: Vec<OsString>,

    /// Format files in place; search directories recursively for .resin files.
    #[arg(short = 'f', long, conflicts_with_all = ["destination"])]
    format: bool,

    /// Render exported declarations and doc comments as Markdown without a compiler service.
    #[arg(long, conflicts_with_all = ["format", "lsp", "embed", "check", "program_args", "include_roots"])]
    doc: bool,

    /// With --format, check without writing; exit 1 on differences or file/syntax errors.
    #[arg(long, requires = "format")]
    check: bool,

    /// Serve Language Server Protocol requests over stdin/stdout for this directory.
    #[arg(long, conflicts_with_all = ["format", "check", "destination", "program_args"])]
    lsp: bool,

    /// Convert a binary file into an aligned C byte array without adding a terminator.
    #[arg(long, value_name = "INPUT", requires_all = ["symbol", "destination"], conflicts_with_all = ["paths", "format", "lsp", "check", "program_args"])]
    embed: Option<PathBuf>,

    /// C array identifier for --embed; also defines <SYMBOL>_length.
    #[arg(long, requires = "embed")]
    symbol: Option<String>,

    #[command(flatten)]
    compile: CompileOptions,
}

#[derive(clap::Args)]
struct CompileOptions {
    /// Destination file, or directory for host executables. Executables use -O3 and are not run.
    #[arg(short = 'o', long = "output", visible_alias = "out")]
    destination: Option<PathBuf>,

    /// Upload this complete header directory, in include search order. May be repeated.
    #[arg(short = 'I', long = "include-root")]
    include_roots: Vec<PathBuf>,
}

impl Cli {
    fn mode(self, environment: &Environment) -> Result<Mode> {
        if let Some(input) = &self.embed {
            return Ok(Mode::Embed {
                input: environment.directory.join(input),
                output: environment
                    .directory
                    .join(self.compile.destination.as_ref().expect("required output")),
                symbol: self.symbol.expect("required symbol"),
            });
        }
        if self.lsp {
            return self.language_server(environment);
        }
        if self.format {
            return self.formatter(environment);
        }
        if self.doc {
            let [input] = self.paths.as_slice() else {
                return Err("--doc requires one Resin source file".into());
            };
            return Ok(Mode::Documentation {
                input: environment.directory.join(input),
                output: self
                    .compile
                    .destination
                    .map(|path| environment.directory.join(path)),
            });
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
        Ok(Mode::LanguageServer {
            directory,
            include_roots: self
                .compile
                .include_roots
                .iter()
                .map(|path| environment.directory.join(path))
                .collect(),
        })
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
        let input = self.input()?;
        Ok(Mode::Interpreter {
            request: Box::new(self.compile.request(input, environment)?),
            args: self.program_args,
        })
    }

    fn input(&self) -> Result<source::Input> {
        let [path] = self.paths.as_slice() else {
            Self::command().error(clap::error::ErrorKind::WrongNumberOfValues,
                "running or compiling requires exactly one FILE[:ENTRY]; use --format for multiple paths").exit();
        };
        let input = source::parse(path.as_os_str()).unwrap_or_else(|error| {
            Self::command()
                .error(clap::error::ErrorKind::InvalidValue, error)
                .exit()
        });
        Ok(input)
    }
}

impl CompileOptions {
    fn request(self, input: source::Input, environment: &Environment) -> Result<Request> {
        let destination = self.destination;
        let profile = if destination.is_none() {
            BuildProfile::Debug
        } else {
            BuildProfile::Release
        };
        Request::new(
            input,
            destination,
            Options {
                directory: environment.directory.clone(),
                profile,
                include_roots: self
                    .include_roots
                    .into_iter()
                    .map(|path| environment.directory.join(path))
                    .collect(),
                temporary: environment.temporary.clone(),
            },
        )
    }
}
