#![allow(dead_code)]
use super::{frontend, shaders};
use resin_executor::Cancellation;
use resin_types::prelude::*;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::Arc,
};
use tempfile::TempDir;

type Error = Box<dyn std::error::Error>;

/// Compiler artifacts and native fixtures kept alive by one language test.
pub struct Project {
    pub generated: Generated,
    checked: Arc<resin_lir::VerifiedModule>,
    entry: Option<String>,
    directory: TempDir,
    native: String,
}

impl std::fmt::Debug for Project {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Project")
            .field("entry", &self.entry)
            .field("directory", &self.directory)
            .finish()
    }
}

#[derive(Debug)]
pub struct Generated {
    program: Option<PathBuf>,
    shaders: Vec<Shader>,
}

impl Generated {
    pub fn program(&self) -> Option<&Path> {
        self.program.as_deref()
    }
    pub fn shaders(&self) -> &[Shader] {
        &self.shaders
    }
}

#[derive(Debug)]
pub struct Shader {
    function: FunctionId,
    stage: Stage,
    unoptimized: PathBuf,
    optimized: PathBuf,
}

impl Shader {
    pub fn function(&self) -> FunctionId {
        self.function
    }
    pub fn stage(&self) -> Stage {
        self.stage
    }
    pub fn unoptimized_spirv(&self) -> &Path {
        &self.unoptimized
    }
    pub fn spirv(&self) -> &Path {
        &self.optimized
    }
}

pub struct Built {
    directory: TempDir,
    executable: Option<resin_toolchain::Executable>,
}

impl Built {
    pub fn path(&self, relative: impl AsRef<Path>) -> PathBuf {
        self.directory.path().join(relative)
    }
    pub fn executable(
        &self,
        relative: impl AsRef<Path>,
    ) -> Result<resin_toolchain::Executable, Error> {
        if relative.as_ref() != Path::new(&program_name()) {
            return Err("unknown fixture executable".into());
        }
        self.executable
            .clone()
            .ok_or_else(|| "shader-only fixture has no executable".into())
    }
}

impl Project {
    pub fn new(module: &resin_lir::Module, entry: Option<&str>) -> Result<Self, Error> {
        if let Some(entry) = entry
            && !module.entries.contains_key(entry)
        {
            return Err(format!("entry {entry} is not exported; add export {{ {entry} }};").into());
        }
        if entry.is_none() && module.shaders.is_empty() {
            return Err("no shader entries were requested".into());
        }
        let directory = TempDir::new()?;
        let checked = Arc::new(resin_lir::VerifiedModule::new(module.clone())?);
        let mut shaders = Vec::new();
        for (&function, shader) in &module.shaders {
            if entry.is_some() && !shader.embedded {
                continue;
            }
            let bytes = frontend::block_on(resin_codegen::generate_spirv(
                checked.clone(),
                function,
                frontend::execution(),
                &Cancellation::new(),
            ))?;
            let unoptimized = directory
                .path()
                .join(format!("shader_{}.unoptimized.spv", function.index()));
            fs::write(&unoptimized, bytes)?;
            shaders.push(Shader {
                function,
                stage: shader.stage.parse()?,
                unoptimized,
                optimized: PathBuf::from(format!("shader_{}.spv", function.index())),
            });
        }
        let generated = Generated {
            program: entry.map(|_| PathBuf::from(program_name())),
            shaders,
        };
        Ok(Self {
            generated,
            checked,
            entry: entry.map(str::to_owned),
            directory,
            native: String::new(),
        })
    }

    /// Include explicit native test shims before the ordinary header declarations.
    pub fn with_native(mut self, native: impl Into<String>) -> Self {
        self.native = native.into();
        self
    }

    pub fn run(&self) -> Output {
        let executable = self.build_executable();
        Command::new(executable.path()).output().unwrap()
    }

    pub fn build_executable(&self) -> resin_toolchain::Executable {
        let mut environment = resin_toolchain::Environment::capture().unwrap();
        environment.directory = self.directory.path().into();
        environment.executable = env!("CARGO_BIN_EXE_resin").into();
        self.build(&environment.toolchain(None, None))
            .unwrap()
            .executable
            .unwrap()
    }

    pub fn build(&self, tools: &resin_toolchain::Toolchain) -> Result<Built, Error> {
        frontend::block_on(async {
            let directory = TempDir::new()?;
            let cancellation = Cancellation::new();
            let execution = frontend::execution();
            let mut inputs = resin_codegen::NativeInputs::default();
            for shader in self.generated.shaders() {
                shaders::validate(shader.unoptimized_spirv());
                let raw = fs::read(shader.unoptimized_spirv())?.into();
                let bytes = tools
                    .optimize_shader(raw, directory.path(), execution, &cancellation)
                    .await?;
                let path = directory.path().join(shader.spirv());
                fs::write(&path, &bytes)?;
                shaders::validate(&path);
                inputs.shaders.insert(shader.function, bytes);
            }
            let executable = if let Some(entry) = &self.entry {
                let foreign = self.foreign_inputs(&mut inputs)?;
                let mut objects = Vec::new();
                if !foreign.includes.is_empty() || !foreign.functions.is_empty() {
                    objects.push(
                        tools
                            .compile_foreign(
                                Arc::new(foreign),
                                directory.path(),
                                execution,
                                &cancellation,
                            )
                            .await?
                            .bytes(),
                    );
                }
                let object = resin_codegen::generate_native(
                    self.checked.clone(),
                    entry.clone(),
                    resin_codegen::NativeOptimization::Speed,
                    Arc::new(inputs),
                    execution,
                    &cancellation,
                )
                .await?;
                objects.insert(0, object.shared_bytes());
                // The executable's owned generation outlives this input fixture.
                Some(
                    tools
                        .link_native(
                            resin_toolchain::NativeLink {
                                objects,
                                runtime: true,
                            },
                            &std::env::temp_dir(),
                            execution,
                            &cancellation,
                        )
                        .await?,
                )
            } else {
                None
            };
            Ok(Built {
                directory,
                executable,
            })
        })
    }

    fn foreign_inputs(
        &self,
        bindings: &mut resin_codegen::NativeInputs,
    ) -> Result<resin_toolchain::ForeignInputs, Error> {
        use resin_toolchain::{ForeignFunction, ForeignInputs, ForeignScalar};
        let module = self.checked.view().module();
        let mut inputs = ForeignInputs::default();
        if !module.foreign_headers.is_empty() || !self.native.is_empty() {
            let runtime = Path::new(resin_runtime::INCLUDE_DIR);
            copy_headers(runtime, runtime, "runtime", &mut inputs.files)?;
            inputs.include_directories.push("runtime".into());
        }
        if !self.native.is_empty() {
            inputs
                .files
                .insert("fixture.h".into(), self.native.as_bytes().into());
            inputs.includes.push("fixture.h".into());
            inputs.includes.push("resin_runtime.h".into());
            // This intrinsic is called directly by native code. Existing allocation
            // failure fixtures replace it with a header macro, just like source externs.
            let symbol = "resin_test_gpu_ptr_allocate";
            bindings
                .runtime
                .insert("resin_gpu_ptr_allocate".into(), symbol.into());
            let word = ForeignScalar::Integer {
                bits: 64,
                signed: false,
            };
            inputs.functions.push(ForeignFunction {
                symbol: symbol.into(),
                name: "resin_gpu_ptr_allocate".into(),
                params: vec![
                    ForeignScalar::Pointer,
                    ForeignScalar::Pointer,
                    word.clone(),
                    word,
                    ForeignScalar::Integer {
                        bits: 32,
                        signed: true,
                    },
                    ForeignScalar::Pointer,
                ],
                result: ForeignScalar::Integer {
                    bits: 32,
                    signed: true,
                },
            });
        }
        let mut roots = BTreeMap::<PathBuf, String>::new();
        for header in &module.foreign_headers {
            let path = Path::new(header.spelling.as_ref());
            let include = if path.is_absolute() || path.is_file() {
                let path = fs::canonicalize(path)?;
                let root = path.parent().unwrap();
                let prefix = if let Some(prefix) = roots.get(root) {
                    prefix.clone()
                } else {
                    let prefix = format!("fixture_{}", roots.len());
                    copy_headers(root, root, &prefix, &mut inputs.files)?;
                    inputs.include_directories.push(prefix.clone());
                    roots.insert(root.into(), prefix.clone());
                    prefix
                };
                format!("{prefix}/{}", path.file_name().unwrap().to_str().unwrap())
            } else {
                header.spelling.to_string()
            };
            if !inputs.includes.contains(&include) {
                inputs.includes.push(include);
            }
        }
        for (index, function) in module.functions.iter().enumerate() {
            let Some(foreign) = &function.foreign else {
                continue;
            };
            let symbol = format!("resin_test_foreign_{index}");
            bindings
                .foreign
                .insert(FunctionId::from_index(index), symbol.clone().into());
            inputs.functions.push(ForeignFunction {
                symbol,
                name: function.name.as_ref().unwrap().to_string(),
                params: foreign
                    .params
                    .iter()
                    .map(foreign_scalar)
                    .collect::<Result<_, _>>()?,
                result: foreign_scalar(&function.result)?,
            });
        }
        Ok(inputs)
    }
}

fn foreign_scalar(ty: &Ty) -> Result<resin_toolchain::ForeignScalar, Error> {
    use resin_toolchain::ForeignScalar;
    Ok(match ty {
        Ty::Unit => ForeignScalar::Void,
        Ty::Bool => ForeignScalar::Bool,
        Ty::Pointer { .. } => ForeignScalar::Pointer,
        Ty::Float32 => ForeignScalar::Float { bits: 32 },
        Ty::Float64 => ForeignScalar::Float { bits: 64 },
        ty if ty.is_integer() => ForeignScalar::Integer {
            bits: match ty {
                Ty::Int8 | Ty::UInt8 => 8,
                Ty::Int16 | Ty::UInt16 => 16,
                Ty::Int32 | Ty::UInt32 => 32,
                _ => 64,
            },
            signed: matches!(ty, Ty::Int8 | Ty::Int16 | Ty::Int32 | Ty::Int64),
        },
        _ => return Err(format!("invalid foreign fixture type {ty:?}").into()),
    })
}

fn copy_headers(
    root: &Path,
    directory: &Path,
    prefix: &str,
    files: &mut BTreeMap<Arc<str>, Arc<[u8]>>,
) -> Result<(), Error> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            copy_headers(root, &path, prefix, files)?;
        } else if entry.file_type()?.is_file() {
            let name = path
                .strip_prefix(root)?
                .to_string_lossy()
                .replace('\\', "/");
            files.insert(format!("{prefix}/{name}").into(), fs::read(path)?.into());
        }
    }
    Ok(())
}

fn program_name() -> String {
    format!("program{}", std::env::consts::EXE_SUFFIX)
}
