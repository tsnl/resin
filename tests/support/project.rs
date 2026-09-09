#![allow(dead_code)]
use super::shaders;
use std::process::{Command, Output};
use tempfile::TempDir;

/// Generated files remain available until the fixture is dropped.
pub struct Project {
    pub generated: resin_codegen::GeneratedProject,
    directory: TempDir,
}

impl Project {
    pub fn new(
        module: &resin_lir::Module,
        entry: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let directory = TempDir::new_in(std::env::temp_dir())?;
        let checked = resin_lir::VerifiedModule::new(module.clone())?;
        let generated = resin_codegen::generate(checked.view(), entry, directory.path())?;
        Ok(Self {
            generated,
            directory,
        })
    }

    pub fn run(&self) -> Output {
        let mut environment = resin_toolchain::Environment::capture().unwrap();
        environment.directory = self.directory.path().into();
        environment.executable = env!("CARGO_BIN_EXE_resin").into();
        let artifacts = self.build(&environment.toolchain(None, None)).unwrap();
        let program = self
            .generated
            .program()
            .unwrap()
            .strip_prefix(self.generated.directory())
            .unwrap();
        let executable = artifacts.executable(program).unwrap();
        Command::new(executable.path()).output().unwrap()
    }

    pub fn build(
        &self,
        tools: &resin_toolchain::Toolchain,
    ) -> Result<resin_toolchain::BuiltProject, resin_toolchain::Error> {
        for shader in self.generated.shaders() {
            shaders::validate(shader.unoptimized_spirv());
        }
        let built = tools.build(
            self.generated.directory(),
            self.generated.name(),
            self.generated.entry().unwrap_or("shaders"),
            resin_toolchain::CProfile::Release,
        )?;
        for shader in self.generated.shaders() {
            shaders::validate(&built.path(shader.spirv().file_name().unwrap()));
        }
        Ok(built)
    }
}

impl std::fmt::Debug for Project {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Project")
            .field("directory", &self.directory.path())
            .finish()
    }
}
