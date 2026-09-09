//! AST → HIR: declare symbols, solve types, and elaborate a resolved tree.
use crate::Module;
use crate::ast::{SourceFile, Span};
use crate::lower::context::Context;
use crate::lower::namespaces::SourceModuleId;
use environment::Environment;
use scope::Scopes;

mod annotation;
mod builtin_methods;
mod builtins;
mod check;
pub(crate) mod context;
mod declarations;
mod elaborate;
mod environment;
pub(crate) mod eval;
mod file;
mod functions;
pub(crate) mod infer;
mod methods;
mod modules;
pub(crate) mod namespaces;
pub(crate) mod scope;
pub(crate) mod semantic;
mod typed;
pub use crate::diagnostic::{GenerateError, GenerateErrorKind};
pub use modules::{Compilation, analyze_program, generate_program};

/// Check a standalone source file. Imports require `generate_program`.
pub fn generate(file: &SourceFile) -> Result<Module, GenerateError> {
    if let Some(import) = file.imports.first() {
        return Err(GenerateError {
            span: import.span,
            kind: GenerateErrorKind::UnresolvedImport {
                path: import.val.clone(),
            },
        });
    }
    let mut generator = Generator::new();
    generator.generate_file(file, Scopes::new());
    if !generator.errors.is_empty() {
        return Err(generator.errors.remove(0));
    }
    generator.module.entries = generator.exported_functions(file)?;
    generator.finish()
}

struct Generator {
    module: Module,
    source_path: std::path::PathBuf,
    source_module: SourceModuleId,
    typer: Context,
    environment: Environment,
    errors: Vec<GenerateError>,
}
impl Generator {
    fn new() -> Self {
        Self {
            module: Module::default(),
            source_path: "<source>".into(),
            source_module: SourceModuleId::from_index(0),
            typer: builtins::typer(),
            environment: Environment::new(),
            errors: vec![],
        }
    }
    fn generate_file(&mut self, file: &SourceFile, mut scopes: Scopes) {
        self.errors
            .extend(scopes.prepare(file, &mut self.typer, self.source_module));
        let methods = self.declare_methods(file, &mut scopes);
        let checked = check::file(file, &mut self.typer, scopes, self.source_module, methods);
        self.errors.extend(checked.errors.iter().cloned());
        self.environment.context = checked.context.clone();
        self.declare_checked_functions(file, &checked);
        self.elaborate_functions(file, &checked);
    }

    fn finish(mut self) -> Result<Module, GenerateError> {
        self.module.types = self
            .typer
            .into_definitions()
            .map_err(|error| GenerateError::typing(Span { start: 0, end: 0 }, error))?;
        Ok(self.module)
    }
}
