use crate::lower::namespaces::{SourceModuleId, SourceOrigin};
use crate::lower::semantic::SemanticData;
use resin_common::prelude::*;

use std::{cell::RefCell, rc::Rc};
use std::{collections::BTreeMap, sync::Arc};

use resin_ast::{Program, SourceFile, StmtKind};

use super::scope::Symbol;
use super::{Generator, Scopes};

struct Export {
    origin: SourceOrigin,
    symbol: Symbol,
}

type Exports = BTreeMap<Arc<str>, Export>;

use crate::CheckedProgram;

pub fn generate_program(program: &Program) -> Result<crate::Module, SourceError> {
    let mut compilation = analyze_program(program);
    if !compilation.diagnostics.is_empty() {
        Err(compilation.diagnostics.remove(0))
    } else {
        Ok(compilation.module.expect("successful compilation"))
    }
}

/// Check modules in import order while retaining independent editor facts after errors.
pub fn analyze_program(program: &Program) -> CheckedProgram {
    let diagnostics = invalid_dependencies(program);
    if !diagnostics.is_empty() {
        return CheckedProgram {
            module: None,
            diagnostics,
            semantics: Default::default(),
        };
    }
    let mut builder = ProgramBuilder::new(program);
    for index in 0..program.modules.len() {
        builder.module(index);
    }
    builder.entries();
    builder.finish()
}

fn invalid_dependencies(program: &Program) -> Vec<SourceError> {
    let mut errors = Vec::new();
    for (index, source) in program.modules.iter().enumerate() {
        for &(span, dependency) in &source.imports {
            if dependency >= index {
                errors.push(source.error(span, "import dependency must precede its consumer"));
            }
        }
    }
    errors
}

struct ProgramBuilder<'a> {
    program: &'a Program,
    generator: Generator,
    data: Rc<RefCell<SemanticData>>,
    exports: Vec<Exports>,
    diagnostics: Vec<SourceError>,
}

impl<'a> ProgramBuilder<'a> {
    fn new(program: &'a Program) -> Self {
        Self {
            program,
            generator: Generator::new(),
            data: Default::default(),
            exports: vec![],
            diagnostics: vec![],
        }
    }

    fn module(&mut self, index: usize) {
        let source = &self.program.modules[index];
        self.begin_module(index);
        let mut scopes = Scopes::for_source(source.path.clone(), self.data.clone());
        let mut names = BTreeMap::new();
        self.imports(index, &mut scopes, &mut names);
        self.declarations(index, &mut names);
        self.generator.generate_file(&source.file, scopes);
        self.diagnostics.extend(
            std::mem::take(&mut self.generator.errors)
                .into_iter()
                .map(|e| source.error(e.span, e)),
        );
        self.export_module(index, &names);
    }

    fn begin_module(&mut self, index: usize) {
        let source = &self.program.modules[index];
        self.generator.source_path = source.path.clone();
        self.generator.source_module = SourceModuleId::from_index(index);
        self.generator
            .module
            .origins
            .sources
            .insert(source.path.clone(), source.source.as_str().into());
    }

    fn imports(
        &mut self,
        index: usize,
        scopes: &mut Scopes,
        names: &mut BTreeMap<Arc<str>, SourceOrigin>,
    ) {
        let source = &self.program.modules[index];
        for &(span, dependency) in &source.imports {
            self.data.borrow_mut().imports.insert(
                SourceLocation {
                    path: source.path.clone(),
                    span,
                },
                self.program.modules[dependency].path.clone(),
            );
            self.import_exports(index, dependency, span, scopes, names);
        }
    }

    fn import_exports(
        &mut self,
        index: usize,
        dependency: usize,
        span: Span,
        scopes: &mut Scopes,
        names: &mut BTreeMap<Arc<str>, SourceOrigin>,
    ) {
        for (name, export) in &self.exports[dependency] {
            match bind(self.program, index, names, name, export.origin, span) {
                Ok(false) => continue,
                Err(error) => self.diagnostics.push(error),
                Ok(true) => {}
            }
            scopes.import(name.clone(), export.symbol);
        }
    }

    fn declarations(&mut self, index: usize, names: &mut BTreeMap<Arc<str>, SourceOrigin>) {
        for stmt in self.program.modules[index].file.declarations() {
            let Some(name) = declaration_name(&stmt.val) else {
                continue;
            };
            let origin = SourceOrigin {
                module: SourceModuleId::from_index(index),
                span: name.span,
            };
            if let Err(error) = bind(self.program, index, names, &name.val, origin, name.span) {
                self.diagnostics.push(error);
            }
        }
    }

    fn export_module(&mut self, index: usize, names: &BTreeMap<Arc<str>, SourceOrigin>) {
        let source = &self.program.modules[index];
        let (symbols, errors) = self.generator.exports(&source.file);
        self.diagnostics
            .extend(errors.into_iter().map(|e| source.error(e.span, e)));
        self.exports.push(
            symbols
                .into_iter()
                .filter_map(|(name, symbol)| {
                    names
                        .get(&name)
                        .copied()
                        .map(|origin| (name, Export { origin, symbol }))
                })
                .collect(),
        );
    }

    fn entries(&mut self) {
        let Some(source) = self.program.modules.last() else {
            return;
        };
        match self.generator.exported_functions(&source.file) {
            Ok(entries) => self.generator.module.entries = entries,
            Err(error) => self.diagnostics.push(source.error(error.span, error)),
        }
    }

    fn finish(mut self) -> CheckedProgram {
        let module = if self.diagnostics.is_empty() {
            self.generator
                .finish()
                .map_err(|error| {
                    self.diagnostics.push(program_error(self.program, error));
                })
                .ok()
        } else {
            None
        };
        CheckedProgram {
            module,
            diagnostics: self.diagnostics,
            semantics: crate::Analysis(self.data.borrow().clone()),
        }
    }
}

fn declaration_name(stmt: &StmtKind) -> Option<&Ident> {
    match stmt {
        StmtKind::ForeignType { name }
        | StmtKind::ForeignFunction { name, .. }
        | StmtKind::Function { name, .. }
        | StmtKind::Define { name, .. }
        | StmtKind::DefineType { name, .. }
        | StmtKind::Struct { name, .. }
        | StmtKind::Declare { name, .. } => Some(name),
        _ => None,
    }
}

fn program_error(program: &Program, error: GenerateError) -> SourceError {
    if let Some(source) = program.modules.last() {
        source.error(error.span, error)
    } else {
        SourceError::new(Default::default(), Some(error.span), error.to_string())
    }
}

fn bind(
    program: &Program,
    module: usize,
    names: &mut BTreeMap<Arc<str>, SourceOrigin>,
    name: &Arc<str>,
    origin: SourceOrigin,
    span: Span,
) -> Result<bool, SourceError> {
    if let Some(previous) = names.get(name) {
        if *previous == origin {
            return Ok(false);
        }
        let location =
            |origin: &SourceOrigin| program.modules[origin.module.index()].location(origin.span);
        let mut error = program.modules[module].error(
            span,
            format!(
                "conflicting binding `{name}`\n  first defined at {}\n  also defined at {}",
                location(previous),
                location(&origin),
            ),
        );
        for origin in [previous, &origin] {
            error.related.push(SourceNote {
                location: SourceLocation {
                    path: program.modules[origin.module.index()].path.clone(),
                    span: origin.span,
                },
                message: "defined here".into(),
            });
        }
        return Err(error);
    }
    names.insert(name.clone(), origin);
    Ok(true)
}

impl Generator {
    pub(super) fn exported_functions(
        &self,
        file: &SourceFile,
    ) -> Result<BTreeMap<Arc<str>, FunctionId>, GenerateError> {
        let (symbols, errors) = self.exports(file);
        if let Some(error) = errors.into_iter().next() {
            return Err(error);
        }
        Ok(symbols
            .into_iter()
            .filter_map(|(name, symbol)| Some((name, self.environment.binding(symbol.definition)?)))
            .collect())
    }
    pub(super) fn exports(
        &self,
        file: &SourceFile,
    ) -> (BTreeMap<Arc<str>, Symbol>, Vec<GenerateError>) {
        let mut exports = BTreeMap::new();
        let mut errors = Vec::new();
        for name in &file.exports {
            if exports.contains_key(&name.val) {
                errors.push(GenerateError {
                    span: name.span,
                    kind: GenerateErrorKind::DuplicateExport {
                        name: name.val.clone(),
                    },
                });
            } else if let Some(symbol) = self.environment.symbol(&name.val) {
                exports.insert(name.val.clone(), symbol);
            } else {
                errors.push(GenerateError {
                    span: name.span,
                    kind: GenerateErrorKind::UnknownExport {
                        name: name.val.clone(),
                    },
                });
            }
        }
        (exports, errors)
    }
}
