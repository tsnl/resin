use crate::{
    ast::{SourceLocation, SourceNote},
    ir::generate::semantic::SemanticData,
};
use std::{cell::RefCell, rc::Rc};
use std::{collections::BTreeMap, sync::Arc};

use crate::ast::{Program, SourceError, SourceFile, Span, StmtKind};
use crate::ir::{
    FunctionId, Module,
    typecheck::{SourceModuleId, SourceOrigin},
};

use super::{
    GenerateError, GenerateErrorKind, Generator, Scopes,
    scope::{Symbol, ValueBindingKind},
};

struct Export {
    origin: SourceOrigin,
    symbol: Symbol,
}

type Exports = BTreeMap<Arc<str>, Export>;

pub(crate) struct Compilation {
    pub module: Option<crate::ir::verify::VerifiedModule>,
    pub diagnostics: Vec<SourceError>,
    pub semantics: SemanticData,
}

pub fn generate_program(program: &Program) -> Result<Module, SourceError> {
    let mut compilation = analyze_program(program);
    if !compilation.diagnostics.is_empty() {
        Err(compilation.diagnostics.remove(0))
    } else {
        Ok(compilation
            .module
            .expect("successful compilation")
            .into_module())
    }
}

pub(crate) fn analyze_program(program: &Program) -> Compilation {
    let data = Rc::new(RefCell::new(SemanticData::default()));
    let mut diagnostics = Vec::new();
    let mut generator = Generator::new();
    let mut exports = Vec::<Exports>::new();
    for (index, source) in program.modules.iter().enumerate() {
        generator.source_path = source.path.clone();
        generator.source_module = SourceModuleId::from_index(index);
        generator
            .module
            .origins
            .sources
            .insert(source.path.clone(), source.source.as_str().into());
        let mut scopes = Scopes::for_source(source.path.clone(), data.clone());
        let mut names = BTreeMap::new();
        for &(span, dependency) in &source.imports {
            data.borrow_mut().imports.insert(
                SourceLocation {
                    path: source.path.clone(),
                    span,
                },
                program.modules[dependency].path.clone(),
            );
            for (name, export) in &exports[dependency] {
                match bind(program, index, &mut names, name, export.origin, span) {
                    Ok(false) => continue,
                    Err(error) => diagnostics.push(error),
                    Ok(true) => {}
                }
                scopes.import(name.clone(), export.symbol);
            }
        }
        for stmt in source.file.declarations() {
            let name = match &stmt.val {
                StmtKind::ForeignType { name }
                | StmtKind::ForeignFunction { name, .. }
                | StmtKind::Function { name, .. }
                | StmtKind::Define { name, .. }
                | StmtKind::DefineType { name, .. }
                | StmtKind::Struct { name, .. }
                | StmtKind::Declare { name, .. } => name,
                _ => continue,
            };
            if let Err(error) = bind(
                program,
                index,
                &mut names,
                &name.val,
                SourceOrigin {
                    module: generator.source_module,
                    span: name.span,
                },
                name.span,
            ) {
                diagnostics.push(error);
            }
        }
        generator.generate_file(&source.file, scopes);
        diagnostics.extend(
            std::mem::take(&mut generator.errors)
                .into_iter()
                .map(|e| source.error(e.span, e)),
        );
        let (symbols, errors) = generator.exports(&source.file);
        diagnostics.extend(errors.into_iter().map(|e| source.error(e.span, e)));
        exports.push(
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
    if let Some(source) = program.modules.last() {
        match generator.exported_functions(&source.file) {
            Ok(entries) => generator.module.entries = entries,
            Err(e) => diagnostics.push(source.error(e.span, e)),
        }
    }
    let mut module = None;
    if diagnostics.is_empty() {
        match generator.finish() {
            Ok(checked) => module = Some(checked),
            Err(e) => diagnostics.push(if let Some(source) = program.modules.last() {
                source.error(e.span, e)
            } else {
                SourceError::new(Default::default(), Some(e.span), e.to_string())
            }),
        }
    }
    Compilation {
        module,
        diagnostics,
        semantics: data.borrow().clone(),
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
            .filter_map(
                |(name, symbol)| match self.environment.binding(symbol.definition)?.kind {
                    ValueBindingKind::Function(id) => Some((name, id)),
                    ValueBindingKind::Local(_) => None,
                },
            )
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
