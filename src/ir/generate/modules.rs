use crate::{
    analysis::semantic::{SemanticData, Trace},
    ast::{SourceLocation, SourceNote},
};
use std::{cell::RefCell, rc::Rc};
use std::{collections::BTreeMap, sync::Arc};

use crate::ast::{Program, SourceError, SourceFile, Span, StmtKind};
use crate::ir::{FunctionId, Module};

use super::{
    GenerateError, GenerateErrorKind, Generator, Scopes,
    scope::{Symbol, ValueBindingKind},
};

#[derive(Clone, Copy, PartialEq, Eq)]
struct Origin {
    module: usize,
    span: Span,
}

struct Export {
    origin: Origin,
    symbol: Symbol,
}

type Exports = BTreeMap<Arc<str>, Export>;

pub fn generate_program(program: &Program) -> Result<Module, SourceError> {
    generate_program_with(program, None)
}

pub(crate) fn analyze_program(program: &Program) -> (SemanticData, Result<Module, SourceError>) {
    let data = Rc::new(RefCell::new(SemanticData::default()));
    let result = generate_program_with(program, Some(data.clone()));
    let snapshot = data.borrow().clone();
    (snapshot, result)
}

fn generate_program_with(
    program: &Program,
    trace: Option<Rc<RefCell<SemanticData>>>,
) -> Result<Module, SourceError> {
    let mut generator = Generator::new();
    let mut exports = Vec::<Exports>::new();
    for (index, source) in program.modules.iter().enumerate() {
        generator.scopes = trace.as_ref().map_or_else(Scopes::new, |data| {
            Scopes::traced(Trace {
                path: source.path.clone(),
                data: data.clone(),
            })
        });
        let mut names = BTreeMap::new();
        for (import, &dependency) in source.file.imports.iter().zip(&source.imports) {
            for (name, export) in &exports[dependency] {
                if bind(program, index, &mut names, name, export.origin, import.span)? {
                    generator.scopes.import(name.clone(), export.symbol.clone());
                    generator.scopes.record_import(
                        name.clone(),
                        matches!(export.symbol, Symbol::Type(_)),
                        SourceLocation {
                            path: program.modules[export.origin.module].path.clone(),
                            span: export.origin.span,
                        },
                    );
                }
            }
        }
        for stmt in &source.file.stmts {
            let name = match &stmt.val {
                StmtKind::ForeignType { name }
                | StmtKind::ForeignFunction { name, .. }
                | StmtKind::Function { name, .. }
                | StmtKind::Define { name, .. }
                | StmtKind::DefineType { name, .. }
                | StmtKind::Struct { name, .. }
                | StmtKind::Declare { name, .. } => name,
                StmtKind::Expr { .. } => continue,
            };
            let origin = Origin {
                module: index,
                span: name.span,
            };
            bind(program, index, &mut names, &name.val, origin, name.span)?;
        }
        generator
            .generate_file(&source.file)
            .map_err(|e| source.error(e.span, e))?;
        let symbols = generator
            .exports(&source.file)
            .map_err(|e| source.error(e.span, e))?;
        exports.push(
            symbols
                .into_iter()
                .map(|(name, symbol)| {
                    let origin = names[&name];
                    (name, Export { origin, symbol })
                })
                .collect(),
        );
    }
    if let Some(source) = program.modules.last() {
        generator.module.entries = generator
            .exported_functions(&source.file)
            .map_err(|e| source.error(e.span, e))?;
    }
    generator.finish().map_err(|e| {
        if let Some(source) = program.modules.last() {
            source.error(e.span, e)
        } else {
            SourceError::new(Default::default(), Some(e.span), e.to_string())
        }
    })
}

fn bind(
    program: &Program,
    module: usize,
    names: &mut BTreeMap<Arc<str>, Origin>,
    name: &Arc<str>,
    origin: Origin,
    span: Span,
) -> Result<bool, SourceError> {
    if let Some(previous) = names.get(name) {
        if *previous == origin {
            return Ok(false);
        }
        let location = |origin: &Origin| program.modules[origin.module].location(origin.span);
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
                    path: program.modules[origin.module].path.clone(),
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
        Ok(self
            .exports(file)?
            .into_iter()
            .filter_map(|(name, symbol)| match symbol {
                Symbol::Value(binding) => match binding.kind {
                    ValueBindingKind::Function(id) => Some((name, id)),
                    ValueBindingKind::Local(_) => unreachable!("module export"),
                },
                Symbol::Type(_) => None,
            })
            .collect())
    }

    pub(super) fn exports(
        &self,
        file: &SourceFile,
    ) -> Result<BTreeMap<Arc<str>, Symbol>, GenerateError> {
        let mut exports = BTreeMap::new();
        for name in &file.exports {
            if exports.contains_key(&name.val) {
                return Err(GenerateError {
                    span: name.span,
                    kind: GenerateErrorKind::DuplicateExport {
                        name: name.val.clone(),
                    },
                });
            }
            let symbol = self.scopes.symbol(&name.val).ok_or_else(|| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::UnknownExport {
                    name: name.val.clone(),
                },
            })?;
            self.scopes
                .record_reference(name, matches!(symbol, Symbol::Type(_)));
            exports.insert(name.val.clone(), symbol);
        }
        Ok(exports)
    }
}
