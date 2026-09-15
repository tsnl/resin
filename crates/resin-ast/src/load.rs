//! Assemble parsed modules from a frozen source graph; no source acquisition occurs here.
use crate::{BuiltProgram, ModuleDocument, Program, SourceModule};
use resin_source::prelude::*;
use std::{collections::BTreeMap, sync::Arc};

pub(super) fn assemble(
    inputs: resin_source::SourceGraph,
    documents: BTreeMap<Source, Arc<ModuleDocument>>,
    cancellation: &resin_executor::Cancellation,
) -> BuiltProgram {
    let source = inputs.entry().clone();
    let mut traversal = Traversal {
        inputs: &inputs,
        available: &documents,
        cancellation,
        program: Program {
            modules: Vec::new(),
        },
        diagnostics: Vec::new(),
        documents: BTreeMap::new(),
        loaded: BTreeMap::new(),
        active: Vec::new(),
    };
    if let Err(error) = traversal.visit(source.clone()) {
        traversal.diagnostics.push(error);
    }
    let Traversal {
        program,
        diagnostics,
        documents,
        ..
    } = traversal;
    BuiltProgram {
        source,
        inputs,
        program,
        documents,
        diagnostics,
    }
}

struct Traversal<'a> {
    inputs: &'a resin_source::SourceGraph,
    available: &'a BTreeMap<Source, Arc<ModuleDocument>>,
    cancellation: &'a resin_executor::Cancellation,
    program: Program,
    diagnostics: Vec<SourceError>,
    documents: BTreeMap<Source, Arc<ModuleDocument>>,
    loaded: BTreeMap<SourceId, usize>,
    active: Vec<Source>,
}

impl Traversal<'_> {
    fn visit(&mut self, source: Source) -> Result<usize, SourceError> {
        // Execution discards the candidate after cancellation. Stop graph work promptly.
        if self.cancellation.is_cancelled() {
            return Err(SourceError::new(
                source,
                None,
                "source assembly cancelled".into(),
            ));
        }
        if self.active.iter().any(|active| active.id() == source.id()) {
            let chain = self
                .active
                .iter()
                .chain(std::iter::once(&source))
                .map(Source::name)
                .collect::<Vec<_>>()
                .join(" -> ");
            return Err(SourceError::new(
                source,
                None,
                format!("cyclic source import: {chain}"),
            ));
        }
        if let Some(&id) = self.loaded.get(&source.id()) {
            return Ok(id);
        }
        let document = self.document(&source)?;
        let mut module = SourceModule {
            source: source.clone(),
            file: document.file.clone(),
            imports: Vec::new(),
        };
        self.diagnostics.extend(
            document
                .errors
                .iter()
                .map(|(span, error)| module.error(*span, error)),
        );
        self.documents.insert(source.clone(), document);
        self.active.push(source.clone());
        self.visit_imports(&mut module);
        self.active.pop();
        let id = self.program.modules.len();
        self.program.modules.push(module);
        self.loaded.insert(source.id(), id);
        Ok(id)
    }

    fn document(&self, source: &Source) -> Result<Arc<ModuleDocument>, SourceError> {
        let document = self.available.get(source).ok_or_else(|| {
            SourceError::new(
                source.clone(),
                None,
                "parsed source was not supplied for this input".into(),
            )
        })?;
        if document.source != *source || document.syntax.source() != source.text() {
            return Err(SourceError::new(
                source.clone(),
                None,
                "parsed source does not match the selected source contents".into(),
            ));
        }
        Ok(document.clone())
    }

    fn visit_imports(&mut self, module: &mut SourceModule) {
        for import in &module.file.imports {
            let start = self.diagnostics.len();
            let imported = self
                .inputs
                .resolve(&module.source.id(), &import.val)
                .cloned()
                .ok_or_else(|| {
                    module.error(
                        import.span,
                        format!("unresolved source import: {}", import.val),
                    )
                })
                .and_then(|source| self.visit(source));
            match imported {
                Ok(id) => module.imports.push((import.span, id)),
                Err(error) => self.diagnostics.push(error),
            }
            for error in &mut self.diagnostics[start..] {
                imported_at(error, module, import.span);
            }
        }
    }
}

fn imported_at(error: &mut SourceError, importer: &SourceModule, span: Span) {
    if error.span.is_none() {
        *error = importer.error(span, &*error);
    } else if error.source != importer.source {
        error.message = format!("{}: {}", importer.location(span), error.message).into();
        error.related.push(SourceNote {
            location: SourceLocation {
                source: importer.source.clone(),
                span,
            },
            message: "imported here".into(),
        });
    }
}
