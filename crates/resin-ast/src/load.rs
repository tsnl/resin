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
        // Import depth uses heap storage, not one worker-stack frame per module.
        let mut pending = vec![self.begin(source)?];
        while let Some(frame) = pending.last_mut() {
            // Execution discards cancelled candidates; also stop cached-edge work.
            if self.cancellation.is_cancelled() {
                return Err(SourceError::new(
                    frame.module.source.clone(),
                    None,
                    "source assembly cancelled".into(),
                ));
            }
            if let Some(import) = frame.module.file.imports.get(frame.next_import).cloned() {
                frame.next_import += 1;
                let source = self.resolve(&frame.module, &import);
                match source {
                    Ok(source) if self.loaded.contains_key(&source.id()) => {
                        frame
                            .module
                            .imports
                            .push((import.span, self.loaded[&source.id()]));
                    }
                    source => match source.and_then(|source| self.begin(source)) {
                        Ok(child) => pending.push(child),
                        Err(mut error) => {
                            imported_at(&mut error, &frame.module, import.span);
                            self.diagnostics.push(error);
                        }
                    },
                }
                continue;
            }
            let completed = pending.pop().unwrap();
            self.active.pop();
            let id = self.program.modules.len();
            self.loaded.insert(completed.module.source.id(), id);
            self.program.modules.push(completed.module);
            let Some(parent) = pending.last_mut() else {
                return Ok(id);
            };
            let span = parent.module.file.imports[parent.next_import - 1].span;
            parent.module.imports.push((span, id));
            for error in &mut self.diagnostics[completed.diagnostics_start..] {
                imported_at(error, &parent.module, span);
            }
        }
        unreachable!("the root module completes the traversal")
    }

    fn resolve(
        &self,
        module: &SourceModule,
        import: &Spanned<Arc<str>>,
    ) -> Result<Source, SourceError> {
        self.inputs
            .resolve(&module.source.id(), &import.val)
            .cloned()
            .ok_or_else(|| {
                module.error(
                    import.span,
                    format!("unresolved source import: {}", import.val),
                )
            })
    }

    fn begin(&mut self, source: Source) -> Result<Frame, SourceError> {
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
        let document = self.document(&source)?;
        let module = SourceModule {
            source: source.clone(),
            file: document.file.clone(),
            imports: Vec::new(),
        };
        let diagnostics_start = self.diagnostics.len();
        self.diagnostics.extend(
            document
                .errors
                .iter()
                .map(|(span, error)| module.error(*span, error)),
        );
        self.documents.insert(source.clone(), document);
        self.active.push(source);
        Ok(Frame {
            module,
            next_import: 0,
            diagnostics_start,
        })
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
}

struct Frame {
    module: SourceModule,
    next_import: usize,
    diagnostics_start: usize,
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
