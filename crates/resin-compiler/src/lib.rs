//! Compile immutable named sources, retaining phase products and editor queries.
//!
//! Loaders discover imports. The compiler has no filesystem, editor buffers, or
//! change notifications. Each call resolves the dependency graph before reusing
//! work. Codegen and native builds consume its verified output separately.
//!
//! ```compile_fail,E0603
//! use resin_compiler::{ImportTraversal, ParsedDocument};
//! ```
use resin_source::prelude::*;
use std::{
    collections::BTreeMap,
    sync::{Arc, Weak},
};

//
// Compilation
//

/// Reusable compilation caches. Sources and compilations are immutable.
#[derive(Default)]
pub struct Compiler {
    parsed: BTreeMap<SourceId, Weak<ParsedDocument>>,
    checked: BTreeMap<SourceId, Arc<Compilation>>,
}
impl Compiler {
    pub fn new() -> Self {
        Self::default()
    }
    /// Resolve all imports, then reuse or construct the complete compilation.
    /// Source and import errors are retained as diagnostics in the result.
    pub fn compile(
        &mut self,
        entry: Source,
        loader: &mut resin_source::Loader,
    ) -> Arc<Compilation> {
        let loaded = self.load_sources(entry.clone(), loader);
        self.checked
            .retain(|id, old| *id == entry.id() || Arc::strong_count(old) > 1);
        let old = self.checked.get(&entry.id());
        if loaded.errors.is_empty()
            && let Some(old) = old.filter(|old| old.matches(&loaded))
        {
            return old.clone();
        }
        let result = Arc::new(Compilation::new(entry.clone(), loaded));
        self.checked.insert(entry.id(), result.clone());
        self.parsed
            .retain(|_, document| document.strong_count() > 0);
        result
    }
}

//
// Retained results and editor queries
//

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub location: SourceLocation,
    pub message: String,
    pub related: Vec<SourceNote>,
}

/// Immutable phase products and editor facts for an entry and its imports.
/// Failed later passes preserve successfully completed earlier products.
///
/// ```compile_fail,E0596
/// fn edit(compilation: &mut resin_compiler::Compilation) {
///     compilation.module().unwrap().functions.clear();
/// }
/// ```
pub struct Compilation {
    entry: Source,
    documents: BTreeMap<Source, Arc<ParsedDocument>>,
    syntax: BTreeMap<Source, Arc<resin_cst::Document>>,
    diagnostics: Vec<Diagnostic>,
    semantics: resin_hir::Analysis,
    graph: Vec<(Source, Vec<(Span, usize)>)>,
    program: Result<resin_ast::Program, SourceError>,
    hir: Result<resin_hir::Module, SourceError>,
    module: Result<resin_lir::VerifiedModule, SourceError>,
}
impl Compilation {
    pub fn entry(&self) -> &Source {
        &self.entry
    }
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
    pub fn sources(&self) -> impl Iterator<Item = &Source> {
        self.documents.keys()
    }
    pub fn program(&self) -> Result<&resin_ast::Program, SourceError> {
        self.program.as_ref().map_err(Clone::clone)
    }
    pub fn hir(&self) -> Result<&resin_hir::Module, SourceError> {
        self.hir.as_ref().map_err(Clone::clone)
    }
    pub fn module(&self) -> Result<&resin_lir::Module, SourceError> {
        self.module
            .as_ref()
            .map(|checked| checked.view().module())
            .map_err(Clone::clone)
    }
    /// Borrow the verified LIR certificate required by code generation.
    pub fn verified(&self) -> Result<resin_lir::Verified<'_>, SourceError> {
        self.module
            .as_ref()
            .map(|module| module.view())
            .map_err(Clone::clone)
    }
    pub fn recovered_file(&self, source: &Source) -> Option<&resin_ast::SourceFile> {
        self.documents.get(source).map(|document| &document.file)
    }
    pub fn definition(&self, source: &Source, offset: usize) -> Option<SourceLocation> {
        self.semantics.definition(&self.syntax, source, offset)
    }
    pub fn hover(&self, source: &Source, offset: usize) -> Option<resin_hir::Hover> {
        self.semantics.hover(&self.syntax, source, offset)
    }
    pub fn completions(&self, source: &Source, offset: usize) -> Vec<resin_hir::Completion> {
        self.semantics.completions(&self.syntax, source, offset)
    }
}

//
// Parsing and import traversal
//

impl Compiler {
    fn load_sources(&mut self, entry: Source, loader: &mut resin_source::Loader) -> Loaded {
        let mut traversal = ImportTraversal {
            loader,
            compiler: self,
            result: Loaded {
                program: resin_ast::Program {
                    modules: Vec::new(),
                },
                errors: Vec::new(),
                documents: BTreeMap::new(),
            },
            loaded: BTreeMap::new(),
            active: Vec::new(),
            versions: BTreeMap::new(),
            resolutions: BTreeMap::new(),
        };
        if let Err(error) = traversal.visit(entry) {
            traversal.result.errors.push(error);
        }
        traversal.result
    }

    fn parse(&mut self, source: &Source) -> Arc<ParsedDocument> {
        let old = self.parsed.get(&source.id()).and_then(Weak::upgrade);
        if let Some(old) = old.as_ref().filter(|old| old.source == *source) {
            return old.clone();
        }
        let document = Arc::new(ParsedDocument::reparse(source.clone(), old.as_deref()));
        self.parsed.insert(source.id(), Arc::downgrade(&document));
        document
    }
}

impl Compilation {
    fn matches(&self, loaded: &Loaded) -> bool {
        self.program.is_ok() && self.graph == loaded.graph()
    }

    fn new(entry: Source, loaded: Loaded) -> Self {
        let graph = loaded.graph();
        let load_error = loaded.errors.first().cloned();
        let mut checked = resin_hir::analyze_program(&loaded.program);
        let hir_error = load_error
            .clone()
            .or_else(|| checked.diagnostics.first().cloned());
        let mut lowered = None;
        if let Some(hir) = &checked.module {
            match lower_to_verified_lir(&loaded.program, hir) {
                Ok(module) => lowered = Some(module),
                Err(errors) => checked.diagnostics.extend(errors),
            }
        }
        let errors = loaded
            .errors
            .into_iter()
            .chain(checked.diagnostics)
            .collect::<Vec<_>>();
        let compile_error = errors.first().cloned();
        Self {
            entry,
            syntax: loaded
                .documents
                .iter()
                .map(|(source, document)| (source.clone(), document.syntax.clone()))
                .collect(),
            documents: loaded.documents,
            graph,
            diagnostics: errors.into_iter().map(diagnostic).collect(),
            semantics: checked.semantics,
            program: load_error.map_or(Ok(loaded.program), Err),
            hir: hir_error.map_or_else(|| Ok(checked.module.expect("successful HIR")), Err),
            module: compile_error.map_or_else(
                || Ok(lowered.expect("successful compilation has verified LIR")),
                Err,
            ),
        }
    }
}

fn diagnostic(error: SourceError) -> Diagnostic {
    Diagnostic {
        location: SourceLocation {
            source: error.source,
            span: error.span.unwrap_or(Span { start: 0, end: 0 }),
        },
        message: error.diagnostic,
        related: error.related,
    }
}

struct ParsedDocument {
    source: Source,
    syntax: Arc<resin_cst::Document>,
    file: resin_ast::SourceFile,
    errors: Vec<(Span, String)>,
}
impl ParsedDocument {
    fn reparse(source: Source, previous: Option<&Self>) -> Self {
        let syntax =
            resin_cst::Document::reparse(source.text().into(), previous.map(|d| d.syntax.as_ref()));
        let resin_ast::Parsed { file, errors } = resin_ast::recover(&syntax);
        Self {
            source,
            syntax: Arc::new(syntax),
            file,
            errors,
        }
    }
}

struct Loaded {
    program: resin_ast::Program,
    errors: Vec<SourceError>,
    documents: BTreeMap<Source, Arc<ParsedDocument>>,
}

impl Loaded {
    fn graph(&self) -> Vec<(Source, Vec<(Span, usize)>)> {
        self.program
            .modules
            .iter()
            .map(|module| (module.source.clone(), module.imports.clone()))
            .collect()
    }
}

struct ImportTraversal<'a> {
    loader: &'a mut resin_source::Loader,
    compiler: &'a mut Compiler,
    result: Loaded,
    loaded: BTreeMap<SourceId, usize>,
    active: Vec<Source>,
    versions: BTreeMap<SourceId, Source>,
    resolutions: BTreeMap<(SourceId, String), Result<Source, String>>,
}

impl ImportTraversal<'_> {
    fn check_version(&mut self, source: &Source) -> Result<(), SourceError> {
        if let Some(old) = self.versions.get(&source.id())
            && old != source
        {
            return Err(SourceError::new(
                source.clone(),
                None,
                "loader returned different versions of the same module during one compilation"
                    .into(),
            ));
        }
        self.versions.insert(source.id(), source.clone());
        Ok(())
    }

    fn visit(&mut self, source: Source) -> Result<usize, SourceError> {
        self.check_version(&source)?;
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
        let document = self.compiler.parse(&source);
        let mut module = resin_ast::SourceModule {
            source: source.clone(),
            file: document.file.clone(),
            imports: Vec::new(),
        };
        self.result.errors.extend(
            document
                .errors
                .iter()
                .map(|(span, error)| module.error(*span, error)),
        );
        self.result.documents.insert(source.clone(), document);
        self.active.push(source.clone());
        self.imports(&mut module);
        self.active.pop();
        let id = self.result.program.modules.len();
        self.result.program.modules.push(module);
        self.loaded.insert(source.id(), id);
        Ok(id)
    }

    fn resolve(&mut self, source: &Source, reference: &str) -> Result<Source, String> {
        self.resolutions
            .entry((source.id(), reference.into()))
            .or_insert_with(|| {
                self.loader
                    .load_import(source, reference)
                    .map_err(|error| error.to_string())
            })
            .clone()
    }

    fn imports(&mut self, module: &mut resin_ast::SourceModule) {
        for import in &module.file.imports {
            let start = self.result.errors.len();
            let imported = self
                .resolve(&module.source, &import.val)
                .map_err(|error| module.error(import.span, error))
                .and_then(|source| self.visit(source));
            match imported {
                Ok(id) => module.imports.push((import.span, id)),
                Err(error) => self.result.errors.push(error),
            }
            for error in &mut self.result.errors[start..] {
                imported_at(error, module, import.span);
            }
        }
    }
}

fn imported_at(error: &mut SourceError, importer: &resin_ast::SourceModule, span: Span) {
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

//
// Lowering and diagnostics
//

fn lower_to_verified_lir(
    program: &resin_ast::Program,
    hir: &resin_hir::Module,
) -> Result<resin_lir::VerifiedModule, Vec<SourceError>> {
    let lir = resin_lir::analyze(hir).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| lowering_error(program, hir, error))
            .collect::<Vec<_>>()
    })?;
    resin_lir::VerifiedModule::new(lir).map_err(|error| {
        vec![SourceError::new(
            program.modules.last().expect("entry module").source.clone(),
            None,
            format!("invalid LIR: {error}"),
        )]
    })
}

fn lowering_error(
    program: &resin_ast::Program,
    hir: &resin_hir::Module,
    error: resin_lir::Error,
) -> SourceError {
    let origin = hir.functions[error.function.index()]
        .location
        .as_ref()
        .expect("generated HIR carries source locations");
    if let Some(source) = program
        .modules
        .iter()
        .find(|source| source.source == origin.source)
    {
        return source.error(error.span, &error);
    }
    SourceError::new(origin.source.clone(), Some(error.span), error.to_string())
}
