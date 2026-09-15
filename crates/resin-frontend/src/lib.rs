//! Compile immutable named sources, retaining phase products and editor queries.
//!
//! Loaders discover imports. This frontend has no filesystem, editor buffers, or
//! change notifications. Each call resolves the dependency graph before reusing
//! work. Codegen and native builds consume its verified output separately.
//!
//! ```compile_fail,E0603
//! use resin_frontend::{ImportTraversal, ParsedDocument};
//! ```
use resin_source::prelude::*;
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    sync::{Arc, Weak},
};

//
// Frontend output
//

/// Immutable resource limits for every compilation made by this frontend.
#[derive(Debug, Clone)]
pub struct FrontendConfig {
    pub max_monomorphs_per_function: NonZeroUsize,
}
impl Default for FrontendConfig {
    fn default() -> Self {
        Self {
            max_monomorphs_per_function: NonZeroUsize::new(16 * 1024).unwrap(),
        }
    }
}

/// Exported entry and Host/Shader profile to build as LIR.
/// Selected when building LIR, not while building HIR.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Target {
    Host { entry: Arc<str> },
    Shader { entry: Arc<str> },
}

/// Reusable analysis caches. Sources and outputs are immutable.
#[derive(Default)]
pub struct Frontend {
    config: FrontendConfig,
    parsed: BTreeMap<SourceId, Weak<ParsedDocument>>,
    outputs: BTreeMap<SourceId, Arc<FrontendOutput>>,
}
impl Frontend {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_config(config: FrontendConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }
    /// Build HIR and retained editor facts for every declaration, without LIR.
    pub fn build_hir(
        &mut self,
        source: Source,
        loader: &mut resin_source::Loader,
    ) -> Arc<FrontendOutput> {
        let loaded = self.load_sources(source.clone(), loader);
        let mut outputs = live_outputs(&self.outputs, source.id());
        if loaded.errors.is_empty()
            && let Some(old) = outputs.get(&source.id()).filter(|old| old.matches(&loaded))
        {
            let reused = old.clone();
            self.outputs = outputs;
            self.parsed = live_documents(&self.parsed);
            return reused;
        }
        let result = Arc::new(FrontendOutput::from_loaded(
            source.clone(),
            loaded,
            self.config.clone(),
        ));
        outputs.insert(source.id(), result.clone());
        self.outputs = outputs;
        self.parsed = live_documents(&self.parsed);
        result
    }
}

fn live_outputs(
    previous: &BTreeMap<SourceId, Arc<FrontendOutput>>,
    source: SourceId,
) -> BTreeMap<SourceId, Arc<FrontendOutput>> {
    previous
        .iter()
        .filter(|(id, output)| **id == source || Arc::strong_count(output) > 1)
        .map(|(id, output)| (*id, output.clone()))
        .collect()
}

fn live_documents(
    previous: &BTreeMap<SourceId, Weak<ParsedDocument>>,
) -> BTreeMap<SourceId, Weak<ParsedDocument>> {
    previous
        .iter()
        .filter(|(_, document)| document.strong_count() > 0)
        .map(|(id, document)| (*id, document.clone()))
        .collect()
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

/// Immutable HIR and editor facts for a source and its imports.
/// Failed later passes preserve successfully completed earlier products.
///
/// ```compile_fail,E0596
/// fn edit(output: &mut resin_frontend::FrontendOutput) {
///     output.hir().unwrap().functions.clear();
/// }
/// ```
pub struct FrontendOutput {
    source: Source,
    documents: BTreeMap<Source, Arc<ParsedDocument>>,
    syntax: BTreeMap<Source, Arc<resin_cst::Document>>,
    diagnostics: Vec<Diagnostic>,
    semantics: resin_hir::Analysis,
    graph: Vec<(Source, Vec<(Span, usize)>)>,
    program: Result<resin_ast::Program, SourceError>,
    hir: Result<resin_hir::Module, SourceError>,
    config: FrontendConfig,
}
impl FrontendOutput {
    pub fn source(&self) -> &Source {
        &self.source
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
    /// Build verified LIR for the requested host and shader entries.
    pub fn build_lir(
        &self,
        targets: &[Target],
    ) -> Result<resin_lir::VerifiedModule, Vec<SourceError>> {
        let program = self.program.as_ref().map_err(|error| vec![error.clone()])?;
        let hir = self.hir.as_ref().map_err(|error| vec![error.clone()])?;
        build_verified_lir(program, hir, &self.config, targets)
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

impl Frontend {
    fn load_sources(&mut self, source: Source, loader: &mut resin_source::Loader) -> Loaded {
        let mut traversal = ImportTraversal {
            loader,
            frontend: self,
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
        if let Err(error) = traversal.visit(source) {
            traversal.result.errors.push(error);
        }
        traversal.result
    }

    fn parse(&mut self, source: &Source) -> Arc<ParsedDocument> {
        let old = self.parsed.get(&source.id()).and_then(Weak::upgrade);
        if let Some(old) = old.as_ref().filter(|old| old.source == *source) {
            return old.clone();
        }
        let document = Arc::new(ParsedDocument::build(source.clone(), old.as_deref()));
        self.parsed.insert(source.id(), Arc::downgrade(&document));
        document
    }
}

impl FrontendOutput {
    fn matches(&self, loaded: &Loaded) -> bool {
        self.program.is_ok() && self.graph == loaded.graph()
    }

    fn from_loaded(source: Source, loaded: Loaded, config: FrontendConfig) -> Self {
        let graph = loaded.graph();
        let syntax = loaded
            .documents
            .iter()
            .map(|(source, document)| (source.clone(), document.syntax.clone()))
            .collect();
        let load_error = loaded.errors.first().cloned();
        let analysis = build_checked_hir(&loaded.program);
        let hir_error = load_error.clone().or(analysis.hir_error);
        let errors = loaded
            .errors
            .into_iter()
            .chain(analysis.diagnostics)
            .collect::<Vec<_>>();
        Self {
            source,
            syntax,
            documents: loaded.documents,
            graph,
            diagnostics: errors.into_iter().map(diagnostic).collect(),
            semantics: analysis.semantics,
            program: load_error.map_or(Ok(loaded.program), Err),
            hir: hir_error.map_or_else(|| Ok(analysis.hir.expect("successful HIR")), Err),
            config,
        }
    }
}

struct Analyzed {
    semantics: resin_hir::Analysis,
    diagnostics: Vec<SourceError>,
    hir: Option<resin_hir::Module>,
    hir_error: Option<SourceError>,
}

fn build_checked_hir(program: &resin_ast::Program) -> Analyzed {
    let checked = resin_hir::build_hir_program_checked(program);
    Analyzed {
        hir_error: checked.diagnostics.first().cloned(),
        semantics: checked.semantics,
        diagnostics: checked.diagnostics,
        hir: checked.module,
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
    fn build(source: Source, previous: Option<&Self>) -> Self {
        let syntax = resin_cst::build_cst(source.text(), previous.map(|d| d.syntax.as_ref()));
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
    frontend: &'a mut Frontend,
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
        let document = self.frontend.parse(&source);
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
        self.visit_imports(&mut module);
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

    fn visit_imports(&mut self, module: &mut resin_ast::SourceModule) {
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

fn build_verified_lir(
    program: &resin_ast::Program,
    hir: &resin_hir::Module,
    config: &FrontendConfig,
    targets: &[Target],
) -> Result<resin_lir::VerifiedModule, Vec<SourceError>> {
    let options = resin_lir::LoweringOptions {
        max_monomorphs_per_function: config.max_monomorphs_per_function,
    };
    let entries = entry_requests(program, hir, targets).map_err(|error| vec![error])?;
    let lir = resin_lir::build_lir(hir, &entries, &options).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| lowering_error(program, error))
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

fn entry_requests(
    program: &resin_ast::Program,
    hir: &resin_hir::Module,
    targets: &[Target],
) -> Result<Vec<resin_lir::Entry>, SourceError> {
    let source = &program.modules.last().expect("entry module").source;
    if targets.is_empty() {
        return Err(SourceError::new(
            source.clone(),
            None,
            "compilation requires at least one target; use build_hir for declaration analysis"
                .into(),
        ));
    }
    targets
        .iter()
        .map(|target| {
            let (name, profile) = match target {
                Target::Host { entry } => (entry, resin_lir::Profile::Host),
                Target::Shader { entry } => (entry, resin_lir::Profile::Shader),
            };
            let function = hir.entries.get(name).ok_or_else(|| {
                SourceError::new(
                    source.clone(),
                    None,
                    format!("entry {name} is not exported as a function"),
                )
            })?;
            Ok(resin_lir::Entry {
                name: name.clone(),
                function: *function,
                arguments: vec![],
                profile,
            })
        })
        .collect()
}

fn lowering_error(program: &resin_ast::Program, error: resin_lir::Error) -> SourceError {
    let mut diagnostic = lowering_origin(program, &error);
    diagnostic
        .related
        .extend(error.applications.into_iter().filter_map(|application| {
            application.location.map(|location| SourceNote {
                location,
                message: format!(
                    "while instantiating {} with {:?} for {:?}",
                    application.function, application.arguments, application.profile
                ),
            })
        }));
    diagnostic
}

fn lowering_origin(program: &resin_ast::Program, error: &resin_lir::Error) -> SourceError {
    let Some(source) = &error.source else {
        return SourceError::new(
            program.modules.last().expect("entry module").source.clone(),
            None,
            error.to_string(),
        );
    };
    if let Some(module) = program
        .modules
        .iter()
        .find(|module| module.source == *source)
    {
        return module.error(error.span, error);
    }
    SourceError::new(source.clone(), Some(error.span), error.to_string())
}
