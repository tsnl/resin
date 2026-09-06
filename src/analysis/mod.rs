//! Editor-independent source analysis.

pub(crate) mod semantic;
mod source;
pub(crate) mod syntax;

use crate::{
    ast::{self, SourceLocation, SourceNote, SourceProvider, Span},
    ir,
};
use semantic::SemanticData;
use source::Snapshot;
pub use source::{Sources, normalize_path};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};
pub use syntax::{Definition, DefinitionKind};
use syntax::{Document, contains, span};

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub location: SourceLocation,
    pub message: String,
    pub related: Vec<SourceNote>,
}

#[derive(Debug, Clone)]
pub struct Hover {
    pub span: Span,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub name: String,
    pub detail: String,
    pub kind: DefinitionKind,
    pub replace: Span,
}

/// An immutable analysis of one entry and its transitive dependencies.
pub struct Analysis {
    pub(crate) documents: BTreeMap<PathBuf, Arc<Document>>,
    pub diagnostics: Vec<Diagnostic>,
    pub dependencies: BTreeSet<PathBuf>,
    semantics: SemanticData,
    stdlib: PathBuf,
    program: Option<ast::Program>,
    module: Option<ir::Module>,
    load_error: Option<ast::SourceError>,
    compile_error: Option<ast::SourceError>,
}

impl Analysis {
    pub fn new(entry: &Path, sources: &impl SourceProvider, stdlib: &Path) -> Self {
        Self::build(entry, sources, stdlib, &mut BTreeMap::new())
    }

    pub(crate) fn build(
        entry: &Path,
        sources: &impl SourceProvider,
        stdlib: &Path,
        parsed: &mut BTreeMap<PathBuf, Arc<Document>>,
    ) -> Self {
        let mut result = Self {
            documents: BTreeMap::new(),
            diagnostics: Vec::new(),
            dependencies: BTreeSet::new(),
            semantics: SemanticData::default(),
            stdlib: stdlib.to_path_buf(),
            program: None,
            module: None,
            load_error: None,
            compile_error: None,
        };
        let mut pending = vec![entry.to_path_buf()];
        while let Some(path) = pending.pop() {
            let path = match sources.resolve(&path) {
                Ok(path) => path,
                Err(error) => {
                    result.error(path, Span { start: 0, end: 0 }, error.to_string());
                    continue;
                }
            };
            if !result.dependencies.insert(path.clone()) {
                continue;
            }
            let text = match sources.read(&path) {
                Ok(text) => text,
                Err(_) => continue, // The shared loader reports this at the importing clause.
            };
            let document = if let Some(old) = parsed.get(&path).filter(|d| d.text == text) {
                old.clone()
            } else {
                Arc::new(Document::reparse(
                    &path,
                    text,
                    parsed.get(&path).map(Arc::as_ref),
                ))
            };
            parsed.insert(path.clone(), document.clone());
            for (span, message) in &document.errors {
                result.error(path.clone(), *span, message.clone());
            }
            for import in &document.imports {
                if let Ok(path) = ast::resolve_import(&path, &import.text, stdlib) {
                    pending.push(path);
                }
            }
            result.documents.insert(path, document);
        }
        let loaded = ast::load_with(entry, stdlib, &Snapshot(&result.documents));
        if let Err(error) = &loaded {
            result.load_error = Some(error.clone());
        }
        let checked = loaded.and_then(|program| {
            let (semantics, checked) = ir::analyze_program(&program);
            result.semantics = semantics;
            result.program = Some(program);
            checked
        });
        if let Err(error) = checked {
            result.compile_error = Some(error.clone());
            let span = error.span.unwrap_or(Span { start: 0, end: 0 });
            if result
                .documents
                .get(&error.path)
                .is_none_or(|d| d.errors.is_empty())
            {
                result.diagnostics.push(Diagnostic {
                    location: SourceLocation {
                        path: error.path,
                        span,
                    },
                    message: error.diagnostic,
                    related: error.related,
                });
            }
        } else if let Ok(module) = checked {
            result.module = Some(module);
        }
        if result.module.is_none()
            && let Ok(program) = ast::load_with(
                entry,
                &result.stdlib,
                &source::RecoverySnapshot(&result.documents),
            )
        {
            let recovered = ir::analyze_recovering(&program);
            // Strict observations win wherever the compiler reached valid code.
            for (location, ty) in recovered.types {
                result.semantics.types.entry(location).or_insert(ty);
            }
            for (location, origin) in recovered.references {
                result
                    .semantics
                    .references
                    .entry(location)
                    .or_insert(origin);
            }
            for (location, fields) in recovered.fields {
                result.semantics.fields.entry(location).or_insert(fields);
            }
        }
        result
    }

    /// Editor AST, possibly containing holes. `program` and `module` stay strict.
    pub fn recovered_file(&self, path: &Path) -> Option<&ast::SourceFile> {
        self.documents.get(path).map(|d| &d.recovered_file)
    }

    pub fn program(&self) -> Result<&ast::Program, ast::SourceError> {
        self.program.as_ref().ok_or_else(|| {
            self.load_error
                .clone()
                .expect("failed source loading has a diagnostic")
        })
    }

    pub fn module(&self) -> Result<&ir::Module, ast::SourceError> {
        self.module.as_ref().ok_or_else(|| {
            self.compile_error
                .clone()
                .expect("failed compilation has a diagnostic")
        })
    }

    pub fn syntax_tree(&self, path: &Path) -> Option<&tree_sitter::Tree> {
        self.documents.get(path).map(|d| &d.tree)
    }

    fn error(&mut self, path: PathBuf, span: Span, message: String) {
        self.diagnostics.push(Diagnostic {
            location: SourceLocation { path, span },
            message,
            related: Vec::new(),
        });
    }

    pub fn sources(&self) -> impl Iterator<Item = (&Path, &str)> {
        self.documents
            .iter()
            .map(|(path, doc)| (path.as_path(), doc.text.as_str()))
    }

    fn imported_path(&self, source: &Path, import: &str) -> Option<PathBuf> {
        normalize_path(&ast::resolve_import(source, import, &self.stdlib).ok()?).ok()
    }

    fn module_bindings(
        &self,
        path: &Path,
        visiting: &mut BTreeSet<PathBuf>,
    ) -> BTreeMap<String, Option<Definition>> {
        if !visiting.insert(path.to_path_buf()) {
            return BTreeMap::new();
        }
        let mut bindings = BTreeMap::new();
        if let Some(document) = self.documents.get(path) {
            for import in &document.imports {
                if let Some(imported) = self.imported_path(path, &import.text) {
                    let exported = self.module_bindings(&imported, visiting);
                    if let Some(dependency) = self.documents.get(&imported) {
                        for name in &dependency.exports {
                            if let Some(definition) = exported.get(name) {
                                merge_binding(&mut bindings, name.clone(), definition.clone());
                            }
                        }
                    }
                }
            }
            for definition in document.definitions.iter().filter(|d| d.top_level) {
                merge_binding(
                    &mut bindings,
                    definition.name.clone(),
                    Some(definition.clone()),
                );
            }
        }
        visiting.remove(path);
        bindings
    }

    fn visible(&self, path: &Path, offset: usize, exports: bool) -> Vec<Definition> {
        let Some(document) = self.documents.get(path) else {
            return Vec::new();
        };
        let in_function = document.in_function(offset);
        let mut bindings = self.module_bindings(path, &mut BTreeSet::new());
        bindings.retain(|_, definition| {
            definition.as_ref().is_some_and(|d| {
                d.location.path != path
                    || in_function
                    || exports
                    || (d.kind == DefinitionKind::Type
                        && !document.in_module_type_definition(offset))
                    || d.visible_after <= offset
            })
        });
        let mut locals = document
            .definitions
            .iter()
            .filter(|d| !d.top_level && contains(d.scope, offset) && d.visible_after <= offset)
            .collect::<Vec<_>>();
        // Apply outer scopes before inner scopes. Same-scope duplicates remain
        // ambiguous instead of selecting an arbitrary declaration.
        // Parameters and the outer function body occupy distinct compiler scopes
        // even though their source ranges cover the same body.
        locals.sort_by_key(|d| {
            (
                std::cmp::Reverse(d.scope.end - d.scope.start),
                d.kind != DefinitionKind::Parameter,
                d.location.span.start,
            )
        });
        let mut local_origins = BTreeMap::new();
        for definition in locals {
            let key = (
                definition.scope,
                definition.kind == DefinitionKind::Parameter,
                definition.name.clone(),
            );
            let duplicate = local_origins
                .insert(key, definition.location.clone())
                .is_some();
            bindings.insert(
                definition.name.clone(),
                if duplicate {
                    None
                } else {
                    Some(definition.clone())
                },
            );
        }
        bindings
            .into_values()
            .flatten()
            .filter(|d| !matches!(d.name.as_str(), "print" | "shader"))
            .collect()
    }

    pub fn definition(&self, path: &Path, offset: usize) -> Option<SourceLocation> {
        let document = self.documents.get(path)?;
        let token = document.token(offset)?;
        for import in &document.imports {
            if contains(import.span, offset) {
                let path = self.imported_path(path, &import.text)?;
                return self
                    .documents
                    .contains_key(&path)
                    .then_some(SourceLocation {
                        path,
                        span: Span { start: 0, end: 0 },
                    });
            }
        }
        if !document.reference(token) {
            return None;
        }
        let location = SourceLocation {
            path: path.to_path_buf(),
            span: span(token),
        };
        if document.definitions.iter().any(|d| d.location == location) {
            return Some(location);
        }
        if let Some(origin) = self.semantics.references.get(&location) {
            return Some(origin.clone());
        }
        let exports = token.parent().is_some_and(|p| p.kind() == "export_clause");
        self.visible(path, offset, exports)
            .into_iter()
            .find(|d| d.name == document.text(token))
            .map(|d| d.location)
    }

    fn definition_label(&self, definition: &Definition) -> String {
        if matches!(
            definition.kind,
            DefinitionKind::Variable | DefinitionKind::Parameter
        ) && let Some(ty) = self.semantics.types.get(&definition.location)
        {
            return format!("{}: {ty}", definition.name);
        }
        definition.label.clone()
    }

    pub fn hover(&self, path: &Path, offset: usize) -> Option<Hover> {
        let document = self.documents.get(path)?;
        let token = document.token(offset)?;
        let text = document.text(token);
        if let Some((_, help, _)) = BUILTINS.iter().find(|(name, _, _)| *name == text)
            && (document.reference(token)
                || matches!(token.kind(), "builtin_type" | "Ptr" | "Span"))
        {
            return Some(Hover {
                span: span(token),
                text: help.to_string(),
            });
        }
        let origin = self.definition(path, offset)?;
        let definition = self
            .documents
            .get(&origin.path)?
            .definitions
            .iter()
            .find(|d| d.location == origin)?;
        Some(Hover {
            span: span(token),
            text: self.definition_label(definition),
        })
    }

    pub fn completions(&self, path: &Path, offset: usize) -> Vec<Completion> {
        let Some(document) = self.documents.get(path) else {
            return Vec::new();
        };
        if !document.text.is_char_boundary(offset) {
            return Vec::new();
        }
        if document.token(offset).is_some_and(|n| match n.kind() {
            "string" => offset < n.end_byte(),
            "comment" => offset < n.end_byte() || document.text(n).starts_with("//"),
            _ => false,
        }) {
            return Vec::new();
        }
        let bytes = document.text.as_bytes();
        let identifier = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        let mut start = offset;
        let mut end = offset;
        while start > 0 && identifier(bytes[start - 1]) {
            start -= 1;
        }
        while end < bytes.len() && identifier(bytes[end]) {
            end += 1;
        }
        if document.text[..start].trim_end().ends_with('.') {
            return self.field_completions(path, Span { start, end }, offset);
        }
        let prefix = &document.text[start..offset];
        let types = document.type_context(offset);
        let replace = Span { start, end };
        let mut items = self
            .visible(path, offset, false)
            .into_iter()
            .filter(|d| !types || d.kind == DefinitionKind::Type)
            .map(|d| Completion {
                detail: self.definition_label(&d),
                name: d.name,
                kind: d.kind,
                replace,
            })
            .collect::<Vec<_>>();
        for (name, help, kind) in BUILTINS {
            if !types || *kind == DefinitionKind::Type {
                items.push(Completion {
                    name: name.to_string(),
                    detail: help.to_string(),
                    kind: *kind,
                    replace,
                });
            }
        }
        items.retain(|item| item.name.starts_with(prefix));
        items.sort_by(|a, b| a.name.cmp(&b.name));
        items.dedup_by(|a, b| a.name == b.name);
        items
    }

    fn field_completions(&self, path: &Path, replace: Span, offset: usize) -> Vec<Completion> {
        let document = &self.documents[path];
        let prefix = &document.text[replace.start..offset];
        let dot = document.text[..replace.start].trim_end().len() - 1;
        let fields = self.semantics.fields.iter().find_map(|(location, fields)| {
            (location.path == path
                && location.span.start > dot
                && location.span.start <= replace.start)
                .then_some(fields)
        });
        let mut items = fields
            .into_iter()
            .flatten()
            .filter(|(name, _)| name.starts_with(prefix))
            .map(|(name, ty)| Completion {
                name: name.clone(),
                detail: format!("{name}: {ty}"),
                kind: DefinitionKind::Field,
                replace,
            })
            .collect::<Vec<_>>();
        items.sort_by(|a, b| a.name.cmp(&b.name));
        items
    }
}

fn merge_binding(
    bindings: &mut BTreeMap<String, Option<Definition>>,
    name: String,
    definition: Option<Definition>,
) {
    use std::collections::btree_map::Entry;
    match bindings.entry(name) {
        Entry::Vacant(entry) => {
            entry.insert(definition);
        }
        Entry::Occupied(mut entry) => {
            let same = entry
                .get()
                .as_ref()
                .zip(definition.as_ref())
                .is_some_and(|(a, b)| a.location == b.location);
            if !same {
                entry.insert(None);
            }
        }
    }
}

const BUILTINS: &[(&str, &str, DefinitionKind)] = &[
    (
        "print",
        "print(format, arguments)\n\nPrint formatted values on the host. Numbered placeholders use {0}, {1}, ….",
        DefinitionKind::Function,
    ),
    (
        "shader",
        "shader(named_function, \"compute\" | \"vertex\" | \"fragment\")\n\nCompile-time shader entry selection.",
        DefinitionKind::Function,
    ),
    (
        "Ptr",
        "Ptr<T>\n\nAn unchecked pointer to T.",
        DefinitionKind::Type,
    ),
    (
        "Span",
        "Span<T>\n\nA pointer and length describing elements of T.",
        DefinitionKind::Type,
    ),
    ("bool", "bool", DefinitionKind::Type),
    (
        "sbyte",
        "sbyte — signed 8-bit integer",
        DefinitionKind::Type,
    ),
    (
        "short",
        "short — signed 16-bit integer",
        DefinitionKind::Type,
    ),
    ("int", "int — signed 32-bit integer", DefinitionKind::Type),
    ("long", "long — signed 64-bit integer", DefinitionKind::Type),
    (
        "ubyte",
        "ubyte — unsigned 8-bit integer",
        DefinitionKind::Type,
    ),
    (
        "ushort",
        "ushort — unsigned 16-bit integer",
        DefinitionKind::Type,
    ),
    (
        "uint",
        "uint — unsigned 32-bit integer",
        DefinitionKind::Type,
    ),
    (
        "ulong",
        "ulong — unsigned 64-bit integer",
        DefinitionKind::Type,
    ),
    ("float32", "float32", DefinitionKind::Type),
    ("float64", "float64", DefinitionKind::Type),
    ("export", "export { name };", DefinitionKind::Keyword),
    (
        "import",
        "import { \"path.resin\" };",
        DefinitionKind::Keyword,
    ),
    (
        "extern",
        "extern — foreign function or opaque type declaration",
        DefinitionKind::Keyword,
    ),
    (
        "def",
        "def name(parameters) -> Type = { body };\n\nOmitted result annotations default to ().",
        DefinitionKind::Keyword,
    ),
    (
        "var",
        "var name = value;\nvar name: Type;",
        DefinitionKind::Keyword,
    ),
    (
        "type",
        "type Name = Type;\nextern type Name;",
        DefinitionKind::Keyword,
    ),
    (
        "if",
        "if (condition) { value } else { value }",
        DefinitionKind::Keyword,
    ),
    ("else", "else { value }", DefinitionKind::Keyword),
    (
        "while",
        "while (condition) { body };",
        DefinitionKind::Keyword,
    ),
];
