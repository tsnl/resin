//! Editor queries over retained compiler contexts and type information.
pub use crate::compiler::{Diagnostic, Snapshot as Analysis, Sources, normalize_path};
pub use crate::ir::generate::semantic::{Definition, DefinitionKind};
use crate::{
    ast::{SourceLocation, Span},
    compiler::syntax::{contains, span},
    ir::generate::semantic,
};
use std::path::Path;

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

impl Analysis {
    pub fn definition(&self, path: &Path, offset: usize) -> Option<SourceLocation> {
        let document = self.documents.get(path)?;
        let token = document.token(offset)?;
        for (location, target) in &self.semantics.imports {
            if location.path == path && contains(location.span, offset) {
                return Some(SourceLocation {
                    path: target.clone(),
                    span: Span { start: 0, end: 0 },
                });
            }
        }
        let location = SourceLocation {
            path: path.to_path_buf(),
            span: span(token),
        };
        if self
            .semantics
            .contexts
            .definitions
            .iter()
            .any(|d| d.location == location)
        {
            return Some(location);
        }
        if let Some(origin) = self
            .semantics
            .fields
            .get(&location)
            .and_then(|members| {
                members
                    .iter()
                    .find(|member| member.name == document.text(token))
            })
            .and_then(|member| member.origin)
        {
            return Some(self.semantics.contexts.definitions[origin].location.clone());
        }
        if !document.reference(token) {
            return None;
        }
        self.semantics
            .contexts
            .definition(path, offset, document.text(token), token.kind() == "uid")
            .map(|d| d.location.clone())
    }

    fn definition_label(&self, definition: &Definition) -> String {
        if matches!(
            definition.kind,
            DefinitionKind::Variable | DefinitionKind::Parameter
        ) {
            let ty = definition
                .ty
                .as_ref()
                .map(|ty| semantic::format_type(ty, &self.semantics.typer))
                .unwrap_or_else(|| "?".into());
            return format!("{}: {ty}", definition.name);
        }
        let Some(document) = self.documents.get(&definition.location.path) else {
            return definition.label.clone();
        };
        let Some(node) = document
            .token(definition.location.span.start)
            .and_then(|node| node.parent())
        else {
            return definition.label.clone();
        };
        match definition.kind {
            DefinitionKind::Function => node
                .child_by_field_name("result")
                .map(|result| document.text[node.start_byte()..result.end_byte()].to_owned())
                .or_else(|| {
                    node.children(&mut node.walk())
                        .find(|child| child.kind() == ")" && !child.is_missing())
                        .map(|end| {
                            format!(
                                "{} -> ()",
                                &document.text[node.start_byte()..end.end_byte()]
                            )
                        })
                }),
            DefinitionKind::Type => node
                .child_by_field_name("init")
                .map(|init| format!("{} = {}", definition.name, document.text(init))),
            _ => None,
        }
        .unwrap_or_else(|| definition.label.clone())
    }

    pub fn hover(&self, path: &Path, offset: usize) -> Option<Hover> {
        let document = self.documents.get(path)?;
        let token = document.token(offset)?;
        let text = document.text(token);
        if let Some((_, help, _)) = BUILTINS.iter().find(|(name, _, _)| *name == text)
            && (document.reference(token)
                || matches!(
                    token.kind(),
                    "builtin_type" | "Ptr" | "Span" | "Result" | "Arc" | "Weak" | "None"
                ))
        {
            return Some(Hover {
                span: span(token),
                text: help.to_string(),
            });
        }
        let Some(origin) = self.definition(path, offset) else {
            let location = SourceLocation {
                path: path.to_path_buf(),
                span: span(token),
            };
            let member = self
                .semantics
                .fields
                .get(&location)?
                .iter()
                .find(|member| member.name == text)?;
            return Some(Hover {
                span: span(token),
                text: format!("{}: {}", member.name, member.ty),
            });
        };
        let definition = self
            .semantics
            .contexts
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
            .semantics
            .contexts
            .visible(path, offset)
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
            .filter(|member| member.name.starts_with(prefix))
            .map(|member| Completion {
                name: member.name.clone(),
                detail: format!("{}: {}", member.name, member.ty),
                kind: member.kind,
                replace,
            })
            .collect::<Vec<_>>();
        items.sort_by(|a, b| {
            (a.kind != DefinitionKind::Field, &a.name)
                .cmp(&(b.kind != DefinitionKind::Field, &b.name))
        });
        items
    }
}

const BUILTINS: &[(&str, &str, DefinitionKind)] = &[
    (
        "fmt",
        "fmt(format, arguments) -> String\n\nFormat a tuple using numbered placeholders {0}, {1}, … into an owned String. Host-only.",
        DefinitionKind::Function,
    ),
    (
        "String",
        "String\n\nOwned bytes, wrapping Arc<Span<ubyte>>. Copies retain the allocation. String literals have type Span<ubyte>.",
        DefinitionKind::Type,
    ),
    (
        "print",
        "print(text) -> ()\n\nWrite a String or Span<ubyte> to stdout verbatim and flush.",
        DefinitionKind::Function,
    ),
    (
        "Ptr",
        "Ptr<T>\n\nAn unchecked pointer to T.",
        DefinitionKind::Type,
    ),
    (
        "Span",
        "Span<T>\n\nA pointer and length describing elements of T. Calling span.at(index: ulong) returns Ptr<T>; shader indexing is unchecked.",
        DefinitionKind::Type,
    ),
    (
        "Result",
        "Result<T, E> — success or a typed error; postfix ? propagates errors.",
        DefinitionKind::Type,
    ),
    (
        "Never",
        "Never — the empty union, with no possible values.",
        DefinitionKind::Type,
    ),
    (
        "ok",
        "ok(value) — construct a successful Result.",
        DefinitionKind::Function,
    ),
    (
        "err",
        "err(error) — construct a failed Result.",
        DefinitionKind::Function,
    ),
    (
        "struct",
        "struct Name { field: Type }; — a nominal record type.",
        DefinitionKind::Keyword,
    ),
    (
        "match",
        "match (value) { Variant(name) => { body }, ... }",
        DefinitionKind::Keyword,
    ),
    (
        "impl",
        "impl T { def method(self: Ptr<T>) = {}; } — inherent methods.",
        DefinitionKind::Keyword,
    ),
    (
        "None",
        "None — singleton value and type; T | None permits absence, postfix ! excludes it or traps.",
        DefinitionKind::Type,
    ),
    (
        "Arc",
        "Arc<T> — a copyable shared owner; copying retains the allocation.",
        DefinitionKind::Type,
    ),
    (
        "Weak",
        "Weak<T> — a weak handle; upgrade() returns Arc<T> | None.",
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
