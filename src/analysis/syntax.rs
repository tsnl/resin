use crate::ast::{SourceLocation, Span};
use std::path::Path;
use tree_sitter::{InputEdit, Node, Parser, Point, Tree};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefinitionKind {
    Function,
    Variable,
    Parameter,
    Type,
    Keyword,
}

#[derive(Debug, Clone)]
pub struct Definition {
    pub name: String,
    pub location: SourceLocation,
    pub kind: DefinitionKind,
    pub label: String,
    pub(crate) scope: Span,
    pub(crate) visible_after: usize,
    pub(crate) top_level: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Import {
    pub span: Span,
    pub text: String,
}

pub(crate) struct Document {
    pub text: String,
    pub tree: Tree,
    pub file: Result<crate::ast::SourceFile, crate::ast::AstError>,
    pub definitions: Vec<Definition>,
    pub imports: Vec<Import>,
    pub exports: Vec<String>,
    pub errors: Vec<(Span, String)>,
    function_bodies: Vec<Span>,
}

pub(crate) fn span(node: Node<'_>) -> Span {
    Span {
        start: node.start_byte(),
        end: node.end_byte(),
    }
}
pub(crate) fn contains(span: Span, offset: usize) -> bool {
    span.start <= offset && offset <= span.end
}

impl Document {
    pub fn reparse(path: &Path, text: String, previous: Option<&Self>) -> Self {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_resin::LANGUAGE.into())
            .expect("Resin grammar");
        let previous_tree = previous.map(|old| {
            let mut tree = old.tree.clone();
            let mut start = old
                .text
                .bytes()
                .zip(text.bytes())
                .take_while(|(a, b)| a == b)
                .count();
            while !old.text.is_char_boundary(start) || !text.is_char_boundary(start) {
                start -= 1;
            }
            let common_end = old.text[start..]
                .bytes()
                .rev()
                .zip(text[start..].bytes().rev())
                .take_while(|(a, b)| a == b)
                .count();
            let mut old_end = old.text.len() - common_end;
            let mut new_end = text.len() - common_end;
            while !old.text.is_char_boundary(old_end) || !text.is_char_boundary(new_end) {
                old_end += 1;
                new_end += 1;
            }
            tree.edit(&InputEdit {
                start_byte: start,
                old_end_byte: old_end,
                new_end_byte: new_end,
                start_position: point(&old.text, start),
                old_end_position: point(&old.text, old_end),
                new_end_position: point(&text, new_end),
            });
            tree
        });
        let tree = parser
            .parse(&text, previous_tree.as_ref())
            .expect("parser language is set");
        let file = crate::ast::AstGen::new(&text).gen_source_file(tree.root_node());
        let mut result = Self {
            text,
            tree: tree.clone(),
            file,
            definitions: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            errors: Vec::new(),
            function_bodies: Vec::new(),
        };
        let root = tree.root_node();
        result.visit(path, root, span(root), true);
        // Tree-sitter can represent an unfinished function as one ERROR node.
        // Closing only unmatched delimiters in a separate syntax tree recovers
        // its parameters/scopes. This tree is never sent to the type checker.
        if root.has_error() {
            let mut closers = Vec::new();
            unmatched(root, &mut closers);
            if !closers.is_empty() && closers.len() <= 64 {
                let mut recovery = result.text.clone();
                recovery.extend(closers.into_iter().rev());
                recovery.push(';');
                let repaired = parser.parse(&recovery, None).expect("Resin grammar");
                let file = crate::ast::AstGen::new(&recovery).gen_source_file(repaired.root_node());
                let mut recovered = Self {
                    text: recovery,
                    tree: repaired.clone(),
                    file,
                    definitions: Vec::new(),
                    imports: Vec::new(),
                    exports: Vec::new(),
                    errors: Vec::new(),
                    function_bodies: Vec::new(),
                };
                recovered.visit(path, repaired.root_node(), span(repaired.root_node()), true);
                if recovered.function_bodies.len() > result.function_bodies.len() {
                    let end = result.text.len();
                    recovered.definitions.retain(|d| d.location.span.end <= end);
                    for definition in &mut recovered.definitions {
                        definition.scope.end = definition.scope.end.min(end);
                    }
                    result.definitions = recovered.definitions;
                    result.function_bodies = recovered
                        .function_bodies
                        .into_iter()
                        .map(|s| Span {
                            start: s.start,
                            end: s.end.min(end),
                        })
                        .collect();
                }
            }
        }
        result
    }

    pub fn text(&self, node: Node<'_>) -> &str {
        &self.text[node.byte_range()]
    }

    fn definition(
        &mut self,
        path: &Path,
        node: Node<'_>,
        kind: DefinitionKind,
        scope: Span,
        top_level: bool,
        hoisted: bool,
    ) {
        let Some(name) = node.child_by_field_name("name").filter(|n| !n.is_missing()) else {
            return;
        };
        let label = match kind {
            DefinitionKind::Function => node
                .child_by_field_name("result")
                .map(|result| self.text[node.start_byte()..result.end_byte()].to_string()),
            DefinitionKind::Type => node
                .child_by_field_name("init")
                .map(|init| format!("{} = {}", self.text(name), self.text(init))),
            _ => node
                .child_by_field_name("ann")
                .map(|ann| format!("{}: {}", self.text(name), self.text(ann))),
        }
        .unwrap_or_else(|| self.text(name).to_string());
        self.definitions.push(Definition {
            name: self.text(name).into(),
            location: SourceLocation {
                path: path.to_path_buf(),
                span: span(name),
            },
            kind,
            label,
            scope,
            visible_after: if hoisted {
                scope.start
            } else {
                name.end_byte()
            },
            top_level,
        });
    }

    fn visit(&mut self, path: &Path, node: Node<'_>, scope: Span, top_level: bool) {
        if node.is_missing() {
            self.errors
                .push((span(node), format!("expected `{}`", node.kind())));
            return;
        }
        if node.is_error() {
            self.errors.push((span(node), "unexpected syntax".into()));
        }
        match node.kind() {
            "import_clause" => {
                for path_node in node.children_by_field_name("path", &mut node.walk()) {
                    if !path_node.has_error()
                        && let Some(text) = decode_string(self.text(path_node))
                    {
                        self.imports.push(Import {
                            span: span(path_node),
                            text,
                        });
                    }
                }
            }
            "export_clause" => {
                self.exports = node
                    .children_by_field_name("name", &mut node.walk())
                    .filter(|n| !n.is_missing())
                    .map(|n| self.text(n).to_string())
                    .collect();
            }
            "function_definition" | "foreign_function" => {
                self.definition(path, node, DefinitionKind::Function, scope, true, true);
                if let Some(body) = node.child_by_field_name("body") {
                    self.function_bodies.push(span(body));
                    for param in node.children_by_field_name("params", &mut node.walk()) {
                        self.definition(
                            path,
                            param,
                            DefinitionKind::Parameter,
                            span(body),
                            false,
                            true,
                        );
                    }
                    // Visit all children for syntax errors, but declarations in the
                    // signature are not bindings in the enclosing module.
                    for child in node.children(&mut node.walk()) {
                        if child.id() == body.id() {
                            self.visit(path, child, span(body), false);
                        } else {
                            self.syntax_errors(child);
                        }
                    }
                } else {
                    self.syntax_errors_children(node);
                }
                return;
            }
            "foreign_type" => {
                self.definition(path, node, DefinitionKind::Type, scope, top_level, true)
            }
            "type_define" => {
                self.definition(path, node, DefinitionKind::Type, scope, top_level, false)
            }
            "term_define" if node.parent().is_none_or(|p| p.kind() != "record_term") => self
                .definition(
                    path,
                    node,
                    DefinitionKind::Variable,
                    scope,
                    top_level,
                    false,
                ),
            "declare" if node.parent().is_some_and(|p| p.kind() == "statement") => self.definition(
                path,
                node,
                DefinitionKind::Variable,
                scope,
                top_level,
                false,
            ),
            _ => {}
        }
        let block = matches!(node.kind(), "block_body" | "chain_term");
        let child_scope = if block { span(node) } else { scope };
        for child in node.children(&mut node.walk()) {
            self.visit(path, child, child_scope, top_level && !block);
        }
    }

    fn syntax_errors_children(&mut self, node: Node<'_>) {
        for child in node.children(&mut node.walk()) {
            self.syntax_errors(child);
        }
    }
    fn syntax_errors(&mut self, node: Node<'_>) {
        if node.is_missing() {
            self.errors
                .push((span(node), format!("expected `{}`", node.kind())));
        } else if node.is_error() {
            self.errors.push((span(node), "unexpected syntax".into()));
        }
        self.syntax_errors_children(node);
    }

    pub fn in_function(&self, offset: usize) -> bool {
        self.function_bodies.iter().any(|s| contains(*s, offset))
    }

    pub fn in_module_type_definition(&self, offset: usize) -> bool {
        let Some(mut node) = self
            .tree
            .root_node()
            .descendant_for_byte_range(offset, offset)
        else {
            return false;
        };
        loop {
            if node.kind() == "type_define"
                && node
                    .parent()
                    .and_then(|n| n.parent())
                    .and_then(|n| n.parent())
                    .is_some_and(|n| n.kind() == "source_file")
            {
                return true;
            }
            match node.parent() {
                Some(parent) => node = parent,
                None => return false,
            }
        }
    }

    pub fn token(&self, offset: usize) -> Option<Node<'_>> {
        if offset > self.text.len() {
            return None;
        }
        for at in [offset, offset.saturating_sub(1)] {
            let node = self.tree.root_node().descendant_for_byte_range(at, at)?;
            if contains(span(node), offset)
                && matches!(
                    node.kind(),
                    "lid" | "uid" | "builtin_type" | "string" | "comment" | "Ptr" | "Span"
                )
            {
                return Some(node);
            }
        }
        None
    }

    pub fn reference(&self, node: Node<'_>) -> bool {
        if !matches!(node.kind(), "lid" | "uid") {
            return false;
        }
        if let Some(parent) = node.parent() {
            if parent.kind() == "field_access" {
                return false;
            }
            if parent
                .child_by_field_name("name")
                .is_some_and(|name| name.id() == node.id())
            {
                if parent.kind() == "term_define"
                    && parent.parent().is_some_and(|n| n.kind() == "record_term")
                {
                    return false;
                }
                if parent.kind() == "declare"
                    && parent.parent().is_some_and(|n| n.kind() == "record_type")
                {
                    return false;
                }
            }
        }
        true
    }

    pub fn type_context(&self, offset: usize) -> bool {
        let Some(mut node) = self
            .tree
            .root_node()
            .descendant_for_byte_range(offset.min(self.text.len()), offset.min(self.text.len()))
        else {
            return false;
        };
        loop {
            if matches!(
                node.kind(),
                "type" | "unary_type" | "primary_type" | "infix_type"
            ) {
                return true;
            }
            match node.parent() {
                Some(parent) => node = parent,
                None => break,
            }
        }
        self.text[..offset.min(self.text.len())]
            .trim_end()
            .ends_with(':')
    }
}

fn point(text: &str, offset: usize) -> Point {
    let prefix = &text[..offset];
    Point {
        row: prefix.bytes().filter(|b| *b == b'\n').count(),
        column: prefix.rsplit('\n').next().unwrap().len(),
    }
}

fn unmatched(node: Node<'_>, closers: &mut Vec<char>) {
    if node.is_missing() {
        return;
    }
    match node.kind() {
        "(" => closers.push(')'),
        "[" => closers.push(']'),
        "{" => closers.push('}'),
        ")" | "]" | "}" if closers.last().copied() == node.kind().chars().next() => {
            closers.pop();
        }
        _ => {
            for child in node.children(&mut node.walk()) {
                unmatched(child, closers);
            }
        }
    }
}

fn decode_string(text: &str) -> Option<String> {
    let inner = text.strip_prefix('"')?.strip_suffix('"')?;
    let mut result = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        result.push(if c == '\\' {
            match chars.next()? {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '0' => '\0',
                '"' => '"',
                '\\' => '\\',
                _ => return None,
            }
        } else {
            c
        });
    }
    Some(result)
}
