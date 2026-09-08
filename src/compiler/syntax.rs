use crate::ast::Span;
use tree_sitter::{InputEdit, Node, Parser, Point, Tree};
pub(crate) struct Document {
    pub text: String,
    pub tree: Tree,
    pub file: crate::ast::SourceFile,
    pub errors: Vec<(Span, String)>,
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
    pub fn reparse(text: String, previous: Option<&Self>) -> Self {
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
        let errors = crate::ast::AstGen::new(&text)
            .errors(tree.root_node())
            .into_iter()
            .map(|error| (error.span, error.to_string()))
            .collect();
        let mut file = crate::ast::AstGen::new(&text).source_file(tree.root_node());
        if tree.root_node().has_error() {
            let mut closers = Vec::new();
            unmatched(tree.root_node(), &mut closers);
            if !closers.is_empty() && closers.len() <= 64 {
                let mut repaired = text.clone();
                repaired.extend(closers.into_iter().rev());
                repaired.push(';');
                let recovery = parser.parse(&repaired, None).expect("Resin grammar");
                file = crate::ast::AstGen::bounded(&repaired, text.len())
                    .source_file(recovery.root_node());
            }
        }
        Self {
            text,
            tree,
            file,
            errors,
        }
    }
    pub fn text(&self, node: Node<'_>) -> &str {
        &self.text[node.byte_range()]
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
            if matches!(parent.kind(), "field_access" | "method_call") {
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

pub(super) fn unmatched(node: Node<'_>, closers: &mut Vec<char>) {
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
