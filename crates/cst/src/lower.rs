//! Source text → concrete syntax, with optional incremental reparsing.
use super::Document;
use crate::source::Span;
use tree_sitter::{InputEdit, Node, Point, Tree};
pub fn span(node: Node<'_>) -> Span {
    Span {
        start: node.start_byte(),
        end: node.end_byte(),
    }
}
pub fn contains(span: Span, offset: usize) -> bool {
    span.start <= offset && offset <= span.end
}
impl Document {
    pub fn reparse(text: String, previous: Option<&Self>) -> Self {
        let previous_tree = previous.map(|old| edited_tree(old, &text));
        let tree = crate::parser()
            .parse(&text, previous_tree.as_ref())
            .expect("parser language is set");
        Self { text, tree }
    }
    pub fn recovery(&self) -> Option<Self> {
        if !self.tree.root_node().has_error() {
            return None;
        }
        let mut closers = vec![];
        unmatched(self.tree.root_node(), &mut closers);
        if closers.is_empty() || closers.len() > 64 {
            return None;
        }
        let mut text = self.text.clone();
        text.extend(closers.into_iter().rev());
        text.push(';');
        Some(Self::reparse(text, None))
    }
}

fn edited_tree(old: &Document, text: &str) -> Tree {
    let mut tree = old.tree.clone();
    tree.edit(&edit(&old.text, text));
    tree
}

fn edit(old: &str, new: &str) -> InputEdit {
    let start = common_start(old, new);
    let (old_end, new_end) = changed_ends(old, new, start);
    InputEdit {
        start_byte: start,
        old_end_byte: old_end,
        new_end_byte: new_end,
        start_position: point(old, start),
        old_end_position: point(old, old_end),
        new_end_position: point(new, new_end),
    }
}

fn common_start(old: &str, new: &str) -> usize {
    let mut start = old
        .bytes()
        .zip(new.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    while !old.is_char_boundary(start) || !new.is_char_boundary(start) {
        start -= 1;
    }
    start
}

fn changed_ends(old: &str, new: &str, start: usize) -> (usize, usize) {
    let suffix = old[start..]
        .bytes()
        .rev()
        .zip(new[start..].bytes().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let (mut old_end, mut new_end) = (old.len() - suffix, new.len() - suffix);
    while !old.is_char_boundary(old_end) || !new.is_char_boundary(new_end) {
        old_end += 1;
        new_end += 1;
    }
    (old_end, new_end)
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
