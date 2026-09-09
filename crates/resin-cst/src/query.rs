//! Syntax-only token and context queries.
use super::{Document, contains, span};
use tree_sitter::Node;
impl Document {
    pub fn node_text(&self, node: Node<'_>) -> &str {
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
