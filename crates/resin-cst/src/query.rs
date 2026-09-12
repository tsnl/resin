//! Syntax-only token and context queries.
use super::{Document, contains, span};
use tree_sitter::Node;
pub(super) fn token(document: &Document, offset: usize) -> Option<Node<'_>> {
    if !document.text.is_char_boundary(offset) {
        return None;
    }
    for at in [offset, offset.saturating_sub(1)] {
        let node = document
            .tree
            .root_node()
            .descendant_for_byte_range(at, at)?;
        if contains(span(node), offset)
            && matches!(
                node.kind(),
                "lid"
                    | "uid"
                    | "builtin_type"
                    | "string"
                    | "comment"
                    | "Ptr"
                    | "Span"
                    | "GpuPtr"
                    | "GpuSpan"
                    | "GpuComputePipeline"
                    | "GpuGraphicsPipeline"
            )
        {
            return Some(node);
        }
    }
    None
}

pub(super) fn reference(node: Node<'_>) -> bool {
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

pub(super) fn type_context(document: &Document, offset: usize) -> bool {
    if !document.text.is_char_boundary(offset) {
        return false;
    }
    let Some(mut node) = document.tree.root_node().descendant_for_byte_range(
        offset.min(document.text.len()),
        offset.min(document.text.len()),
    ) else {
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
    document.text[..offset.min(document.text.len())]
        .trim_end()
        .ends_with(':')
}
