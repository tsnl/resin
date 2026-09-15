//! Read dependency declarations from complete or recovering concrete syntax.

use super::*;

pub(super) fn read(document: &Document) -> Preamble {
    let mut result = Preamble::default();
    collect(document, document.source().len(), &mut result, true);
    if !result.diagnostics.is_empty()
        && let Some(recovered) = document.recovery()
    {
        collect(&recovered, document.source().len(), &mut result, false);
    }
    result.imports.sort_by_key(|entry| entry.span);
    result.imports.dedup_by_key(|entry| entry.span);
    result.headers.sort_by_key(|entry| entry.span);
    result.headers.dedup_by_key(|entry| entry.span);
    result
}

fn collect(document: &Document, length: usize, result: &mut Preamble, errors: bool) {
    let root = document.tree().root_node();
    let mut body = false;
    for child in root.children(&mut root.walk()) {
        if starts_declaration(child) {
            body = true;
        }
        if body {
            if errors && is_preamble_clause(document, child) {
                result.diagnostics.push(Spanned::new(
                    "source preamble must precede declarations".into(),
                    span(child),
                ));
            }
            continue;
        }
        match child.kind() {
            "import_clause" => strings(document, child, length, &mut result.imports),
            "extern_clause" => headers(document, child, length, &mut result.headers),
            "ERROR" => match first_token(child).map(|token| token.kind()) {
                Some("import") => strings(document, child, length, &mut result.imports),
                Some("extern") => headers(document, child, length, &mut result.headers),
                _ => {}
            },
            _ => {}
        }
        if errors {
            diagnostics(child, length, &mut result.diagnostics);
        }
    }
}

fn is_preamble_clause(document: &Document, node: Node<'_>) -> bool {
    if matches!(
        node.kind(),
        "export_clause" | "extern_clause" | "import_clause"
    ) {
        return true;
    }
    node.is_error()
        && first_token(node).is_some_and(|token| {
            // Recovery may classify a misplaced keyword as an identifier.
            matches!(document.node_text(token), "export" | "extern" | "import")
        })
}

fn starts_declaration(node: Node<'_>) -> bool {
    if matches!(
        node.kind(),
        "function_definition"
            | "const_declaration"
            | "foreign_type"
            | "intrinsic_function"
            | "type_definition"
            | "struct_definition"
    ) {
        return true;
    }
    node.is_error()
        && first_token(node).is_some_and(|token| {
            matches!(
                token.kind(),
                "const" | "def" | "type" | "struct" | "intrinsic" | "@"
            )
        })
}

fn first_token(node: Node<'_>) -> Option<Node<'_>> {
    if node.kind() == "comment" || node.is_missing() {
        return None;
    }
    if node.child_count() == 0 {
        return Some(node);
    }
    node.children(&mut node.walk()).find_map(first_token)
}

fn strings(
    document: &Document,
    node: Node<'_>,
    length: usize,
    entries: &mut Vec<Spanned<Arc<str>>>,
) {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "string"
            && child.end_byte() <= length
            && let Some(value) = decode_string(document.node_text(child))
        {
            entries.push(Spanned::new(value.into(), span(child)));
        }
    }
}

fn headers(
    document: &Document,
    node: Node<'_>,
    length: usize,
    entries: &mut Vec<Spanned<Arc<str>>>,
) {
    // Recovery may leave a group's header directly beneath ERROR, without its group.
    strings(document, node, length, entries);
    for child in node.children(&mut node.walk()) {
        if child.kind() == "foreign_group" || child.is_error() {
            headers(document, child, length, entries);
        }
    }
}

fn diagnostics(node: Node<'_>, length: usize, errors: &mut Vec<Spanned<String>>) {
    if node.is_error() || node.is_missing() {
        let message = if node.is_missing() {
            format!("expected {} in source preamble", node.kind())
        } else {
            "unexpected or incomplete source preamble".into()
        };
        let span = Span {
            start: node.start_byte().min(length),
            end: node.end_byte().min(length),
        };
        errors.push(Spanned::new(message, span));
    }
    for child in node.children(&mut node.walk()) {
        diagnostics(child, length, errors);
    }
}
