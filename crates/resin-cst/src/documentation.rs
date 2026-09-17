//! Attach Markdown to syntax declarations. No name resolution or source loading.
use crate::{DeclarationDocumentation, Document, Documentation, Node, span};
use resin_source::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn read(document: &Document) -> Documentation {
    let root = document.tree().root_node();
    let exports = exports(document, root);
    let mut result = Documentation::default();
    let mut targets = BTreeMap::new();
    declarations(document, root, &exports, &mut result, &mut targets);
    let mut tokens = Vec::new();
    leaves(root, &mut tokens);
    attach(document, &tokens, &targets, &mut result);
    result
}

fn exports<'a>(document: &'a Document, root: Node<'_>) -> BTreeSet<&'a str> {
    root.child_by_field_name("exports")
        .map_or_else(BTreeSet::new, |clause| {
            clause
                .children_by_field_name("name", &mut clause.walk())
                .map(|node| document.node_text(node))
                .collect()
        })
}

fn declarations(
    document: &Document,
    node: Node<'_>,
    exports: &BTreeSet<&str>,
    result: &mut Documentation,
    targets: &mut BTreeMap<usize, Vec<usize>>,
) {
    let first = result.declarations.len();
    if let Some(name) = declaration_name(node) {
        add(document, node, name, exports, result, targets);
    } else if node.kind() == "const_spec" {
        for name in node.children_by_field_name("name", &mut node.walk()) {
            if name.kind() != "discard" {
                add(document, node, name, exports, result, targets);
            }
        }
    }
    for child in node.named_children(&mut node.walk()) {
        declarations(document, child, exports, result, targets);
    }
    if node.kind() == "const_declaration" {
        let entries = (first..result.declarations.len())
            .filter(|&i| {
                node.children_by_field_name("spec", &mut node.walk())
                    .any(|spec| result.declarations[i].span == span(spec))
            })
            .collect();
        targets.insert(node.start_byte(), entries);
    }
}

fn declaration_name(node: Node<'_>) -> Option<Node<'_>> {
    match node.kind() {
        "function_definition"
        | "intrinsic_function"
        | "foreign_function"
        | "foreign_type"
        | "struct_definition" => node.child_by_field_name("name"),
        "type_definition" => node
            .child_by_field_name("definition")?
            .child_by_field_name("name"),
        "define" => node
            .child_by_field_name("type")?
            .child_by_field_name("name"),
        "declare" if node.parent()?.kind() == "struct_definition" => {
            node.child_by_field_name("name")
        }
        _ => None,
    }
}

fn add(
    document: &Document,
    node: Node<'_>,
    name: Node<'_>,
    exports: &BTreeSet<&str>,
    result: &mut Documentation,
    targets: &mut BTreeMap<usize, Vec<usize>>,
) {
    let owner = (node.kind() == "declare").then(|| span(node.parent().unwrap()));
    let end = node
        .child_by_field_name("body")
        .map_or(node.end_byte(), |body| body.start_byte());
    let signature = document.source()[node.start_byte()..end].trim().to_string();
    let signature = if node.kind() == "const_spec" {
        format!("const {signature};")
    } else {
        signature
    };
    let exported =
        owner.is_none() && module_declaration(node) && exports.contains(document.node_text(name));
    targets
        .entry(node.start_byte())
        .or_default()
        .push(result.declarations.len());
    result.declarations.push(DeclarationDocumentation {
        name: Spanned::new(document.node_text(name).to_string(), span(name)),
        span: span(node),
        signature,
        markdown: String::new(),
        exported,
        field_owner: owner,
    });
}

fn module_declaration(mut node: Node<'_>) -> bool {
    while let Some(parent) = node.parent() {
        match parent.kind() {
            "source_file" => return true,
            "foreign_group" | "extern_clause" | "const_declaration" => node = parent,
            _ => return false,
        }
    }
    false
}

fn leaves<'a>(node: Node<'a>, output: &mut Vec<Node<'a>>) {
    if node.child_count() == 0 {
        if node.kind() != "source_file" {
            output.push(node);
        }
    } else {
        for child in node.children(&mut node.walk()) {
            leaves(child, output);
        }
    }
}

fn attach(
    document: &Document,
    tokens: &[Node<'_>],
    targets: &BTreeMap<usize, Vec<usize>>,
    result: &mut Documentation,
) {
    let mut pending = Vec::new();
    let mut preamble = true;
    for &token in tokens {
        match token.kind() {
            "comment" => {}
            "doc_comment" => {
                let text = document.node_text(token);
                if text.starts_with("//!") || text.starts_with("/*!") {
                    if preamble && pending.is_empty() {
                        append(&mut result.module, &markdown(text));
                    } else {
                        result.diagnostics.push(Spanned::new("module documentation must precede all declarations and preamble clauses".into(), span(token)));
                    }
                } else {
                    pending.push(token);
                }
            }
            _ => {
                preamble = false;
                if !pending.is_empty() {
                    finish(document, &pending, targets.get(&token.start_byte()), result);
                    pending.clear();
                }
            }
        }
    }
    finish(document, &pending, None, result);
}

fn finish(
    document: &Document,
    pending: &[Node<'_>],
    targets: Option<&Vec<usize>>,
    result: &mut Documentation,
) {
    for &comment in pending {
        if let Some(targets) = targets.filter(|targets| !targets.is_empty()) {
            for &index in targets {
                append(
                    &mut result.declarations[index].markdown,
                    &markdown(document.node_text(comment)),
                );
            }
        } else {
            result.diagnostics.push(Spanned::new(
                "documentation comment must precede a declaration or struct field".into(),
                span(comment),
            ));
        }
    }
}

fn append(output: &mut String, text: &str) {
    if !output.is_empty() {
        output.push('\n');
    }
    output.push_str(text);
}

fn markdown(comment: &str) -> String {
    if comment.starts_with("//") {
        let text = &comment[3..];
        return text
            .strip_prefix(' ')
            .unwrap_or(text)
            .trim_end_matches('\r')
            .into();
    }
    let text = &comment[3..comment.len() - 2];
    let lines: Vec<_> = text.lines().collect();
    let decorated = lines
        .iter()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .all(|line| line.trim_start().starts_with('*'));
    let mut lines: Vec<_> = lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            if i == 0 {
                line.trim_start().to_string()
            } else if decorated {
                let line = line.trim_start().strip_prefix('*').unwrap_or("");
                line.strip_prefix(' ').unwrap_or(line).to_string()
            } else {
                (*line).to_string()
            }
        })
        .collect();
    if !decorated {
        let indent = lines
            .iter()
            .skip(1)
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                line.bytes()
                    .take_while(|byte| matches!(byte, b' ' | b'\t'))
                    .count()
            })
            .min()
            .unwrap_or(0);
        for line in lines.iter_mut().skip(1) {
            let prefix = line
                .bytes()
                .take(indent)
                .take_while(|byte| matches!(byte, b' ' | b'\t'))
                .count();
            line.drain(..prefix);
        }
    }
    let start = lines
        .iter()
        .position(|line| !line.trim().is_empty())
        .unwrap_or(lines.len());
    let end = lines
        .iter()
        .rposition(|line| !line.trim().is_empty())
        .map_or(start, |index| index + 1);
    lines[start..end].join("\n")
}
