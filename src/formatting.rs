//! Canonical source formatting. Normalizes whitespace and numeric suffixes.
//!
//! Indentation uses hard tabs. A trailing comma forces a delimiter group onto
//! multiple lines, except for singleton tuples. Invalid syntax is left alone;
//! semantic analysis is unnecessary.

use tree_sitter::{Node, Parser};

/// Format a complete source buffer, or return `None` when it contains syntax errors.
pub fn format_source(source: &str) -> Option<String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .expect("Resin grammar");
    let tree = parser.parse(source, None)?;
    if tree.root_node().has_error() {
        return None;
    }
    let mut tokens = Vec::new();
    leaves(tree.root_node(), &mut tokens);
    let mut groups = vec![None; tokens.len()];
    let mut stack = Vec::new();
    for (i, node) in tokens.iter().enumerate() {
        let generic = node.parent().is_some_and(|p| p.kind() == "unary_type");
        match node.kind() {
            "(" | "[" | "{" => stack.push(i),
            "<" if generic => stack.push(i),
            ")" | "]" | "}" => finish_group(source, &tokens, &mut groups, &mut stack, i),
            ">" if generic => finish_group(source, &tokens, &mut groups, &mut stack, i),
            _ => {}
        }
    }
    let mut writer = Writer::default();
    let mut active: Vec<Group> = Vec::new();
    let mut pending_break = false;
    for (i, node) in tokens.iter().enumerate() {
        let text = &source[node.byte_range()];
        let previous = i.checked_sub(1).map(|i| tokens[i]);
        let gap = previous.map_or(&source[..node.start_byte()], |p| {
            &source[p.end_byte()..node.start_byte()]
        });
        let closing = active.last().is_some_and(|g| g.end == i);
        let opening = groups[i];
        let just_opened = previous.is_some_and(|_| groups[i - 1].is_some());
        if closing && active.last().unwrap().multiline {
            writer.indent -= 1;
            writer.newline();
        }
        if node.kind() == "comment" && gap.contains('\n') && !writer.output.is_empty() {
            writer.newline();
        }
        // Keep a single intentional blank line between statements/items/comments.
        // Never retain padding just inside a delimiter or at file boundaries.
        if writer.output.ends_with('\n')
            && !closing
            && !just_opened
            && gap.bytes().filter(|b| *b == b'\n').count() >= 2
        {
            writer.blank_line();
        }
        if let Some(prev) = previous
            && space_between(prev, *node)
        {
            writer.space();
        }
        if node.kind() == "number" {
            writer.write(&crate::ir::literal::format(text));
        } else {
            writer.write(text);
        }
        if closing {
            active.pop();
        }
        if let Some(group) = opening {
            active.push(group);
            if group.multiline {
                writer.indent += 1;
                // A comment on the opener's line stays attached to it.
                if !trailing_comment(source, &tokens, i) {
                    writer.newline();
                } else {
                    pending_break = true;
                }
            }
        }
        let break_after = match node.kind() {
            "lid" if node.parent().is_some_and(|p| p.kind() == "decorator") => true,
            "comment" => {
                text.starts_with("//")
                    || tokens.get(i + 1).is_some_and(|next| {
                        source[node.end_byte()..next.start_byte()].contains('\n')
                    })
            }
            ";" => true,
            "}" if node.parent().is_some_and(|p| p.kind() == "impl_definition") => true,
            "," => active.last().is_some_and(|g| g.multiline),
            _ => false,
        };
        pending_break |= break_after;
        if pending_break && !trailing_comment(source, &tokens, i) {
            writer.newline();
            pending_break = false;
        }
    }
    while writer.output.ends_with('\n') {
        writer.output.pop();
    }
    if !writer.output.is_empty() {
        writer.output.push('\n');
    }
    Some(writer.output)
}

fn leaves<'a>(node: Node<'a>, result: &mut Vec<Node<'a>>) {
    if node.child_count() == 0 {
        if node.kind() != "source_file" {
            result.push(node);
        }
    } else {
        for child in node.children(&mut node.walk()) {
            leaves(child, result);
        }
    }
}

#[derive(Clone, Copy)]
struct Group {
    end: usize,
    multiline: bool,
}

fn finish_group(
    source: &str,
    tokens: &[Node<'_>],
    groups: &mut [Option<Group>],
    stack: &mut Vec<usize>,
    end: usize,
) {
    let start = stack.pop().expect("balanced syntax");
    let body = &tokens[start + 1..end];
    let block = tokens[start].kind() == "{"
        && tokens[start].parent().is_some_and(|p| {
            matches!(
                p.kind(),
                "block_body" | "chain_term" | "match_term" | "impl_definition"
            )
        });
    let singleton_tuple = tokens[start].parent().is_some_and(|parent| {
        matches!(parent.kind(), "tuple_term" | "tuple_type")
            && parent
                .children_by_field_name("elems", &mut parent.walk())
                .take(2)
                .count()
                == 1
    });
    let trailing_comma = !singleton_tuple
        && body
            .iter()
            .rev()
            .find(|n| n.kind() != "comment")
            .is_some_and(|n| n.kind() == ",");
    let comments = body.iter().enumerate().any(|(i, n)| {
        n.kind() == "comment"
            && (source[n.byte_range()].starts_with("//")
                || source[n.byte_range()].contains('\n')
                || source[tokens[start].end_byte()..n.start_byte()].contains('\n')
                || source[n.end_byte()..tokens[start + i + 2].start_byte()].contains('\n'))
    });
    let nested_multiline = groups[start + 1..end].iter().flatten().any(|g| g.multiline);
    groups[start] = Some(Group {
        end,
        multiline: (block && !body.is_empty()) || trailing_comma || comments || nested_multiline,
    });
}

fn trailing_comment(source: &str, tokens: &[Node<'_>], i: usize) -> bool {
    tokens.get(i + 1).is_some_and(|next| {
        next.kind() == "comment" && !source[tokens[i].end_byte()..next.start_byte()].contains('\n')
    })
}

fn space_between(left: Node<'_>, right: Node<'_>) -> bool {
    if left.kind() == "@" {
        return false;
    }
    let a = left.kind();
    let b = right.kind();
    if right.parent().is_some_and(|p| p.kind() == "unwrap_suffix") {
        return false;
    }
    if matches!(
        b,
        "," | ";" | ")" | "]" | ":" | "." | "pointer_deref" | "try_suffix" | "unwrap_suffix"
    ) {
        return false;
    }
    if a == "comment" || b == "comment" {
        return true;
    }
    if matches!(a, "(" | "[" | ".") {
        return false;
    }
    let generic_left = left.parent().is_some_and(|p| p.kind() == "unary_type");
    let generic_right = right.parent().is_some_and(|p| p.kind() == "unary_type");
    if (a == "<" && generic_left) || (matches!(b, "<" | ">") && generic_right) {
        return false;
    }
    if a == "{" && b == "}" {
        return false;
    }
    if left.parent().is_some_and(|p| p.kind() == "unary_term") && is_operator(left) {
        // Two address-of operators must not become the logical-and token.
        return a == "&" && b == "&";
    }
    if b == "(" || b == "[" {
        return matches!(
            a,
            "if" | "while" | "match" | "else" | "=" | ":=" | "->" | "," | ":"
        ) || (a == ")" && left.parent().is_some_and(|p| p.kind() == "if_term"))
            || (is_operator(left) && !(a == ">" && generic_left));
    }
    true
}

fn is_operator(node: Node<'_>) -> bool {
    matches!(
        node.kind(),
        "+" | "-"
            | "*"
            | "/"
            | "%"
            | "!"
            | "~"
            | "&"
            | "|"
            | "^"
            | "&&"
            | "||"
            | "=="
            | "!="
            | "<"
            | ">"
            | "<="
            | ">="
            | "<<"
            | ">>"
    )
}

#[derive(Default)]
struct Writer {
    output: String,
    indent: usize,
}

impl Writer {
    fn write(&mut self, text: &str) {
        if self.output.is_empty() || self.output.ends_with('\n') {
            self.output.extend(std::iter::repeat_n('\t', self.indent));
        }
        self.output.push_str(text);
    }
    fn space(&mut self) {
        if !self.output.is_empty() && !self.output.ends_with(['\n', ' ', '\t']) {
            self.output.push(' ');
        }
    }
    fn newline(&mut self) {
        if !self.output.ends_with('\n') {
            self.output.push('\n');
        }
    }
    fn blank_line(&mut self) {
        if !self.output.is_empty() && !self.output.ends_with("\n\n") {
            self.output.push('\n');
        }
    }
}
