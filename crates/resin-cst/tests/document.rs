use resin_cst::{Document, Node};

fn nodes(node: Node<'_>) -> Vec<(String, usize, usize)> {
    let mut result = vec![(node.kind().into(), node.start_byte(), node.end_byte())];
    for child in node.children(&mut node.walk()) {
        result.extend(nodes(child));
    }
    result
}

#[test]
fn incremental_edits_match_fresh_parsing_including_utf8_boundaries() {
    let texts = [
        "",
        "// é",
        "// ê\n",
        "// 🌲\n",
        "// 🌳\r\n",
        "def main() = { print(\"é🌲\") };",
        "def main() = { print(\"ê🌳\") };",
        "def main() = { print(\"ê🌳\")",
        "def main() = { print(\"",
        "def main() = { var value: int = 4; value };",
    ];
    for before in texts {
        let original = Document::reparse(before.into(), None);
        for after in texts {
            let incremental = Document::reparse(after.into(), Some(&original));
            let fresh = Document::reparse(after.into(), None);
            assert_eq!(
                nodes(incremental.tree().root_node()),
                nodes(fresh.tree().root_node())
            );
            assert_eq!(original.source(), before);
        }
    }
}

#[test]
fn queries_reject_invalid_utf8_and_out_of_range_offsets() {
    let source = "// é🌲\ndef main() = {};";
    let document = Document::reparse(source.into(), None);
    for offset in 0..=source.len() + 1 {
        let token = document.token(offset);
        let types = document.type_context(offset);
        if !source.is_char_boundary(offset) {
            assert!(token.is_none());
            assert!(!types);
        }
    }
    assert!(document.token(usize::MAX).is_none());
    assert!(!document.type_context(usize::MAX));
}

#[test]
fn delimiter_recovery_preserves_source_and_ignores_string_contents() {
    let source = "def main() = { print(\"[({é\")";
    let document = Document::reparse(source.into(), None);
    let recovered = document
        .recovery()
        .expect("the block needs a closing brace");
    assert!(recovered.source().starts_with(source));
    assert!(!recovered.tree().root_node().has_error());
    assert!(document.tree().root_node().has_error());
    assert_eq!(document.source(), source);
    assert!(recovered.recovery().is_none());
}

#[test]
fn formatting_needs_no_semantic_context() {
    let source = "def unknown() = { missing_name(42) };";
    let formatted = resin_cst::format_source(source).unwrap();
    assert_eq!(resin_cst::format_source(&formatted).unwrap(), formatted);
    assert!(resin_cst::format_source("def broken( = {").is_none());
}
