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

#[test]
fn source_wrapper_names_preserve_token_queries_and_formatting() {
    let source = "def first(values: GpuSpan<uint>) -> GpuPtr<uint> = { values.at(0_ul) };";
    let document = Document::reparse(source.into(), None);
    assert!(!document.tree().root_node().has_error());
    for former in ["GpuSpan", "GpuPtr"] {
        let offset = source.find(former).unwrap() + 2;
        assert_eq!(document.token(offset).unwrap().kind(), "uid");
        assert!(document.type_context(offset));
    }
    let formatted = resin_cst::format_source(source).unwrap();
    assert_eq!(resin_cst::format_source(&formatted).unwrap(), formatted);
    assert!(formatted.contains("GpuSpan<uint>"));
    assert!(formatted.contains("GpuPtr<uint>"));
}

#[test]
fn gpu_pipeline_annotations_preserve_type_queries_and_formatting() {
    let source = "def pipeline(value:GpuComputePipeline<Root,ArcPtr<Owner>>)->GpuGraphicsPipeline<None,_> = {value};";
    let document = Document::reparse(source.into(), None);
    assert!(!document.tree().root_node().has_error());
    for former in ["GpuComputePipeline", "GpuGraphicsPipeline"] {
        let offset = source.find(former).unwrap() + 2;
        assert_eq!(document.token(offset).unwrap().kind(), "uid");
        assert!(document.type_context(offset));
    }
    let formatted = resin_cst::format_source(source).unwrap();
    assert_eq!(resin_cst::format_source(&formatted).unwrap(), formatted);
    assert!(formatted.contains("GpuComputePipeline<Root, ArcPtr<Owner>>"));
    assert!(formatted.contains("GpuGraphicsPipeline<None, _>"));
}

#[test]
fn associated_method_references_format_and_preserve_type_queries() {
    let source = "def main()={var f=Cell<int>.select :: <ulong>;var g=Factory.create :: <Ptr<int>>;Cell<int>.select :: <ulong>(7,42);};";
    let document = Document::reparse(source.into(), None);
    assert!(!document.tree().root_node().has_error());
    assert!(document.type_context(source.find("ulong").unwrap()));
    assert_eq!(
        document
            .token(source.find("select").unwrap())
            .unwrap()
            .kind(),
        "lid"
    );
    let formatted = resin_cst::format_source(source).unwrap();
    assert!(
        formatted.contains("Cell<int>.select::<ulong>"),
        "{formatted}"
    );
    assert!(
        formatted.contains("Factory.create::<Ptr<int>>"),
        "{formatted}"
    );
    assert_eq!(resin_cst::format_source(&formatted).unwrap(), formatted);
    assert!(
        !Document::reparse(formatted, None)
            .tree()
            .root_node()
            .has_error()
    );
}

#[test]
fn editing_method_reference_arguments_matches_fresh_parsing() {
    let sources = [
        "def main() = { var f = Cell<int>.",
        "def main() = { var f = Cell<int>.select::",
        "def main() = { var f = Cell<int>.select::<",
        "def main() = { var f = Cell<int>.select::<ulong",
        "def main() = { var f = Cell<int>.select::<ulong>; };",
        "def main() = { var f = Cell<int>.select::<ulong>(7, 42); };",
    ];
    for before in sources {
        let original = Document::reparse(before.into(), None);
        for after in sources {
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
