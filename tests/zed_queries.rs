use std::{collections::BTreeSet, path::Path};
use tree_sitter::{Query, QueryCursor, StreamingIterator};

const QUERIES: &[(&str, &str)] = &[
    (
        "highlights",
        include_str!("../zed-resin/languages/resin/highlights.scm"),
    ),
    (
        "brackets",
        include_str!("../zed-resin/languages/resin/brackets.scm"),
    ),
    (
        "indents",
        include_str!("../zed-resin/languages/resin/indents.scm"),
    ),
    (
        "outline",
        include_str!("../zed-resin/languages/resin/outline.scm"),
    ),
    (
        "overrides",
        include_str!("../zed-resin/languages/resin/overrides.scm"),
    ),
    (
        "textobjects",
        include_str!("../zed-resin/languages/resin/textobjects.scm"),
    ),
];

fn captures(query: &str, source: &str) -> BTreeSet<(String, String)> {
    let language = tree_sitter_resin::LANGUAGE.into();
    let query = Query::new(&language, query).unwrap();
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&language).unwrap();
    let tree = parser.parse(source, None).unwrap();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, tree.root_node(), source.as_bytes());
    let mut captures = BTreeSet::new();
    while let Some(m) = matches.next() {
        for capture in m.captures() {
            captures.insert((
                query.capture_names()[capture.index as usize].into(),
                source[capture.node.byte_range()].into(),
            ));
        }
    }
    captures
}

#[test]
fn queries_capture_resin_constructs() {
    let source = "export { main, Number }; import { \"std/core.resin\" };\n\
        extern type Handle; extern \"lib.h\" native (arg: int) -> int;\n\
        Number = {field: int}; global = 1;\n\
        // a function\n\
        main (parameter: int) -> int = {\n\
            local = {field = 2}; pointer: Ptr(int); values = [1, 2];\n\
            native(parameter) + local.field + pointer.*\n\
        };";
    let results = QUERIES
        .iter()
        .map(|(name, query)| (*name, captures(query, source)))
        .collect::<std::collections::BTreeMap<_, _>>();
    for (query, capture, text) in [
        ("highlights", "function", "main"),
        ("highlights", "function", "native"),
        ("highlights", "variable.parameter", "parameter"),
        ("highlights", "type", "Number"),
        ("highlights", "type", "Handle"),
        ("highlights", "type.builtin", "Ptr"),
        ("highlights", "type.builtin", "int"),
        ("highlights", "property", "field"),
        ("highlights", "keyword", "export"),
        ("highlights", "operator", ".*"),
        ("brackets", "open", "["),
        ("brackets", "close", "]"),
        ("indents", "end", "}"),
        ("outline", "name", "main"),
        ("outline", "name", "global"),
        ("outline", "name", "Number"),
        ("overrides", "string", "\"std/core.resin\""),
        ("overrides", "comment.inclusive", "// a function"),
        ("textobjects", "comment.around", "// a function"),
    ] {
        assert!(
            results[query].contains(&(capture.into(), text.into())),
            "{query}: missing {capture} on {text}"
        );
    }
    assert!(!results["outline"].contains(&("name".into(), "local".into())));
    assert!(
        results["textobjects"]
            .iter()
            .any(|(kind, text)| kind == "function.inside" && text.contains("local ="))
    );
}

#[test]
fn queries_run_on_examples_stdlib_and_incomplete_code() {
    fn visit(path: &Path, sources: &mut Vec<String>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, sources);
            } else if path.extension().is_some_and(|ext| ext == "resin") {
                sources.push(std::fs::read_to_string(path).unwrap());
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources = vec!["main (argument: int) -> int = { pri".into()];
    visit(&root.join("examples"), &mut sources);
    visit(&root.join("stdlib"), &mut sources);
    for source in sources {
        for (_, query) in QUERIES {
            captures(query, &source);
        }
    }
}
