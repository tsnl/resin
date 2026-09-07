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
fn declaration_keywords_are_visible_in_outlines_and_struct_textobjects() {
    let source = "struct Point { x: float32, y: float32, }; type Position = Point;";
    let outline = captures(QUERIES[3].1, source);
    for (kind, text) in [
        ("context", "struct"),
        ("context", "type"),
        ("name", "Point"),
        ("name", "Position"),
    ] {
        assert!(
            outline.contains(&(kind.into(), text.into())),
            "{kind}: {text}"
        );
    }
    let objects = captures(QUERIES[5].1, source);
    assert!(objects.contains(&(
        "class.around".into(),
        "struct Point { x: float32, y: float32, };".into()
    )));
    for field in ["x: float32", "y: float32"] {
        assert!(objects.contains(&("class.inside".into(), field.into())));
    }
}

#[test]
fn reserved_words_have_highlight_rules() {
    let source = "export { f }; import { \"x.resin\" }; extern type Handle; struct S { value: int }; type T = S; def f() -> Result<(), Never> = { var x: Span<Ptr<ubyte>>; defer free(x); while (0 < 1) { if (0 == 1) { () } else { () }; }; match (value) { ok(v) => { ok(v) }, err(e) => { err(e) } } };";
    let highlighted = captures(QUERIES[0].1, source);
    for word in [
        "export", "import", "extern", "type", "struct", "def", "var", "if", "else", "while",
        "match", "defer",
    ] {
        assert!(
            highlighted.contains(&("keyword".into(), word.into())),
            "missing {word}"
        );
    }
}

#[test]
fn result_syntax_is_highlighted_and_structs_have_outlines() {
    let source = "struct Broken { code: int }; def fail() -> Result<int, Never> = { match (value) { ok(n) => { ok(n?) }, err(error) => { err(error) } } };";
    let captured = captures(QUERIES[0].1, source);
    for (kind, text) in [
        ("keyword", "struct"),
        ("keyword", "match"),
        ("type.builtin", "Result"),
        ("type.builtin", "Never"),
        ("operator", "?"),
        ("function.builtin", "ok"),
        ("variable.parameter", "error"),
    ] {
        assert!(
            captured.contains(&(kind.into(), text.into())),
            "{kind}: {text}"
        );
    }
    assert!(captures(QUERIES[3].1, source).contains(&("name".into(), "Broken".into())));
}

#[test]
fn inference_holes_are_highlighted_as_types() {
    let source = "def f(p: Ptr<int>) -> Ptr<_> = { var value: _; value := p; value };";
    let captured = captures(QUERIES[0].1, source);
    assert!(captured.contains(&("type.builtin".into(), "_".into())));
}

#[test]
fn defer_is_highlighted_as_a_keyword() {
    let source = "def f() = { defer print(\"done\", ()); };";
    assert!(captures(QUERIES[0].1, source).contains(&("keyword".into(), "defer".into())));
}

#[test]
fn queries_capture_resin_constructs() {
    let source = "export { main, Number }; import { \"std/core.resin\" };\n\
        extern type Handle; extern \"lib.h\" def native (arg: int) -> int;\n\
        struct Number {field: int};\n\
        // a function\n\
        def main (parameter: int) -> int = {\n\
            var local = {field = 2}; var pointer: Ptr<int>; var values = [1, 2];\n\
            native(parameter) + local.field + pointer.*\n\
        }; def reset() = {}; extern \"lib.h\" def finish();";
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .unwrap();
    assert!(!parser.parse(source, None).unwrap().root_node().has_error());
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
        ("highlights", "keyword", "def"),
        ("highlights", "keyword", "var"),
        ("highlights", "keyword", "type"),
        ("highlights", "function", "reset"),
        ("highlights", "function", "finish"),
        ("highlights", "operator", ".*"),
        ("brackets", "open", "["),
        ("brackets", "close", "]"),
        ("brackets", "open", "<"),
        ("brackets", "close", ">"),
        ("indents", "end", "}"),
        ("outline", "name", "main"),
        ("outline", "name", "Number"),
        ("outline", "name", "reset"),
        ("outline", "name", "finish"),
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
    let mut sources = vec!["def main (argument: int) -> int = { pri".into()];
    visit(&root.join("examples"), &mut sources);
    visit(&root.join("stdlib"), &mut sources);
    for source in sources {
        for (_, query) in QUERIES {
            captures(query, &source);
        }
    }
}

#[test]
fn comparison_operators_are_not_type_brackets() {
    let source = "def main () -> () = { var less = 1 < 2; var greater = 2 > 1; };";
    let query = include_str!("../zed-resin/languages/resin/brackets.scm");
    assert!(
        !captures(query, source)
            .iter()
            .any(|(_, text)| text == "<" || text == ">")
    );
}

#[test]
fn shader_decorators_are_highlighted_as_attributes() {
    let source = "@compute_shader def kernel(i: uint) -> uint = { i };";
    let highlighted = captures(QUERIES[0].1, source);
    assert!(highlighted.contains(&("attribute".into(), "@".into())));
    assert!(highlighted.contains(&("attribute".into(), "compute_shader".into())));
}
