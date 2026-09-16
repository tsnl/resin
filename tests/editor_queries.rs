use std::{collections::BTreeSet, path::Path};
use tree_sitter::{Query, QueryCursor, StreamingIterator};

const ZED_QUERIES: &[(&str, &str)] = &[
    (
        "highlights",
        include_str!("../editors/zed/languages/resin/highlights.scm"),
    ),
    (
        "brackets",
        include_str!("../editors/zed/languages/resin/brackets.scm"),
    ),
    (
        "indents",
        include_str!("../editors/zed/languages/resin/indents.scm"),
    ),
    (
        "outline",
        include_str!("../editors/zed/languages/resin/outline.scm"),
    ),
    (
        "overrides",
        include_str!("../editors/zed/languages/resin/overrides.scm"),
    ),
    (
        "textobjects",
        include_str!("../editors/zed/languages/resin/textobjects.scm"),
    ),
];

const HELIX_QUERIES: &[(&str, &str)] = &[
    (
        "highlights",
        include_str!("../editors/helix/runtime/queries/resin/highlights.scm"),
    ),
    (
        "indents",
        include_str!("../editors/helix/runtime/queries/resin/indents.scm"),
    ),
    (
        "textobjects",
        include_str!("../editors/helix/runtime/queries/resin/textobjects.scm"),
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
fn constants_and_sizeof_are_highlighted_and_outlined() {
    let source = "const ( size = sizeof(int); next = iota; );";
    for queries in [ZED_QUERIES, HELIX_QUERIES] {
        let highlighted = captures(queries[0].1, source);
        for (kind, text) in [
            ("keyword", "const"),
            ("constant", "size"),
            ("constant.builtin", "iota"),
            ("function.builtin", "sizeof"),
        ] {
            assert!(
                highlighted.contains(&(kind.into(), text.into())),
                "{kind}: {text}"
            );
        }
    }
    let outline = captures(ZED_QUERIES[3].1, source);
    assert!(outline.contains(&("name".into(), "size".into())));
}

#[test]
fn declaration_keywords_are_visible_in_outlines_and_struct_textobjects() {
    let source = "struct Point { x: float32; y: float32; } type Position = Point;";
    let outline = captures(ZED_QUERIES[3].1, source);
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
    let objects = captures(ZED_QUERIES[5].1, source);
    assert!(objects.contains(&(
        "class.around".into(),
        "struct Point { x: float32; y: float32; }".into()
    )));
    for field in ["x: float32", "y: float32"] {
        assert!(objects.contains(&("class.inside".into(), field.into())));
    }
}

#[test]
fn reserved_words_have_highlight_rules() {
    let source = "export { f }; import { \"x.resin\" }; extern type Handle; struct S { value: int; } type T = S; fn f() -> (() | Err<Never>)  { let mut x: Span<Ptr<ubyte>>; free(x); while (0 < 1) { if (0 == 1) { () } else { () }; }; match (value) { ()(v) => { (v) }, Err(e) => { Err(e) } } }";
    let highlighted = captures(ZED_QUERIES[0].1, source);
    for word in [
        "export", "import", "extern", "type", "struct", "fn", "let", "mut", "if", "else", "while",
        "match",
    ] {
        assert!(
            highlighted.contains(&("keyword".into(), word.into())),
            "missing {word}"
        );
    }
}

#[test]
fn result_syntax_is_highlighted_and_structs_have_outlines() {
    let source = "struct Broken { code: int; } fn fail() -> (int | Err<Never>)  { match (value) { int(n) => { (n?) }, Err(error) => { Err(error) } } }";
    let captured = captures(ZED_QUERIES[0].1, source);
    for (kind, text) in [
        ("keyword", "struct"),
        ("keyword", "match"),
        ("type.builtin", "Err"),
        ("type.builtin", "Never"),
        ("operator", "?"),
        ("variable.parameter", "error"),
    ] {
        assert!(
            captured.contains(&(kind.into(), text.into())),
            "{kind}: {text}"
        );
    }
    assert!(captures(ZED_QUERIES[3].1, source).contains(&("name".into(), "Broken".into())));
}

#[test]
fn none_and_postfix_unwrapping_are_highlighted() {
    let source = "fn f(o: int | None) -> int  { let mut empty: int | None; empty = None; o! }";
    let captured = captures(ZED_QUERIES[0].1, source);
    for (kind, text) in [("type.builtin", "None"), ("operator", "!")] {
        assert!(
            captured.contains(&(kind.into(), text.into())),
            "{kind}: {text}"
        );
    }
}

#[test]
fn inference_holes_are_highlighted_as_types() {
    let source = "fn f(p: Ptr<int>) -> Ptr<_>  { let mut value: _; value = p; value }";
    let captured = captures(ZED_QUERIES[0].1, source);
    assert!(captured.contains(&("type.builtin".into(), "_".into())));
}

#[test]
fn queries_capture_resin_constructs() {
    let source = r#"export { main, Number };
        extern {
            "lib.h": {
                fn native (arg: int) -> int;
                fn finish();
            },
        };
        import { "$/core.resin" };
        struct FieldsField<T0> { field: T0; }
extern type Handle;
        struct Number {field: int;}
        // a function
        fn main (parameter: int) -> int  {
            let mut local = FieldsField<_> {field = 2}; let mut pointer: Ptr<int>; let mut values = [1, 2];
            native(parameter) + local.field + pointer.*
        }
        fn reset()  {}
    "#;
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .unwrap();
    assert!(!parser.parse(source, None).unwrap().root_node().has_error());
    let results = ZED_QUERIES
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
        ("highlights", "keyword", "fn"),
        ("highlights", "keyword", "let"),
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
        ("overrides", "string", "\"$/core.resin\""),
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
fn queries_run_on_examples_libraries_and_incomplete_code() {
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
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), ""));
    let mut sources = vec!["fn main (argument: int) -> int = { pri".into()];
    visit(&root.join("examples"), &mut sources);
    visit(&root.join("resin"), &mut sources);
    for source in sources {
        for (_, query) in ZED_QUERIES.iter().chain(HELIX_QUERIES) {
            captures(query, &source);
        }
    }
}

#[test]
fn comparison_operators_are_not_type_brackets() {
    let source = "fn main () -> ()  { let mut less = 1 < 2; let mut greater = 2 > 1; }";
    let query = include_str!("../editors/zed/languages/resin/brackets.scm");
    assert!(
        !captures(query, source)
            .iter()
            .any(|(_, text)| text == "<" || text == ">")
    );
}

#[test]
fn shader_decorators_are_highlighted_as_attributes() {
    let source = "@compute_shader fn kernel(i: uint) -> uint  { i }";
    let highlighted = captures(ZED_QUERIES[0].1, source);
    assert!(highlighted.contains(&("attribute".into(), "@".into())));
    assert!(highlighted.contains(&("attribute".into(), "compute_shader".into())));
}

#[test]
fn intrinsic_declarations_have_function_navigation_and_parameter_highlights() {
    let source = r#"intrinsic "pointer_index" fn at<T>(data: Ptr<T>, length: ulong, index: ulong) -> Ptr<T>;"#;
    let highlights = captures(ZED_QUERIES[0].1, source);
    for (kind, text) in [
        ("keyword", "intrinsic"),
        ("function", "at"),
        ("variable.parameter", "data"),
    ] {
        assert!(highlights.contains(&(kind.into(), text.into())));
    }
    assert!(captures(ZED_QUERIES[3].1, source).contains(&("name".into(), "at".into())));
    assert!(
        captures(ZED_QUERIES[5].1, source).contains(&("function.around".into(), source.into()))
    );
}

#[test]
fn helix_queries_capture_highlights_indentation_and_textobjects() {
    let source = r#"struct Cell<T> { value: T; }
        // increment
        fn increment(value: int) -> int {
            let cell = Cell<int> { value = 1 };
            let less = value < 10;
            cell.value + value
        }"#;
    let highlights = captures(HELIX_QUERIES[0].1, source);
    for (kind, text) in [
        ("keyword", "struct"),
        ("keyword", "fn"),
        ("type", "Cell"),
        ("type.builtin", "int"),
        ("function", "increment"),
        ("variable.parameter", "value"),
        ("variable.other.member", "value"),
        ("constant.numeric", "1"),
        ("operator", "<"),
    ] {
        assert!(
            highlights.contains(&(kind.into(), text.into())),
            "{kind}: {text}"
        );
    }
    let indents = captures(HELIX_QUERIES[1].1, source);
    for text in ["}", ")", ">"] {
        assert!(indents.contains(&("outdent".into(), text.into())));
    }
    assert!(!indents.contains(&("outdent".into(), "<".into())));
    let comparison = "fn compare(value: int) -> bool  { value > 10 }";
    assert!(!captures(HELIX_QUERIES[1].1, comparison).contains(&("outdent".into(), ">".into())));
    let objects = captures(HELIX_QUERIES[2].1, source);
    for (kind, text) in [
        ("class.around", "struct Cell<T> { value: T; }"),
        ("class.inside", "value: T"),
        ("comment.around", "// increment"),
    ] {
        assert!(objects.contains(&(kind.into(), text.into())));
    }
    assert!(
        objects
            .iter()
            .any(|(kind, text)| kind == "function.inside" && text.contains("cell.value"))
    );
}
