use resin::formatting::format_source;
use std::path::Path;
use tree_sitter::{Node, Parser};

fn syntax(source: &str) -> Vec<(String, String)> {
    fn visit(node: Node<'_>, source: &str, out: &mut Vec<(String, String)>) {
        // Include the tree structure, not just tokens: whitespace must not change parsing.
        out.push((
            node.kind().into(),
            if node.child_count() == 0 {
                source[node.byte_range()].into()
            } else {
                String::new()
            },
        ));
        for child in node.children(&mut node.walk()) {
            visit(child, source, out);
        }
        out.push(("end".into(), String::new()));
    }
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .unwrap();
    let tree = parser.parse(source, None).unwrap();
    assert!(!tree.root_node().has_error(), "invalid syntax: {source}");
    let mut result = Vec::new();
    visit(tree.root_node(), source, &mut result);
    result
}

fn check(source: &str, expected: &str) {
    let formatted = format_source(source).expect("valid input");
    assert_eq!(formatted, expected);
    assert_eq!(syntax(source), syntax(&formatted));
    assert_eq!(
        format_source(&formatted).unwrap(),
        formatted,
        "not idempotent"
    );
}

#[test]
fn trailing_commas_and_nested_lists() {
    check(
        "export {main,}; import {\"std/test.resin\",}; def main(a:int,b:float32,)={var xs=[1,2,3,]; call(a,b,); var r={x=1,y=2,}; ();};",
        "export {\n\tmain,\n};\nimport {\n\t\"std/test.resin\",\n};\ndef main(\n\ta: int,\n\tb: float32,\n) = {\n\tvar xs = [\n\t\t1,\n\t\t2,\n\t\t3,\n\t];\n\tcall(\n\t\ta,\n\t\tb,\n\t);\n\tvar r = {\n\t\tx = 1,\n\t\ty = 2,\n\t};\n\t();\n};\n",
    );
    check(
        "type Pair={x:int,y:float32,}; extern \"x.h\" def call(x:(int,),); def main()={f([1,2,],3);};",
        "type Pair = {\n\tx: int,\n\ty: float32,\n};\nextern \"x.h\" def call(\n\tx: (int,),\n);\ndef main() = {\n\tf(\n\t\t[\n\t\t\t1,\n\t\t\t2,\n\t\t],\n\t\t3\n\t);\n};\n",
    );
}

#[test]
fn compact_lists_and_canonical_spacing() {
    check(
        "\n\nexport{main};\n\n\n\ndef main ( ) -> ( ) ={\n\nvar xs = [\n1,\n2\n];\n\n\nvar r={x=1,y=2};var p=Ptr<Ptr<int>>( & &xs);var x= -(1+2)*3;while(x<2){x:=x+1;};if(x>0){x}else{0}\n\n};\n\n",
        "export { main };\n\ndef main() -> () = {\n\tvar xs = [1, 2];\n\n\tvar r = { x = 1, y = 2 };\n\tvar p = Ptr<Ptr<int>>(& &xs);\n\tvar x = -(1 + 2) * 3;\n\twhile (x < 2) {\n\t\tx := x + 1;\n\t};\n\tif (x > 0) {\n\t\tx\n\t} else {\n\t\t0\n\t}\n};\n",
    );
}

#[test]
fn comments_keep_contents_and_attachment() {
    check(
        "// header 😀\n\n\n\ndef main()={ // body\nvar a=[1, // first\n2, /* last */]; // array\n\n\n/* standalone\n   keep this indentation\n\n   and blank line */\nvar b=1; /* trailing */ var c=2;\ncall(\n// argument\nb);\n}; // end",
        "// header 😀\n\ndef main() = { // body\n\tvar a = [\n\t\t1, // first\n\t\t2, /* last */\n\t]; // array\n\n\t/* standalone\n   keep this indentation\n\n   and blank line */\n\tvar b = 1; /* trailing */\n\tvar c = 2;\n\tcall(\n\t\t// argument\n\t\tb\n\t);\n}; // end\n",
    );
    check(
        "def main()={var a=[1,/* comma */2,/* end */];};",
        "def main() = {\n\tvar a = [\n\t\t1, /* comma */\n\t\t2, /* end */\n\t];\n};\n",
    );
}

#[test]
fn pointer_dereference_stays_attached_to_its_operand() {
    check(
        "def f(p:Ptr<int>)={p .* := 1;var x=p .*+1;p .*(x);};",
        "def f(p: Ptr<int>) = {\n\tp.* := 1;\n\tvar x = p.* + 1;\n\tp.*(x);\n};\n",
    );
}

#[test]
fn structs_results_match_and_defer() {
    check(
        "struct Empty{};struct Item{x:int,};type Errors=Empty|Item;def run()->Result<_,_>={defer cleanup();defer (cleanup());defer {cleanup();};var value=read() ?;match(value){ok(v)=>{v},err(e)=>{0},}};",
        "struct Empty {};\nstruct Item {\n\tx: int,\n};\ntype Errors = Empty | Item;\ndef run() -> Result<_, _> = {\n\tdefer cleanup();\n\tdefer (cleanup());\n\tdefer {\n\t\tcleanup();\n\t};\n\tvar value = read()?;\n\tmatch (value) {\n\t\tok(v) => {\n\t\t\tv\n\t\t},\n\t\terr(e) => {\n\t\t\t0\n\t\t},\n\t}\n};\n",
    );
    check(
        "def main()={if(true)(1)else(0);match(x){ok(v)=>{},err(e)=>{}};defer [f(),g()];};",
        "def main() = {\n\tif (true) (1) else (0);\n\tmatch (x) {\n\t\tok(v) => {},\n\t\terr(e) => {}\n\t};\n\tdefer [f(), g()];\n};\n",
    );
}

#[test]
fn empty_invalid_and_literal_buffers() {
    for source in ["", " \t\r\n\n"] {
        assert_eq!(format_source(source).as_deref(), Some(""));
    }
    check("/* only */", "/* only */\n");
    check("def main()={};", "def main() = {};\n");
    check(
        "def main()={var s=\"😀  // /* \\\" \\n\";var n=0x0f;};\r\n",
        "def main() = {\n\tvar s = \"😀  // /* \\\" \\n\";\n\tvar n = 0x0f;\n};\n",
    );
    for source in [
        "def main( = {",
        "def main() = { var x = ; };",
        "/* unfinished",
    ] {
        assert!(format_source(source).is_none());
    }
}

#[test]
fn comments_at_every_token_boundary_preserve_syntax() {
    fn ends(node: Node<'_>, offsets: &mut Vec<usize>) {
        if node.child_count() == 0 {
            offsets.push(node.end_byte());
        } else {
            for child in node.children(&mut node.walk()) {
                ends(child, offsets);
            }
        }
    }
    let source = "export {main}; import {\"missing.resin\"}; struct T {x: Ptr<int>, y: (int,)}; extern \"x.h\" def ext(x: int); def main(a: int) -> Result<_, Never> = { defer f(); var p = & &a; var x = f([1, 2,], {x = 3})?; while (x < 3) { x := x + 1; }; match (x) { ok(v) => { if (v == 3) { v } else { -v } }, err(e) => { 0 } } };";
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .unwrap();
    let tree = parser.parse(source, None).unwrap();
    let mut offsets = vec![0];
    ends(tree.root_node(), &mut offsets);
    for offset in offsets {
        for comment in [
            " /* marker */ ",
            "\n\n// marker 😀\n\n",
            " /* marker */\n",
            " /* block\n\ncontents */ ",
        ] {
            let mut input = source.to_owned();
            input.insert_str(offset, comment);
            let formatted = format_source(&input).expect(&input);
            assert_eq!(syntax(&input), syntax(&formatted), "{input}");
            assert_eq!(format_source(&formatted).unwrap(), formatted, "{input}");
        }
    }
}

#[test]
fn repository_sources_preserve_syntax_and_are_idempotent() {
    fn visit(path: &Path, count: &mut usize) {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, count);
            } else if path.extension().is_some_and(|e| e == "resin") {
                let source = std::fs::read_to_string(&path).unwrap();
                // Examples may deliberately demonstrate rejected syntax.
                if let Some(formatted) = format_source(&source) {
                    assert_eq!(syntax(&source), syntax(&formatted), "{}", path.display());
                    assert_eq!(
                        format_source(&formatted).unwrap(),
                        formatted,
                        "{}",
                        path.display()
                    );
                    *count += 1;
                }
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut count = 0;
    for folder in ["examples", "stdlib"] {
        visit(&root.join(folder), &mut count);
    }
    assert!(
        count > 20,
        "expected coverage of the source corpus, got {count}"
    );
}

#[test]
fn shader_decorators_stay_on_their_own_lines() {
    check(
        "@compute_shader def kernel(i:uint)->uint={i};",
        "@compute_shader\ndef kernel(i: uint) -> uint = {\n\ti\n};\n",
    );
}

#[test]
fn numeric_suffixes_and_one_armed_if_keep_their_spelling() {
    check(
        "def main()={var a=42L;var b=-42l;if(a>0L){var c=1.5f;};};",
        "def main() = {\n\tvar a = 42L;\n\tvar b = -42l;\n\tif (a > 0L) {\n\t\tvar c = 1.5f;\n\t};\n};\n",
    );
}

#[test]
fn singleton_tuples_do_not_expand_for_their_trailing_comma() {
    check(
        "type Single=(int,);def main(x:(int,))={var a=(\n1,\n);var b=((1,2),);var c=((1,),);f(x,);};",
        "type Single = (int,);\ndef main(x: (int,)) = {\n\tvar a = (1,);\n\tvar b = ((1, 2),);\n\tvar c = ((1,),);\n\tf(x,);\n};\n",
    );
    check(
        "def main()={var a=(1,/* note */);var b=(1,// note\n);var c=([1,],);var d=(1,2,);};",
        "def main() = {\n\tvar a = (1, /* note */);\n\tvar b = (\n\t\t1, // note\n\t);\n\tvar c = (\n\t\t[\n\t\t\t1,\n\t\t],\n\t);\n\tvar d = (\n\t\t1,\n\t\t2,\n\t);\n};\n",
    );
}

#[test]
fn optional_unwrap_suffix_stays_attached_and_preserves_operators() {
    check(
        "def f(x: int | None) -> bool = { x ! != 0 };",
        "def f(x: int | None) -> bool = {\n\tx! != 0\n};\n",
    );
}
