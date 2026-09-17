use resin_cst::format_source;
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
fn operator_method_names_remain_attached_to_the_parameter_list() {
    check(
        "struct A{x:i32,}\nfn __add__(a:A,b:A)->A{a}\n\nfn __neg__(a:A)->A{a}\n\nfn __lshift__(a:A,b:i32)->A{a}\n",
        "struct A {\n\tx: i32,\n}\nfn __add__(a: A, b: A) -> A {\n\ta\n}\n\nfn __neg__(a: A) -> A {\n\ta\n}\n\nfn __lshift__(a: A, b: i32) -> A {\n\ta\n}\n",
    );
}

#[test]
fn constant_groups_keep_specifications_and_comments_together() {
    check(
        "const answer:i32=42;const(a,b:u32=1<<iota,8<<iota;_,_=iota,iota;c,d:u32=1<<iota,8<<iota;);fn f(){const(local=sizeof(i32);next=sizeof(i32););local}",
        "const answer: i32 = 42;\nconst (\n\ta, b: u32 = 1 << iota, 8 << iota;\n\t_, _ = iota, iota;\n\tc, d: u32 = 1 << iota, 8 << iota;\n);\nfn f() {\n\tconst (\n\t\tlocal = sizeof(i32);\n\t\tnext = sizeof(i32);\n\t);\n\tlocal\n}\n",
    );
    check(
        "const(a=iota; // zero\nb=iota;);",
        "const (\n\ta = iota; // zero\n\tb = iota;\n);\n",
    );
}

#[test]
fn trailing_commas_and_nested_lists() {
    check(
        "export {main,}; import {\"$/test.resin\",}; struct FieldsXY<T0, T1> { x: T0, y: T1, }\nfn main(a:i32,b:f32,){let mut xs=[1,2,3,]; call(a,b,); let mut r=FieldsXY<_, _> {x=1,y=2,}; ();}",
        "export {\n\tmain,\n};\nimport {\n\t\"$/test.resin\",\n};\nstruct FieldsXY<T0, T1> {\n\tx: T0,\n\ty: T1,\n}\nfn main(\n\ta: i32,\n\tb: f32,\n) {\n\tlet mut xs = [\n\t\t1,\n\t\t2,\n\t\t3,\n\t];\n\tcall(\n\t\ta,\n\t\tb,\n\t);\n\tlet mut r = FieldsXY<_, _> {\n\t\tx = 1,\n\t\ty = 2,\n\t};\n\t();\n}\n",
    );
    check(
        "extern {\"x.h\":{fn call(x:(i32,),);}}; struct FieldsXY<T0, T1> { x: T0, y: T1, }\ntype Pair=FieldsXY<i32, f32>; fn main(){f([1,2,],3);}",
        "extern {\n\t\"x.h\": {\n\t\tfn call(\n\t\t\tx: (i32,),\n\t\t);\n\t}\n};\nstruct FieldsXY<T0, T1> {\n\tx: T0,\n\ty: T1,\n}\ntype Pair = FieldsXY<i32, f32>;\nfn main() {\n\tf(\n\t\t[\n\t\t\t1,\n\t\t\t2,\n\t\t],\n\t\t3\n\t);\n}\n",
    );
}

#[test]
fn compact_lists_and_canonical_spacing() {
    check(
        "export{main};\n\n\n\nstruct FieldsXY<T0, T1> { x: T0, y: T1, }\nfn main ( ) -> ( ) {\n\nlet mut xs = [\n1,\n2\n];\n\n\nlet mut r=FieldsXY<_, _> {x=1,y=2};let mut p=Ptr<Ptr<i32>>( & &xs);let mut x= -(1+2)*3;while(x<2){x=x+1;};if(x>0){x}else{0}\n\n}\n\n",
        "export { main };\n\nstruct FieldsXY<T0, T1> {\n\tx: T0,\n\ty: T1,\n}\nfn main() -> () {\n\tlet mut xs = [1, 2];\n\n\tlet mut r = FieldsXY<_, _> { x = 1, y = 2 };\n\tlet mut p = Ptr<Ptr<i32>>(& &xs);\n\tlet mut x = -(1 + 2) * 3;\n\twhile (x < 2) {\n\t\tx = x + 1;\n\t};\n\tif (x > 0) {\n\t\tx\n\t} else {\n\t\t0\n\t}\n}\n",
    );
}

#[test]
fn comments_keep_contents_and_attachment() {
    check(
        "// header 😀\n\n\n\nfn main(){ // body\nlet mut a=[1, // first\n2, /* last */]; // array\n\n\n/* standalone\n   keep this indentation\n\n   and blank line */\nlet mut b=1; /* trailing */ let mut c=2;\ncall(\n// argument\nb);\n} // end",
        "// header 😀\n\nfn main() { // body\n\tlet mut a = [\n\t\t1, // first\n\t\t2, /* last */\n\t]; // array\n\n\t/* standalone\n   keep this indentation\n\n   and blank line */\n\tlet mut b = 1; /* trailing */\n\tlet mut c = 2;\n\tcall(\n\t\t// argument\n\t\tb\n\t);\n} // end\n",
    );
    check(
        "fn main(){let mut a=[1,/* comma */2,/* end */];}",
        "fn main() {\n\tlet mut a = [\n\t\t1, /* comma */\n\t\t2, /* end */\n\t];\n}\n",
    );
}

#[test]
fn pointer_dereference_stays_attached_to_its_operand() {
    check(
        "fn f(p:Ptr<i32>){p.* = 1;let mut x=p.*+1;p.*(x);}",
        "fn f(p: Ptr<i32>) {\n\tp.* = 1;\n\tlet mut x = p.* + 1;\n\tp.*(x);\n}\n",
    );
}

#[test]
fn structs_results_and_match() {
    check(
        "struct Empty{}struct Item{x:i32,}type Errors=Empty|Item;fn run()->(_ | Err<_>){cleanup();(cleanup());{cleanup();};let mut value=read()?;match(value){i32(v)=>{v},Err(e)=>{0},}}",
        "struct Empty {}\nstruct Item {\n\tx: i32,\n}\ntype Errors = Empty | Item;\nfn run() -> (_ | Err<_>) {\n\tcleanup();\n\t(cleanup());\n\t{\n\t\tcleanup();\n\t};\n\tlet mut value = read()?;\n\tmatch (value) {\n\t\ti32(v) => {\n\t\t\tv\n\t\t},\n\t\tErr(e) => {\n\t\t\t0\n\t\t},\n\t}\n}\n",
    );
    check(
        "fn main(){if(true)(1)else(0);match(x){i32(v)=>{},Err(e)=>{}};[f(),g()];}",
        "fn main() {\n\tif (true) (1) else (0);\n\tmatch (x) {\n\t\ti32(v) => {},\n\t\tErr(e) => {}\n\t};\n\t[f(), g()];\n}\n",
    );
}

#[test]
fn empty_invalid_and_literal_buffers() {
    for source in ["", " \t\r\n\n"] {
        assert_eq!(format_source(source).as_deref(), Some(""));
    }
    check("/* only */", "/* only */\n");
    check("fn main(){}", "fn main() {}\n");
    check(
        "fn main(){let mut s=\"😀  // /* \\\" \\n\";let mut n=0x0f;}\r\n",
        "fn main() {\n\tlet mut s = \"😀  // /* \\\" \\n\";\n\tlet mut n = 0x0f;\n}\n",
    );
    for source in [
        "fn main( = {",
        "fn main() { let mut x = ; }",
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
    let source = "export {main}; extern {\"x.h\": {fn ext(x: i32);}}; import {\"missing.resin\"}; struct FieldsX<T0> { x: T0, }\nstruct T {x: Ptr<i32>, y: (i32,),} fn main(a: i32) -> (_ | Err<Never>) { f(); let mut p = & &a; let mut x = f([1, 2,], FieldsX<_> {x = 3})?; while (x < 3) { x = x + 1; }; match (x) { i32(v) => { if (v == 3) { v } else { -v } }, Err(e) => { 0 } } }";
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
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), ""));
    let mut count = 0;
    for folder in ["examples", "resin"] {
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
        "@compute_shader fn kernel(i:u32)->u32{i}",
        "@compute_shader\nfn kernel(i: u32) -> u32 {\n\ti\n}\n",
    );
}

#[test]
fn else_if_chains_keep_else_and_if_together() {
    check(
        "fn f(x:i32)->i32{if(x==0){1}else if(x==1){2}else{3}}",
        "fn f(x: i32) -> i32 {\n\tif (x == 0) {\n\t\t1\n\t} else if (x == 1) {\n\t\t2\n\t} else {\n\t\t3\n\t}\n}\n",
    );
}

#[test]
fn numeric_annotations_and_one_armed_if_keep_their_spelling() {
    check(
        "fn main(){let mut a: u64=42;let mut b: i64=-42;if(a>u64(0)){let mut c: f32=1.5;};}",
        "fn main() {\n\tlet mut a: u64 = 42;\n\tlet mut b: i64 = -42;\n\tif (a > u64(0)) {\n\t\tlet mut c: f32 = 1.5;\n\t};\n}\n",
    );
}

#[test]
fn singleton_tuples_do_not_expand_for_their_trailing_comma() {
    check(
        "type Single=(i32,);fn main(x:(i32,)){let mut a=(\n1,\n);let mut b=((1,2),);let mut c=((1,),);f((x,));}",
        "type Single = (i32,);\nfn main(x: (i32,)) {\n\tlet mut a = (1,);\n\tlet mut b = ((1, 2),);\n\tlet mut c = ((1,),);\n\tf((x,));\n}\n",
    );
    check(
        "fn main(){let mut a=(1,/* note */);let mut b=(1,// note\n);let mut c=([1,],);let mut d=(1,2,);}",
        "fn main() {\n\tlet mut a = (1, /* note */);\n\tlet mut b = (\n\t\t1, // note\n\t);\n\tlet mut c = (\n\t\t[\n\t\t\t1,\n\t\t],\n\t);\n\tlet mut d = (\n\t\t1,\n\t\t2,\n\t);\n}\n",
    );
}

#[test]
fn optional_unwrap_suffix_stays_attached_and_preserves_operators() {
    check(
        "fn f(x: i32 | None) -> bool { x! != 0 }",
        "fn f(x: i32 | None) -> bool {\n\tx! != 0\n}\n",
    );
}

#[test]
fn structs_and_free_functions_have_independent_bodies() {
    check(
        "struct N{v:i32,  }\nfn n_new(v:i32)->N{N{v=v}}\n\nfn read(self:N)->i32{self.v}\nfn main()->i32{n_new(42):read()}",
        "struct N {\n\tv: i32,\n}\nfn n_new(v: i32) -> N {\n\tN { v = v }\n}\n\nfn read(self: N) -> i32 {\n\tself.v\n}\nfn main() -> i32 {\n\tn_new(42):read()\n}\n",
    );
}

#[test]
fn numeric_spelling_is_preserved() {
    for (input, output) in [
        ("1_000", "1_000"),
        ("1E3", "1E3"),
        ("0xAB", "0xAB"),
        ("0x1_F", "0x1_F"),
        ("0xdead", "0xdead"),
        ("-128", "-128"),
        ("123", "123"),
        ("1.5", "1.5"),
    ] {
        let source = format!("fn main(){{let mut n={input};}}");
        let expected = format!("fn main() {{\n\tlet mut n = {output};\n}}\n");
        let formatted = format_source(&source).unwrap();
        assert_eq!(formatted, expected, "{input}");
        assert_eq!(format_source(&formatted).unwrap(), formatted);
        // Whitespace formatting must preserve the parsed tree structure.
        assert_eq!(
            syntax(&source)
                .iter()
                .map(|(kind, _)| kind)
                .collect::<Vec<_>>(),
            syntax(&formatted)
                .iter()
                .map(|(kind, _)| kind)
                .collect::<Vec<_>>()
        );
    }
    check(
        "// 42UL\nfn main(){let mut text=\"42UL 1.5F\";}",
        "// 42UL\nfn main() {\n\tlet mut text = \"42UL 1.5F\";\n}\n",
    );
}

#[test]
fn template_delimiters_and_turbofish_are_attached() {
    check(
        "fn id <T>(x:T)->T{x}fn main(){let mut f=id:: <Ptr<i32>>;gpu:alloc:: < Pair<i32,i64> >(4);let mut n=2>1;}",
        "fn id<T>(x: T) -> T {\n\tx\n}\nfn main() {\n\tlet mut f = id::<Ptr<i32>>;\n\tgpu:alloc::<Pair<i32, i64>>(4);\n\tlet mut n = 2 > 1;\n}\n",
    );
}

#[test]
fn intrinsic_declarations_format_without_semantic_lookup() {
    check(
        "intrinsic \"pointer_index\" fn at <T>(data:Ptr<T>,length:u64,index:u64)->Ptr<T>;",
        "intrinsic \"pointer_index\" fn at<T>(data: Ptr<T>, length: u64, index: u64) -> Ptr<T>;\n",
    );
}

#[test]
fn reference_types_and_initialized_binding_annotations() {
    check(
        "fn identity<T>(value:Ref<T>)->Ref<T>{value}fn main(){let mut value:i32=1;let mut alias:Ref<_>=identity(value);alias=2;}",
        "fn identity<T>(value: Ref<T>) -> Ref<T> {\n\tvalue\n}\nfn main() {\n\tlet mut value: i32 = 1;\n\tlet mut alias: Ref<_> = identity(value);\n\talias = 2;\n}\n",
    );
}

#[test]
fn error_constructors_and_type_arguments_preserve_syntax() {
    let source = "fn f() -> Err<str> { Err(\"failure\") }";
    let formatted = format_source(source).unwrap();
    assert_eq!(syntax(source), syntax(&formatted));
    assert_eq!(format_source(&formatted).unwrap(), formatted);
}
