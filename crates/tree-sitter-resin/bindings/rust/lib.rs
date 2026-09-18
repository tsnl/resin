//! This crate provides Resin language support for the [tree-sitter] parsing library.
//!
//! Typically, you will use the [`LANGUAGE`] constant to add this language to a
//! tree-sitter [`Parser`], and then use the parser to parse some code:
//!
//! ```
//! let code = r#"
//! "#;
//! let mut parser = tree_sitter::Parser::new();
//! let language = tree_sitter_resin::LANGUAGE;
//! parser
//!     .set_language(&language.into())
//!     .expect("Error loading Resin parser");
//! let tree = parser.parse(code, None).unwrap();
//! assert!(!tree.root_node().has_error());
//! ```
//!
//! [`Parser`]: https://docs.rs/tree-sitter/0.27.0/tree_sitter/struct.Parser.html
//! [tree-sitter]: https://tree-sitter.github.io/

use tree_sitter_language::LanguageFn;

unsafe extern "C" {
    fn tree_sitter_resin() -> *const ();
}

/// The tree-sitter [`LanguageFn`] for this grammar.
pub const LANGUAGE: LanguageFn = unsafe { LanguageFn::from_raw(tree_sitter_resin) };

/// The content of the [`node-types.json`] file for this grammar.
///
/// [`node-types.json`]: https://tree-sitter.github.io/tree-sitter/using-parsers/6-static-node-types
pub const NODE_TYPES: &str = include_str!("../../src/node-types.json");

#[cfg(with_highlights_query)]
/// The syntax highlighting query for this grammar.
pub const HIGHLIGHTS_QUERY: &str = include_str!("../../queries/highlights.scm");

#[cfg(with_injections_query)]
/// The language injection query for this grammar.
pub const INJECTIONS_QUERY: &str = include_str!("../../queries/injections.scm");

#[cfg(with_locals_query)]
/// The local variable query for this grammar.
pub const LOCALS_QUERY: &str = include_str!("../../queries/locals.scm");

#[cfg(with_tags_query)]
/// The symbol tagging query for this grammar.
pub const TAGS_QUERY: &str = include_str!("../../queries/tags.scm");

#[cfg(test)]
mod tests {
    #[test]
    fn struct_fields_are_comma_separated_with_an_optional_trailing_comma() {
        for source in [
            "struct Empty {}",
            "struct Cell<T> { value: T }",
            "struct Cell<T> { value: T, }",
            "struct Point<T> { x: T, y: T }",
            "struct Point<T> { x: T, y: T, }",
            "struct Nested<T> { pair: (T, T), next: PtrMut<Nested<T>> }",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "struct Point<T> { x: T; y: T; }",
            "struct Point<T> { x: T y: T }",
            "struct Point<T> { x: T,, y: T }",
            "struct Point<T> { x = T, y = T }",
            "struct Point { fn get() -> i32 { 0 } }",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn records_require_named_structs() {
        for source in [
            "type Point = { x: i32, y: i32 };",
            "fn value()  { let mut point = { x = 1, y = 2 }; }",
            "fn value(point: { x: i32 })  {}",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "struct Point { x: i32, y: i32, } fn value() -> Point  { Point { x = 1, y = 2 } }",
            "fn pair() -> (i32, bool)  { (1, true) }",
            "fn empty() -> ()  {}",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
    }

    fn parse(source: &str) -> tree_sitter::Tree {
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&super::LANGUAGE.into()).unwrap();
        parser.parse(source, None).unwrap()
    }

    #[test]
    fn numeric_literals_reject_suffixes() {
        for literal in [
            "1u", "1_u", "1uf", "1_UD", "1_u_i", "1lu", "1_uuL", "0xffuf", "1B", "1uB", "1_UB",
            "1Uh", "1_UI", "1_uL", "1F", "1_D", "0xff_UB", "42_i32", "42_u64", "1.0_f32", "1f64",
        ] {
            let source = format!("fn main()  {{ let mut n = {literal}; }}");
            assert!(parse(&source).root_node().has_error(), "{literal}");
        }
        for literal in ["1", "1_000", "1.25", "1e3", "1.0E-2", "0x7f_b", "0xff_AB"] {
            let source = format!("fn main()  {{ let mut n = {literal}; }}");
            assert!(!parse(&source).root_node().has_error(), "{literal}");
        }
    }

    #[test]
    fn parses_unit_and_tuple_function_types() {
        for source in [
            "type F = () -> i32; fn f () -> i32  { 1 } fn main() -> ()  { let mut x = f(); }",
            "type F = (i32, i32) -> i32; fn f (a: i32, b: i32) -> i32  { a + b } fn main() -> ()  { let mut x = f(1, 2); }",
            "type F = ((i32, i32), ()) -> ();",
            "type Unit = (); fn main() -> ()  { let mut x = Unit(()); }",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        let tree = parse("fn main() -> ()  { let mut x = (()); }");
        let root = tree.root_node();
        let function = root.child_by_field_name("stmt").unwrap();
        let sexp = function.child_by_field_name("body").unwrap().to_sexp();
        assert!(sexp.contains("unit_term"));
        assert!(!sexp.contains("unit_type"));
    }

    #[test]
    fn records_accept_only_value_members() {
        assert!(
            parse("fn main() -> ()  { let mut x = Item { T = i32 }; }")
                .root_node()
                .has_error()
        );
        assert!(
            parse("fn main() -> ()  { let mut x = Item { a = 1, T = i32 }; }")
                .root_node()
                .has_error()
        );
        assert!(
            !parse("struct FieldsAB<T0, T1> { a: T0, b: T1, }\nfn main() -> ()  { let mut x = FieldsAB<_, _> { a = 1, b = 2 }; }")
                .root_node()
                .has_error()
        );
        assert!(
            !parse("fn main() -> ()  { let mut x = { type T = i32; T(1) }; }")
                .root_node()
                .has_error()
        );
    }

    #[test]
    fn parses_strings_and_single_element_tuples() {
        for source in [
            r#"fn main() -> ()  { print("x = {0}\n", (n,)); }"#,
            r#"fn main() -> ()  { let mut x = "a\"b\\c\t\r\0"; }"#,
            "fn main() -> ()  { let mut x = \"héllo\"; }",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            r#"fn main() -> () = { let mut x = "\q"; };"#,
            "fn main() -> () = { let mut x = \"unterminated; };",
            "fn main() -> () = { let mut x = \"line\nbreak\"; };",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn parses_module_items_and_address_of() {
        for source in [
            "export { Gpu, create }; extern { \"runtime.h\": { fn create (gpu: PtrMut<PtrMut<Gpu>>) -> i32; } }; import { \"runtime.resin\" }; extern type Gpu;",
            "fn empty () -> ()  {} fn identity (n: i32) -> i32  { n }",
            "fn main () -> ()  { let mut n = 0; let mut p = &n; p.* = 1; }",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "fn main() -> ()  { let mut f = (n: i32) => n; }",
            "fn outer () -> i32 = { fn inner() -> i32  { 1 } inner() };",
            "fn f (n)  { n }",
            "fn f (n: i32) -> i32 { n };",
            "fn f (n: i32): i32  { n }",
            "def f(n: i32) -> i32 = { n };",
            "fn f () -> ()  { include \"runtime.resin\"; }",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn module_clauses_are_optional_unique_and_ordered() {
        for source in [
            "export {};",
            "import {};",
            "export { f, Item, }; import { \"a.resin\", \"b.resin\", };",
            "// heading\nexport {}; /* between */ import {};",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "export {}; export {};",
            "import {}; import {};",
            "import {}; export {};",
            "fn x() -> i32  { 1 } export { x };",
            "fn x() -> i32  { 1 } import {};",
            "export { \"f\" };",
            "import { foo };",
            "export { f f };",
            "import { \"a\" \"b\" };",
            "export {}",
            "import {}",
            "fn f () -> ()  { export {}; }",
            "fn f () -> ()  { import {}; }",
            "include \"legacy.resin\";",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn source_files_only_contain_declarations() {
        for statement in [
            "let x = 1;",
            "let mut x: i32;",
            "x = 2;",
            "f();",
            "();",
            "while (ready) {};",
        ] {
            assert!(parse(statement).root_node().has_error(), "{statement}");
            let body = format!("fn main() -> ()  {{ {statement} }}");
            assert!(!parse(&body).root_node().has_error(), "{body}");
        }
        assert!(
            !parse("export { main, demo }; type Item = i32; fn main() -> ()  {} fn demo() -> i32  { 42 }")
                .root_node()
                .has_error()
        );
    }

    #[test]
    fn keywords_are_reserved_but_intrinsics_remain_identifiers() {
        for keyword in [
            "export", "import", "extern", "type", "fn", "let", "mut", "if", "else", "while",
            "bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64",
        ] {
            for source in [
                format!("fn main() -> ()  {{ let mut {keyword} = 1; }}"),
                format!("fn {keyword} () -> ()  {{}}"),
                format!("fn f ({keyword}: i32) -> ()  {{}}"),
                format!("struct R {{ {keyword}: i32 }}"),
                format!("fn main() -> ()  {{ let mut r = R {{ {keyword} = 1 }}; }}"),
                format!("fn main() -> ()  {{ r.{keyword}; }}"),
                format!("export {{ {keyword} }};"),
            ] {
                assert!(parse(&source).root_node().has_error(), "{source}");
            }
            let source = format!(
                "fn main() -> ()  {{ let mut {keyword}_value = 1; let mut _{keyword} = 2; }}"
            );
            assert!(!parse(&source).root_node().has_error(), "{source}");
        }
        for name in [
            "Ptr",
            "PtrMut",
            "Ref",
            "RefMut",
            "Err",
            "GpuArguments",
            "StrongOwner",
            "WeakOwner",
            "GpuView",
            "GpuPipelineContract",
        ] {
            for source in [
                format!("type {name} = int;"),
                format!("extern type {name};"),
                format!("export {{ {name} }};"),
            ] {
                assert!(parse(&source).root_node().has_error(), "{source}");
            }
            let source =
                format!("type {name}Value = i32; fn main() -> ()  {{ type _{name} = i32; }}");
            assert!(!parse(&source).root_node().has_error(), "{source}");
        }
        for source in [
            "fn main() -> ()  { let mut print = 1; let mut shader = 2; }",
            "fn print (n: i32) -> i32  { n } fn shader () -> ()  {}",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn type_formers_use_angle_brackets_without_conflicting_with_operators() {
        for source in [
            "type P = PtrMut<i32>; type Pp = PtrMut<PtrMut<i32>>; type S = SpanMut<PtrMut<i32>>;",
            "struct SpanMut<T> { data: PtrMut<T>, length: u64, } struct ArcPtr<T> { owner: StrongOwner, } type GpuSpanMut<T> = SpanMut<T>;",
            "type P = GpuSpanMut<i32, i32>;",
            "struct FieldsX<T0> { x: T0, }\ntype G = GpuPtrMut<i32>; type S = GpuSpanMut<FieldsX<u32>>; type Nested = PtrMut<GpuSpanMut<GpuPtrMut<i32>>>;",
            "fn launch(args: GpuArguments) -> GpuArguments  { args }",
            "fn first(p: GpuSpanMut<i32>) -> GpuPtrMut<i32>  { p:at(u64(0)) } fn make() -> GpuPtrMut<_>  { gpu:create::<i32>(i32(3)) }",
            "struct FieldsX<T0> { x: T0, }\ntype P = PtrMut<()>; type S = SpanMut<(i32, i32)>; type R = PtrMut<FieldsX<i32>>; type F = PtrMut<(i32) -> i32>;",
            "fn main() -> ()  { let mut p = PtrMut<i32>(u64(0)); let mut x = u64(p) > u64(0); let mut y = 8 >> 1; let mut z = 1 < 2; }",
            "fn main() -> ()  { let mut p = Ptr < Ptr < i32 > >(u64(0)); }",
            "fn fibonacci(n: i32) -> i32  { n } fn main() -> ()  { let mut x = fibonacci(2); let mut y = fibonacci(3); }",
            "fn main() -> ()  { let mut x = Name { value = 1 }; let mut y = Converter([1, 2]); }",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "type P = Ptr(i32);",
            "type P = Span (i32);",
            "type P = Ptr i32;",
            "type P = PtrMut<1>;",
            "type P = PtrMut<>;",
            "type P = PtrMut<i32, i32>;",
            "type P = PtrMut<PtrMut<i32>;",
            "type P = GpuPtrMut<>;",
            "type P = GpuPtrMut(i32);",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn definition_keywords_belong_to_statements_not_fields_or_parameters() {
        for source in [
            "struct FieldsXY<T0, T1> { x: T0, y: T1, }\ntype Point = FieldsXY<i32, i32>; fn make(x: i32) -> Point  { let mut y: i32; y = x + 1; Point { x = x, y = y } }",
            "struct FieldsAB<T0, T1> { a: T0, b: T1, }\nstruct FieldsC<T0> { c: T0, }\nfn main() -> ()  { let mut record = FieldsAB<_, _> { a = { let mut x = 1; x }, b = FieldsC<_> { c = 2 } }; }",
            "fn main() -> ()  { type T = i32; let mut x = T(1); let mut callback = main; }",
            "extern { \"stdlib.h\": { fn abs(n: i32) -> i32; } };",
            "fn/* comment */main() -> ()  { let mut/* comment */n = 1; }",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "main() -> () = {};",
            "Point = { x: i32 };",
            "var Point = { x: i32 };",
            "fn Point = { x: i32 };",
            "type point = i32;",
            "fn main() -> ()  { x: i32; }",
            "fn main() -> ()  { let mut T = i32; }",
            "fn main() -> ()  { fn T = i32; }",
            "fn main() -> ()  { type x = 1; }",
            "fn main() -> ()  { let mut x := 1; }",
            "fn main() -> ()  { fn x = 1; }",
            "var main() -> () = {};",
            "fn f(let mut x: i32) -> i32  { x }",
            "type R = { let mut x: i32 };",
            "fn main() -> ()  { let mut r = Item { let mut x = 1 }; }",
            "fn main() -> ()  { let mut r = { x = 1, let mut y = 2 }; }",
            "extern { \"stdlib.h\": { abs(n: i32) -> i32; } };",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn function_results_can_be_omitted_but_not_incomplete() {
        for source in [
            "fn empty()  {} fn explicit() -> ()  { () }",
            "fn greet(n: i32)  { print(\"{0}\", (n,)); }",
            "fn f(n: i32)  { n }",
            "extern { \"stdlib.h\": { fn free(p: PtrMut<u8>); } };",
            "fn apply(f: () -> ())  { f() }",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "fn empty() ->  {}",
            "extern { \"stdlib.h\": { fn free(p: PtrMut<u8>) ->; } };",
            "fn empty() {};",
            "def empty() = {};",
            "type Callback = (i32) ->;",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn foreign_groups_belong_to_the_optional_source_preamble() {
        for source in [
            "",
            "extern {};",
            "extern { \"empty.h\": {} };",
            "extern { \"first.h\": { fn first(); fn second(); }, \"empty.h\": {}, };",
            "export { call }; extern { \"native.h\": { fn call(value: PtrMut<Handle>); } }; import { \"types.resin\" }; extern type Handle;",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "extern \"native.h\" fn call();",
            "import {}; extern {};",
            "extern {}; export {};",
            "extern {}; extern {};",
            "fn main()  {} extern {};",
            "extern { \"native.h\" { fn call(); } };",
            "extern { \"native.h\": { fn call() = {}; } };",
            "extern { \"native.h\": { fn call(); } \"next.h\": {} };",
            "extern { \"native.h\": {} }",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn test_can_load_grammar() {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&super::LANGUAGE.into())
            .expect("Error loading Resin parser");
    }
}
