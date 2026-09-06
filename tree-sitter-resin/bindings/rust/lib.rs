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
    fn parse(source: &str) -> tree_sitter::Tree {
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&super::LANGUAGE.into()).unwrap();
        parser.parse(source, None).unwrap()
    }

    #[test]
    fn parses_unit_and_tuple_function_types() {
        for source in [
            "F = () -> int; f () -> int = { 1 }; main() -> () = { x = f(); };",
            "F = (int, int) -> int; f (a: int, b: int) -> int = { a + b }; main() -> () = { x = f(1, 2); };",
            "F = ((int, int), ()) -> ();",
            "Unit = (); main() -> () = { x = Unit (()); };",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        let tree = parse("main() -> () = { x = (()); };");
        let root = tree.root_node();
        let function = root.child_by_field_name("stmt").unwrap();
        let sexp = function.child_by_field_name("body").unwrap().to_sexp();
        assert!(sexp.contains("unit_term"));
        assert!(!sexp.contains("unit_type"));
    }

    #[test]
    fn records_accept_only_value_members() {
        assert!(
            parse("main() -> () = { x = { T = int }; };")
                .root_node()
                .has_error()
        );
        assert!(
            parse("main() -> () = { x = { a = 1, T = int }; };")
                .root_node()
                .has_error()
        );
        assert!(
            !parse("main() -> () = { x = { a = 1, b = 2 }; };")
                .root_node()
                .has_error()
        );
        assert!(
            !parse("main() -> () = { x = { T = int; T (1) }; };")
                .root_node()
                .has_error()
        );
    }

    #[test]
    fn parses_strings_and_single_element_tuples() {
        for source in [
            r#"main() -> () = { print("x = {0}\n", (n,)); };"#,
            r#"main() -> () = { x = "a\"b\\c\t\r\0"; };"#,
            "main() -> () = { x = \"héllo\"; };",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            r#"main() -> () = { x = "\q"; };"#,
            "main() -> () = { x = \"unterminated; };",
            "main() -> () = { x = \"line\nbreak\"; };",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn parses_module_items_and_address_of() {
        for source in [
            "export { Gpu, create }; import { \"runtime.resin\" }; extern type Gpu; extern \"runtime.h\" create (gpu: Ptr<Ptr<Gpu>>) -> int;",
            "empty () -> () = {}; identity (n: int) -> int = { n };",
            "main () -> () = { n = 0; p = &n; p.* := 1; };",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "f = (n: int) => n;",
            "outer () -> int = { inner () -> int = { 1 }; inner() };",
            "f (n: int) = { n };",
            "f (n: int) -> int { n };",
            "f (n: int): int = { n };",
            "f (n: int) -> int = { n }",
            "def f(n: int) -> int { n }",
            "f () -> () = { include \"runtime.resin\"; };",
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
            "x() -> int = { 1 }; export { x };",
            "x() -> int = { 1 }; import {};",
            "export { \"f\" };",
            "import { foo };",
            "export { f f };",
            "import { \"a\" \"b\" };",
            "export {}",
            "import {}",
            "f () -> () = { export {}; };",
            "f () -> () = { import {}; };",
            "include \"legacy.resin\";",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn source_files_only_contain_declarations() {
        for statement in [
            "x = 1;",
            "x: int;",
            "x := 2;",
            "f();",
            "();",
            "while (ready) {};",
        ] {
            assert!(parse(statement).root_node().has_error(), "{statement}");
            let body = format!("main() -> () = {{ {statement} }};");
            assert!(!parse(&body).root_node().has_error(), "{body}");
        }
        assert!(
            !parse("export { main, demo }; Item = int; main() -> () = {}; demo() -> int = { 42 };")
                .root_node()
                .has_error()
        );
    }

    #[test]
    fn keywords_are_reserved_but_intrinsics_remain_identifiers() {
        for keyword in [
            "export", "import", "extern", "type", "if", "else", "while", "bool", "sbyte", "short",
            "int", "long", "ubyte", "ushort", "uint", "ulong", "float32", "float64",
        ] {
            for source in [
                format!("main() -> () = {{ {keyword} = 1; }};"),
                format!("{keyword} () -> () = {{}};"),
                format!("f ({keyword}: int) -> () = {{}};"),
                format!("R = {{ {keyword}: int }};"),
                format!("main() -> () = {{ r.{keyword}; }};"),
                format!("export {{ {keyword} }};"),
            ] {
                assert!(parse(&source).root_node().has_error(), "{source}");
            }
            let source = format!("main() -> () = {{ {keyword}_value = 1; _{keyword} = 2; }};");
            assert!(!parse(&source).root_node().has_error(), "{source}");
        }
        for name in ["Ptr", "Span"] {
            for source in [
                format!("{name} = int;"),
                format!("extern type {name};"),
                format!("export {{ {name} }};"),
            ] {
                assert!(parse(&source).root_node().has_error(), "{source}");
            }
            let source = format!("{name}Value = int; main() -> () = {{ _{name} = int; }};");
            assert!(!parse(&source).root_node().has_error(), "{source}");
        }
        for source in [
            "main() -> () = { print = 1; shader = 2; };",
            "print (n: int) -> int = { n }; shader () -> () = {};",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn type_formers_use_angle_brackets_without_conflicting_with_operators() {
        for source in [
            "P = Ptr<int>; Pp = Ptr<Ptr<int>>; S = Span<Ptr<int>>;",
            "P = Ptr<()>; S = Span<(int, int)>; R = Ptr<{ x: int }>; F = Ptr<(int) -> int>;",
            "main() -> () = { p = Ptr<int>(ulong(0)); x = ulong(p) > ulong(0); y = 8 >> 1; z = 1 < 2; };",
            "main() -> () = { p = Ptr < Ptr < int > > (ulong (0)); };",
            "fibonacci(n: int) -> int = { n }; main() -> () = { x = fibonacci(2); y = fibonacci (3); };",
            "main() -> () = { x = Name { value = 1 }; y = Converter [1, 2]; };",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "P = Ptr(int);",
            "P = Span (int);",
            "P = Ptr int;",
            "P = Ptr<1>;",
            "P = Ptr<>;",
            "P = Ptr<int, int>;",
            "P = Ptr<Ptr<int>;",
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
