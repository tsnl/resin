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
            "type F = () -> int; def f () -> int = { 1 }; def main() -> () = { var x = f(); };",
            "type F = (int, int) -> int; def f (a: int, b: int) -> int = { a + b }; def main() -> () = { var x = f(1, 2); };",
            "type F = ((int, int), ()) -> ();",
            "type Unit = (); def main() -> () = { var x = Unit (()); };",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        let tree = parse("def main() -> () = { var x = (()); };");
        let root = tree.root_node();
        let function = root.child_by_field_name("stmt").unwrap();
        let sexp = function.child_by_field_name("body").unwrap().to_sexp();
        assert!(sexp.contains("unit_term"));
        assert!(!sexp.contains("unit_type"));
    }

    #[test]
    fn records_accept_only_value_members() {
        assert!(
            parse("def main() -> () = { var x = { T = int }; };")
                .root_node()
                .has_error()
        );
        assert!(
            parse("def main() -> () = { var x = { a = 1, T = int }; };")
                .root_node()
                .has_error()
        );
        assert!(
            !parse("def main() -> () = { var x = { a = 1, b = 2 }; };")
                .root_node()
                .has_error()
        );
        assert!(
            !parse("def main() -> () = { var x = { type T = int; T (1) }; };")
                .root_node()
                .has_error()
        );
    }

    #[test]
    fn parses_strings_and_single_element_tuples() {
        for source in [
            r#"def main() -> () = { print("x = {0}\n", (n,)); };"#,
            r#"def main() -> () = { var x = "a\"b\\c\t\r\0"; };"#,
            "def main() -> () = { var x = \"héllo\"; };",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            r#"def main() -> () = { var x = "\q"; };"#,
            "def main() -> () = { var x = \"unterminated; };",
            "def main() -> () = { var x = \"line\nbreak\"; };",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn parses_module_items_and_address_of() {
        for source in [
            "export { Gpu, create }; import { \"runtime.resin\" }; extern type Gpu; extern \"runtime.h\" def create (gpu: Ptr<Ptr<Gpu>>) -> int;",
            "def empty () -> () = {}; def identity (n: int) -> int = { n };",
            "def main () -> () = { var n = 0; var p = &n; p.* := 1; };",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "def main() -> () = { var f = (n: int) => n; };",
            "def outer () -> int = { def inner () -> int = { 1 }; inner() };",
            "def f (n) = { n };",
            "def f (n: int) -> int { n };",
            "def f (n: int): int = { n };",
            "def f (n: int) -> int = { n }",
            "def f(n: int) -> int { n }",
            "def f () -> () = { include \"runtime.resin\"; };",
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
            "def x() -> int = { 1 }; export { x };",
            "def x() -> int = { 1 }; import {};",
            "export { \"f\" };",
            "import { foo };",
            "export { f f };",
            "import { \"a\" \"b\" };",
            "export {}",
            "import {}",
            "def f () -> () = { export {}; };",
            "def f () -> () = { import {}; };",
            "include \"legacy.resin\";",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn source_files_only_contain_declarations() {
        for statement in [
            "var x = 1;",
            "var x: int;",
            "x := 2;",
            "f();",
            "();",
            "while (ready) {};",
        ] {
            assert!(parse(statement).root_node().has_error(), "{statement}");
            let body = format!("def main() -> () = {{ {statement} }};");
            assert!(!parse(&body).root_node().has_error(), "{body}");
        }
        assert!(
            !parse("export { main, demo }; type Item = int; def main() -> () = {}; def demo() -> int = { 42 };")
                .root_node()
                .has_error()
        );
    }

    #[test]
    fn keywords_are_reserved_but_intrinsics_remain_identifiers() {
        for keyword in [
            "export", "import", "extern", "type", "def", "var", "if", "else", "while", "bool",
            "sbyte", "short", "int", "long", "ubyte", "ushort", "uint", "ulong", "float32",
            "float64",
        ] {
            for source in [
                format!("def main() -> () = {{ var {keyword} = 1; }};"),
                format!("def {keyword} () -> () = {{}};"),
                format!("def f ({keyword}: int) -> () = {{}};"),
                format!("type R = {{ {keyword}: int }};"),
                format!("def main() -> () = {{ var r = {{ {keyword} = 1 }}; }};"),
                format!("def main() -> () = {{ r.{keyword}; }};"),
                format!("export {{ {keyword} }};"),
            ] {
                assert!(parse(&source).root_node().has_error(), "{source}");
            }
            let source =
                format!("def main() -> () = {{ var {keyword}_value = 1; var _{keyword} = 2; }};");
            assert!(!parse(&source).root_node().has_error(), "{source}");
        }
        for name in ["Ptr", "Span"] {
            for source in [
                format!("type {name} = int;"),
                format!("extern type {name};"),
                format!("export {{ {name} }};"),
            ] {
                assert!(parse(&source).root_node().has_error(), "{source}");
            }
            let source =
                format!("type {name}Value = int; def main() -> () = {{ type _{name} = int; }};");
            assert!(!parse(&source).root_node().has_error(), "{source}");
        }
        for source in [
            "def main() -> () = { var print = 1; var shader = 2; };",
            "def print (n: int) -> int = { n }; def shader () -> () = {};",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn type_formers_use_angle_brackets_without_conflicting_with_operators() {
        for source in [
            "type P = Ptr<int>; type Pp = Ptr<Ptr<int>>; type S = Span<Ptr<int>>;",
            "type P = Ptr<()>; type S = Span<(int, int)>; type R = Ptr<{ x: int }>; type F = Ptr<(int) -> int>;",
            "def main() -> () = { var p = Ptr<int>(ulong(0)); var x = ulong(p) > ulong(0); var y = 8 >> 1; var z = 1 < 2; };",
            "def main() -> () = { var p = Ptr < Ptr < int > > (ulong (0)); };",
            "def fibonacci(n: int) -> int = { n }; def main() -> () = { var x = fibonacci(2); var y = fibonacci (3); };",
            "def main() -> () = { var x = Name { value = 1 }; var y = Converter [1, 2]; };",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "type P = Ptr(int);",
            "type P = Span (int);",
            "type P = Ptr int;",
            "type P = Ptr<1>;",
            "type P = Ptr<>;",
            "type P = Ptr<int, int>;",
            "type P = Ptr<Ptr<int>;",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn definition_keywords_belong_to_statements_not_fields_or_parameters() {
        for source in [
            "type Point = { x: int, y: int }; def make(x: int) -> Point = { var y: int; y := x + 1; Point { x = x, y = y } };",
            "def main() -> () = { var record = { a = { var x = 1; x }, b = { c = 2 } }; };",
            "def main() -> () = { type T = int; var x = T(1); var callback = main; };",
            "extern \"stdlib.h\" def abs(n: int) -> int;",
            "def/* comment */main() -> () = { var/* comment */n = 1; };",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "main() -> () = {};",
            "Point = { x: int };",
            "var Point = { x: int };",
            "def Point = { x: int };",
            "type point = int;",
            "def main() -> () = { x = 1; };",
            "def main() -> () = { x: int; };",
            "def main() -> () = { T = int; };",
            "def main() -> () = { var T = int; };",
            "def main() -> () = { def T = int; };",
            "def main() -> () = { type x = 1; };",
            "def main() -> () = { var x := 1; };",
            "def main() -> () = { def x = 1; };",
            "var main() -> () = {};",
            "def f(var x: int) -> int = { x };",
            "type R = { var x: int };",
            "def main() -> () = { var r = { var x = 1 }; };",
            "def main() -> () = { var r = { x = 1, var y = 2 }; };",
            "extern \"stdlib.h\" abs(n: int) -> int;",
        ] {
            assert!(parse(source).root_node().has_error(), "{source}");
        }
    }

    #[test]
    fn function_results_can_be_omitted_but_not_incomplete() {
        for source in [
            "def empty() = {}; def explicit() -> () = { () };",
            "def greet(n: int) = { print(\"{0}\", (n,)); };",
            "def f(n: int) = { n };",
            "extern \"stdlib.h\" def free(p: Ptr<ubyte>);",
            "def apply(f: () -> ()) = { f() };",
        ] {
            assert!(!parse(source).root_node().has_error(), "{source}");
        }
        for source in [
            "def empty() -> = {};",
            "extern \"stdlib.h\" def free(p: Ptr<ubyte>) ->;",
            "def empty() {};",
            "def empty() = {}",
            "type Callback = (int) ->;",
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
