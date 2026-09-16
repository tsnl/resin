mod common;
use common::hir_module;

fn accepts(source: &str) {
    hir_module(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
}

fn rejects(source: &str, message: &str) {
    let error = hir_module(source).unwrap_err().to_string().to_lowercase();
    assert!(
        error.contains(message),
        "{source}\nexpected {message:?}, got {error}"
    );
}

#[test]
fn module_constants_resolve_dependencies_and_default_before_use() {
    accepts(
        "const result = first + second; const second = 2; const first = 40; fn value() -> long  { result }",
    );
    rejects("const value = 42; fn use() -> int  { value }", "type");
    accepts("const value: int = 42; fn use() -> int  { value }");
}

#[test]
fn groups_use_explicit_initializers_and_reset_iota() {
    accepts(
        "const ( a, b: uint = 1 << iota, 8 << iota; _, _ = iota, iota; c, d: uint = 1 << iota, 8 << iota; ); const reset = iota; fn use() -> uint  { a + b + c + d + uint(reset) }",
    );
    accepts("const (); const _ = 12;");
    accepts("const ( first: uint = iota; second = iota; ); fn use() -> long  { second }");
    rejects("const ( a, b = 1; );", "equal counts");
    rejects("const ( a = 1; b, c = 2; );", "equal counts");
}

#[test]
fn local_constants_obey_lexical_scope_and_specification_visibility() {
    accepts(
        "const outer = 7; fn use() -> long  { const outer, other = outer + 1, outer; { const outer = 9; outer; }; outer + other }",
    );
    rejects(
        "fn use() -> long  { { const local = 1; }; local }",
        "unbound",
    );
    rejects("fn use()  { const a = b; const b = 1; }", "not a constant");
    rejects(
        "fn use()  { let mut value = 1; const result = value; }",
        "not a constant",
    );
}

#[test]
fn constants_have_no_storage_including_unused_functions() {
    for declaration in [
        "const value = 1;",
        "const value = 1 == 1;",
        "const value = \"text\";",
    ] {
        rejects(
            &format!("{declaration} fn unused()  {{ &value; }}"),
            "constant",
        );
        rejects(
            &format!("{declaration} fn unused()  {{ value = value; }}"),
            "constant",
        );
    }
    rejects(
        "const value = 1; fn unused()  { let mut reference: Ref<long> = value; }",
        "reference binding requires",
    );
    accepts("const value = 1; fn use() -> long  { let mut copy = value; copy = 2; copy }");
    rejects(
        "const text = \"hello\"; fn unused()  { text.length = 1; }",
        "constant",
    );
    rejects(
        "const text = \"hello\"; fn unused()  { &text.length; }",
        "constant",
    );
}

#[test]
fn invalid_and_cyclic_constants_fail_even_when_unused() {
    for source in ["const a = a;", "const a = b; const b = a;"] {
        rejects(source, "cyclic constant");
    }
    rejects(
        "fn runtime() -> long  { 1 } const a = runtime();",
        "const initializer",
    );
    rejects("const a = [1, 2];", "scalar constant");
    rejects("const a = 1; const a = 2;", "conflicting binding");
    rejects("const _ = 1 / 0;", "division by zero");
    rejects(
        "const unused = (1 == 0) && (1 / 0 == 0);",
        "division by zero",
    );
    rejects("fn unused()  { const a = 255_ub + 1_ub; }", "overflows");
    rejects("const value = 1 << 100;", "shift count");
    rejects("const value = 1_ul << 64_ul;", "shift count");
    rejects("const value = 9223372036854775807 + 1;", "overflows");
    rejects("const value = 1.0 / 0.0;", "division by zero");
    rejects("const value = 1e308 * 10.0;", "non-finite");
}

#[test]
fn iota_is_restricted_to_constant_initializers() {
    rejects(
        "fn unused()  { iota; }",
        "only available in a const initializer",
    );
    rejects("const iota = 1;", "reserved");
    rejects("fn f(iota: long)  {}", "reserved");
    rejects("fn f()  { const n = iota; iota; }", "only available");
}

#[test]
fn sizeof_accepts_only_types_and_preserves_generic_queries() {
    accepts(
        "struct Pair<T> { first: T; second: T; } const bytes = sizeof(Pair<int>); fn size<T>() -> ulong  { sizeof(T) } fn main() -> ulong  { bytes + size::<int>() }",
    );
    accepts("struct Node { next: Ptr<Node>; value: int; } const bytes = sizeof(Node);");
    accepts(
        "struct FieldsXY<T0, T1> { x: T0; y: T1; }\ntype Number = int; const bytes = sizeof(Number); fn value() -> ulong  { sizeof(FieldsXY<int, long>) }",
    );
    for source in [
        "fn unused()  { sizeof(1); }",
        "fn unused(value: int)  { sizeof(value); }",
        "fn unused()  { sizeof(int, long); }",
    ] {
        let parsed = common::run(resin_ast::build_ast(
            common::syntax(source),
            &resin_executor::Execution::default(),
            &resin_executor::Cancellation::new(),
        ))
        .unwrap();
        assert!(
            !parsed.errors.is_empty(),
            "sizeof accepted a value operand: {source}"
        );
    }
    rejects("fn unused()  { sizeof(Ref<int>); }", "ref<t>");
    rejects(
        "extern type Opaque; fn unused()  { sizeof(Opaque); }",
        "no known value layout",
    );
    rejects("fn unused<T>()  { const bytes = sizeof(T); }", "type");
}

#[test]
fn constant_checking_does_not_default_surrounding_inference() {
    accepts("const text = str(\"hello\"); const flag = bool(1 == 1); const number = long(1);");
    accepts("fn use() -> int  { let mut value = 1; const other = 2; value + int(other) }");
    accepts("fn identity<T>(value: T) -> T  { const size = sizeof(int); value }");
}

#[test]
fn former_workgroup_builtin_uses_ordinary_constant_lookup() {
    accepts(
        "const compute_workgroup_size = 7; const value = compute_workgroup_size; fn use() -> long  { value }",
    );
    rejects(
        "fn unused()  { let mut compute_workgroup_size = 1; const value = compute_workgroup_size; }",
        "not a constant",
    );
}
