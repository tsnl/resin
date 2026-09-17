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
fn block_bindings_and_read_only_captures_are_ordinary_typed_places() {
    accepts(
        r#"
        fn read(value: Ref<i64>) -> i64 { value }
        fn main() -> i64 {
            let mut scale = 2;
            let ys = parallel_map([1, 2, 3]) |mut x| {
                x = x * read(scale);
                let mut local = x;
                local = local + 1;
                local
            };
            scale = 4;
            parallel_reduce(ys, 0) |a, b| { a + b }
        }
    "#,
    );
    accepts("fn f(value: RefMut<i64>) { parallel_map([1]) |x| { x + value }; }");
    accepts("fn f(pointer: Ptr<i64>) { parallel_map([1]) |x| { pointer.* = x; x }; }");
}

#[test]
fn captures_cannot_be_written_or_borrowed_as_mutable() {
    for body in [
        "value = x; x",
        "let alias: RefMut<i64> = value; x",
        "write(value); x",
    ] {
        rejects(
            &format!(
                "fn write(value: RefMut<i64>) {{ value = 2; }} fn f() {{ let mut value = 0; parallel_map([1]) |x| {{ {body} }}; }}"
            ),
            "captures are read-only",
        );
    }
    rejects(
        "struct Cell { value: i64 } fn f() { let mut cell = Cell { value = 0 }; parallel_map([1]) |x| { cell.value = x; x }; }",
        "captures are read-only",
    );
    rejects(
        "fn f(value: RefMut<i64>) { parallel_map([1]) |x| { value = x; x }; }",
        "captures are read-only",
    );
    rejects(
        "fn f() { let mut xs = [0]; parallel_map([1]) |x| { xs:at_mut(0) = x; x }; }",
        "captures are read-only",
    );
    rejects(
        "fn f() { let value: i64; parallel_map([1]) |x| { value = x; x }; }",
        "captures are read-only",
    );
}

#[test]
fn nested_regions_and_shadowing_have_distinct_bindings() {
    accepts(
        "fn f() { let value = 10; parallel_map([1, 2]) |value| { parallel_reduce([value, value], 0) |a, b| { a + b } }; }",
    );
    rejects(
        "fn f() { parallel_map([1]) |mut x| { parallel_map([2]) |y| { x = y; y } }; }",
        "captures are read-only",
    );
    rejects("fn f() { parallel_map([1]) |x| { x }; x; }", "unbound");
    rejects(
        "fn f() { parallel_reduce([1], 0) |x, x| { x }; }",
        "duplicate",
    );
}

#[test]
fn nonlocal_exits_are_rejected_but_iteration_local_loops_work() {
    for body in ["return 1;", "let value = Err(1); value?;"] {
        rejects(
            &format!("fn f() -> i64 | Err<i64> {{ parallel_map([1]) |x| {{ {body} x }}; 0 }}"),
            "parallel block",
        );
    }
    for exit in ["break", "continue"] {
        rejects(
            &format!("fn f() {{ while (true) {{ parallel_map([1]) |x| {{ {exit}; x }}; }}; }}"),
            "require a loop body",
        );
    }
    accepts("fn f() { parallel_map([1]) |x| { while (true) { break; }; x }; }");
}

#[test]
fn arrays_and_reduction_types_are_checked_even_in_unused_functions() {
    rejects("fn f() { parallel_map(1) |x| { x }; }", "inline array");
    rejects(
        "fn f() { parallel_reduce([1], 0) |a, b| { true }; }",
        "type",
    );
    accepts("fn f() -> i64 { parallel_reduce([], 0) |a, b| { a + b } }");
    rejects(
        "fn f() { let x: i64; parallel_map([1]) |a| { x + a }; }",
        "uninitialized",
    );
}

#[test]
fn parallel_captures_do_not_move_owners_out_of_the_enclosing_scope() {
    let owner = "struct Owner {} fn drop(value: RefMut<Owner>) {}";
    rejects(
        &format!(
            "{owner} fn f() {{ let owner = Owner {{}}; parallel_map([1]) |x| {{ let moved = owner; x }}; }}"
        ),
        "cannot move",
    );
    rejects(
        &format!("{owner} fn f() {{ parallel_map([Owner {{}}]) |x| {{ x }}; }}"),
        "copyable",
    );
    accepts(&format!(
        "{owner} fn inspect(owner: Ref<Owner>) -> i64 {{ 1 }} fn f() {{ let owner = Owner {{}}; parallel_map([1]) |x| {{ inspect(owner) + x }}; }}"
    ));
}
