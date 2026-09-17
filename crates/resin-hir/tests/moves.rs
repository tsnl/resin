mod common;
use common::hir_module;

fn accepts(source: &str) {
    hir_module(source).unwrap();
}
fn rejects(source: &str, message: &str) {
    let error = hir_module(source).unwrap_err().to_string();
    assert!(error.contains(message), "{error}");
}

#[test]
fn values_move_and_primitives_copy() {
    accepts("fn f(value: i32) -> i32 { value + value }");
    rejects(
        "struct Item {} fn f(value: Item) { let other = value; value; }",
        "moved",
    );
    accepts("struct Item {} fn f(mut value: Item) { let other = value; value = Item {}; value; }");
}

#[test]
fn widening_copyable_values_does_not_move_them() {
    accepts(
        "struct Item {} fn f(value: i32) -> i32 | Item { let widened: i32 | Item = value; value }",
    );
    accepts("struct Item {} fn f(value: Ref<i32>) -> i32 | Item { value }");
    rejects(
        "struct Item {} fn f(value: Ref<Item>) -> i32 | Item { value }",
        "reference",
    );
}

#[test]
fn assignment_requires_mut_and_returns_unit() {
    rejects("fn f() { let value = 1; value = 2; }", "immutable");
    rejects("fn f(value: i32) { value = 2; }", "immutable");
    accepts("fn f() { let value: i32; value = 2; }");
    rejects(
        "fn f(flag: bool) { let value: i32; if (flag) { value = 2; }; value = 3; }",
        "immutable",
    );
    accepts("fn f(mut value: i32) { value = 2 }");
    accepts(
        "fn f(flag: bool) -> i32 { let value: i32; if (flag) { value = 1; } else { value = 2; }; value }",
    );
    rejects(
        "fn f(flag: bool) { let value: i32; while (flag) { value = 1; }; }",
        "immutable",
    );
    rejects(
        "struct Item {} fn f() { let value = Item {}; let moved = value; value = Item {}; }",
        "immutable",
    );
    rejects(
        "struct Item { value: i32 } fn f(item: Item) { item.value = 2; }",
        "immutable",
    );
}

#[test]
fn match_binders_are_immutable_unless_marked_mut() {
    rejects(
        "fn f(value: i32 | None) { match (value) { i32(number) => { number = 2; }, None => {} }; }",
        "immutable",
    );
    accepts(
        "fn f(value: i32 | None) { match (value) { i32(mut number) => { number = 2; }, None => {} }; }",
    );
}

#[test]
fn field_moves_preserve_siblings_and_block_whole_reads() {
    accepts(
        "struct Item {} struct Pair { a: Item, b: Item, } fn f(value: Pair) { let a = value.a; let b = value.b; }",
    );
    rejects(
        "struct Item {} struct Pair { a: Item, b: Item, } fn f(value: Pair) { let a = value.a; value; }",
        "moved",
    );
    accepts(
        "struct Item {} struct Pair { a: Item, b: Item, } fn f(mut value: Pair) { let a = value.a; value.a = Item {}; value; }",
    );
}

#[test]
fn branches_and_loop_backedges_check_moves() {
    rejects(
        "struct Item {} fn f(value: Item, flag: bool) { if (flag) { let other = value; }; value; }",
        "moved",
    );
    rejects(
        "struct Item {} fn f(value: Item, flag: bool) { while (flag) { let other = value; }; }",
        "moved",
    );
    accepts(
        "struct Item {} fn f(mut value: Item, flag: bool) { while (flag) { let other = value; value = Item {}; }; value; }",
    );
}

#[test]
fn references_do_not_consume_their_referents() {
    accepts(
        "struct Item { value: i32, } fn inspect(value: Ref<Item>) -> i32 { value.value } fn f(value: Item) -> i32 { inspect(value) + inspect(value) }",
    );
    rejects(
        "struct Item {} fn f(value: Ref<Item>) -> Item { value }",
        "reference",
    );
}

#[test]
fn early_exits_join_ownership_only_on_reachable_paths() {
    accepts(
        "struct Item {} fn f(value: Item, flag: bool) -> Item { if (flag) { return value; }; value }",
    );
    accepts(
        "struct Item {} fn f(value: Item, flag: bool) -> Item { while (flag) { return value; }; value }",
    );
    rejects(
        "struct Item {} fn f(value: Item, flag: bool) { while (flag) { let other = value; break; }; value; }",
        "moved",
    );
    rejects(
        "struct Item {} fn f(value: Item, flag: bool) { while (flag) { let other = value; continue; }; }",
        "moved",
    );
    accepts(
        "struct Item {} fn f(mut value: Item, flag: bool) { while (flag) { let other = value; value = Item {}; continue; }; value; }",
    );
}

#[test]
fn error_union_payloads_preserve_move_rules() {
    accepts(
        "struct Item {} fn f(value: Item, flag: bool) -> Item | Err<Item> { if (flag) { return Err(value); }; value }",
    );
    rejects(
        "struct Item {} fn f(value: Item) { let error = Err(value); value; }",
        "moved",
    );
    rejects(
        "struct Item {} fn f(value: Err<Item>) { let other = value; value; }",
        "moved",
    );
    accepts("fn f(value: Err<i32>) { let other = value; value; }");
}
