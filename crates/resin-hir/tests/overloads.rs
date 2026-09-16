mod common;
use common::hir_module;

#[test]
fn overloads_use_all_arguments_and_expected_results() {
    hir_module(
        r#"fn combine(left: int, right: bool) -> int { left }
        fn combine(left: int, right: int) -> int { left + right }
        fn make() -> int { 1 }
        fn make() -> bool { 1 == 1 }
        fn main() -> int {
            let choice: bool = make();
            combine(combine(1_i, 2_i), choice)
        }
    "#,
    )
    .unwrap();
}

#[test]
fn generic_overloads_substitute_signatures() {
    hir_module(
        r#"fn select<T>(value: T, flag: bool) -> T { value }
        fn select<T>(value: T, count: int) -> T { value }
        fn main() -> long { select::<long>(42, 1 == 1) }
    "#,
    )
    .unwrap();
}

#[test]
fn selected_body_errors_do_not_disappear() {
    assert!(
        hir_module(
            r#"fn select(value: int) -> int { unknown }
        fn select(value: bool) -> bool { value }
        fn main() -> int { select(1_i) }
    "#
        )
        .is_err()
    );
}

#[test]
fn ambiguous_overloads_are_rejected() {
    assert!(
        hir_module(
            r#"fn select<T>(value: T) -> T { value }
        fn select(value: int) -> int { value }
        fn main() -> int { select(1_i) }
    "#
        )
        .is_err()
    );
}

#[test]
fn receiver_calls_are_free_function_calls() {
    hir_module(
        r#"
        struct Item { value: int, }
        fn read(value: Ref<Item>) -> int { value.value }
        fn read(value: int) -> int { value }
        fn main() -> int { let value = Item { value= 21 }; value:read() + read(value) }
    "#,
    )
    .unwrap();
}

#[test]
fn local_function_values_shadow_primitive_operations() {
    hir_module("fn identity(value: int) -> int { value } fn main() -> int { let at = identity; let replace = identity; at(20) + 22:replace() }").unwrap();
}

#[test]
fn numeric_literals_receive_context_from_source_overloads_of_primitive_names() {
    hir_module("fn replace(value: int) -> int { value + 1 } fn use() -> bool { replace(1) == 2 }")
        .unwrap();
}
