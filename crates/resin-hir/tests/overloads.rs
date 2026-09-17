mod common;
use common::hir_module;

#[test]
fn overloads_use_all_arguments_and_expected_results() {
    hir_module(
        r#"fn combine(left: i32, right: bool) -> i32 { left }
        fn combine(left: i32, right: i32) -> i32 { left + right }
        fn make() -> i32 { 1 }
        fn make() -> bool { 1 == 1 }
        fn main() -> i32 {
            let choice: bool = make();
            combine(combine(i32(1), i32(2)), choice)
        }
    "#,
    )
    .unwrap();
}

#[test]
fn generic_overloads_substitute_signatures() {
    hir_module(
        r#"fn select<T>(value: T, flag: bool) -> T { value }
        fn select<T>(value: T, count: i32) -> T { value }
        fn main() -> i64 { select::<i64>(42, 1 == 1) }
    "#,
    )
    .unwrap();
}

#[test]
fn selected_body_errors_do_not_disappear() {
    assert!(
        hir_module(
            r#"fn select(value: i32) -> i32 { unknown }
        fn select(value: bool) -> bool { value }
        fn main() -> i32 { select(i32(1)) }
    "#
        )
        .is_err()
    );
}

#[test]
fn receiver_calls_are_free_function_calls() {
    hir_module(
        r#"
        struct Item { value: i32, }
        fn read(value: Ref<Item>) -> i32 { value.value }
        fn read(value: i32) -> i32 { value }
        fn main() -> i32 { let value = Item { value= 21 }; value:read() + read(value) }
    "#,
    )
    .unwrap();
}

#[test]
fn local_function_values_shadow_primitive_operations() {
    hir_module("fn identity(value: i32) -> i32 { value } fn main() -> i32 { let at = identity; let replace = identity; at(20) + 22:replace() }").unwrap();
}

#[test]
fn numeric_literals_receive_context_from_source_overloads_of_primitive_names() {
    hir_module("fn replace(value: i32) -> i32 { value + 1 } fn use() -> bool { replace(1) == 2 }")
        .unwrap();
}

#[test]
fn rejected_candidates_do_not_constrain_later_trials() {
    let candidates = [
        "fn choose(value: i64, flag: bool) -> i64 { value }",
        "fn choose(value: i32, count: i32) -> i32 { value + count }",
    ];
    for order in [[0, 1], [1, 0]] {
        hir_module(&format!(
            "{} {} fn main() -> i32 {{ choose(20, i32(22)) }}",
            candidates[order[0]], candidates[order[1]],
        ))
        .unwrap();
    }
}

#[test]
fn unresolved_overloads_report_ambiguity_at_the_solver_fixed_point() {
    let error = hir_module(
        "fn choose<T>(value: T) -> T { value }
         fn choose(value: i32) -> i32 { value }
         fn main() -> i32 { choose(i32(1)) }",
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("ambiguous overload of `choose`"),
        "{error}"
    );
}
