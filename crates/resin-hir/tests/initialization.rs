//! Initialization is a source guarantee, including definitions with no LIR request.
use resin_hir::Module;

mod common;
use common::hir_module;

fn lower(source: &str) -> Result<Module, resin_source::SourceError> {
    hir_module(source)
}
fn valid(source: &str) {
    lower(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
}
fn uninitialized(source: &str) {
    let error = lower(source).unwrap_err();
    assert!(
        error.diagnostic.contains("UninitializedValue")
            || error.diagnostic.contains("uninitialized"),
        "{source}\n{error}"
    );
}

#[test]
fn unused_definitions_still_require_initialized_reads() {
    uninitialized(
        "export { main }; fn unused() -> int  { let mut value: int; value } fn main()  {}",
    );
    for expression in ["value + 1", "{ let mut pointer = &value; 1 }"] {
        let source = format!("fn unused() -> int  {{ let mut value = {expression}; value }}");
        assert!(
            lower(&source)
                .unwrap_err()
                .diagnostic
                .contains("EagerRecursion"),
            "{source}"
        );
    }
}

#[test]
fn address_acquisition_and_whole_value_assignment_do_not_read_storage() {
    valid(
        "fn main() -> int  { let mut value: int; let mut pointer = &value; value = 42; pointer.* }",
    );
    uninitialized("struct R { value: int; } fn main()  { let mut record: R; record.value = 42; }");
    uninitialized(
        "struct R { value: int; } fn main()  { let mut record: R; let mut address = &record.value; }",
    );
}

#[test]
fn branching_requires_initialization_on_every_path() {
    valid(
        "fn choose(flag: bool) -> int  { let mut value: int; if (flag) { value = 1; } else { value = 2; }; value }",
    );
    uninitialized(
        "fn choose(flag: bool) -> int  { let mut value: int; if (flag) { value = 1; }; value }",
    );
    uninitialized(
        "fn choose(flag: bool) -> int  { let mut value: int; if (flag) { value = 1; value } else { value } }",
    );
}

#[test]
fn loop_conditions_execute_before_the_optional_body() {
    valid("fn main() -> int  { let mut value: int; while ({ value = 1; value } == 0) {}; value }");
    uninitialized("fn main() -> int  { let mut value: int; while (1 == 0) { value = 1; }; value }");
    uninitialized("fn main()  { let mut value: int; while (value == 0) { value = 1; }; }");
}

#[test]
fn short_circuiting_obeys_evaluation_order_and_conditional_effects() {
    valid("fn main() -> bool  { let mut value: int; ({ value = 1; value } == 1) && (value == 1) }");
    uninitialized(
        "fn main() -> bool  { let mut value: int; (value == 1) && ({ value = 1; value } == 1) }",
    );
    for operator in ["&&", "||"] {
        uninitialized(&format!(
            "fn main() -> int  {{ let mut value: int; (1 == 1) {operator}({{ value = 1; value }} == 1); value }}"
        ));
    }
}

#[test]
fn match_scrutinees_precede_arms_and_pattern_bindings_start_initialized() {
    valid(
        "fn choose(option: int | None) -> int  { let mut value: int; match ({ value = 1; option }) { int(number) => { number + value }, None => { value } } }",
    );
    valid(
        "fn choose(option: int | None) -> int  { let mut value: int; match (option) { int(number) => { value = number; }, None => { value = 0; } }; value }",
    );
    uninitialized(
        "fn choose(option: int | None) -> int  { let mut value: int; match (option) { int(number) => { value = number; }, None => {} }; value }",
    );
    uninitialized(
        "fn choose(option: int | None) -> int  { let mut value: int; match (option) { int(number) => { value = number; value }, None => { value } } }",
    );
}

#[test]
fn layout_queries_neither_read_nor_initialize_their_operands() {
    valid("fn size() -> ulong  { let mut value: int; size_of(value) }");
    valid("fn size() -> ulong  { let mut value = size_of(value); value }");
    uninitialized("fn main() -> int  { let mut value: int; size_of({ value = 1; value }); value }");
}

#[test]
fn eager_self_reference_is_rejected_before_its_type_is_known() {
    for expression in [
        "value + 1",
        "-value",
        "&value",
        "{ let mut nested = value; nested }",
    ] {
        let source = format!("fn unused()  {{ let mut value = {expression}; }}");
        let error = lower(&source).unwrap_err();
        assert!(
            error.diagnostic.contains("EagerRecursion"),
            "{source}\n{error}"
        );
    }
    valid("fn shadow() -> int  { let mut value = { let mut value = 42; value }; value }");
    valid("fn size() -> ulong  { size_of({ let mut value: int = value + 1; 0_i }) }");
}
