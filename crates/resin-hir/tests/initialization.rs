//! Initialization is a source guarantee, including definitions with no LIR request.
use resin_hir::{GenerateErrorKind, Module};

fn lower(source: &str) -> Result<Module, resin_hir::GenerateError> {
    let syntax = resin_cst::Document::reparse(source.into(), None);
    let file = resin_ast::generate(&syntax).unwrap();
    resin_hir::generate(&file)
}
fn valid(source: &str) {
    lower(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
}
fn uninitialized(source: &str) {
    let error = lower(source).unwrap_err();
    assert!(
        matches!(error.kind, GenerateErrorKind::UninitializedValue { .. }),
        "{source}\n{error}"
    );
}

#[test]
fn unused_definitions_still_require_initialized_reads() {
    uninitialized(
        "export { main }; def unused() -> int = { var value: int; value }; def main() = {};",
    );
    for expression in ["value + 1", "{ var pointer = &value; 1 }"] {
        let source = format!("def unused() -> int = {{ var value = {expression}; value }};");
        assert!(matches!(
            lower(&source).unwrap_err().kind,
            GenerateErrorKind::EagerRecursion { .. }
        ));
    }
}

#[test]
fn address_acquisition_and_whole_value_assignment_do_not_read_storage() {
    valid("def main() -> int = { var value: int; var pointer = &value; value := 42; pointer.* };");
    uninitialized("struct R { value: int }; def main() = { var record: R; record.value := 42; };");
    uninitialized(
        "struct R { value: int }; def main() = { var record: R; var address = &record.value; };",
    );
}

#[test]
fn branching_requires_initialization_on_every_path() {
    valid(
        "def choose(flag: bool) -> int = { var value: int; if (flag) { value := 1; } else { value := 2; }; value };",
    );
    uninitialized(
        "def choose(flag: bool) -> int = { var value: int; if (flag) { value := 1; }; value };",
    );
    uninitialized(
        "def choose(flag: bool) -> int = { var value: int; if (flag) { value := 1 } else { value } };",
    );
}

#[test]
fn loop_conditions_execute_before_the_optional_body() {
    valid("def main() -> int = { var value: int; while ((value := 1) == 0) {}; value };");
    uninitialized("def main() -> int = { var value: int; while (1 == 0) { value := 1; }; value };");
    uninitialized("def main() = { var value: int; while (value == 0) { value := 1; }; };");
}

#[test]
fn short_circuiting_obeys_evaluation_order_and_conditional_effects() {
    valid("def main() -> bool = { var value: int; ((value := 1) == 1) && (value == 1) };");
    uninitialized("def main() -> bool = { var value: int; (value == 1) && ((value := 1) == 1) };");
    for operator in ["&&", "||"] {
        uninitialized(&format!(
            "def main() -> int = {{ var value: int; (1 == 1) {operator} ((value := 1) == 1); value }};"
        ));
    }
}

#[test]
fn match_scrutinees_precede_arms_and_pattern_bindings_start_initialized() {
    valid(
        "def choose(option: int | None) -> int = { var value: int; match ({ value := 1; option }) { int(number) => { number + value }, None => { value } } };",
    );
    valid(
        "def choose(option: int | None) -> int = { var value: int; match (option) { int(number) => { value := number; }, None => { value := 0; } }; value };",
    );
    uninitialized(
        "def choose(option: int | None) -> int = { var value: int; match (option) { int(number) => { value := number; }, None => {} }; value };",
    );
    uninitialized(
        "def choose(option: int | None) -> int = { var value: int; match (option) { int(number) => { value := number }, None => { value } } };",
    );
}

#[test]
fn layout_queries_neither_read_nor_initialize_their_operands() {
    valid("def size() -> ulong = { var value: int; size_of(value) };");
    uninitialized("def main() -> int = { var value: int; size_of(value := 1); value };");
}
