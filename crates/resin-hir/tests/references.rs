mod common;
use common::hir_module;

#[test]
fn reference_parameters_results_and_initialized_annotations_complete() {
    for source in [
        "def identity<T>(value: Ref<T>) -> Ref<T> = { value };",
        "def refer<T>(value: Ptr<T>) -> Ref<T> = { value.* };",
        "def main() -> int = { var x: int = 1; var r: Ref<int> = x; r := 2; var copy = r; copy };",
        "def id(value: Ref<int>) -> Ref<int> = { value }; def main() -> int = { var x = 1_i; id(x) := 2; x };",
        "def choose(a: Ref<int>, b: Ref<int>, flag: bool) -> Ref<int> = { if (flag) { a } else { b } };",
    ] {
        hir_module(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
}

#[test]
fn reference_binding_requires_an_initialized_place() {
    for (source, diagnostic) in [
        (
            "def main() = { var reference: Ref<int>; };",
            "require an initializer",
        ),
        (
            "def main() = { var reference: Ref<int> = 1_i; };",
            "reference binding requires",
        ),
        (
            "def main() = { var value: int; var reference: Ref<int> = value; };",
            "UninitializedValue",
        ),
        (
            "def get() -> int = { 1 }; def main() = { var reference: Ref<int> = get(); };",
            "reference binding requires",
        ),
    ] {
        let error = hir_module(source).unwrap_err();
        assert!(error.diagnostic.contains(diagnostic), "{source}\n{error}");
    }
}

#[test]
fn references_cannot_be_hidden_in_value_storage_or_generic_arguments() {
    for source in [
        "struct Invalid { value: Ref<int> };",
        "type R = Ref<int>; struct Invalid { value: R };",
        "def invalid(value: Ptr<Ref<int>>) = {};",
        "def invalid(value: Ref<Ref<int>>) = {};",
        "def invalid(value: Result<Ref<int>, None>) = {};",
        "def invalid(value: Ref<int> | None) = {};",
        "struct Cell<T> { value: T }; def invalid(value: Cell<Ref<int>>) = {};",
        "def identity<T>(value: T) -> T = { value }; def invalid() = { var x = 1_i; identity::<Ref<int>>(x); };",
        "struct Cell { def accept<T>(value: T) = {}; }; def invalid() = { var x = 1_i; Cell.accept::<Ref<int>>(x); };",
        "def invalid() = { var value: Ref<int> = Ref<int>(1_i); };",
        "extern \"test.h\" def invalid(value: Ref<int>);",
        "extern \"test.h\" def invalid() -> Ref<int>;",
    ] {
        assert!(
            hir_module(source).is_err(),
            "accepted unsupported reference storage: {source}"
        );
    }
}

#[test]
fn reference_binding_does_not_widen_or_implicitly_dereference_pointers() {
    for source in [
        "def invalid() = { var x = 1_i; var r: Ref<int | None> = x; };",
        "def invalid(p: Ptr<int>) = { var r: Ref<int> = p; };",
        "def accept(value: Ref<int>) = {}; def invalid() = { accept(1_i); };",
        "struct Cell { value: int, def get(self: Ref<Cell>) -> Ref<int> = { self.value }; }; def invalid() = { Cell { value = 1 }.get(); };",
    ] {
        assert!(
            hir_module(source).is_err(),
            "accepted invalid binding: {source}"
        );
    }
}
