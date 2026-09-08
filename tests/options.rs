#[path = "support/toolchain.rs"]
mod config;
mod support;

use resin::{
    backend::c,
    ir,
    toolchain::{self, TempDir},
};
use std::{
    ffi::OsString,
    process::{Command, Output},
};

fn run(source: &str) -> Output {
    let module = support::module(&format!("export {{ main }}; {source}"));
    let source = c::emit(&module, "main").unwrap();
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let executable = temp
        .path()
        .join(format!("option{}", std::env::consts::EXE_SUFFIX));
    let cc =
        std::env::var_os("CC").unwrap_or_else(|| OsString::from(toolchain::DEFAULT_C_COMPILER));
    toolchain::compile_c(&source, &executable, &config::c(&cc)).unwrap();
    Command::new(executable).output().unwrap()
}

#[test]
fn optional_values_match_and_unwrap_once() {
    let output = run(r#"
        struct Item { number: int };
        def make(calls: Ptr<int>) -> Item | None = { calls.* := calls.* + 1; Item { number = 41 } };
        def main() -> int = {
            var calls = 0;
            var number = make(&calls)!.number + 1;
            var empty: int | None; empty := None;
            var absent = match (empty) { int(n) => { 1 == 0 }, None => { 1 == 1 } };
            var condition: bool | None; condition := 1 == 0;
            if (absent && calls == 1 && number == 42 && !condition!) { 0 } else { 1 }
        };
    "#);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn unwrapped_values_widen_in_return_argument_and_field_contexts() {
    let output = run(r#"
        struct A { number: int }; struct B {};
        struct Record { item: A | B };
        def widen(o: A | None) -> A | B = { o! };
        def read(value: A | B) -> int = { match (value) { A(a) => { a.number }, B(b) => { 0 } } };
        def main() -> int = {
            var o: A | None; o := A { number = 14 };
            var record = Record { item = o! };
            if (read(widen(o)) + read(o!) + read(record.item) == 42) { 0 } else { 1 }
        };
    "#);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn none_elimination_preserves_all_other_union_members() {
    let output = run(r#"
        type OptionalInt = int | None;
        type Choice = None | bool | OptionalInt | int;
        def choose(flag: bool) -> Choice = { if (flag) { 42 } else { 1 == 0 } };
        def read(value: int | bool) -> int = {
            match (value) { int(n) => { n }, bool(b) => { if (b) { 1 } else { 0 } } }
        };
        def main() -> int = { if (read(choose(1 == 1)!) + read(choose(1 == 0)!) == 42) { 0 } else { 1 } };
    "#);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let module = support::module(
        "type OptionalInt = int | None; def f(x: OptionalInt | None | int) -> None | int = { x };",
    );
    assert_eq!(
        module.functions[0].result,
        ir::Ty::union_of([ir::Ty::None, ir::Ty::Int32])
    );
}

#[test]
fn result_cases_remain_distinct_inside_optional_unions() {
    let output = run(r#"
        struct Item { number: int };
        type Outcome = Result<Item, Item>;
        def make(fail: bool) -> Outcome = { if (fail) { err(Item { number = 2 }) } else { ok(Item { number = 40 }) } };
        def optional(fail: bool) -> Outcome | None = { make(fail) };
        def read(o: Outcome | None) -> int = {
            match (o!) { ok(item) => { item.number }, err(item) => { item.number } }
        };
        def main() -> int = { if (read(optional(1 == 1)) + read(optional(1 == 0)) == 42) { 0 } else { 1 } };
    "#);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn structural_members_keep_their_identity_across_union_widening() {
    let output = run(r#"
        type Callback = (int) -> int;
        type Record = { number: int };
        def increment(n: int) -> int = { n + 1 };
        def select(p: Ptr<int>, use_pointer: bool) -> Ptr<int> | Callback = { if (use_pointer) { p } else { increment } };
        def widen(value: Ptr<int> | Callback) -> None | Record | Callback | Ptr<int> = { value };
        def read(value: None | Record | Callback | Ptr<int>) -> int = {
            match (value) {
                None => { 0 }, Record(r) => { r.number }, Callback(f) => { f(19) }, Ptr<int>(p) => { p.* }
            }
        };
        def main() -> int = {
            var n = 22;
            if (read(widen(select(&n, 1 == 1))) + read(widen(select(&n, 1 == 0))) == 42) { 0 } else { 1 }
        };
    "#);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn all_value_producers_widen_at_the_consumer() {
    let output = run(r#"
        type Record = { number: uint };
        def record() -> Record | None = { { number = 42_ui } };
        def field(r: Record) -> uint | None = { r.number };
        def load(p: Ptr<uint>) -> uint | None = { p.* };
        def literal() -> uint | None = { 42 };
        def operator(n: uint) -> uint | None = { n + 1_ui };
        def compare(n: uint) -> bool | None = { n == 42_ui };
        def main() -> int = {
            var n = record()!.number;
            if (field(record()!)! == literal()! && load(&n)! == operator(41_ui)! && compare(n)!) { 0 } else { 1 }
        };
    "#);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn unwrapping_none_traps_before_following_side_effects() {
    for value in ["absent", "None"] {
        let output = run(&format!(
            r#"
            def main() -> int = {{
                var absent: int | None; absent := None;
                var value: int; value := {value}!;
                print("not reached");
                value
            }};
        "#
        ));
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("cannot unwrap None"));
    }
}

#[test]
fn optional_patterns_and_unwrap_are_checked() {
    for source in [
        "def f(value: int | None) -> int = { match (value) { int(n) => { n } } };",
        "def f(value: int | None) -> int = { match (value) { ok(n) => { n }, None => { 0 } } };",
        "def f(value: int) -> int = { value! };",
        "struct E {}; def f(r: Result<int, E>) -> int = { r! };",
        "struct A {}; struct B {}; def f(o: Ptr<A> | None) -> Ptr<A | B> = { o! };",
        "def f(o: Span<int> | None) -> Span<int | None> = { o! };",
        "def f(x: int | None) = { match (x) { int(n) => {}, None => {}, None => {} } };",
        "type Both = int | bool; def f(x: Both) = { match (x) { Both(v) => {} } };",
    ] {
        assert!(ir::generate(&support::parse(source)).is_err(), "{source}");
    }
}
