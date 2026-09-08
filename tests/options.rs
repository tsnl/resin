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
fn options_infer_match_and_unwrap_nested_payloads_once() {
    let output = run(r#"
        struct Item { number: int };
        def make(calls: Ptr<int>) -> Option<Item> = { calls.* := calls.* + 1; some(Item { number = 41 }) };
        def main() -> int = {
            var calls = 0;
            var number = make(&calls)!.number + 1;
            var empty: Option<int>; empty := none();
            var absent = match (empty) { some(n) => { 1 == 0 }, none(n) => { 1 == 1 } };
            var nested = some(some(number));
            if (absent && calls == 1 && nested!! == 42 && !some(1 == 0)!) { 0 } else { 1 }
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
    let output = run(r#"
        def main() -> int = {
            var absent: Option<int>; absent := none();
            var value = absent!;
            print("not reached", ());
            value
        };
    "#);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot unwrap none"));
}

#[test]
fn option_patterns_and_unwrap_are_checked() {
    for source in [
        "def f(value: Option<int>) -> int = { match (value) { some(n) => { n } } };",
        "def f(value: Option<int>) -> int = { match (value) { ok(n) => { n }, none(n) => { 0 } } };",
        "def f(value: int) -> int = { value! };",
        "def f() -> Option<int> = { none(1) };",
    ] {
        assert!(ir::generate(&support::parse(source)).is_err(), "{source}");
    }
}
