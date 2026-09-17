use resin_types::prelude::*;
use support::pipeline;
mod support;

use std::process::Output;

fn run(source: &str) -> Output {
    let module = support::module(&format!("export {{ main }}; {source}"));
    support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run()
}

#[test]
fn optional_values_match_and_unwrap_once() {
    let output = run(r#"
import { "$/shared.resin" };
struct Item { number: int, }
        fn make(calls: Ptr<int>) -> Item | None  { calls.* = calls.* + 1; Item { number = 41 } }
        fn main() -> int | Err<_> {
            let calls_owner = arc_ptr_alloc(0)?; let calls: Ref<_> = calls_owner:get().*;
            let mut number = make(calls_owner:get())!.number + 1;
            let mut empty: int | None; empty = None;
            let mut absent = match (empty) { int(n) => { 1 == 0 }, None => { 1 == 1 } };
            let mut condition: bool | None; condition = 1 == 0;
            if (absent && calls == 1 && number == 42 && !condition!) { 0 } else { 1 }
        }
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
    let output = run(r#"struct A { number: int, } struct B {}
        struct Record { item: A | B, }
        fn optional() -> A | None { A { number = 14 } }
        fn widen(o: A | None) -> A | B  { o! }
        fn read(value: A | B) -> int  { match (value) { A(a) => { a.number }, B(b) => { 0 } } }
        fn main() -> int  {
            let mut o: A | None; o = A { number = 14 };
            let mut record = Record { item = o! };
            if (read(widen(optional())) + read(optional()!) + read(record.item) == 42) { 0 } else { 1 }
        }
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
    let output = run(r#"type OptionalInt = int | None;
        type Choice = None | bool | OptionalInt | int;
        fn choose(flag: bool) -> Choice  { if (flag) { 42 } else { 1 == 0 } }
        fn read(value: int | bool) -> int  {
            match (value) { int(n) => { n }, bool(b) => { if (b) { 1 } else { 0 } } }
        }
        fn main() -> int  { if (read(choose(1 == 1)!) + read(choose(1 == 0)!) == 42) { 0 } else { 1 } }
    "#);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let module = support::module(
        "type OptionalInt = int | None; fn f(x: OptionalInt | None | int) -> None | int  { x }",
    );
    assert_eq!(
        module.functions[0].result,
        Ty::union_of([Ty::None, Ty::Int32])
    );
}

#[test]
fn result_cases_remain_distinct_inside_optional_unions() {
    let output = run(r#"struct Item { number: int, }
        type Outcome = (Item | Err<Item>);
        fn make(fail: bool) -> Outcome  { if (fail) { Err(Item { number = 2 }) } else { (Item { number = 40 }) } }
        fn optional(fail: bool) -> Outcome | None  { make(fail) }
        fn read(o: Outcome | None) -> int  {
            match (o!) { Item(item) => { item.number }, Err(item) => { item.number } }
        }
        fn main() -> int  { if (read(optional(1 == 1)) + read(optional(1 == 0)) == 42) { 0 } else { 1 } }
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
import { "$/shared.resin" };
struct FieldsNumber<T0> { number: T0, }
type Callback = (int) -> int;
        type Record = FieldsNumber<int>;
        fn increment(n: int) -> int  { n + 1 }
        fn select(p: Ptr<int>, use_pointer: bool) -> Ptr<int> | Callback  { if (use_pointer) { p } else { increment } }
        fn widen(value: Ptr<int> | Callback) -> None | Record | Callback | Ptr<int>  { value }
        fn read(value: None | Record | Callback | Ptr<int>) -> int  {
            match (value) {
                None => { 0 }, Record(r) => { r.number }, Callback(f) => { f(19) }, Ptr<int>(p) => { p.* }
            }
        }
        fn main() -> int | Err<_> {
            let n_owner = arc_ptr_alloc(22)?; let n: Ref<_> = n_owner:get().*;
            if (read(widen(select(n_owner:get(), 1 == 1))) + read(widen(select(n_owner:get(), 1 == 0))) == 42) { 0 } else { 1 }
        }
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
import { "$/shared.resin" };
struct FieldsNumber<T0> { number: T0, }
type Record = FieldsNumber<uint>;
        fn record() -> Record | None  { FieldsNumber<_> { number = 42_ui } }
        fn field(r: Record) -> uint | None  { r.number }
        fn load(p: Ptr<uint>) -> uint | None  { p.* }
        fn literal() -> uint | None  { 42 }
        fn operator(n: uint) -> uint | None  { n + 1_ui }
        fn compare(n: uint) -> bool | None  { n == 42_ui }
        fn main() -> int | Err<_> {
            let n_owner = arc_ptr_alloc(record()!.number)?; let n: Ref<_> = n_owner:get().*;
            if (field(record()!)! == literal()! && load(n_owner:get())! == operator(41_ui)! && compare(n)!) { 0 } else { 1 }
        }
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
            r#"import {{ "$/string.resin" }};
            fn main() -> int  {{
                let mut absent: int | None; absent = None;
                let mut value: int; value = {value}!;
                print("not reached");
                value
            }}
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
        "fn f(value: int | None) -> int  { match (value) { int(n) => { n } } }",
        "fn f(value: int | None) -> int  { match (value) { bool(n) => { n }, None => { 0 } } }",
        "fn f(value: int) -> int  { value! }",
        "struct E {} fn f(r: (int | Err<E>)) -> int  { r! }",
        "struct A {} struct B {} fn f(o: Ptr<A> | None) -> Ptr<A | B>  { o! }",
        "struct Span<T> { data: Ptr<T>, length: ulong, } fn f(o: Span<int> | None) -> Span<int | None>  { o! }",
        "fn f(x: int | None)  { match (x) { int(n) => {}, None => {}, None => {} } }",
        "type Both = int | bool; fn f(x: Both)  { match (x) { Both(v) => {} } }",
    ] {
        assert!(
            pipeline::generate(&support::parse(source)).is_err(),
            "{source}"
        );
    }
}
