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
struct Item { number: i32, }
        fn make(calls: Ptr<i32>) -> Item | None  { calls.* = calls.* + 1; Item { number = 41 } }
        fn main() -> i32 | Err<_> {
            let calls_owner = arc_ptr_alloc(0)?; let calls: Ref<_> = calls_owner:get().*;
            let mut number = make(calls_owner:get())!.number + 1;
            let mut empty: i32 | None; empty = None;
            let mut absent = match (empty) { i32(n) => { 1 == 0 }, None => { 1 == 1 } };
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
    let output = run(r#"struct A { number: i32, } struct B {}
        struct Record { item: A | B, }
        fn optional() -> A | None { A { number = 14 } }
        fn widen(o: A | None) -> A | B  { o! }
        fn read(value: A | B) -> i32  { match (value) { A(a) => { a.number }, B(b) => { 0 } } }
        fn main() -> i32  {
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
    let output = run(r#"type OptionalInt = i32 | None;
        type Choice = None | bool | OptionalInt | i32;
        fn choose(flag: bool) -> Choice  { if (flag) { 42 } else { 1 == 0 } }
        fn read(value: i32 | bool) -> i32  {
            match (value) { i32(n) => { n }, bool(b) => { if (b) { 1 } else { 0 } } }
        }
        fn main() -> i32  { if (read(choose(1 == 1)!) + read(choose(1 == 0)!) == 42) { 0 } else { 1 } }
    "#);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let module = support::module(
        "type OptionalInt = i32 | None; fn f(x: OptionalInt | None | i32) -> None | i32  { x }",
    );
    assert_eq!(
        module.functions[0].result,
        Ty::union_of([Ty::None, Ty::Int32])
    );
}

#[test]
fn result_cases_remain_distinct_inside_optional_unions() {
    let output = run(r#"struct Item { number: i32, }
        type Outcome = (Item | Err<Item>);
        fn make(fail: bool) -> Outcome  { if (fail) { Err(Item { number = 2 }) } else { (Item { number = 40 }) } }
        fn optional(fail: bool) -> Outcome | None  { make(fail) }
        fn read(o: Outcome | None) -> i32  {
            match (o!) { Item(item) => { item.number }, Err(item) => { item.number } }
        }
        fn main() -> i32  { if (read(optional(1 == 1)) + read(optional(1 == 0)) == 42) { 0 } else { 1 } }
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
type Callback = (i32) -> i32;
        type Record = FieldsNumber<i32>;
        fn increment(n: i32) -> i32  { n + 1 }
        fn select(p: Ptr<i32>, use_pointer: bool) -> Ptr<i32> | Callback  { if (use_pointer) { p } else { increment } }
        fn widen(value: Ptr<i32> | Callback) -> None | Record | Callback | Ptr<i32>  { value }
        fn read(value: None | Record | Callback | Ptr<i32>) -> i32  {
            match (value) {
                None => { 0 }, Record(r) => { r.number }, Callback(f) => { f(19) }, Ptr<i32>(p) => { p.* }
            }
        }
        fn main() -> i32 | Err<_> {
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
type Record = FieldsNumber<u32>;
        fn record() -> Record | None  { FieldsNumber<_> { number = u32(42) } }
        fn field(r: Record) -> u32 | None  { r.number }
        fn load(p: Ptr<u32>) -> u32 | None  { p.* }
        fn literal() -> u32 | None  { 42 }
        fn operator(n: u32) -> u32 | None  { n + u32(1) }
        fn compare(n: u32) -> bool | None  { n == u32(42) }
        fn main() -> i32 | Err<_> {
            let n_owner = arc_ptr_alloc(record()!.number)?; let n: Ref<_> = n_owner:get().*;
            if (field(record()!)! == literal()! && load(n_owner:get())! == operator(u32(41))! && compare(n)!) { 0 } else { 1 }
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
            r#"import {{ "$/string.resin", "$/stdio.resin" }};
            fn main() -> i32  {{
                let mut absent: i32 | None; absent = None;
                let mut value: i32; value = {value}!;
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
        "fn f(value: i32 | None) -> i32  { match (value) { i32(n) => { n } } }",
        "fn f(value: i32 | None) -> i32  { match (value) { bool(n) => { n }, None => { 0 } } }",
        "fn f(value: i32) -> i32  { value! }",
        "struct E {} fn f(r: (i32 | Err<E>)) -> i32  { r! }",
        "struct A {} struct B {} fn f(o: Ptr<A> | None) -> Ptr<A | B>  { o! }",
        "struct Span<T> { data: Ptr<T>, length: u64, } fn f(o: Span<i32> | None) -> Span<i32 | None>  { o! }",
        "fn f(x: i32 | None)  { match (x) { i32(n) => {}, None => {}, None => {} } }",
        "type Both = i32 | bool; fn f(x: Both)  { match (x) { Both(v) => {} } }",
    ] {
        assert!(
            pipeline::generate(&support::parse(source)).is_err(),
            "{source}"
        );
    }
}
