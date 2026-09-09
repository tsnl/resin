#[path = "support/toolchain.rs"]
mod toolchain;
use support::pipeline;
mod support;

use resin_common::TempDir;
use std::{ffi::OsString, process::Command};

const RESOURCE: &str = r#"
struct Resource { trace: Ptr<int>, digit: int };
def optional_resource(trace: Ptr<int>, digit: int) -> Resource | None = { Resource.make(trace, digit) };
def optional_shared(trace: Ptr<int>, digit: int) -> Arc<Resource> | None = { Arc<Resource> { trace = trace, digit = digit } };
def optional_int(value: int) -> int | None = { value };
impl Resource {
    def drop(dying: Ptr<Resource>) = {
        if (dying.digit != 0) { dying.trace.* := dying.trace.* * 10 + dying.digit; };
    };
    def read(self: Ptr<Resource>) -> int = { self.digit };
    def make(trace: Ptr<int>, digit: int) -> Resource = { Resource { trace = trace, digit = digit } };
}
"#;

fn run(source: &str) {
    let source = format!("export {{ main }}; {RESOURCE} {source}");
    let module = support::module(&source);
    let c = resin_codegen::emit_c(&module, "main").unwrap();
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let output = temp
        .path()
        .join(format!("shared{}", std::env::consts::EXE_SUFFIX));
    let cc = std::env::var_os("CC")
        .unwrap_or_else(|| OsString::from(resin_platform_toolchain::DEFAULT_C_COMPILER));
    toolchain::c(&cc)
        .compile_c(&c, &output)
        .unwrap_or_else(|e| panic!("{e}\n{c}"));
    let result = Command::new(output).output().unwrap();
    assert_eq!(
        result.status.code(),
        Some(0),
        "{source}\n{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

fn rejects(source: &str, message: &str) {
    let source = format!("{RESOURCE} {source}");
    let error = pipeline::generate(&support::parse(&source))
        .unwrap_err()
        .to_string();
    assert!(error.contains(message), "expected {message:?}: {error}");
}

#[test]
fn at_indexing_preserves_shared_array_owners_and_overwrite_cleanup() {
    run(r#"
    def main() -> int = {
        var trace = 0; var weak = Weak<Resource>();
        {
            var holder = { values = [Arc<Resource> { trace = &trace, digit = 1 }] };
            weak := holder.values.at(0).*.downgrade();
            var saved = holder.values.at(0).*;
            holder.values.at(0).* := Arc<Resource> { trace = &trace, digit = 2 };
            if (saved.read() != 1 || holder.values.at(0).*.read() != 2 || trace != 0) { trace := 9; };
        };
        var expired = match (weak.upgrade()) { Arc<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
        if (expired && trace == 12) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn assignment_branches_preserve_initialization_and_overwrite_cleanup() {
    run(r#"
    struct Tracked { drops: Ptr<int>, value: int };
    impl Tracked { def drop(self: Ptr<Tracked>) = { self.drops.* := self.drops.* + 1; }; }
    def main() -> int = {
        var drops = 0; var index = 0; var valid = 1 == 1;
        while (index < 2) {
            var value: Tracked;
            value := if (index == 0) { Tracked { drops = &drops, value = 1 } } else { Tracked { drops = &drops, value = 1 } };
            value := match (optional_int(index)) {
                int(n) => { Tracked { drops = &drops, value = 2 } },
                None => { Tracked { drops = &drops, value = 3 } }
            };
            valid := valid && value.value == 2;
            index := index + 1;
        };
        if (valid && drops == 8) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn shared_assignment_branches_release_every_owner() {
    run(r#"
    def main() -> int = {
        var trace = 0; var weak = Weak<Resource>();
        {
            var value: Arc<Resource>;
            value := if (trace == 0) { Arc<Resource> { trace = &trace, digit = 1 } } else { Arc<Resource> { trace = &trace, digit = 9 } };
            value := match (optional_int(2)) {
                int(n) => { Arc<Resource> { trace = &trace, digit = n } },
                None => { Arc<Resource> { trace = &trace, digit = 9 } }
            };
            weak := value.downgrade();
        };
        var expired = match (weak.upgrade()) { Arc<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
        if (expired && trace == 12) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn assignment_propagation_tracks_success_and_error_cleanup() {
    run(r#"
    struct Failed {};
    def acquire(trace: Ptr<int>, fail: bool) -> Result<Arc<Resource>, Failed> = {
        if (fail) { err(Failed {}) } else { ok(Arc<Resource> { trace = trace, digit = 2 }) }
    };
    def work(trace: Ptr<int>, fail: bool) -> Result<(), Failed> = {
        var first = Arc<Resource> { trace = trace, digit = 1 };
        var pending: Arc<Resource>;
        pending := acquire(trace, fail)?;
        ok(())
    };
    def main() -> int = {
        var trace = 0;
        var succeeded = match (work(&trace, 1 == 0)) { ok(n) => { 1 == 1 }, err(e) => { 1 == 0 } };
        var failed = match (work(&trace, 1 == 1)) { ok(n) => { 1 == 0 }, err(e) => { 1 == 1 } };
        if (succeeded && failed && trace == 211) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn temporary_projection_keeps_nominal_and_nested_destructors() {
    run(r#"
    struct Outer { trace: Ptr<int>, inner: Arc<Resource> };
    impl Outer { def drop(self: Ptr<Outer>) = { self.trace.* := self.trace.* * 10 + 2; }; }
    def make(trace: Ptr<int>) -> Outer = { Outer { trace = trace, inner = Arc<Resource> { trace = trace, digit = 3 } } };
    def main() -> int = {
        var trace = 0;
        var number = Resource.make(&trace, 1).digit;
        var valid = number == 1 && trace == 1;
        {
            var inner = make(&trace).inner;
            valid := valid && trace == 12 && inner.digit == 3;
        };
        if (valid && trace == 123) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn function_fields_named_drop_remain_callable() {
    run(r#"
    def increment(value: int) -> int = { value + 1 };
    struct Callback { drop: (int) -> int };
    def main() -> int = {
        var record = { drop = increment };
        var nominal = Callback { drop = increment };
        if ((record.drop)(20) + (nominal.drop)(20) == 42) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn all_applications_consume_fresh_arguments_without_an_extra_drop() {
    run(r#"
    def consume(value: Resource) -> int = { value.digit };
    def consume_pair(first: Resource, second: Resource) = {};
    def main() -> int = {
        var trace = 0;
        consume(Resource.make(&trace, 1));
        consume(Resource(Resource.make(&trace, 2)));
        { var value = Resource.make(&trace, 3); };
        { var shared = Arc<Resource>(Resource.make(&trace, 4)); };
        consume_pair(Resource.make(&trace, 5), Resource.make(&trace, 6));
        if (trace == 123465) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn named_values_are_copied_even_when_the_type_has_a_destructor() {
    run(r#"
    def consume(value: Resource) = {};
    def main() -> int = {
        var trace = 0;
        {
            var original = Resource.make(&trace, 1);
            { var copied = original; consume(original); };
            if (trace != 11 || original.read() != 1) { trace := 9; };
        };
        if (trace == 111) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn shared_copies_reassignment_weak_upgrade_and_expiration() {
    run(r#"
    struct Shared { value: Arc<Resource> };
    def main() -> int = {
        var trace = 0;
        var weak = Weak<Resource>();
        {
            var a = Arc<Resource> { trace = &trace, digit = 1 };
            var b = Shared { value = a };
            weak := b.value.downgrade();
            a := Arc<Resource> { trace = &trace, digit = 2 };
            match (weak.upgrade()) {
                Arc<Resource>(owner) => { if (owner.read() != 1) { trace := 9; }; },
                None => { trace := 9; }
            };
            if (trace != 0) { trace := 9; };
        };
        var expired = match (weak.upgrade()) { Arc<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
        if (expired && trace == 12) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn native_wrapper_can_transfer_a_handle_by_disarming_the_source() {
    run(r#"
    def move(value: Ptr<Resource>) -> Resource = {
        Resource { trace = value.trace, digit = (&value.digit).replace(0) }
    };
    def main() -> int = {
        var trace = 0;
        {
            var value = Resource.make(&trace, 1);
            { var shared = Arc<Resource>(move(&value)); };
            if (trace != 1 || value.digit != 0) { trace := 9; };
        };
        if (trace == 1) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn early_errors_destroy_only_acquired_owners() {
    run(r#"
    struct Failed {};
    def fail() -> Result<int, Failed> = { err(Failed {}) };
    def work(trace: Ptr<int>) -> Result<Resource, Failed> = {
        var first = Resource.make(trace, 1);
        var second = Resource.make(trace, 2);
        var unused = fail()?;
        ok(Resource.make(trace, 3))
    };
    def main() -> int = {
        var trace = 0;
        var failed = match (work(&trace)) { ok(owner) => { 1 == 0 }, err(error) => { 1 == 1 } };
        if (failed && trace == 21) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn arrays_options_and_methods_consume_fresh_payloads() {
    run(r#"
    struct Pair { value: Resource };
    struct Sink {};
    impl Sink {
        def consume(self: Ptr<Sink>, a: Resource, b: Resource) = {};
    }
    def main() -> int = {
        var trace = 0;
        {
            var pair = Pair { value = Resource.make(&trace, 1) };
            var array = [Resource.make(&trace, 2), Resource.make(&trace, 3)];
            var option = optional_resource(&trace, 4);
            // Reading this named union copies its payload. Both copies are destroyed.
            match (option) { Resource(value) => {}, None => {} };
            var sink = Sink {};
            sink.consume(Resource.make(&trace, 5), Resource.make(&trace, 6));
        };
        if (trace == 4654321) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn indexing_preserves_array_storage_and_only_value_reads_copy_elements() {
    run(r#"
    def main() -> int = {
        var trace = 0;
        {
            var array = [Resource.make(&trace, 1), Resource.make(&trace, 2)];
            var element = array(0);
            if (element.digit != 1 || array(1).digit != 2 || trace != 0) { trace := 9; };
            { var copied = array(1).*; };
            if (trace != 2 || array(1).digit != 2) { trace := 9; };
        };
        if (trace == 221) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn destruction_preserves_results_and_runs_per_scope_and_iteration() {
    run(r#"
    struct Set { target: Ptr<int>, value: int };
    impl Set { def drop(self: Ptr<Set>) = { self.target.* := self.value; }; }
    def result_before_cleanup() -> int = {
        var value = 42;
        var cleanup = Set { target = &value, value = 99 };
        value
    };
    def main() -> int = {
        var trace = 0;
        var index = 1;
        while (index < 4) {
            var outer = Resource.make(&trace, index);
            { var inner = Resource.make(&trace, index + 3); };
            index := index + 1;
        };
        if (1 == 1) { var branch = Resource.make(&trace, 7); };
        match (optional_int(8)) {
            int(n) => { var arm = Resource.make(&trace, n); },
            None => { var unused = Resource.make(&trace, 9); }
        };
        if (trace == 41526378 && result_before_cleanup() == 42) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn pointer_returning_index_wrappers_preserve_nested_places() {
    run(r#"
    struct Payload { value: int };
    struct Entry { nested: Payload };
    def at(items: Span<Entry>, index: ulong) -> Ptr<Entry> = { items(index) };
    def main() -> int = {
        var items = [Entry { nested = Payload { value = 1 } }, Entry { nested = Payload { value = 2 } }];
        var span = Span<Entry> { data = Ptr<Entry>(&items), length = 2_ul };
        at(span, 1_ul).nested.value := 42;
        var p = &at(span, 1_ul).*.nested.value;
        p.* := p.* + 1;
        var copied = at(span, 1_ul).*.nested;
        copied.value := 99;
        if (items(1).nested.value == 43 && copied.value == 99 && items(0).nested.value == 1) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn former_defer_keyword_can_name_an_ordinary_immediate_call() {
    run(r#"
    def defer(trace: Ptr<int>) = { trace.* := trace.* + 1; };
    def main() -> int = {
        var trace = 0;
        defer(&trace);
        if (trace == 1) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn weak_cycles_and_nested_pointer_handle_access() {
    run(r#"
    struct Node { trace: Ptr<int>, parent: Weak<Node> };
    impl Node { def drop(self: Ptr<Node>) = { self.trace.* := self.trace.* + 1; }; }
    def main() -> int = {
        var trace = 0;
        var weak = {
            var node = Arc<Node> { trace = &trace, parent = Weak<Node>() };
            node.parent := node.downgrade();
            var address = &node;
            var nested = &address;
            if (nested.trace.* != 0) { trace := 9; };
            node.downgrade()
        };
        var expired = match (weak.upgrade()) { Arc<Node>(node) => { 1 == 0 }, None => { 1 == 1 } };
        if (expired && trace == 1) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn named_argument_to_arc_copies_the_pointee_value() {
    run(r#"
    def main() -> int = {
        var trace = 0;
        {
            var value = Resource.make(&trace, 1);
            { var shared = Arc<Resource>(value); };
            if (value.digit != 1 || trace != 1) { trace := 9; };
        };
        if (trace == 11) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn methods_require_a_compatible_receiver_and_valid_destructor() {
    rejects(
        "struct Item {}; impl Item { def shared(self: Arc<Item>) = {}; } def f(item: Item) = { item.shared(); };",
        "method receiver does not match",
    );
    rejects(
        "struct Item {}; impl Item { def drop(self: Item) = {}; }",
        "drop must have signature",
    );
}

#[test]
fn option_unwrap_transfers_fresh_payloads_and_copies_named_options() {
    run(r#"
    def main() -> int = {
        var trace = 0;
        { var unwrapped = optional_resource(&trace, 1)!; };
        {
            var named = optional_resource(&trace, 2);
            { var copied = named!; };
        };
        var weak = Weak<Resource>();
        {
            var shared = optional_shared(&trace, 3)!;
            weak := shared.downgrade();
            { var upgraded = weak.upgrade()!; };
            if (trace != 122) { trace := 9; };
        };
        var expired = match (weak.upgrade()) { Arc<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
        if (trace == 1223 && expired) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn destruction_hooks_are_ordinary_calls_and_remain_automatic() {
    run(r#"
    struct Manual { trace: Ptr<int>, digit: int };
    impl Manual {
        def drop(self: Ptr<Manual>) = {
            if (self.digit != 0) { self.trace.* := self.trace.* * 10 + self.digit; };
            self.digit := 0;
        };
    }
    def main() -> int = {
        var trace = 0;
        {
            var a = Manual { trace = &trace, digit = 1 };
            var b = Manual { trace = &trace, digit = 2 };
            var c = Manual { trace = &trace, digit = 3 };
            a.drop();
            Manual.drop(&b);
        };
        if (trace == 123) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn generated_methods_use_ordinary_calls_and_borrow_fresh_receivers() {
    run(r#"
    def main() -> int = {
        var trace = 0;
        {
            var pointer = Arc<Resource> { trace = &trace, digit = 1 }.get();
            if (trace != 0 || pointer.digit != 1) { trace := 9; };
            var second = Arc<Resource> { trace = &trace, digit = 2 };
            var other = Arc<Resource>.get(&second);
            var weak = Arc<Resource>.downgrade(second);
            { var upgraded = Weak<Resource>.upgrade(weak)!; };
            if (trace != 0 || other.digit != 2) { trace := 9; };
            var number = Resource.make(&trace, 3).read();
            if (trace != 0 || number != 3) { trace := 9; };
        };
        if (trace == 321) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn weak_payloads_require_upgrade_before_dereference_or_field_access() {
    for source in [
        "def f(value: Weak<Resource>) = { value.*; };",
        "def f(value: Weak<Resource>) = { value.digit; };",
        "def f(value: Ptr<Weak<Resource>>) = { value.digit; };",
        "def f(value: Weak<Resource>) = { value.read(); };",
    ] {
        assert!(
            pipeline::generate(&support::parse(&format!("{RESOURCE} {source}"))).is_err(),
            "{source}"
        );
    }
    run(r#"
        def read(value: Weak<Resource>) -> int = { value.upgrade()!.digit };
        def main() -> int = {
            var trace = 0;
            var value = Arc<Resource> { trace = &trace, digit = 1 };
            if (read(value.downgrade()) == 1) { 0 } else { 1 }
        };
    "#);
}

#[test]
fn pointer_replace_transfers_managed_values_and_leaves_ordinary_names_available() {
    run(r#"
    def replace(value: int) -> int = { value + 1 };
    def main() -> int = {
        var trace = 0;
        {
            var value = Resource.make(&trace, 1);
            { var old = (&value).replace(Resource.make(&trace, 2)); };
            if (trace != 1 || value.digit != 2) { trace := 9; };
            { var old = Ptr<Resource>.replace(&value, Resource.make(&trace, 3)); };
            if (trace != 12 || value.digit != 3) { trace := 9; };
        };
        if (trace == 123 && replace(1) == 2) { 0 } else { 1 }
    };
    "#);
}
