#[path = "support/toolchain.rs"]
mod config;
mod support;

use resin::{
    backend::c,
    ir,
    toolchain::{self, TempDir},
};
use std::{ffi::OsString, process::Command};

const RESOURCE: &str = r#"
struct Resource { trace: Ptr<int>, digit: int };
type OptionalResource = Resource | None;
type OptionalShared = Arc<Resource> | None;
type OptionalInt = int | None;
impl Resource {
    def drop(self: Ptr<Resource>) = {
        if (self.digit != 0) { self.trace.* := self.trace.* * 10 + self.digit; };
    };
    def read(self: Ptr<Resource>) -> int = { self.digit };
    def make(trace: Ptr<int>, digit: int) -> Resource = { Resource { trace = trace, digit = digit } };
}
"#;

fn run(source: &str) {
    let source = format!("export {{ main }}; {RESOURCE} {source}");
    let module = support::module(&source);
    let c = c::emit(&module, "main").unwrap();
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let output = temp
        .path()
        .join(format!("shared{}", std::env::consts::EXE_SUFFIX));
    let cc =
        std::env::var_os("CC").unwrap_or_else(|| OsString::from(toolchain::DEFAULT_C_COMPILER));
    toolchain::compile_c(&c, &output, &config::c(&cc)).unwrap_or_else(|e| panic!("{e}\n{c}"));
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
    let error = ir::generate(&support::parse(&source))
        .unwrap_err()
        .to_string();
    assert!(error.contains(message), "expected {message:?}: {error}");
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
        Resource { trace = value.trace, digit = replace(&value.digit, 0) }
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
            var option: Resource | None = Resource.make(&trace, 4);
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
        match (OptionalInt(8)) {
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
        var span = Span<Entry> { data = Ptr<Entry>(&items), length = 2L };
        at(span, 1L).nested.value := 42;
        var p = &at(span, 1L).*.nested.value;
        p.* := p.* + 1;
        var copied = at(span, 1L).*.nested;
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
    rejects(
        "def f(a: Resource) = { a.drop(); };",
        "compiler-invoked destruction hook",
    );
}

#[test]
fn option_unwrap_transfers_fresh_payloads_and_copies_named_options() {
    run(r#"
    def main() -> int = {
        var trace = 0;
        { var unwrapped = OptionalResource(Resource.make(&trace, 1))!; };
        {
            var named: Resource | None = Resource.make(&trace, 2);
            { var copied = named!; };
        };
        var weak = Weak<Resource>();
        {
            var shared = OptionalShared(Arc<Resource> { trace = &trace, digit = 3 })!;
            weak := shared.downgrade();
            { var upgraded = weak.upgrade()!; };
            if (trace != 122) { trace := 9; };
        };
        var expired = match (weak.upgrade()) { Arc<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
        if (trace == 1223 && expired) { 0 } else { 1 }
    };
    "#);
}
