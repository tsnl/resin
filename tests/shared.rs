use support::pipeline;
mod support;

const RESOURCE: &str = r#"
import { "$/shared.resin" };
struct Resource { trace: Ptr<int>, digit: int,
    def drop(dying: Ptr<Resource>) = {
        if (dying.digit != 0) { dying.trace.* := dying.trace.* * 10 + dying.digit; };
    };
    def read(self: Ptr<Resource>) -> int = { self.digit };
    def make(trace: Ptr<int>, digit: int) -> Resource = { Resource { trace = trace, digit = digit } };
};
def optional_resource(trace: Ptr<int>, digit: int) -> Resource | None = { Resource.make(trace, digit) };
def shared_resource(trace: Ptr<int>, digit: int) -> ArcPtr<Resource> = {
    var optional: ArcPtr<Resource> | None;
    optional := match (ArcPtr<Resource>.alloc(Resource { trace = trace, digit = 0 })) {
        ArcPtr<Resource>(value) => { value }, Err(error) => { None },
    };
    var owner = optional!;
    owner.get().digit := digit;
    owner
};
def optional_shared(trace: Ptr<int>, digit: int) -> ArcPtr<Resource> | None = { shared_resource(trace, digit) };
def optional_int(value: int) -> int | None = { value };

"#;

fn run(source: &str) {
    let source = format!("export {{ main }}; {RESOURCE} {source}");
    let module = support::module(&source);
    let result = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
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
    let error = pipeline::source_module(&source).unwrap_err().to_string();
    assert!(error.contains(message), "expected {message:?}: {error}");
}

#[test]
fn at_indexing_preserves_shared_array_owners_and_overwrite_cleanup() {
    run(r#"
    def main() -> int = {
        var trace = 0; var weak = WeakPtr<Resource>.empty();
        {
            var holder = { values = [shared_resource(&trace, 1)] };
            weak := holder.values.at(0).downgrade();
            var saved = holder.values.at(0);
            holder.values.at(0) := shared_resource(&trace, 2);
            if (saved.get().read() != 1 || holder.values.at(0).get().read() != 2 || trace != 0) { trace := 9; };
        };
        var expired = match (weak.upgrade()) { ArcPtr<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
        if (expired && trace == 12) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn assignment_branches_preserve_initialization_and_overwrite_cleanup() {
    run(r#"
    struct Tracked { drops: Ptr<int>, value: int,
        def drop(self: Ptr<Tracked>) = { self.drops.* := self.drops.* + 1; };
    };

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
        var trace = 0; var weak = WeakPtr<Resource>.empty();
        {
            var value: ArcPtr<Resource>;
            value := if (trace == 0) { shared_resource(&trace, 1) } else { shared_resource(&trace, 9) };
            value := match (optional_int(2)) {
                int(n) => { shared_resource(&trace, n) },
                None => { shared_resource(&trace, 9) }
            };
            weak := value.downgrade();
        };
        var expired = match (weak.upgrade()) { ArcPtr<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
        if (expired && trace == 12) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn assignment_propagation_tracks_success_and_error_cleanup() {
    run(r#"
    struct Failed {};
    def acquire(trace: Ptr<int>, fail: bool) -> (ArcPtr<Resource> | Err<Failed>) = {
        if (fail) { Err(Failed {}) } else { (shared_resource(trace, 2)) }
    };
    def work(trace: Ptr<int>, fail: bool) -> (() | Err<Failed>) = {
        var first = shared_resource(trace, 1);
        var pending: ArcPtr<Resource>;
        pending := acquire(trace, fail)?;
        (())
    };
    def main() -> int = {
        var trace = 0;
        var succeeded = match (work(&trace, 1 == 0)) { ()(n) => { 1 == 1 }, Err(e) => { 1 == 0 } };
        var failed = match (work(&trace, 1 == 1)) { ()(n) => { 1 == 0 }, Err(e) => { 1 == 1 } };
        if (succeeded && failed && trace == 211) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn temporary_projection_keeps_nominal_and_nested_destructors() {
    run(r#"
    struct Outer { trace: Ptr<int>, inner: ArcPtr<Resource>,
        def drop(self: Ptr<Outer>) = { self.trace.* := self.trace.* * 10 + 2; };
    };

    def make(trace: Ptr<int>) -> Outer = { Outer { trace = trace, inner = shared_resource(trace, 3) } };
    def main() -> int = {
        var trace = 0;
        var number = Resource.make(&trace, 1).digit;
        var valid = number == 1 && trace == 1;
        {
            var inner = make(&trace).inner;
            valid := valid && trace == 12 && inner.get().digit == 3;
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
        { var shared = shared_resource(&trace, 4); };
        consume_pair(Resource.make(&trace, 5), Resource.make(&trace, 6));
        if (trace == 123465) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn generic_record_initializers_preserve_layout_and_cleanup_on_partial_failure() {
    run(r#"
    struct Pair<T> { first: T, second: T };
    struct Failed {};
    def ready(fail: bool) -> (() | Err<Failed>) = {
        if (fail) { Err(Failed {}) } else { (()) }
    };
    def build<T>(create: (Ptr<int>, int) -> T, trace: Ptr<int>, fail: bool) -> (Pair<T> | Err<Failed>) = {
        (Pair<T> {
            second = create(trace, 2),
            first = {
                ready(fail)?;
                create(trace, 1)
            },
        })
    };
    def main() -> int = {
        var trace = 0;
        var ordered = match (build(shared_resource, &trace, 1 == 0)) {
            Pair<ArcPtr<Resource>>(pair) => { pair.first.get().digit == 1 && pair.second.get().digit == 2 },
            Err(error) => { 1 == 0 },
        };
        var destroyed = trace == 21;
        var failed = match (build(shared_resource, &trace, 1 == 1)) {
            Pair<ArcPtr<Resource>>(pair) => { 1 == 0 },
            Err(error) => { 1 == 1 },
        };
        if (ordered && destroyed && failed && trace == 212) { 0 } else { 1 }
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
    struct Shared { value: ArcPtr<Resource> };
    def main() -> int = {
        var trace = 0;
        var weak = WeakPtr<Resource>.empty();
        {
            var a = shared_resource(&trace, 1);
            var b = Shared { value = a };
            weak := b.value.downgrade();
            a := shared_resource(&trace, 2);
            match (weak.upgrade()) {
                ArcPtr<Resource>(owner) => { if (owner.get().read() != 1) { trace := 9; }; },
                None => { trace := 9; }
            };
            if (trace != 0) { trace := 9; };
        };
        var expired = match (weak.upgrade()) { ArcPtr<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
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
            {
                var target = shared_resource(&trace, 0);
                target.get().replace(move(&value));
            };
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
    def fail() -> (int | Err<Failed>) = { Err(Failed {}) };
    def work(trace: Ptr<int>) -> (Resource | Err<Failed>) = {
        var first = Resource.make(trace, 1);
        var second = Resource.make(trace, 2);
        var unused = fail()?;
        (Resource.make(trace, 3))
    };
    def main() -> int = {
        var trace = 0;
        var failed = match (work(&trace)) { Resource(owner) => { 1 == 0 }, Err(error) => { 1 == 1 } };
        if (failed && trace == 21) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn arrays_options_and_methods_consume_fresh_payloads() {
    run(r#"
    struct Pair { value: Resource };
    struct Sink {
        def consume(self: Ptr<Sink>, a: Resource, b: Resource) = {};
    };

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
            var element: Ref<Resource> = array(0);
            if (element.digit != 1 || array(1).digit != 2 || trace != 0) { trace := 9; };
            { var copied = array(1); };
            if (trace != 2 || array(1).digit != 2) { trace := 9; };
        };
        if (trace == 221) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn destruction_preserves_results_and_runs_per_scope_and_iteration() {
    run(r#"
    struct Set { target: Ptr<int>, value: int,
        def drop(self: Ptr<Set>) = { self.target.* := self.value; };
    };

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
    struct Node { trace: Ptr<int>, live: bool, parent: WeakPtr<Node>,
        def drop(self: Ptr<Node>) = { if (self.live) { self.trace.* := self.trace.* + 1; }; };
    };

    def main() -> int = {
        var trace = 0;
        var weak = {
            var optional: ArcPtr<Node> | None;
            optional := match (ArcPtr<Node>.alloc(Node { trace = &trace, live = 1 == 0, parent = WeakPtr<Node>.empty() })) {
                ArcPtr<Node>(value) => { value }, Err(error) => { None },
            };
            var node = optional!;
            node.get().live := 1 == 1;
            node.get().parent := node.downgrade();
            var address = &node;
            var nested = &address;
            if (nested.*.get().trace.* != 0) { trace := 9; };
            node.downgrade()
        };
        var expired = match (weak.upgrade()) { ArcPtr<Node>(node) => { 1 == 0 }, None => { 1 == 1 } };
        if (expired && trace == 1) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn source_allocation_copies_shared_initializers_without_cloning_payloads() {
    run(r#"
    def main() -> int = {
        var trace = 0;
        {
            var original = shared_resource(&trace, 1);
            {
                var optional: ArcPtr<ArcPtr<Resource>> | None;
                optional := match (ArcPtr<ArcPtr<Resource>>.alloc(original)) {
                    ArcPtr<ArcPtr<Resource>>(value) => { value }, Err(error) => { None },
                };
                var nested = optional!;
                if (nested.get().get().digit != 1 || trace != 0) { trace := 9; };
            };
            if (original.get().digit != 1 || trace != 0) { trace := 9; };
        };
        if (trace == 1) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn methods_require_a_compatible_receiver_and_valid_destructor() {
    rejects(
        "struct Item { def shared_resource(self: ArcPtr<Item>) = {}; };  def f(item: Item) = { item.shared_resource(); };",
        "method receiver does not match",
    );
    rejects(
        "struct Item { def drop(self: Item) = {}; }; ",
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
        var weak = WeakPtr<Resource>.empty();
        {
            var shared = optional_shared(&trace, 3)!;
            weak := shared.downgrade();
            { var upgraded = weak.upgrade()!; };
            if (trace != 122) { trace := 9; };
        };
        var expired = match (weak.upgrade()) { ArcPtr<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
        if (trace == 1223 && expired) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn destruction_hooks_are_ordinary_calls_and_remain_automatic() {
    run(r#"
    struct Manual { trace: Ptr<int>, digit: int,
        def drop(self: Ptr<Manual>) = {
            if (self.digit != 0) { self.trace.* := self.trace.* * 10 + self.digit; };
            self.digit := 0;
        };
    };

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
fn source_methods_use_ordinary_calls_and_borrow_fresh_receivers() {
    run(r#"
    def main() -> int = {
        var trace = 0;
        {
            var pointer = shared_resource(&trace, 1).get();
            if (trace != 0 || pointer.digit != 1) { trace := 9; };
            var second = shared_resource(&trace, 2);
            var other = ArcPtr<Resource>.get(&second);
            var weak = ArcPtr<Resource>.downgrade(&second);
            { var upgraded = WeakPtr<Resource>.upgrade(&weak)!; };
            if (trace != 0 || other.digit != 2) { trace := 9; };
            var number = Resource.make(&trace, 3).read();
            if (trace != 0 || number != 3) { trace := 9; };
        };
        if (trace == 321) { 0 } else { 1 }
    };
    "#);
}

#[test]
fn temporary_single_and_sequence_owners_live_until_scope_exit_and_error_cleanup() {
    let source = r#"
        export { main };
        import { "$/shared.resin", "$/status.resin" };
        struct Stop {};
        struct Marker { trace: Ptr<int>, digit: int,
            def drop(self: Ptr<Marker>) = {
                if (self.digit != 0_i) { self.trace.* := self.trace.* * 10_i + self.digit; };
            };
        };
        def single(trace: Ptr<int>) -> (ArcPtr<Marker> | Err<OutOfMemory>) = {
            var owner = ArcPtr<Marker>.alloc(Marker { trace = trace, digit = 0_i })?;
            owner.get().digit := 3_i;
            (owner)
        };
        def sequence(trace: Ptr<int>) -> (ArcSpan<Marker> | Err<OutOfMemory>) = {
            var owner = ArcSpan<Marker>.alloc(2_ul, Marker { trace = trace, digit = 0_i })?;
            owner.get().at(0_ul).digit := 1_i;
            owner.get().at(1_ul).digit := 2_i;
            (owner)
        };
        def stop_if(fail: bool) -> (() | Err<Stop>) = {
            if (fail) { Err(Stop {}) } else { (()) }
        };
        def read(trace: Ptr<int>, fail: bool) -> (int | Err<_>) = {
            var pointer = single(trace)?.get();
            var span = sequence(trace)?.get();
            // Observe destruction directly: freed bytes could retain old values.
            if (trace.* != 0_i) { trace.* := 1000_i; };
            stop_if(fail)?;
            (pointer.digit + span.at(0_ul).digit + span.at(1_ul).digit)
        };
        def main() -> (int | Err<_>) = {
            var trace = 0_i;
            var value = read(&trace, 1 == 0)?;
            var valid = value == 6_i && trace == 213_i;
            trace := 0_i;
            var stopped = match (read(&trace, 1 == 1)) {
                int(value) => { 1 == 0 },
                Err(error) => { match (error) { Stop(stop) => { 1 == 1 }, OutOfMemory(error) => { 1 == 0 } } },
            };
            (if (valid && stopped && trace == 213_i) { 0_i } else { 1_i })
        };
    "#;
    let output = support::project::Project::new(&support::module(source), Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn weak_payloads_require_upgrade_before_dereference_or_field_access() {
    for source in [
        "def f(value: WeakPtr<Resource>) = { value.*; };",
        "def f(value: WeakPtr<Resource>) = { value.digit; };",
        "def f(value: Ptr<WeakPtr<Resource>>) = { value.digit; };",
        "def f(value: WeakPtr<Resource>) = { value.read(); };",
    ] {
        assert!(
            pipeline::source_module(&format!("{RESOURCE} {source}")).is_err(),
            "{source}"
        );
    }
    run(r#"
        def read(value: WeakPtr<Resource>) -> int = { value.upgrade()!.get().digit };
        def main() -> int = {
            var trace = 0;
            var value = shared_resource(&trace, 1);
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

#[test]
fn loop_condition_and_body_errors_preserve_scope_cleanup() {
    run(r#"
    struct Stopped { code: uint };
    def check(stop: bool) -> (() | Err<Stopped>) = {
        if (stop) { Err(Stopped { code = 7_ui }) } else { (()) }
    };
    def exercise(trace: Ptr<int>, mode: int) -> (int | Err<Stopped>) = {
        var owner = Resource.make(trace, 1);
        var index = 0;
        while ({
            var condition = Resource.make(trace, 2);
            check(mode == 0)?;
            index < 1
        }) {
            var body = Resource.make(trace, 3);
            check(mode == 1)?;
            index := index + 1;
        };
        (trace.*)
    };
    def main() -> int = {
        var condition_trace = 0;
        var body_trace = 0;
        var normal_trace = 0;
        var a = match (exercise(&condition_trace, 0)) { int(v) => { 0_ui }, Err(e) => { e.code } };
        var b = match (exercise(&body_trace, 1)) { int(v) => { 0_ui }, Err(e) => { e.code } };
        var c = match (exercise(&normal_trace, 2)) { int(v) => { v }, Err(e) => { 0 } };
        if (a == 7_ui && b == 7_ui && c == 232 &&
            condition_trace == 21 && body_trace == 231 && normal_trace == 2321) { 0 } else { 1 }
    };
    "#);
}
