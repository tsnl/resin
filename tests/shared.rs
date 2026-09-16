use support::pipeline;
mod support;

const RESOURCE: &str = r#"import { "$/shared.resin" };
struct Resource { trace: Ptr<int>; digit: int;
    
    
    
}
fn drop(dying: Ptr<Resource>)  {
        if (dying.digit != 0) { dying.trace.* = dying.trace.* * 10 + dying.digit; };
    }

fn read(self: Ptr<Resource>) -> int  { self.digit }

fn resource_make(trace: Ptr<int>, digit: int) -> Resource  { Resource { trace = trace, digit = digit } }

fn optional_resource(trace: Ptr<int>, digit: int) -> Resource | None  { resource_make(trace, digit) }
fn shared_resource(trace: Ptr<int>, digit: int) -> ArcPtr<Resource>  {
    let mut optional: ArcPtr<Resource> | None;
    optional = match (arc_ptr_alloc::<Resource>(Resource { trace = trace, digit = 0 })) {
        ArcPtr<Resource>(value) => { value }, Err(error) => { None },
    };
    let mut owner = optional!;
    owner:get().digit = digit;
    owner
}
fn optional_shared(trace: Ptr<int>, digit: int) -> ArcPtr<Resource> | None  { shared_resource(trace, digit) }
fn optional_int(value: int) -> int | None  { value }

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
    run(r#"struct FieldsValues<T0> { values: T0; }
fn main() -> int  {
        let mut trace = 0; let mut weak = weak_ptr_empty::<Resource>();
        {
            let mut holder = FieldsValues<_> { values = [shared_resource(&trace, 1)] };
            weak = holder.values:at(0):downgrade();
            let mut saved = holder.values:at(0);
            holder.values:at(0) = shared_resource(&trace, 2);
            if (saved:get():read() != 1 || holder.values:at(0):get():read() != 2 || trace != 0) { trace = 9; };
        };
        let mut expired = match (weak:upgrade()) { ArcPtr<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
        if (expired && trace == 12) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn assignment_branches_preserve_initialization_and_overwrite_cleanup() {
    run(r#"struct Tracked { drops: Ptr<int>; value: int;
        
    }
fn drop(self: Ptr<Tracked>)  { self.drops.* = self.drops.* + 1; }


    fn main() -> int  {
        let mut drops = 0; let mut index = 0; let mut valid = 1 == 1;
        while (index < 2) {
            let mut value: Tracked;
            value = if (index == 0) { Tracked { drops = &drops, value = 1 } } else { Tracked { drops = &drops, value = 1 } };
            value = match (optional_int(index)) {
                int(n) => { Tracked { drops = &drops, value = 2 } },
                None => { Tracked { drops = &drops, value = 3 } }
            };
            valid = valid && value.value == 2;
            index = index + 1;
        };
        if (valid && drops == 8) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn shared_assignment_branches_release_every_owner() {
    run(r#"fn main() -> int  {
        let mut trace = 0; let mut weak = weak_ptr_empty::<Resource>();
        {
            let mut value: ArcPtr<Resource>;
            value = if (trace == 0) { shared_resource(&trace, 1) } else { shared_resource(&trace, 9) };
            value = match (optional_int(2)) {
                int(n) => { shared_resource(&trace, n) },
                None => { shared_resource(&trace, 9) }
            };
            weak = value:downgrade();
        };
        let mut expired = match (weak:upgrade()) { ArcPtr<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
        if (expired && trace == 12) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn assignment_propagation_tracks_success_and_error_cleanup() {
    run(r#"struct Failed {}
    fn acquire(trace: Ptr<int>, fail: bool) -> (ArcPtr<Resource> | Err<Failed>)  {
        if (fail) { Err(Failed {}) } else { (shared_resource(trace, 2)) }
    }
    fn work(trace: Ptr<int>, fail: bool) -> (() | Err<Failed>)  {
        let mut first = shared_resource(trace, 1);
        let mut pending: ArcPtr<Resource>;
        pending = acquire(trace, fail)?;
        (())
    }
    fn main() -> int  {
        let mut trace = 0;
        let mut succeeded = match (work(&trace, 1 == 0)) { ()(n) => { 1 == 1 }, Err(e) => { 1 == 0 } };
        let mut failed = match (work(&trace, 1 == 1)) { ()(n) => { 1 == 0 }, Err(e) => { 1 == 1 } };
        if (succeeded && failed && trace == 211) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn temporary_projection_keeps_nominal_and_nested_destructors() {
    run(r#"struct Outer { trace: Ptr<int>; inner: ArcPtr<Resource>;
        
    }
fn drop(self: Ptr<Outer>)  { self.trace.* = self.trace.* * 10 + 2; }


    fn make(trace: Ptr<int>) -> Outer  { Outer { trace = trace, inner = shared_resource(trace, 3) } }
    fn main() -> int  {
        let mut trace = 0;
        let mut number = make(&trace, 1).digit;
        let mut valid = number == 1 && trace == 1;
        {
            let mut inner = make(&trace).inner;
            valid = valid && trace == 12 && inner:get().digit == 3;
        };
        if (valid && trace == 123) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn function_fields_named_drop_remain_callable() {
    run(r#"struct FieldsDrop<T0> { drop: T0; }
fn increment(value: int) -> int  { value + 1 }
    struct Callback { drop: (int) -> int; }
    fn main() -> int  {
        let mut record = FieldsDrop<_> { drop = increment };
        let mut nominal = Callback { drop = increment };
        if ((record.drop)(20) + (nominal.drop)(20) == 42) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn all_applications_consume_fresh_arguments_without_an_extra_drop() {
    run(r#"fn consume(value: Resource) -> int  { value.digit }
    fn consume_pair(first: Resource, second: Resource)  {}
    fn main() -> int  {
        let mut trace = 0;
        consume(make(&trace, 1));
        consume(Resource(make(&trace, 2)));
        { let mut value = make(&trace, 3); };
        { let mut shared = shared_resource(&trace, 4); };
        consume_pair(make(&trace, 5), make(&trace, 6));
        if (trace == 123465) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn generic_record_initializers_preserve_layout_and_cleanup_on_partial_failure() {
    run(r#"struct Pair<T> { first: T; second: T; }
    struct Failed {}
    fn ready(fail: bool) -> (() | Err<Failed>)  {
        if (fail) { Err(Failed {}) } else { (()) }
    }
    fn build<T>(create: (Ptr<int>, int) -> T, trace: Ptr<int>, fail: bool) -> (Pair<T> | Err<Failed>)  {
        (Pair<T> {
            second = create(trace, 2),
            first = {
                ready(fail)?;
                create(trace, 1)
            },
        })
    }
    fn main() -> int  {
        let mut trace = 0;
        let mut ordered = match (build(shared_resource, &trace, 1 == 0)) {
            Pair<ArcPtr<Resource>>(pair) => { pair.first:get().digit == 1 && pair.second:get().digit == 2 },
            Err(error) => { 1 == 0 },
        };
        let mut destroyed = trace == 21;
        let mut failed = match (build(shared_resource, &trace, 1 == 1)) {
            Pair<ArcPtr<Resource>>(pair) => { 1 == 0 },
            Err(error) => { 1 == 1 },
        };
        if (ordered && destroyed && failed && trace == 212) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn named_values_are_copied_even_when_the_type_has_a_destructor() {
    run(r#"fn consume(value: Resource)  {}
    fn main() -> int  {
        let mut trace = 0;
        {
            let mut original = make(&trace, 1);
            { let mut copied = original; consume(original); };
            if (trace != 11 || original:read() != 1) { trace = 9; };
        };
        if (trace == 111) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn shared_copies_reassignment_weak_upgrade_and_expiration() {
    run(r#"struct Shared { value: ArcPtr<Resource>; }
    fn main() -> int  {
        let mut trace = 0;
        let mut weak = weak_ptr_empty::<Resource>();
        {
            let mut a = shared_resource(&trace, 1);
            let mut b = Shared { value = a };
            weak = b.value:downgrade();
            a = shared_resource(&trace, 2);
            match (weak:upgrade()) {
                ArcPtr<Resource>(owner) => { if (owner:get():read() != 1) { trace = 9; }; },
                None => { trace = 9; }
            };
            if (trace != 0) { trace = 9; };
        };
        let mut expired = match (weak:upgrade()) { ArcPtr<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
        if (expired && trace == 12) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn native_wrapper_can_transfer_a_handle_by_disarming_the_source() {
    run(r#"fn move(value: Ptr<Resource>) -> Resource  {
        Resource { trace = value.trace, digit = (&value.digit):replace(0) }
    }
    fn main() -> int  {
        let mut trace = 0;
        {
            let mut value = make(&trace, 1);
            {
                let mut target = shared_resource(&trace, 0);
                target:get():replace(move(&value));
            };
            if (trace != 1 || value.digit != 0) { trace = 9; };
        };
        if (trace == 1) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn early_errors_destroy_only_acquired_owners() {
    run(r#"struct Failed {}
    fn fail() -> (int | Err<Failed>)  { Err(Failed {}) }
    fn work(trace: Ptr<int>) -> (Resource | Err<Failed>)  {
        let mut first = make(trace, 1);
        let mut second = make(trace, 2);
        let mut unused = fail()?;
        (make(trace, 3))
    }
    fn main() -> int  {
        let mut trace = 0;
        let mut failed = match (work(&trace)) { Resource(owner) => { 1 == 0 }, Err(error) => { 1 == 1 } };
        if (failed && trace == 21) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn arrays_options_and_methods_consume_fresh_payloads() {
    run(r#"struct Pair { value: Resource; }
    struct Sink {
        
    }
fn consume(self: Ptr<Sink>, a: Resource, b: Resource)  {}


    fn main() -> int  {
        let mut trace = 0;
        {
            let mut pair = Pair { value = make(&trace, 1) };
            let mut array = [make(&trace, 2), make(&trace, 3)];
            let mut option = optional_resource(&trace, 4);
            // Reading this named union copies its payload. Both copies are destroyed.
            match (option) { Resource(value) => {}, None => {} };
            let mut sink = Sink {};
            sink:consume(make(&trace, 5), make(&trace, 6));
        };
        if (trace == 4654321) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn indexing_preserves_array_storage_and_only_value_reads_copy_elements() {
    run(r#"fn main() -> int  {
        let mut trace = 0;
        {
            let mut array = [make(&trace, 1), make(&trace, 2)];
            let mut element: Ref<Resource> = array(0);
            if (element.digit != 1 || array(1).digit != 2 || trace != 0) { trace = 9; };
            { let mut copied = array(1); };
            if (trace != 2 || array(1).digit != 2) { trace = 9; };
        };
        if (trace == 221) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn destruction_preserves_results_and_runs_per_scope_and_iteration() {
    run(r#"struct Set { target: Ptr<int>; value: int;
        
    }
fn drop(self: Ptr<Set>)  { self.target.* = self.value; }


    fn result_before_cleanup() -> int  {
        let mut value = 42;
        let mut cleanup = Set { target = &value, value = 99 };
        value
    }
    fn main() -> int  {
        let mut trace = 0;
        let mut index = 1;
        while (index < 4) {
            let mut outer = make(&trace, index);
            { let mut inner = make(&trace, index + 3); };
            index = index + 1;
        };
        if (1 == 1) { let mut branch = make(&trace, 7); };
        match (optional_int(8)) {
            int(n) => { let mut arm = make(&trace, n); },
            None => { let mut unused = make(&trace, 9); }
        };
        if (trace == 41526378 && result_before_cleanup() == 42) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn former_defer_keyword_can_name_an_ordinary_immediate_call() {
    run(r#"fn defer(trace: Ptr<int>)  { trace.* = trace.* + 1; }
    fn main() -> int  {
        let mut trace = 0;
        defer(&trace);
        if (trace == 1) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn weak_cycles_and_nested_pointer_handle_access() {
    run(
        r#"struct Node { trace: Ptr<int>; live: bool; parent: WeakPtr<Node>;
        
    }
fn drop(self: Ptr<Node>)  { if (self.live) { self.trace.* = self.trace.* + 1; }; }


    fn main() -> int  {
        let mut trace = 0;
        let mut weak = {
            let mut optional: ArcPtr<Node> | None;
            optional = match (arc_ptr_alloc::<Node>(Node { trace = &trace, live = 1 == 0, parent = weak_ptr_empty::<Node>() })) {
                ArcPtr<Node>(value) => { value }, Err(error) => { None },
            };
            let mut node = optional!;
            node:get().live = 1 == 1;
            node:get().parent = node:downgrade();
            let mut address = &node;
            let mut nested = &address;
            if (nested.*:get().trace.* != 0) { trace = 9; };
            node:downgrade()
        };
        let mut expired = match (weak:upgrade()) { ArcPtr<Node>(node) => { 1 == 0 }, None => { 1 == 1 } };
        if (expired && trace == 1) { 0 } else { 1 }
    }
    "#,
    );
}

#[test]
fn source_allocation_copies_shared_initializers_without_cloning_payloads() {
    run(r#"fn main() -> int  {
        let mut trace = 0;
        {
            let mut original = shared_resource(&trace, 1);
            {
                let mut optional: ArcPtr<ArcPtr<Resource>> | None;
                optional = match (arc_ptr_alloc::<ArcPtr<Resource>>(original)) {
                    ArcPtr<ArcPtr<Resource>>(value) => { value }, Err(error) => { None },
                };
                let mut nested = optional!;
                if (nested:get():get().digit != 1 || trace != 0) { trace = 9; };
            };
            if (original:get().digit != 1 || trace != 0) { trace = 9; };
        };
        if (trace == 1) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn methods_require_a_compatible_receiver_and_valid_destructor() {
    rejects(
        "struct Item {  }\nfn item_shared_resource(self: ArcPtr<Item>)  {}\n  fn f(item: Item)  { item:shared_resource(); }",
        "method receiver does not match",
    );
    rejects(
        "struct Item {  }\nfn drop(self: Item)  {}\n ",
        "drop must have signature",
    );
}

#[test]
fn option_unwrap_transfers_fresh_payloads_and_copies_named_options() {
    run(r#"fn main() -> int  {
        let mut trace = 0;
        { let mut unwrapped = optional_resource(&trace, 1)!; };
        {
            let mut named = optional_resource(&trace, 2);
            { let mut copied = named!; };
        };
        let mut weak = weak_ptr_empty::<Resource>();
        {
            let mut shared = optional_shared(&trace, 3)!;
            weak = shared:downgrade();
            { let mut upgraded = weak:upgrade()!; };
            if (trace != 122) { trace = 9; };
        };
        let mut expired = match (weak:upgrade()) { ArcPtr<Resource>(owner) => { 1 == 0 }, None => { 1 == 1 } };
        if (trace == 1223 && expired) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn destruction_hooks_are_ordinary_calls_and_remain_automatic() {
    run(r#"struct Manual { trace: Ptr<int>; digit: int;
        
    }
fn drop(self: Ptr<Manual>)  {
            if (self.digit != 0) { self.trace.* = self.trace.* * 10 + self.digit; };
            self.digit = 0;
        }


    fn main() -> int  {
        let mut trace = 0;
        {
            let mut a = Manual { trace = &trace, digit = 1 };
            let mut b = Manual { trace = &trace, digit = 2 };
            let mut c = Manual { trace = &trace, digit = 3 };
            a:drop();
            drop(&b);
        };
        if (trace == 123) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn source_methods_use_ordinary_calls_and_borrow_fresh_receivers() {
    run(r#"fn main() -> int  {
        let mut trace = 0;
        {
            let mut pointer = shared_resource(&trace, 1):get();
            if (trace != 0 || pointer.digit != 1) { trace = 9; };
            let mut second = shared_resource(&trace, 2);
            let mut other = get::<Resource>(&second);
            let mut weak = downgrade::<Resource>(&second);
            { let mut upgraded = upgrade::<Resource>(&weak)!; };
            if (trace != 0 || other.digit != 2) { trace = 9; };
            let mut number = make(&trace, 3):read();
            if (trace != 0 || number != 3) { trace = 9; };
        };
        if (trace == 321) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn temporary_single_and_sequence_owners_live_until_scope_exit_and_error_cleanup() {
    let source = r#"export { main };
        import { "$/shared.resin", "$/status.resin" };
        struct Stop {}
        struct Marker { trace: Ptr<int>; digit: int;
            
        }
fn drop(self: Ptr<Marker>)  {
                if (self.digit != 0_i) { self.trace.* = self.trace.* * 10_i + self.digit; };
            }

        fn single(trace: Ptr<int>) -> (ArcPtr<Marker> | Err<OutOfMemory>)  {
            let mut owner = arc_ptr_alloc::<Marker>(Marker { trace = trace, digit = 0_i })?;
            owner:get().digit = 3_i;
            (owner)
        }
        fn sequence(trace: Ptr<int>) -> (ArcSpan<Marker> | Err<OutOfMemory>)  {
            let mut owner = arc_span_alloc::<Marker>(2_ul, Marker { trace = trace, digit = 0_i })?;
            owner:get():at(0_ul).digit = 1_i;
            owner:get():at(1_ul).digit = 2_i;
            (owner)
        }
        fn stop_if(fail: bool) -> (() | Err<Stop>)  {
            if (fail) { Err(Stop {}) } else { (()) }
        }
        fn read(trace: Ptr<int>, fail: bool) -> (int | Err<_>)  {
            let mut pointer = single(trace)?:get();
            let mut span = sequence(trace)?:get();
            // Observe destruction directly: freed bytes could retain old values.
            if (trace.* != 0_i) { trace.* = 1000_i; };
            stop_if(fail)?;
            (pointer.digit + span:at(0_ul).digit + span:at(1_ul).digit)
        }
        fn main() -> (int | Err<_>)  {
            let mut trace = 0_i;
            let mut value = read(&trace, 1 == 0)?;
            let mut valid = value == 6_i && trace == 213_i;
            trace = 0_i;
            let mut stopped = match (read(&trace, 1 == 1)) {
                int(value) => { 1 == 0 },
                Err(error) => { match (error) { Stop(stop) => { 1 == 1 }, OutOfMemory(error) => { 1 == 0 } } },
            };
            (if (valid && stopped && trace == 213_i) { 0_i } else { 1_i })
        }
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
        "fn f(value: WeakPtr<Resource>)  { value.*; }",
        "fn f(value: WeakPtr<Resource>)  { value.digit; }",
        "fn f(value: Ptr<WeakPtr<Resource>>)  { value.digit; }",
        "fn f(value: WeakPtr<Resource>)  { value:read(); }",
    ] {
        assert!(
            pipeline::source_module(&format!("{RESOURCE} {source}")).is_err(),
            "{source}"
        );
    }
    run(
        r#"fn read(value: WeakPtr<Resource>) -> int  { value:upgrade()!:get().digit }
        fn main() -> int  {
            let mut trace = 0;
            let mut value = shared_resource(&trace, 1);
            if (read(value:downgrade()) == 1) { 0 } else { 1 }
        }
    "#,
    );
}

#[test]
fn pointer_replace_transfers_managed_values_and_leaves_ordinary_names_available() {
    run(r#"fn replace(value: int) -> int  { value + 1 }
    fn main() -> int  {
        let mut trace = 0;
        {
            let mut value = make(&trace, 1);
            { let mut old = (&value):replace(make(&trace, 2)); };
            if (trace != 1 || value.digit != 2) { trace = 9; };
            { let mut old = replace::<Resource>(&value, make(&trace, 3)); };
            if (trace != 12 || value.digit != 3) { trace = 9; };
        };
        if (trace == 123 && replace(1) == 2) { 0 } else { 1 }
    }
    "#);
}

#[test]
fn loop_condition_and_body_errors_preserve_scope_cleanup() {
    run(r#"struct Stopped { code: uint; }
    fn check(stop: bool) -> (() | Err<Stopped>)  {
        if (stop) { Err(Stopped { code = 7_ui }) } else { (()) }
    }
    fn exercise(trace: Ptr<int>, mode: int) -> (int | Err<Stopped>)  {
        let mut owner = make(trace, 1);
        let mut index = 0;
        while ({
            let mut condition = make(trace, 2);
            check(mode == 0)?;
            index < 1
        }) {
            let mut body = make(trace, 3);
            check(mode == 1)?;
            index = index + 1;
        };
        (trace.*)
    }
    fn main() -> int  {
        let mut condition_trace = 0;
        let mut body_trace = 0;
        let mut normal_trace = 0;
        let mut a = match (exercise(&condition_trace, 0)) { int(v) => { 0_ui }, Err(e) => { e.code } };
        let mut b = match (exercise(&body_trace, 1)) { int(v) => { 0_ui }, Err(e) => { e.code } };
        let mut c = match (exercise(&normal_trace, 2)) { int(v) => { v }, Err(e) => { 0 } };
        if (a == 7_ui && b == 7_ui && c == 232 &&
            condition_trace == 21 && body_trace == 231 && normal_trace == 2321) { 0 } else { 1 }
    }
    "#);
}
