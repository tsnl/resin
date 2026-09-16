//! Native aggregate snapshots, internal ABI, and managed lifetime behavior.
mod support;

use resin_codegen::NativeOptimization;
use resin_executor::Cancellation;
use std::{process::Command, sync::Arc};

fn equivalent(source: &str, expected: i32) {
    let module = support::module(source);
    let checked = Arc::new(resin_lir::VerifiedModule::new(module).unwrap());
    let directory = tempfile::TempDir::new().unwrap();
    let mut environment = resin_toolchain::Environment::capture().unwrap();
    environment.directory = directory.path().into();
    environment.executable = env!("CARGO_BIN_EXE_resin").into();
    let tools = environment.toolchain(None, None);
    for optimization in [NativeOptimization::None, NativeOptimization::Speed] {
        let object = support::frontend::block_on(resin_codegen::generate_native(
            checked.clone(),
            "main".into(),
            optimization,
            Arc::new(Default::default()),
            support::frontend::execution(),
            &Cancellation::new(),
        ))
        .unwrap();
        let executable = support::frontend::block_on(tools.link_native(
            resin_toolchain::NativeLink {
                objects: vec![object.shared_bytes()],
                runtime: true,
            },
            directory.path(),
            support::frontend::execution(),
            &Cancellation::new(),
        ))
        .unwrap();
        let native = Command::new(executable.path()).output().unwrap();
        assert_eq!(
            native.status.code(),
            Some(expected),
            "{optimization:?}: {source}\n{}",
            String::from_utf8_lossy(&native.stderr)
        );
        assert!(native.stdout.is_empty());
        assert!(native.stderr.is_empty());
    }
}

#[test]
fn record_parameters_and_results_preserve_independent_snapshots() {
    equivalent(
        r#"export { main };
        struct Pair { left: int, right: int };
        def swap(value: Pair) -> Pair = { Pair { left = value.right, right = value.left } };
        def main() -> int = {
            var original = Pair { left = 7, right = 11 };
            var copy = original;
            original.left := 30;
            var function = swap;
            var swapped = function(copy);
            swapped.left + swapped.right + original.left
        };"#,
        48,
    );
}

#[test]
fn arrays_nested_records_and_pointer_replacement_preserve_values() {
    equivalent(
        r#"export { main };
        def main() -> int = {
            var rows = [[1_i, 2_i], [3_i, 4_i]];
            var old = rows;
            rows.at(1_ul).at(0_ul) := 20;
            var record = { values = rows, count = 5_i };
            var pointer = &record;
            pointer.values.at(0_ul).at(1_ul) := 9;
            old.at(1_ul).at(0_ul) + rows.at(1_ul).at(0_ul) + record.values.at(0_ul).at(1_ul)
        };"#,
        32,
    );
}

#[test]
fn recursive_aggregate_calls_and_loop_reassignment_transfer_values() {
    equivalent(
        r#"export { main };
        struct Pair { first: int, second: int };
        def recurse(count: int, value: Pair) -> Pair = {
            if (count == 0) { value } else { recurse(count - 1, Pair { first = value.second, second = value.first + value.second }) }
        };
        def main() -> int = {
            var pair = Pair { first = 0, second = 1 };
            var index = 0_i;
            while (index < 4) {
                var next = Pair { first = pair.second, second = pair.first + pair.second };
                pair := next;
                index := index + 1;
            };
            var final = recurse(3, pair);
            final.first + final.second
        };"#,
        34,
    );
}

#[test]
fn strings_are_static_nul_terminated_views_and_copy_by_value() {
    equivalent(
        r#"export { main };
        def text() -> str = { "hello" };
        def main() -> int = {
            var first = text();
            var second = first;
            first := "x";
            int(second.length) + int(second.at(1_ul)) + int(second.data.*) - 100
        };"#,
        110,
    );
}

#[test]
fn option_payloads_and_union_widening_keep_nominal_type_tags() {
    equivalent(
        r#"export { main };
        struct A { number: int };
        struct B { number: int };
        def inspect(value: A | B) -> int = { match (value) { A(a) => { a.number }, B(b) => { b.number + 1 } } };
        def main() -> int = {
            var optional: A | None = A { number = 17 };
            var concrete = optional!;
            optional := None;
            var variant: A | B = concrete;
            inspect(variant) + inspect(B { number = 3 })
        };"#,
        21,
    );
}

#[test]
fn result_payloads_branch_and_propagate_through_aggregate_return_slots() {
    equivalent(
        r#"export { main };
        struct Failed { number: int };
        struct Pair { left: int, right: int };
        def choose(fail: bool) -> Result<Pair, Failed> = {
            if (fail) { err(Failed { number = 8 }) } else { ok(Pair { left = 11, right = 13 }) }
        };
        def forward(fail: bool) -> Result<Pair, Failed> = { var pair = choose(fail)?; ok(pair) };
        def main() -> int = {
            var good = match (forward(1 == 2)) { ok(pair) => { pair.left + pair.right }, err(error) => { 0 } };
            var bad = match (forward(1 == 1)) { ok(pair) => { 0 }, err(error) => { error.number } };
            good + bad
        };"#,
        32,
    );
}

#[test]
fn managed_branch_assignment_tracks_uninitialized_and_replaced_storage() {
    equivalent(
        r#"export { main };
        struct Tracked { drops: Ptr<int>, value: int,
            def drop(self: Ptr<Tracked>) = { self.drops.* := self.drops.* + 1; };
        };
        def optional(value: int) -> int | None = { value };
        def main() -> int = {
            var drops = 0_i; var index = 0_i; var valid = 1 == 1;
            while (index < 2) {
                var value: Tracked;
                value := if (index == 0) { Tracked { drops = &drops, value = 1 } } else { Tracked { drops = &drops, value = 1 } };
                value := match (optional(index)) { int(n) => { Tracked { drops = &drops, value = 2 } }, None => { Tracked { drops = &drops, value = 3 } } };
                valid := valid && value.value == 2;
                index := index + 1;
            };
            if (valid && drops == 8) { 0 } else { 1 }
        };"#,
        0,
    );
}

#[test]
fn shared_owner_copies_keep_payload_alive_until_the_final_release() {
    equivalent(
        r#"export { main };
        intrinsic "owner_allocate" def allocate<T>(count: ulong, initial: T) -> StrongOwner | None;
        intrinsic "owner_data" def data<T>(owner: Ptr<StrongOwner>) -> Ptr<T>;
        intrinsic "owner_downgrade" def downgrade(owner: Ptr<StrongOwner>) -> WeakOwner;
        intrinsic "owner_upgrade" def upgrade(owner: Ptr<WeakOwner>) -> StrongOwner | None;
        intrinsic "weak_empty" def empty() -> WeakOwner;
        struct Tracked { drops: Ptr<int>, live: bool,
            def drop(self: Ptr<Tracked>) = { if (self.live) { self.drops.* := self.drops.* + 1; }; };
        };
        def main() -> int = {
            var drops = 0_i; var weak = empty(); var valid = 1 == 1;
            {
                var owner = allocate(1_ul, Tracked { drops = &drops, live = 1 == 2 })!;
                data::<Tracked>(&owner).live := 1 == 1;
                weak := downgrade(&owner);
                var copied = owner;
                var upgraded = upgrade(&weak);
                valid := drops == 0 && data::<Tracked>(&copied).live;
            };
            var expired = match (upgrade(&weak)) { StrongOwner(value) => { 1 == 2 }, None => { 1 == 1 } };
            if (valid && drops == 1 && expired) { 0 } else { 5 }
        };"#,
        0,
    );
}

#[test]
fn propagation_destroys_completed_values_before_returning_an_error() {
    equivalent(
        r#"export { main };
        struct Failed {};
        struct Tracked { trace: Ptr<int>, digit: int,
            def drop(self: Ptr<Tracked>) = { self.trace.* := self.trace.* * 10 + self.digit; };
        };
        def failing() -> Result<(), Failed> = { err(Failed {}) };
        def work(trace: Ptr<int>) -> Result<(), Failed> = {
            var first = Tracked { trace = trace, digit = 1 };
            var second = Tracked { trace = trace, digit = 2 };
            failing()?;
            ok(())
        };
        def main() -> int = {
            var trace = 0_i;
            var failed = match (work(&trace)) { ok(value) => { 1 == 2 }, err(error) => { 1 == 1 } };
            if (failed && trace == 21) { 0 } else { 1 }
        };"#,
        0,
    );
}
