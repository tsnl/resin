mod support;

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
        def single(trace: Ptr<int>) -> Result<ArcPtr<Marker>, OutOfMemory> = {
            var owner = ArcPtr<Marker>.alloc(Marker { trace = trace, digit = 0_i })?;
            owner.get().digit := 3_i;
            ok(owner)
        };
        def sequence(trace: Ptr<int>) -> Result<ArcSpan<Marker>, OutOfMemory> = {
            var owner = ArcSpan<Marker>.alloc(2_ul, Marker { trace = trace, digit = 0_i })?;
            owner.get().at(0_ul).digit := 1_i;
            owner.get().at(1_ul).digit := 2_i;
            ok(owner)
        };
        def stop_if(fail: bool) -> Result<(), Stop> = {
            if (fail) { err(Stop {}) } else { ok(()) }
        };
        def read(trace: Ptr<int>, fail: bool) -> Result<int, _> = {
            var pointer = single(trace)?.get();
            var span = sequence(trace)?.get();
            // Observe destruction directly: freed bytes could retain old values.
            if (trace.* != 0_i) { trace.* := 1000_i; };
            stop_if(fail)?;
            ok(pointer.digit + span.at(0_ul).digit + span.at(1_ul).digit)
        };
        def main() -> Result<int, _> = {
            var trace = 0_i;
            var value = read(&trace, 1 == 0)?;
            var valid = value == 6_i && trace == 213_i;
            trace := 0_i;
            var stopped = match (read(&trace, 1 == 1)) {
                ok(value) => { 1 == 0 },
                err(error) => { match (error) { Stop(stop) => { 1 == 1 }, OutOfMemory(error) => { 1 == 0 } } },
            };
            ok(if (valid && stopped && trace == 213_i) { 0_i } else { 1_i })
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
