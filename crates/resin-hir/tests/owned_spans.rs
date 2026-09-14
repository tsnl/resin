use resin_hir::{TermKind, Type};

fn generate(source: &str) -> Result<resin_hir::Module, resin_hir::GenerateError> {
    let syntax = resin_cst::Document::reparse(source.into(), None);
    resin_hir::generate(&resin_ast::generate(&syntax).unwrap())
}

#[test]
fn shared_sequences_expose_explicit_borrows_and_weak_handles() {
    let module = generate(
        r#"
        def make() -> WeakSpan<uint> = {
            var owner = ArcSpan<uint>.try_new(4, 0_ui)!;
            owner.get().slice(1, 2).as_bytes();
            owner.downgrade()
        };
        def empty() -> WeakSpan<uint> = { WeakSpan<uint>() };
        def upgrade(value: WeakSpan<uint>) -> ArcSpan<uint> | None = { value.upgrade() };
    "#,
    )
    .unwrap();
    let printed = resin_hir::format_module(&module);
    for operation in [
        "ArcSpanTryNew",
        "ArcSpanGet",
        "SpanSlice",
        "SpanBytes",
        "Downgrade",
        "Upgrade",
    ] {
        assert!(printed.contains(operation), "{printed}");
    }
    let TermKind::Block { tail, .. } = &module.functions[1].body.as_ref().unwrap().kind else {
        panic!("block")
    };
    assert!(
        matches!(&tail.kind, TermKind::WeakEmpty { ty: Type::WeakSpan { element } } if **element == Type::UInt32)
    );
}

#[test]
fn host_allocator_keeps_source_error_policy_and_infers_element_type() {
    let module = generate(
        r#"
        struct OutOfMemory {};
        struct Host {
            @host_allocation_error
            def allocation_error() -> OutOfMemory = { OutOfMemory {} };
        };
        def allocate() -> Result<ArcSpan<uint>, OutOfMemory> = { Host.alloc(4, 0) };
    "#,
    )
    .unwrap();
    let function = module
        .functions
        .iter()
        .find(|function| function.name.as_ref() == "allocate")
        .unwrap();
    let TermKind::Block { tail, .. } = &function.body.as_ref().unwrap().kind else {
        panic!("block")
    };
    assert!(matches!(&tail.kind, TermKind::HostAllocate { .. }));
    assert!(
        matches!(&tail.ty, Type::Result { value, .. } if matches!(&**value, Type::ArcSpan { element } if **element == Type::UInt32))
    );
}

#[test]
fn ownership_and_raw_byte_views_cannot_be_forged_by_conversions() {
    for source in [
        "def bad(value: Span<uint>) -> ArcSpan<uint> = { ArcSpan<uint>(value) };",
        "def bad(value: ArcSpan<uint>) -> Span<uint> = { Span<uint>(value) };",
        "def bad(value: Span<bool>) -> Span<ubyte> = { value.as_bytes() };",
        "def bad(value: Span<ArcPtr<uint>>) -> Span<ubyte> = { value.as_bytes() };",
        "def bad(value: WeakSpan<uint>) -> Span<uint> = { value.get() };",
        "def bad(value: ArcSpan<uint>) -> ulong = { value.length };",
    ] {
        assert!(generate(source).is_err(), "{source}");
    }
}

#[test]
fn host_allocation_error_factory_requires_a_valid_static_signature() {
    for declaration in [
        "def failed(value: ulong) -> Error = { Error {} };",
        "def failed() -> ulong = { 0 };",
    ] {
        let source =
            format!("struct Error {{}}; struct Host {{ @host_allocation_error {declaration} }};");
        assert!(generate(&source).is_err(), "{source}");
    }
}

#[test]
fn generic_shared_operations_preserve_payload_parameters_in_hir() {
    let module = generate(
        r#"
        struct OutOfMemory {};
        struct Host {
            @host_allocation_error
            def allocation_error() -> OutOfMemory = { OutOfMemory {} };
        };
        def allocate<T>(count: ulong, initial: T) -> Result<ArcSpan<T>, OutOfMemory> = {
            Host.alloc(count, initial)
        };
        def optional<T>(count: ulong, initial: T) -> ArcSpan<T> | None = {
            ArcSpan<T>.try_new(count, initial)
        };
        def borrow<T>(owner: Ptr<ArcSpan<T>>) -> Span<T> = { owner.*.get() };
        def direct<T>(owner: Ptr<ArcSpan<T>>) -> Span<T> = { owner.get() };
        def weaken<T>(owner: ArcSpan<T>) -> WeakSpan<T> = { owner.downgrade() };
        def upgrade<T>(owner: WeakSpan<T>) -> ArcSpan<T> | None = { owner.upgrade() };
        def pointer<T>(owner: Ptr<ArcPtr<T>>) -> Ptr<T> = { owner.get() };
        def local<T>(value: T) -> ulong = {
            var owner = ArcSpan<T>.try_new(1, value)!;
            var address = &owner;
            address.get().length
        };
        def part<T>(view: Span<T>, start: ulong, length: ulong) -> Span<T> = {
            view.slice(start, length)
        };
        def element<T>(view: Span<T>, index: ulong) -> Ptr<T> = { view.at(index) };
        def main() -> Result<uint, OutOfMemory> = {
            var owner = allocate(3, 7_ui)?;
            var alias = upgrade(weaken(owner))!;
            var view = direct(&alias);
            ok(element(part(view, 1, 1), 0).*)
        };
    "#,
    )
    .unwrap();
    let allocate = module
        .functions
        .iter()
        .find(|f| f.name.as_ref() == "allocate")
        .unwrap();
    let TermKind::Block { tail, .. } = &allocate.body.as_ref().unwrap().kind else {
        panic!("block")
    };
    let TermKind::HostAllocate { args, .. } = &tail.kind else {
        panic!("host allocation")
    };
    assert!(matches!(args.params[1], Type::Parameter { .. }));
}

#[test]
fn host_allocation_requires_positional_count_and_initial_arguments() {
    let error = generate(
        r#"
        struct OutOfMemory {};
        struct Host {
            @host_allocation_error
            def allocation_error() -> OutOfMemory = { OutOfMemory {} };
        };
        def bad() -> Result<ArcSpan<uint>, OutOfMemory> = {
            Host.alloc({ count = 2_ul, initial = 7_ui })
        };
    "#,
    )
    .unwrap_err();
    assert!(error.to_string().contains("unknown method `alloc`"), "{error}");
}

#[test]
fn allocation_and_upgrade_keep_narrow_results_before_contextual_widening() {
    let module = generate(
        r#"
        struct OutOfMemory {};
        struct Other {};
        struct Host {
            @host_allocation_error
            def allocation_error() -> OutOfMemory = { OutOfMemory {} };
        };
        def allocate() -> Result<ArcSpan<uint>, OutOfMemory | Other> = {
            Host.alloc(2, 7_ui)
        };
        def optional() -> ArcSpan<uint> | None | Other = { ArcSpan<uint>.try_new(2, 7_ui) };
        def upgrade(weak: WeakSpan<uint>) -> ArcSpan<uint> | None | Other = { weak.upgrade() };
    "#,
    )
    .unwrap();
    for name in ["allocate", "optional", "upgrade"] {
        let function = module
            .functions
            .iter()
            .find(|f| f.name.as_ref() == name)
            .unwrap();
        let TermKind::Block { tail, .. } = &function.body.as_ref().unwrap().kind else {
            panic!("block")
        };
        let operation = if let TermKind::Convert { arg } = &tail.kind {
            arg.as_ref()
        } else {
            tail.as_ref()
        };
        assert_ne!(operation.ty, function.signature.result.ty);
        assert!(matches!(
            operation.kind,
            TermKind::HostAllocate { .. } | TermKind::Intrinsic { .. }
        ));
    }
}

#[test]
fn instantiated_generic_unions_expose_flat_variants_to_exhaustive_matches() {
    generate(
        r#"
        struct Other {};
        def upgrade<T>(weak: WeakSpan<T>) -> ArcSpan<T> | None | Other = { weak.upgrade() };
        def main(weak: WeakSpan<uint>) -> ulong = {
            match (upgrade(weak)) {
                ArcSpan<uint>(live) => { live.get().length },
                None => { 0_ul },
                Other(other) => { 0_ul },
            }
        };
    "#,
    )
    .unwrap();
}
