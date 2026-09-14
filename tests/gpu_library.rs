use resin_compiler::Compiler;
use resin_source::{Loader, Source, library_root};

#[test]
fn source_gpu_library_resolves_generic_allocation_and_explicit_access() {
    let source = Source::new(
        "gpu-library.resin",
        r#"
        export { main };
        import { "$/gpu.resin", "$/span.resin" };
        struct Pair { left: int, right: int };
        def main() -> Result<(), _> = {
            var gpu = Gpu.new()?;
            var scalar = gpu.create(Pair { left = 1_i, right = 2_i })?;
            var value = scalar.load();
            value.right := 3_i;
            scalar.store(value);
            scalar.replace(value);
            var values = gpu.alloc::<int>(4_ul)?;
            values.at(1_ul).store(7_i);
            var tail = values.slice(1_ul, 2_ul);
            var alias = tail.data.slice(1_ul, 1_ul);
            var output = [0_i, 0_i];
            tail.read_only().copy_to(Span<int> { data = output.at(0_ul), length = 2_ul });
            var readback = gpu.alloc_in::<ubyte>(64_ul, Memory.readback())?;
            ok(())
        };
    "#,
    );
    let mut loader = Loader::new(library_root());
    let analysis = Compiler::new().analyze(source, &mut loader);
    analysis.hir().unwrap();
}
