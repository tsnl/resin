// Keep pass-reuse assertions in CI at a small scale; nightly benchmarks use the
// full 17-file, 1,041-declaration fixture and measure large maps and execution.
#[allow(dead_code)]
#[path = "../benchmarks/compiler/cache.rs"]
mod cache;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_warm_and_edited_sources_reuse_completed_passes() {
    cache::frontend(2, 2).await.unwrap();
}
