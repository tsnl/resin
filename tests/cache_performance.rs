//! Reproducible Phase 1 measurements, with correctness assertions on actual pass reuse.
use resin_cache::Cache;
use resin_executor::{Cancellation, Execution};
use resin_source::{ImportBinding, Source, SourceGraph};
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone)]
struct Retained {
    syntax: Cache<Source, resin_cst::Document>,
    ast: Cache<Source, resin_ast::ModuleDocument>,
    hir: Cache<SourceGraph, resin_hir::Hir>,
    lir: Cache<resin_lir::LirKey, resin_lir::VerifiedModule>,
}

impl Retained {
    fn new() -> Self {
        Self {
            syntax: Cache::new(4096),
            ast: Cache::new(4096),
            hir: Cache::new(64),
            lir: Cache::new(64),
        }
    }

    async fn update(
        &self,
        graph: SourceGraph,
        counts: &[AtomicUsize; 4],
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<Self> {
        let syntax = self
            .syntax
            .update(
                graph.sources().cloned(),
                |source| {
                    let previous = self
                        .syntax
                        .iter()
                        .find(|(old, _)| old.id() == source.id())
                        .map(|(_, syntax)| syntax.clone());
                    async move {
                        counts[0].fetch_add(1, Ordering::Relaxed);
                        resin_cst::build_cst(
                            source.text().to_owned(),
                            previous.as_deref(),
                            execution,
                            cancellation,
                        )
                        .await
                        .map(Arc::new)
                    }
                },
                execution,
                cancellation,
            )
            .await?;
        let ast = self
            .ast
            .update(
                graph.sources().cloned(),
                |source| {
                    let syntax = syntax.get(&source).unwrap().clone();
                    async move {
                        counts[1].fetch_add(1, Ordering::Relaxed);
                        let parsed =
                            resin_ast::build_ast(syntax.clone(), execution, cancellation).await?;
                        Ok::<_, resin_executor::Error>(Arc::new(resin_ast::ModuleDocument {
                            source,
                            syntax,
                            file: Arc::new(parsed.file),
                            errors: parsed.errors,
                        }))
                    }
                },
                execution,
                cancellation,
            )
            .await?;
        let documents: BTreeMap<_, _> = graph
            .sources()
            .map(|source| (source.clone(), ast.get(source).unwrap().clone()))
            .collect();
        let hir = self
            .hir
            .update(
                [graph.clone()],
                |graph| {
                    let documents = documents.clone();
                    async move {
                        counts[2].fetch_add(1, Ordering::Relaxed);
                        let inputs =
                            resin_ast::build_program(graph, documents, execution, cancellation)
                                .await?;
                        resin_hir::Hir::build(Arc::new(inputs), execution, cancellation)
                            .await
                            .map(Arc::new)
                    }
                },
                execution,
                cancellation,
            )
            .await?;
        let module = hir.get(&graph).unwrap().hir()?.clone();
        let entry = resin_lir::Entry::exported(&module, "main", resin_lir::Profile::Host)?;
        let key = resin_lir::LirKey::new(
            Arc::new(graph),
            [entry],
            resin_lir::LoweringOptions::default(),
        );
        let lir = self
            .lir
            .update(
                [key],
                |key| {
                    let module = module.clone();
                    async move {
                        counts[3].fetch_add(1, Ordering::Relaxed);
                        let lir = resin_lir::build_lir(
                            module,
                            key.entries().to_vec(),
                            key.options().clone(),
                            execution,
                            cancellation,
                        )
                        .await?;
                        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(Arc::new(
                            resin_lir::VerifiedModule::build(lir, execution, cancellation).await?,
                        ))
                    }
                },
                execution,
                cancellation,
            )
            .await
            .map_err(|error| error.to_string())?;
        Ok(Self {
            syntax,
            ast,
            hir,
            lir,
        })
    }
}

fn fixture(changed: bool) -> SourceGraph {
    let count = 16;
    let mut sources = Vec::new();
    for index in 0..count {
        let value = if changed && index == 0 { 43 } else { 42 };
        let mut text =
            format!("export {{ value_{index} }}; fn value_{index}() -> i32  {{ {value} }}\n");
        for helper in 0..64 {
            text.push_str(&format!(
                "fn helper_{helper}(value: i32) -> i32  {{ value + {helper} }}\n"
            ));
        }
        sources.push(Source::new(format!("file{index}.resin"), text));
    }
    let imports = (0..count)
        .map(|index| format!("\"file{index}.resin\""))
        .collect::<Vec<_>>()
        .join(", ");
    let sum = (0..count)
        .map(|index| format!("value_{index}()"))
        .collect::<Vec<_>>()
        .join(" + ");
    let root = Source::new(
        "main.resin",
        format!("export {{ main }}; import {{ {imports} }}; fn main() -> i32  {{ {sum} }}"),
    );
    let bindings: Vec<_> = sources
        .iter()
        .map(|source| ImportBinding {
            source: root.id(),
            reference: source.name().into(),
            target: source.id(),
        })
        .collect();
    // Independent acquisitions can discover the same graph in another order.
    if changed {
        sources.reverse();
    }
    SourceGraph::new(root, sources, bindings).unwrap()
}

fn counts(values: &[AtomicUsize; 4]) -> [usize; 4] {
    std::array::from_fn(|index| values[index].load(Ordering::Relaxed))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_warm_edit_cache_measurements() -> Result<()> {
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let built = std::array::from_fn(|_| AtomicUsize::new(0));
    let started = Instant::now();
    let first = Retained::new()
        .update(fixture(false), &built, &execution, &cancellation)
        .await?;
    let cold = started.elapsed();
    assert_eq!(counts(&built), [17, 17, 1, 1]);
    let memory_cold = resident_kib();
    let started = Instant::now();
    let warm = first
        .update(fixture(false), &built, &execution, &cancellation)
        .await?;
    let unchanged = started.elapsed();
    assert_eq!(counts(&built), [17, 17, 1, 1]);
    let original = first.hir.get(&fixture(false)).unwrap();
    assert!(Arc::ptr_eq(
        original,
        warm.hir.get(&fixture(false)).unwrap()
    ));
    let started = Instant::now();
    let edited = warm
        .update(fixture(true), &built, &execution, &cancellation)
        .await?;
    let edit = started.elapsed();
    assert_eq!(counts(&built), [18, 18, 2, 2]);
    assert!(original.hir().is_ok());
    assert_eq!(first.syntax.len(), 17);
    assert_eq!(edited.syntax.len(), 18);
    println!(
        "phase1 frontend: jobs={}, sources=17, declarations=1041, cold_us={}, warm_us={}, edit_us={}, builds={:?}, rss_cold_kib={:?}, rss_retained_kib={:?}",
        execution.jobs(),
        cold.as_micros(),
        unchanged.as_micros(),
        edit.as_micros(),
        counts(&built),
        memory_cold,
        resident_kib()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn map_copy_rebase_and_retained_memory_measurements() -> Result<()> {
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let before = resident_kib();
    let original = Cache::new(10_000)
        .update(
            0..10_000_u32,
            |_| async { Ok::<_, std::convert::Infallible>(Arc::new([7_u8; 1024])) },
            &execution,
            &cancellation,
        )
        .await?;
    let allocated = resident_kib();
    let start = Instant::now();
    for _ in 0..100 {
        std::hint::black_box(original.clone());
    }
    let copy = start.elapsed();
    let start = Instant::now();
    for _ in 0..10 {
        let next = original
            .update(0..10_000, unexpected_miss, &execution, &cancellation)
            .await?;
        assert_eq!(next.len(), 10_000);
    }
    let rebase = start.elapsed();
    let weak = Arc::downgrade(original.get(&0).unwrap());
    let full = original
        .update(
            10_000..20_000,
            |_| async { Ok::<_, std::convert::Infallible>(Arc::new([9; 1024])) },
            &execution,
            &cancellation,
        )
        .await?;
    assert!(full.get(&0).is_none());
    assert!(weak.upgrade().is_some());
    drop(original);
    assert!(weak.upgrade().is_none());
    println!(
        "phase1 maps: entries=10000, payload_bytes=10240000, mean_clone_us={}, mean_hit_update_us={}, rss_before_kib={:?}, rss_allocated_kib={:?}",
        copy.as_micros() / 100,
        rebase.as_micros() / 10,
        before,
        allocated
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn execution_overlap_and_cancellation_measurements() -> Result<()> {
    let execution = Execution::new(NonZeroUsize::new(4).unwrap());
    let cancellation = Cancellation::new();
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    let workers = (0..4).map(|_| {
        let active = active.clone();
        let peak = peak.clone();
        execution.run(&cancellation, move |_| {
            peak.fetch_max(active.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
            let end = Instant::now() + Duration::from_millis(40);
            while Instant::now() < end {
                std::hint::black_box(13_u64.wrapping_mul(101));
            }
            active.fetch_sub(1, Ordering::SeqCst);
        })
    });
    futures::future::try_join_all(workers).await?;
    let parallel = start.elapsed();
    assert!(peak.load(Ordering::SeqCst) > 1);
    assert!(peak.load(Ordering::SeqCst) <= 4);
    let (started, ready) = tokio::sync::oneshot::channel();
    let work = execution.run(&cancellation, move |cancellation| {
        started.send(()).unwrap();
        while !cancellation.is_cancelled() {
            std::hint::black_box(7_u64.wrapping_mul(101));
        }
    });
    let cancel = async {
        ready.await.unwrap();
        let start = Instant::now();
        cancellation.cancel();
        start
    };
    let (result, cancelled) = tokio::join!(work, cancel);
    assert_eq!(result, Err(resin_executor::Error::Cancelled));
    execution.wait_idle().await;
    println!(
        "phase1 execution: peak_jobs={}, four_40ms_cpu_jobs_us={}, cancel_and_drain_us={}",
        peak.load(Ordering::SeqCst),
        parallel.as_micros(),
        cancelled.elapsed().as_micros()
    );
    Ok(())
}

fn resident_kib() -> Option<usize> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}

async fn unexpected_miss(_: u32) -> std::result::Result<Arc<[u8; 1024]>, std::convert::Infallible> {
    panic!("all keys are retained hits")
}
