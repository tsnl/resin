use resin_cache::{Cache, UpdateError};
use resin_executor::{Cancellation, Execution};
use std::{
    convert::Infallible,
    num::NonZeroUsize,
    sync::{
        Arc, Mutex, Once,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

fn execution(jobs: usize) -> Execution {
    Execution::new(NonZeroUsize::new(jobs).unwrap())
}

async fn update(cache: &Cache<u32, u32>, keys: &[u32]) -> Cache<u32, u32> {
    cache
        .update(
            keys.iter().copied(),
            |key| async move { Ok::<_, Infallible>(Arc::new(key * 10)) },
            &execution(2),
            &Cancellation::new(),
        )
        .await
        .unwrap()
}

fn keys(cache: &Cache<u32, u32>) -> Vec<u32> {
    cache.iter().map(|(key, _)| *key).collect()
}

#[tokio::test]
async fn updates_share_hits_and_leave_predecessors_unchanged() {
    let empty = Cache::new(4);
    let first = update(&empty, &[1, 2]).await;
    let next = update(&first, &[2, 3]).await;
    assert!(empty.is_empty());
    assert_eq!(keys(&first), [1, 2]);
    assert_eq!(keys(&next), [1, 2, 3]);
    assert!(Arc::ptr_eq(first.get(&2).unwrap(), next.get(&2).unwrap()));
}

#[tokio::test]
async fn duplicate_keys_build_once_and_hits_do_not_invoke_the_builder() {
    let calls = AtomicUsize::new(0);
    let first = Cache::new(5)
        .update(
            [3, 1, 3, 2, 1],
            |key| {
                calls.fetch_add(1, Ordering::SeqCst);
                async move { Ok::<_, Infallible>(Arc::new(key)) }
            },
            &execution(3),
            &Cancellation::new(),
        )
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    let next = first
        .update::<_, _, Infallible>(
            [1, 3],
            |_| async { panic!("a hit must not invoke its builder") },
            &execution(1),
            &Cancellation::new(),
        )
        .await
        .unwrap();
    assert!(Arc::ptr_eq(first.get(&1).unwrap(), next.get(&1).unwrap()));
}

#[tokio::test]
async fn hits_refresh_recency_and_ties_use_key_order() {
    let first = update(&Cache::new(3), &[3, 2, 1]).await;
    let hit = update(&first, &[1]).await;
    let added = update(&hit, &[4]).await;
    assert_eq!(keys(&added), [1, 2, 4]);
    let newest = update(&added, &[3]).await;
    assert_eq!(keys(&newest), [1, 3, 4]);
    assert_eq!(keys(&first), [1, 2, 3]);
}

#[tokio::test]
async fn eviction_order_does_not_follow_builder_completion_order() {
    let pool = execution(3);
    let cancellation = Cancellation::new();
    let next = Cache::new(2)
        .update(
            [1, 2, 3],
            |key| async move {
                tokio::time::sleep(Duration::from_millis((4 - key) * 5)).await;
                Ok::<_, Infallible>(Arc::new(key))
            },
            &pool,
            &cancellation,
        )
        .await
        .unwrap();
    let retained = next
        .update::<_, _, Infallible>(
            [],
            |_| async { panic!("an empty update builds nothing") },
            &pool,
            &cancellation,
        )
        .await
        .unwrap();
    assert_eq!(
        retained.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
        [1, 2]
    );
}

#[tokio::test]
async fn concurrent_successors_are_send_and_keep_their_own_requested_sets() {
    let base = Arc::new(update(&Cache::new(2), &[1]).await);
    let mut tasks = tokio::task::JoinSet::new();
    for key in [2, 3] {
        let base = base.clone();
        tasks.spawn(async move { update(&base, &[key]).await });
    }
    let left = tasks.join_next().await.unwrap().unwrap();
    let right = tasks.join_next().await.unwrap().unwrap();
    assert_eq!(keys(&base), [1]);
    assert_eq!(left.len(), 2);
    assert_eq!(right.len(), 2);
    assert_ne!(keys(&left), keys(&right));
    assert!(Arc::ptr_eq(left.get(&1).unwrap(), right.get(&1).unwrap()));
}

#[tokio::test]
async fn overflow_retains_requests_and_later_updates_shrink_it() {
    let first = update(&Cache::new(2), &[9]).await;
    let oversized = update(&first, &[3, 2, 1, 1]).await;
    assert_eq!(oversized.capacity(), 2);
    assert_eq!(keys(&oversized), [1, 2, 3]);
    assert_eq!(keys(&update(&oversized, &[]).await), [1, 2]);
    assert_eq!(keys(&update(&oversized, &[3]).await), [1, 3]);
    assert_eq!(keys(&first), [9]);
    let zero = update(&Cache::new(0), &[1]).await;
    assert_eq!(keys(&zero), [1]);
    assert!(update(&zero, &[]).await.is_empty());
    assert!(update(&Cache::new(0), &[]).await.is_empty());
}

struct Warnings(Mutex<Vec<String>>);

impl log::Log for Warnings {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() == log::Level::Warn && metadata.target() == "resin_cache"
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            self.0.lock().unwrap().push(record.args().to_string());
        }
    }

    fn flush(&self) {}
}

static WARNINGS: Warnings = Warnings(Mutex::new(Vec::new()));
static LOGGER: Once = Once::new();

#[tokio::test]
async fn oversized_updates_automatically_log_their_counts() {
    LOGGER.call_once(|| {
        log::set_logger(&WARNINGS).unwrap();
        log::set_max_level(log::LevelFilter::Warn);
    });
    let keys: Vec<_> = (0..24).collect();
    let next = update(&Cache::new(23), &keys).await;
    assert_eq!(next.len(), 24);
    assert!(WARNINGS.0.lock().unwrap().iter().any(|message| {
        message == "cache capacity exceeded: capacity=23, requested=24, retained=24"
    }));
}

#[tokio::test]
async fn eviction_keeps_live_handles_valid_without_retaining_history() {
    let old = update(&Cache::new(1), &[1]).await;
    let held = old.get(&1).unwrap().clone();
    let weak = Arc::downgrade(&held);
    let next = update(&old, &[2]).await;
    drop(old);
    assert_eq!(*held, 10);
    drop(held);
    assert!(weak.upgrade().is_none());
    assert_eq!(**next.get(&2).unwrap(), 20);
}

#[tokio::test]
async fn builders_overlap_without_holding_nested_cpu_permits() {
    let pool = execution(2);
    let cancellation = Cancellation::new();
    let active = AtomicUsize::new(0);
    let maximum = AtomicUsize::new(0);
    let cache = Cache::new(8);
    let future = cache.update(
        0..8,
        |key| {
            let pool = &pool;
            let cancellation = &cancellation;
            let active = &active;
            let maximum = &maximum;
            async move {
                let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                maximum.fetch_max(count, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(5)).await;
                let value = pool.run(cancellation, move |_| key * 10).await?;
                active.fetch_sub(1, Ordering::SeqCst);
                Ok::<_, resin_executor::Error>(Arc::new(value))
            }
        },
        &pool,
        &cancellation,
    );
    let next = tokio::time::timeout(Duration::from_secs(2), future)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(next.len(), 8);
    assert_eq!(maximum.load(Ordering::SeqCst), 2);
    assert_eq!(active.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn failures_and_cancellation_do_not_change_the_old_cache() {
    let old = update(&Cache::new(4), &[1]).await;
    let error = old
        .update(
            [1, 2, 3],
            |key| async move {
                if key == 3 {
                    Err("failed")
                } else {
                    Ok(Arc::new(key))
                }
            },
            &execution(2),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, UpdateError::Build { error: "failed" }));
    assert_eq!(keys(&old), [1]);

    let cancellation = Cancellation::new();
    let cancel = cancellation.clone();
    let (started, ready) = tokio::sync::oneshot::channel();
    let sender = Mutex::new(Some(started));
    let pool = execution(1);
    let pending = old.update(
        [2],
        |_| {
            sender.lock().unwrap().take().unwrap().send(()).unwrap();
            std::future::pending::<Result<Arc<u32>, Infallible>>()
        },
        &pool,
        &cancellation,
    );
    let cancellation_task = async move {
        ready.await.unwrap();
        cancel.cancel();
    };
    let (result, ()) = tokio::join!(pending, cancellation_task);
    assert!(matches!(result, Err(UpdateError::Cancelled)));
    assert_eq!(keys(&old), [1]);
}

#[tokio::test]
async fn rebasing_reuses_nonclone_values_and_preserves_new_contributions() {
    struct Value(u32);
    let pool = execution(2);
    let cancellation = Cancellation::new();
    let base = Cache::new(3);
    let left = base
        .update(
            [1],
            |key| async move { Ok::<_, Infallible>(Arc::new(Value(key))) },
            &pool,
            &cancellation,
        )
        .await
        .unwrap();
    let right = base
        .update(
            [2],
            |key| async move { Ok::<_, Infallible>(Arc::new(Value(key))) },
            &pool,
            &cancellation,
        )
        .await
        .unwrap();
    let completed = right.get(&2).unwrap().clone();
    let merged = left
        .update(
            [2],
            |_| std::future::ready(Ok::<_, Infallible>(completed.clone())),
            &pool,
            &cancellation,
        )
        .await
        .unwrap();
    assert_eq!(merged.len(), 2);
    assert!(Arc::ptr_eq(merged.get(&2).unwrap(), &completed));
    assert_eq!(merged.get(&1).unwrap().0, 1);
}

#[test]
fn completed_caches_are_shareable() {
    fn shared<T: Send + Sync>() {}
    shared::<Cache<u32, String>>();
}
