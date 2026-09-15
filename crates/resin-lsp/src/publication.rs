use arc_swap::ArcSwap;
use resin_cache::{Cache, UpdateError};
use resin_executor::{Cancellation, Execution};
use std::{collections::BTreeMap, future::Future, sync::Arc};

/// Select requested handles and publish their recency without mutating any generation.
///
/// Miss builders run only in the initial update. A failed CAS rebases all requested
/// handles, including initial hits, onto the current head. Current-head hits win;
/// no old unrequested history or stale eviction decisions are replayed. Callers own
/// the returned handles even when another update immediately evicts their keys.
/// Cancellation before a CAS publishes nothing; cancellation racing a successful
/// CAS cannot revoke the completed values already published by this layer.
pub(crate) async fn select<K, V, F, Fut, E>(
    head: &ArcSwap<Cache<K, V>>,
    requested: Vec<K>,
    build: F,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<BTreeMap<K, Arc<V>>, UpdateError<E>>
where
    K: Ord + Clone + Send + Sync,
    V: Send + Sync,
    F: Fn(K) -> Fut + Sync,
    Fut: Future<Output = Result<Arc<V>, E>> + Send,
{
    // Owned Arcs, never borrowed ArcSwap guards, cross suspension points.
    let mut base = head.load_full();
    let mut candidate = Arc::new(
        base.update(requested.iter().cloned(), &build, execution, cancellation)
            .await?,
    );
    drop(build);
    let completed = requested_values(&candidate, &requested);
    loop {
        cancellation.check().map_err(|_| UpdateError::Cancelled)?;
        let previous = head.compare_and_swap(&base, candidate.clone());
        let published = Arc::ptr_eq(&base, &previous);
        drop(previous);
        if published {
            return Ok(requested_values(&candidate, &requested));
        }
        // Keeping only requested handles avoids retaining rejected whole histories
        // through further work, and preserves a hit evicted by the winning update.
        drop(candidate);
        drop(base);
        tokio::task::yield_now().await;
        cancellation.check().map_err(|_| UpdateError::Cancelled)?;
        base = head.load_full();
        candidate = Arc::new(
            base.update(
                requested.iter().cloned(),
                |key| {
                    let value = completed[&key].clone();
                    async move { Ok::<_, E>(value) }
                },
                execution,
                cancellation,
            )
            .await?,
        );
    }
}

fn requested_values<K: Ord + Clone, V>(
    cache: &Cache<K, V>,
    requested: &[K],
) -> BTreeMap<K, Arc<V>> {
    requested
        .iter()
        .map(|key| {
            let value = cache
                .get(key)
                .expect("a completed cache update retains every requested value");
            (key.clone(), value.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        convert::Infallible,
        num::NonZeroUsize,
        sync::atomic::{AtomicUsize, Ordering},
        time::Instant,
    };
    use tokio::sync::{Barrier, Notify};

    struct Value(u32);

    fn execution() -> Execution {
        Execution::new(NonZeroUsize::new(2).unwrap())
    }

    async fn values(
        head: &ArcSwap<Cache<u32, Value>>,
        requested: Vec<u32>,
    ) -> BTreeMap<u32, Arc<Value>> {
        select(
            head,
            requested,
            |key| async move { Ok::<_, Infallible>(Arc::new(Value(key))) },
            &execution(),
            &Cancellation::new(),
        )
        .await
        .unwrap()
    }

    fn keys(head: &ArcSwap<Cache<u32, Value>>) -> Vec<u32> {
        head.load().iter().map(|(key, _)| *key).collect()
    }

    #[tokio::test]
    async fn competing_disjoint_misses_rebase_without_rebuilding_or_losing_contributions() {
        for capacity in [1, 2] {
            let head = ArcSwap::from_pointee(Cache::new(capacity));
            let barrier = Barrier::new(2);
            let builds = AtomicUsize::new(0);
            let execution = execution();
            let cancellation = Cancellation::new();
            let build = |key| {
                let builds = &builds;
                let barrier = &barrier;
                async move {
                    builds.fetch_add(1, Ordering::SeqCst);
                    barrier.wait().await;
                    Ok::<_, Infallible>(Arc::new(Value(key)))
                }
            };
            let (left, right) = tokio::join!(
                select(&head, vec![1], &build, &execution, &cancellation),
                select(&head, vec![2], &build, &execution, &cancellation),
            );
            assert_eq!(builds.load(Ordering::SeqCst), 2);
            assert_eq!(left.unwrap()[&1].0, 1);
            assert_eq!(right.unwrap()[&2].0, 2);
            assert_eq!(head.load().len(), capacity);
            if capacity == 2 {
                assert_eq!(keys(&head), [1, 2]);
            }
        }
    }

    #[tokio::test]
    async fn competing_equal_misses_return_the_published_handle() {
        let head = ArcSwap::from_pointee(Cache::new(1));
        let barrier = Barrier::new(2);
        let builds = AtomicUsize::new(0);
        let execution = execution();
        let cancellation = Cancellation::new();
        let build = |key| {
            let builds = &builds;
            let barrier = &barrier;
            async move {
                builds.fetch_add(1, Ordering::SeqCst);
                barrier.wait().await;
                Ok::<_, Infallible>(Arc::new(Value(key)))
            }
        };
        let (left, right) = tokio::join!(
            select(&head, vec![1, 1], &build, &execution, &cancellation),
            select(&head, vec![1], &build, &execution, &cancellation),
        );
        let left = left.unwrap();
        let right = right.unwrap();
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert!(Arc::ptr_eq(&left[&1], &right[&1]));
        assert!(Arc::ptr_eq(&left[&1], head.load().get(&1).unwrap()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_tasks_publish_sendable_selections_without_rebuilding() {
        let head = Arc::new(ArcSwap::from_pointee(Cache::new(16)));
        let barrier = Arc::new(Barrier::new(16));
        let builds = Arc::new(AtomicUsize::new(0));
        let mut requests = Vec::new();
        for key in 0..16 {
            let head = head.clone();
            let barrier = barrier.clone();
            let builds = builds.clone();
            requests.push(tokio::spawn(async move {
                select(
                    &head,
                    vec![key],
                    |key| {
                        let barrier = barrier.clone();
                        let builds = builds.clone();
                        async move {
                            builds.fetch_add(1, Ordering::SeqCst);
                            barrier.wait().await;
                            Ok::<_, Infallible>(Arc::new(Value(key)))
                        }
                    },
                    &execution(),
                    &Cancellation::new(),
                )
                .await
                .unwrap()
            }));
        }
        for (key, request) in requests.into_iter().enumerate() {
            assert_eq!(request.await.unwrap()[&(key as u32)].0, key as u32);
        }
        assert_eq!(builds.load(Ordering::SeqCst), 16);
        assert_eq!(keys(&head), (0..16).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn losing_update_keeps_hits_that_were_evicted_before_its_retry() {
        let head = ArcSwap::from_pointee(Cache::new(1));
        let initial = values(&head, vec![0]).await;
        let started = Notify::new();
        let proceed = Notify::new();
        let builds = AtomicUsize::new(0);
        let execution = execution();
        let cancellation = Cancellation::new();
        let slow = select(
            &head,
            vec![0, 1],
            |key| {
                let builds = &builds;
                let started = &started;
                let proceed = &proceed;
                async move {
                    assert_eq!(key, 1, "the evicted initial hit must never rebuild");
                    builds.fetch_add(1, Ordering::SeqCst);
                    started.notify_one();
                    proceed.notified().await;
                    Ok::<_, Infallible>(Arc::new(Value(key)))
                }
            },
            &execution,
            &cancellation,
        );
        let winner = async {
            started.notified().await;
            values(&head, vec![2]).await;
            assert_eq!(keys(&head), [2]);
            proceed.notify_one();
        };
        let (selected, ()) = tokio::join!(slow, winner);
        let selected = selected.unwrap();
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&selected[&0], &initial[&0]));
        assert_eq!(keys(&head), [0, 1]);
        values(&head, vec![2]).await;
        assert_eq!(keys(&head), [2]);
        assert_eq!(selected[&0].0, 0);
    }

    #[tokio::test]
    async fn rebasing_uses_current_recency_and_does_not_restore_evicted_history() {
        let head = ArcSwap::from_pointee(Cache::new(3));
        values(&head, vec![0, 1, 2]).await;
        let started = Notify::new();
        let proceed = Notify::new();
        let execution = execution();
        let cancellation = Cancellation::new();
        let slow = select(
            &head,
            vec![3],
            |key| {
                let started = &started;
                let proceed = &proceed;
                async move {
                    started.notify_one();
                    proceed.notified().await;
                    Ok::<_, Infallible>(Arc::new(Value(key)))
                }
            },
            &execution,
            &cancellation,
        );
        let winner = async {
            started.notified().await;
            values(&head, vec![1]).await;
            values(&head, vec![4]).await;
            assert_eq!(keys(&head), [0, 1, 4]);
            proceed.notify_one();
        };
        let (selected, ()) = tokio::join!(slow, winner);
        assert_eq!(selected.unwrap()[&3].0, 3);
        assert_eq!(keys(&head), [1, 3, 4]);
    }

    #[tokio::test]
    async fn failed_and_cancelled_selection_cannot_replace_the_head() {
        let head = ArcSwap::from_pointee(Cache::new(2));
        values(&head, vec![0]).await;
        let initial = head.load_full();
        let execution = execution();
        let cancellation = Cancellation::new();
        let failed = select(
            &head,
            vec![1, 2],
            |key| async move {
                if key == 2 {
                    Err("failed")
                } else {
                    Ok(Arc::new(Value(key)))
                }
            },
            &execution,
            &cancellation,
        )
        .await;
        assert!(matches!(
            failed,
            Err(UpdateError::Build { error: "failed" })
        ));
        assert!(Arc::ptr_eq(&initial, &head.load_full()));
        let cancelled = select(
            &head,
            vec![1],
            |key| {
                cancellation.cancel();
                async move { Ok::<_, Infallible>(Arc::new(Value(key))) }
            },
            &execution,
            &cancellation,
        )
        .await;
        assert!(matches!(cancelled, Err(UpdateError::Cancelled)));
        assert!(Arc::ptr_eq(&initial, &head.load_full()));
    }

    #[tokio::test]
    async fn evicted_values_outlive_the_head_only_while_consumers_retain_them() {
        let head = ArcSwap::from_pointee(Cache::new(1));
        let selected = values(&head, vec![0]).await;
        let value = Arc::downgrade(&selected[&0]);
        let old = head.load_full();
        let history = Arc::downgrade(&old);
        values(&head, vec![1]).await;
        drop(old);
        assert!(history.upgrade().is_none());
        assert!(value.upgrade().is_some());
        drop(selected);
        assert!(value.upgrade().is_none());
    }

    #[tokio::test]
    async fn large_hit_only_publications_record_copy_cost_without_rebuilding() {
        let head = ArcSwap::from_pointee(Cache::new(10_000));
        values(&head, (0..10_000).collect()).await;
        let execution = execution();
        let cancellation = Cancellation::new();
        let started = Instant::now();
        for _ in 0..10 {
            let selected = select(
                &head,
                (0..10_000).collect(),
                |_| async { panic!("a hit-only publication cannot build") },
                &execution,
                &cancellation,
            )
            .await
                as Result<BTreeMap<u32, Arc<Value>>, UpdateError<Infallible>>;
            assert_eq!(selected.unwrap().len(), 10_000);
        }
        println!(
            "phase2 publication: entries=10000, mean_hit_publish_us={}",
            started.elapsed().as_micros() / 10
        );
    }

    #[tokio::test]
    async fn large_map_race_records_one_rebase_without_repeating_builders() {
        let head = ArcSwap::from_pointee(Cache::new(10_002));
        values(&head, (0..10_000).collect()).await;
        let barrier = Barrier::new(2);
        let builds = AtomicUsize::new(0);
        let execution = execution();
        let cancellation = Cancellation::new();
        let build = |key| {
            let barrier = &barrier;
            let builds = &builds;
            async move {
                builds.fetch_add(1, Ordering::SeqCst);
                barrier.wait().await;
                Ok::<_, Infallible>(Arc::new(Value(key)))
            }
        };
        let left: Vec<_> = (0..10_000).chain([10_000]).collect();
        let right: Vec<_> = (0..10_000).chain([10_001]).collect();
        let started = Instant::now();
        let (left, right) = tokio::join!(
            select(&head, left, &build, &execution, &cancellation),
            select(&head, right, &build, &execution, &cancellation),
        );
        assert_eq!(left.unwrap().len(), 10_001);
        assert_eq!(right.unwrap().len(), 10_001);
        assert_eq!(head.load().len(), 10_002);
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        // The barrier gives both updates the same base. One wins and stops;
        // the other must perform exactly one uncontended rebase to retain both.
        println!(
            "phase2 publication race: initial_entries=10000, final_entries=10002, requests=2, cas_rebases=1, builds=2, elapsed_us={}",
            started.elapsed().as_micros()
        );
    }
}
