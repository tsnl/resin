//! Immutable cache generations with bounded asynchronous construction.
//!
//! An update retains every requested result and fills spare capacity with recently
//! used old results. Applications own publication; this crate knows no compiler
//! phases, global cache pointers, timers, or background tasks.

use futures::{StreamExt, stream};
use resin_executor::{Cancellation, Execution};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    future::Future,
    sync::Arc,
};

//
// Completed cache generations
//

/// A completed immutable mapping. Capacity counts entries, regardless of size.
///
/// Keys describe every semantic input to their values. The miss builder may use
/// predecessors as optimization hints, but equivalent keys must produce equivalent
/// results. Values and retained keys must not retain a chain of prior caches.
#[derive(Debug)]
pub struct Cache<K, V> {
    capacity: usize,
    generation: u64,
    entries: BTreeMap<K, Entry<V>>,
}

#[derive(Debug)]
struct Entry<V> {
    value: Arc<V>,
    used: u64,
}

impl<V> Clone for Entry<V> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            used: self.used,
        }
    }
}

impl<K: Clone, V> Clone for Cache<K, V> {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            generation: self.generation,
            entries: self.entries.clone(),
        }
    }
}

impl<K, V> Cache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            generation: 0,
            entries: BTreeMap::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Reading alone does not refresh recency; request the key in `update()`.
    pub fn get(&self, key: &K) -> Option<&Arc<V>>
    where
        K: Ord,
    {
        self.entries.get(key).map(|entry| &entry.value)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &Arc<V>)> {
        self.entries.iter().map(|(key, entry)| (key, &entry.value))
    }
}

impl<K: Ord + Clone + Send + Sync, V: Send + Sync> Cache<K, V> {
    /// Build an independent successor, preserving this cache and all its values.
    ///
    /// Each missing key is built once per call. At most `execution.jobs()` builder
    /// futures run concurrently; orchestration itself holds no CPU permit, so
    /// builders can use the same execution pool. Completion order never determines
    /// eviction: equally recent entries are retained in ascending key order.
    /// Builders return shared handles so a publication retry can return previously
    /// completed values without cloning their contents or repeating construction.
    ///
    /// Overflow retains all requested keys and logs a warning. Cancellation or a
    /// builder error returns no successor; it drops unfinished builder futures.
    /// Started CPU work keeps its permits until it exits. Applications cancel its
    /// token and await `Execution::wait_idle()` when shutting down.
    pub async fn update<F, Fut, E>(
        &self,
        requested: impl IntoIterator<Item = K>,
        build: F,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<Self, UpdateError<E>>
    where
        F: Fn(K) -> Fut + Sync,
        Fut: Future<Output = Result<Arc<V>, E>> + Send,
    {
        cancellation.check().map_err(|_| UpdateError::Cancelled)?;
        let requested: BTreeSet<K> = requested.into_iter().collect();
        let values = self
            .build_requested(&requested, &build, execution, cancellation)
            .await?;
        cancellation.check().map_err(|_| UpdateError::Cancelled)?;
        Ok(self.successor(values))
    }

    async fn build_requested<F, Fut, E>(
        &self,
        requested: &BTreeSet<K>,
        build: &F,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<BTreeMap<K, Arc<V>>, UpdateError<E>>
    where
        F: Fn(K) -> Fut + Sync,
        Fut: Future<Output = Result<Arc<V>, E>> + Send,
    {
        let mut values = self.hits(requested);
        let missing: Vec<_> = requested
            .iter()
            .filter(|key| !self.entries.contains_key(*key))
            .cloned()
            .collect();
        let mut pending = stream::iter(missing)
            .map(|key| async move {
                let value = build(key.clone())
                    .await
                    .map_err(|error| UpdateError::Build { error })?;
                Ok((key, value))
            })
            .buffer_unordered(execution.jobs());
        loop {
            let completed = tokio::select! {
                biased;
                _ = cancellation.cancelled() => return Err(UpdateError::Cancelled),
                completed = pending.next() => completed,
            };
            let Some(completed) = completed else {
                return Ok(values);
            };
            let (key, value) = completed?;
            values.insert(key, value);
        }
    }

    fn hits(&self, requested: &BTreeSet<K>) -> BTreeMap<K, Arc<V>> {
        requested
            .iter()
            .filter_map(|key| self.get(key).map(|value| (key.clone(), value.clone())))
            .collect()
    }

    fn successor(&self, values: BTreeMap<K, Arc<V>>) -> Self {
        let generation = self
            .generation
            .checked_add(1)
            .expect("cache generation exhausted");
        let requested = values.len();
        let mut entries: BTreeMap<K, Entry<V>> = values
            .into_iter()
            .map(|(key, value)| {
                (
                    key,
                    Entry {
                        value,
                        used: generation,
                    },
                )
            })
            .collect();
        let mut retained: Vec<_> = self
            .entries
            .iter()
            .filter(|(key, _)| !entries.contains_key(*key))
            .collect();
        retained.sort_by(|(left_key, left), (right_key, right)| {
            right
                .used
                .cmp(&left.used)
                .then_with(|| left_key.cmp(right_key))
        });
        for (key, entry) in retained
            .into_iter()
            .take(self.capacity.saturating_sub(requested))
        {
            entries.insert(key.clone(), entry.clone());
        }
        if requested > self.capacity {
            log::warn!(target: "resin_cache", "cache capacity exceeded: capacity={}, requested={}, retained={}", self.capacity, requested, entries.len());
        }
        Self {
            capacity: self.capacity,
            generation,
            entries,
        }
    }
}

//
// Failed updates never produce partial cache generations
//

#[derive(Debug)]
pub enum UpdateError<E> {
    Cancelled,
    Build { error: E },
}

impl<E: fmt::Display> fmt::Display for UpdateError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("cache update cancelled"),
            Self::Build { error } => write!(formatter, "cache entry failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for UpdateError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cancelled => None,
            Self::Build { error } => Some(error),
        }
    }
}
