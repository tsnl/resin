//! Samplers: generate dataset keys for random sampling, shuffling, and augmentation.
//!
//! Unlike a dataset, may be non-deterministic and stateful. Samplers can shuffle,
//! augment, or drop samples.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::dataset::Dataset;

/// Produces batches of keys for one training epoch.
pub trait Sampler {
    type Key;

    /// Next batch of keys, or `None` when the epoch is exhausted.
    fn next_batch_keys(&mut self) -> Option<&[Self::Key]>;

    /// Start a new epoch: reseed the RNG and reshuffle / reset position.
    fn reset(&mut self, seed: u64);

    /// Convenience helper: call `next_batch_keys()` and then `dataset.get(key)` for
    /// each key.
    fn next_batch<'d, D: Dataset<Key = Self::Key>>(
        &mut self,
        dataset: &'d D,
    ) -> Option<Vec<D::Val<'d>>> {
        self.next_batch_keys()
            .map(|keys| keys.iter().map(|k| dataset.get(k)).collect())
    }
}

/// Shuffle sampler over plain row indices (`Key = usize`).
///
/// For richer keys (index + crop/flip/…), write another [`Sampler`] that draws
/// those parameters from its own RNG and leaves the dataset deterministic.
pub struct IndexSampler {
    order: Vec<usize>,
    n: usize,
    batch_size: usize,
    pos: usize,
    drop_last: bool,
    rng: StdRng,
}

impl IndexSampler {
    pub fn new(n: usize, batch_size: usize, seed: u64, drop_last: bool) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let order = shuffle_order(n, batch_size, drop_last, &mut rng);
        Self {
            order,
            n,
            batch_size,
            pos: 0,
            drop_last,
            rng,
        }
    }

    /// Borrow the RNG so a caller can draw aug parameters into richer keys.
    pub fn rng_mut(&mut self) -> &mut StdRng {
        &mut self.rng
    }
}

impl Sampler for IndexSampler {
    type Key = usize;

    fn next_batch_keys(&mut self) -> Option<&[usize]> {
        let start = self.pos;
        let end = (self.pos + self.batch_size).min(self.order.len());
        self.pos = end;
        let slice = &self.order[start..end];
        if slice.is_empty() { None } else { Some(slice) }
    }

    fn reset(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
        self.order = shuffle_order(self.n, self.batch_size, self.drop_last, &mut self.rng);
        self.pos = 0;
    }
}

fn shuffle_order(n: usize, batch_size: usize, drop_last: bool, rng: &mut StdRng) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(rng);
    if drop_last {
        let usable = (n / batch_size) * batch_size;
        order.truncate(usable);
    }
    order
}
