//! Samplers: produce batches of dataset keys for training epochs.
//!
//! Samplers own any RNG used for shuffle / augmentation parameters. Pair a
//! sampler's [`Key`](Sampler::Key) with a [`Dataset`](crate::dataset::Dataset)
//! that uses the same key type; keep [`get`](crate::dataset::Dataset::get) pure.
//!
//! Streaming counterparts (`IterSampler`) can land later next to `IterDataset`.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// Produces batches of keys for one training epoch.
pub trait Sampler {
    type Key;

    /// Next batch of keys, or `None` when the epoch is exhausted.
    fn next_batch(&mut self) -> Option<&[Self::Key]>;

    /// Start a new epoch: reseed the RNG and reshuffle / reset position.
    fn reset(&mut self, seed: u64);
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

    fn next_batch(&mut self) -> Option<&[usize]> {
        if self.pos + self.batch_size > self.order.len() {
            return None;
        }
        let start = self.pos;
        self.pos += self.batch_size;
        Some(&self.order[start..self.pos])
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
