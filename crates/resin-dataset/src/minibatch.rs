//! Session-agnostic minibatch iteration: host dataset → resident buffer slots.

use resin_core::Tree;

// `Tree` derive expands to `resin_tree` paths.
use resin_tree as _;

/// Feature/label pair for one training step (compile-time `View` or runtime buffer slots).
#[derive(Clone, Tree)]
pub struct Minibatch<T> {
    pub xs: T,
    pub ys: T,
}

/// Host dataset that can materialize one index batch as contiguous `f32` tensors.
pub trait BatchDataset {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn batch_f32(&self, indices: &[usize]) -> (Vec<f32>, Vec<f32>);
}

/// Uploads host batch tensors into opaque resident slots (GPU buffers, views, etc.).
pub trait MinibatchWriter<B> {
    type Error;
    fn write_xs(&self, slot: &B, features: &[f32]) -> Result<(), Self::Error>;
    fn write_ys(&self, slot: &B, labels: &[f32]) -> Result<(), Self::Error>;
}

/// Iterates shuffled batches, writing each into pre-admitted [`Minibatch`] slots.
pub struct MinibatchLoader<B: Clone> {
    pub slots: Minibatch<B>,
    batches: crate::BatchIndices,
    n: usize,
    batch_size: usize,
    drop_last: bool,
}

impl<B: Clone> MinibatchLoader<B> {
    pub fn new(
        slots: Minibatch<B>,
        dataset_len: usize,
        batch_size: usize,
        seed: u64,
        drop_last: bool,
    ) -> Self {
        Self {
            slots,
            batches: crate::BatchIndices::new(dataset_len, batch_size, seed, drop_last),
            n: dataset_len,
            batch_size,
            drop_last,
        }
    }

    pub fn rewind(&mut self, seed: u64) {
        self.batches = crate::BatchIndices::new(self.n, self.batch_size, seed, self.drop_last);
    }

    pub fn next_minibatch<D, W>(
        &mut self,
        dataset: &D,
        writer: &W,
    ) -> Result<Option<Minibatch<B>>, W::Error>
    where
        D: BatchDataset,
        W: MinibatchWriter<B>,
    {
        let Some(indices) = self.batches.next_batch() else {
            return Ok(None);
        };
        let (xs, ys) = dataset.batch_f32(indices);
        writer.write_xs(&self.slots.xs, &xs)?;
        writer.write_ys(&self.slots.ys, &ys)?;
        Ok(Some(self.slots.clone()))
    }
}
