//! Host dataset loaders (MNIST and friends).

pub mod cifar10;
pub mod mnist;

pub use cifar10::Cifar10Dataset;
pub use mnist::{IMG_H, IMG_W, IMG_WH, MnistDataset, NUM_CLS};

/// Shuffled batch index iterator (Fisher–Yates with LCG).
pub struct BatchIndices {
    order: Vec<usize>,
    n: usize,
    batch_size: usize,
    pos: usize,
    drop_last: bool,
}

impl BatchIndices {
    pub fn new(n: usize, batch_size: usize, seed: u64, drop_last: bool) -> Self {
        let mut order: Vec<usize> = (0..n).collect();
        let mut state = seed;
        for i in (1..n).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (state as usize) % (i + 1);
            order.swap(i, j);
        }
        if drop_last {
            let usable = (n / batch_size) * batch_size;
            order.truncate(usable);
        }
        Self {
            order,
            n,
            batch_size,
            pos: 0,
            drop_last,
        }
    }

    pub fn rewind(&mut self, seed: u64) {
        *self = Self::new(self.n, self.batch_size, seed, self.drop_last);
    }

    pub fn next_batch(&mut self) -> Option<&[usize]> {
        if self.pos + self.batch_size > self.order.len() {
            return None;
        }
        let start = self.pos;
        self.pos += self.batch_size;
        Some(&self.order[start..self.pos])
    }
}
