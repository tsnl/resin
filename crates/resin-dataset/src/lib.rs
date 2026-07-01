//! Host dataset loaders (MNIST and friends).

/// Placeholder for MNIST byte loading (download / path wiring in a follow-up).
pub struct MnistDataset {
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
    pub n: usize,
}

impl MnistDataset {
    /// Empty dataset used by examples when files are not present.
    pub fn empty() -> Self {
        Self {
            images: vec![],
            labels: vec![],
            n: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}
