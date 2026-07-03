//! Host utilities (datasets and related helpers).

pub mod dataset;

pub use dataset::{
    BatchDataset, BatchIndices, Minibatch, MinibatchLoader, MinibatchWriter, MnistDataset, IMG_WH,
    NUM_CLS,
};
