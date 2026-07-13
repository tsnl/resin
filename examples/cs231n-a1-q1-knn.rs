use std::collections::BinaryHeap;
use std::error::Error;

use resin_extras::dataset::{self, Dataset, Split, cifar10};
use resin_extras::notebook::Notebook;
use resin_extras::notebook::mosaic::{mosaic, plot_image};

//
// Entry point
//

fn main() -> Result<(), Box<dyn Error>> {
    let mut nb = Notebook::create("out/cs231n-a1-q1-knn.html", "CS231N A1 Q1: KNN")?;

    nb.h1("CS231N A1 Q1: KNN")?;
    nb.hr()?;

    // Load the dataset:
    nb.h2("Dataset Loading")?;
    let dataset = dataset::cifar10::Cifar10::load(Split::Train)?;
    nb.hr()?;

    // Visualize some examples from the dataset:
    nb.h2("Dataset Examples")?;
    visualize_dataset_examples(&dataset, &mut nb)?;
    nb.hr()?;

    // Train a naive KNN classifier and evaluate on the test split:
    nb.h2("KNN Classifier")?;
    let distance_metric = KnnDistanceMetric::L2Norm;
    let k = 1;
    nb.p("Training a naive KNN classifier on the CIFAR-10 train split...")?;
    nb.p(&format!("- distance metric: {distance_metric:?}"))?;
    nb.p(&format!("- k: {k:?}"))?;
    let mut classifier = NaiveKnnClassifier::new(distance_metric, k);
    classifier.train(&dataset);
    nb.p("Done")?;
    nb.hr()?;

    // Evaluate the classifier on the test split:
    nb.h2("Evaluation")?;
    let test_dataset = dataset::cifar10::Cifar10::load(Split::Test)?;
    let accuracy = evaluate_classifier(&classifier, &test_dataset, &mut nb)?;
    nb.p(&format!("Test accuracy (k={k}): {:.2}%", accuracy * 100.0))?;
    nb.hr()?;

    // Done
    Ok(())
}

//
// Dataset loading and visualization
//

fn visualize_dataset_examples(
    dataset: &dataset::cifar10::Cifar10,
    nb: &mut Notebook,
) -> Result<(), Box<dyn Error>> {
    const EXAMPLES_PER_CLASS: usize = 10;
    const PIXEL_SCALE: usize = 2;
    const TILE_GAP: usize = 4;

    nb.p(&format!(
        "Loaded CIFAR-10 train split: {} images ({}x{}x{}, planar CHW).",
        dataset.len(),
        cifar10::IMG_C,
        cifar10::IMG_H,
        cifar10::IMG_W,
    ))?;

    let groups = first_n_per_class(dataset, EXAMPLES_PER_CLASS);
    // Row-major tiles: each row is one example index, columns are classes.
    let mut rgb_tiles = Vec::with_capacity(EXAMPLES_PER_CLASS * cifar10::CLASSES.len());
    for row in 0..EXAMPLES_PER_CLASS {
        for class in 0..cifar10::CLASSES.len() {
            rgb_tiles.push(cifar_to_rgb(&groups[class][row]));
        }
    }
    let tile_refs: Vec<&[[u8; 3]]> = rgb_tiles.iter().map(|t| t.as_slice()).collect();
    let z = mosaic(
        &tile_refs,
        cifar10::IMG_W,
        cifar10::IMG_H,
        cifar10::CLASSES.len(),
        PIXEL_SCALE,
        TILE_GAP,
        [255, 255, 255],
    );
    nb.plot(&plot_image(z, "CIFAR-10 train examples"))?;
    nb.p(&format!(
        "Columns (left → right): {}.",
        cifar10::CLASSES.join(", ")
    ))?;

    Ok(())
}

/// First `n` images for each label in dataset order.
fn first_n_per_class(
    dataset: &dataset::cifar10::Cifar10,
    n: usize,
) -> [Vec<cifar10::Image>; cifar10::CLASSES.len()] {
    let mut groups: [Vec<cifar10::Image>; cifar10::CLASSES.len()] =
        std::array::from_fn(|_| Vec::with_capacity(n));

    for ex in dataset.values() {
        let class = ex.label as usize;
        if groups[class].len() < n {
            groups[class].push(*ex.image);
        }
        if groups.iter().all(|g| g.len() >= n) {
            break;
        }
    }
    groups
}

/// Planar CHW CIFAR image → row-major HWC RGB pixels.
fn cifar_to_rgb(img: &cifar10::Image) -> Vec<[u8; 3]> {
    let plane = cifar10::IMG_H * cifar10::IMG_W;
    (0..plane)
        .map(|i| [img[i], img[plane + i], img[2 * plane + i]])
        .collect()
}

//
// Classifier interface:
//

pub trait Classifier {
    fn train(&mut self, dataset: &dataset::cifar10::Cifar10);
    fn predict(&self, image: &[cifar10::Image]) -> Box<[u8]>;
}

/// evaluate_classifier: returns the accuracy of a trained classifier on the test split
/// of CIFAR-10.
fn evaluate_classifier<C: Classifier>(
    c: &C,
    test_dataset: &dataset::cifar10::Cifar10,
    nb: &mut Notebook,
) -> Result<f32, Box<dyn Error>> {
    const PROGRESS_INTERVAL: usize = 100;

    // TODO: batching
    let mut correct = 0;
    for (index, test_image) in test_dataset.values().enumerate() {
        let predicted_label = c.predict(&[*test_image.image])[0];
        let correct_label = test_image.label;
        if predicted_label == correct_label {
            correct += 1;
        }
        if index % PROGRESS_INTERVAL == 0 {
            let count = test_dataset.len();
            let accuracy_pc = 100.0 * correct as f32 / (index + 1) as f32;
            nb.p(&format!("[{index}/{count}]: {accuracy_pc:.2}%",))?;
        }
    }
    Ok(correct as f32 / test_dataset.len() as f32)
}

//
// KNN Classifier
//

#[derive(Debug, Clone, Copy)]
pub enum KnnDistanceMetric {
    L1Norm,
    L2Norm,
}

pub struct NaiveKnnClassifier {
    images: Vec<cifar10::Image>,
    labels: Vec<u8>,
    distance_metric: KnnDistanceMetric,
    k: usize,
}

impl NaiveKnnClassifier {
    pub fn new(distance_metric: KnnDistanceMetric, k: usize) -> Self {
        Self {
            images: Vec::new(),
            labels: Vec::new(),
            distance_metric,
            k,
        }
    }
}
impl Classifier for NaiveKnnClassifier {
    fn train(&mut self, dataset: &dataset::cifar10::Cifar10) {
        self.images.clear();
        self.labels.clear();

        for ex in dataset.values() {
            self.images.push(*ex.image);
            self.labels.push(ex.label);
        }
    }
    fn predict(&self, test_images: &[cifar10::Image]) -> Box<[u8]> {
        // DistanceFunction: lets us select the image-image distance function used
        // outside the hot inner loops.
        type DistanceFunction = fn(im1: &cifar10::Image, im2: &cifar10::Image) -> f32;
        fn distance_l1(im1: &cifar10::Image, im2: &cifar10::Image) -> f32 {
            let mut acc = 0.0;
            for dim in 0..cifar10::IMG_CHW {
                let f1 = im1[dim] as f32;
                let f2 = im2[dim] as f32;
                acc += (f1 - f2).abs();
            }
            acc
        }
        fn distance_l2(im1: &cifar10::Image, im2: &cifar10::Image) -> f32 {
            let mut acc = 0.0;
            for dim in 0..cifar10::IMG_CHW {
                let f1 = im1[dim] as f32;
                let f2 = im2[dim] as f32;
                let dd = f1 - f2;
                acc += dd * dd;
            }
            acc
        }

        /// knn: given a train set and a single test image, returns the indices of the
        /// 'k' nearest neighbors in the train set.
        fn knn(
            train_images: &[cifar10::Image],
            test_image: &cifar10::Image,
            distance_fn: DistanceFunction,
            k: usize,
        ) -> Box<[usize]> {
            #[derive(PartialEq)]
            struct HeapItem {
                distance: f32,
                train_index: usize,
            }
            impl Eq for HeapItem {}
            impl PartialOrd for HeapItem {
                fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                    self.distance.partial_cmp(&other.distance)
                }
            }
            impl Ord for HeapItem {
                fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                    self.distance.total_cmp(&other.distance)
                }
            }

            // Iterate over each train sample, retain the indices of the 'k' train
            // elements with the minimum distance to the test example:
            let mut max_heap = BinaryHeap::with_capacity(k + 1);
            for (train_index, train_image) in train_images.iter().enumerate() {
                // Compute image-image distance:
                let distance = distance_fn(train_image, test_image);
                assert!(distance.is_finite());

                // Insert into heap:
                let pq_item = HeapItem {
                    distance,
                    train_index,
                };
                max_heap.push(pq_item);

                // Pop elements from the max heap until upto k elements remain.
                // This lets us retain the 'k' elements with MIN distance.
                while max_heap.len() > k {
                    _ = max_heap.pop();
                }
            }

            // Return the 'k' nearest neighbors' indices:
            max_heap.into_iter().map(|x| x.train_index).collect()
        }

        /// tally_votes: given knn output, emit the most common label among the 'k'
        /// nearest neighbors.
        fn tally_votes(train_labels: &[u8], knn_indices: &[usize]) -> u8 {
            let mut counts = [0; cifar10::CLASSES.len()];
            for &train_index in knn_indices {
                counts[train_labels[train_index] as usize] += 1;
            }
            let (max_label, _) = counts
                .iter()
                .enumerate()
                .max_by_key(|&(_label, &count)| count)
                .expect("at least one label");
            max_label as u8
        }

        // Choose a distance function, resolve this once outside the hot inner loops:
        let distance_function = {
            match self.distance_metric {
                KnnDistanceMetric::L1Norm => distance_l1,
                KnnDistanceMetric::L2Norm => distance_l2,
            }
        };

        // Putting it all together:
        let mut result = Vec::with_capacity(test_images.len());
        for test_image in test_images {
            let knn_indices = knn(&self.images, test_image, distance_function, self.k);
            let predicted_label = tally_votes(&self.labels, &knn_indices);
            result.push(predicted_label);
        }
        result.into_boxed_slice()
    }
}
