use std::error::Error;

use resin_extras::dataset::{self, Dataset, Split};
use resin_extras::notebook::Notebook;

const CLASSES: &'static [&'static str; 10] = &[
    "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
];

fn main() -> Result<(), Box<dyn Error>> {
    let mut nb = Notebook::create("out/cs231n-a1-q1-knn.html", "CS231N A1 Q1: KNN")?;

    headline(&mut nb)?;

    // let dataset = load_dataset(&mut nb)?;

    Ok(())
}

fn headline(nb: &mut Notebook) -> Result<(), Box<dyn Error>> {
    nb.h1("CS231N A1 Q1: KNN")?;
    nb.hr()?;
    Ok(())
}

// fn load_dataset(nb: &mut Notebook) -> Result<dataset::cifar10::Cifar10, Box<dyn Error>> {
//     const NUM_CLASSES: usize = dataset::cifar10::NUM_CLASSES;
//     const EXAMPLES_PER_CLASS: usize = 10;

//     nb.h2("Load dataset, and show a few examples")?;
//     let dataset = {
//         let dataset = dataset::cifar10::Cifar10::load(Split::Train)?;

//         let layout = plotly::Layout::new()
//             .title("CIFAR-10 Train Set Examples")
//             .grid(
//                 plotly::layout::LayoutGrid::new()
//                     .rows(10)
//                     .columns(EXAMPLES_PER_CLASS)
//                     .pattern(plotly::layout::GridPattern::Independent)
//                     .row_order(plotly::layout::RowOrder::TopToBottom),
//             );
//         let mut plot = plotly::Plot::new();
//         plot.set_layout(layout);
//         plot.add_trace(plotly::Image::new());

//         dataset
//     };

//     Ok(dataset)
// }

// fn group_dataset_images_by_class<const NUM_CLASSES: usize>(
//     dataset: &dataset::cifar10::Cifar10,
//     examples_per_class: usize,
// ) -> [Vec<u8>; NUM_CLASSES] {
//     let mut groups_by_class: [Vec<u8>; NUM_CLASSES] = [const { Vec::default() }; NUM_CLASSES];
//     for class in 0..NUM_CLASSES {
//         let mut count = 0;
//         for (i, label) in dataset.labels.iter().enumerate() {
//             if *label == class as u8 {
//                 groups_by_class[class]
//                     .push(dataset.images[i..i + dataset::cifar10::IMG_CHW].to_vec());
//                 count += 1;
//                 if count >= examples_per_class {
//                     break;
//                 }
//             }
//         }
//     }
//     groups_by_class
// }
