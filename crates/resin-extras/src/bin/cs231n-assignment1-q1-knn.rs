//! CS231n Assignment 1, Q1: k-Nearest Neighbor on CIFAR-10.
//!
//! Port of `docs/cs231n/assignment1/knn.ipynb`. Streams a self-contained HTML
//! notebook (Plotly + sample images) as it runs.
//!
//! ```text
//! # once: cargo install live-server
//! # terminal A — serve + reload on file change:
//! live-server --path . --open out/cs231n-assignment1-q1-knn.html
//!
//! # terminal B — stream the notebook:
//! cargo run -p resin-extras --bin cs231n-assignment1-q1-knn -- --quick
//! cargo run -p resin-extras --bin cs231n-assignment1-q1-knn
//! ```

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use plotly::common::{ColorScale, ColorScalePalette, Mode, Title};
use plotly::layout::Layout;
use plotly::{HeatMap, Plot, Scatter};
use resin_extras::dataset::cifar10::{Cifar10Dataset, IMG_C, IMG_CHW, IMG_H, IMG_W, NUM_CLS};
use resin_extras::notebook::Notebook;

const CLASSES: [&str; NUM_CLS] = [
    "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
];

struct Config {
    num_train: usize,
    num_test: usize,
    samples_per_class: usize,
    k_choices: Vec<usize>,
    num_folds: usize,
    out: PathBuf,
    cache: PathBuf,
}

fn main() {
    let config = parse_args();
    if let Err(e) = run(config) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(config: Config) -> Result<(), String> {
    let mut nb = Notebook::create(&config.out, "CS231n A1 Q1: k-NN")?;
    eprintln!("notebook → {}", nb.path().display());

    nb.h1("k-Nearest Neighbor (kNN) on CIFAR-10")?;
    nb.p(
        "Rust port of CS231n assignment 1, question 1. Host-side k-NN (no resin \
         compiler). Figures are interactive Plotly. View with \
         `live-server` (`cargo install live-server`) for reload-on-write.",
    )?;

    // --- load ---
    nb.h2("Load CIFAR-10")?;
    nb.p(&format!(
        "Loading train/test into cache dir `{}` (downloads on first run)…",
        config.cache.display()
    ))?;
    let t0 = Instant::now();
    let train = Cifar10Dataset::load("train", &config.cache)?;
    let test = Cifar10Dataset::load("test", &config.cache)?;
    nb.p(&format!(
        "Loaded train={} test={} in {:.1}s.",
        train.n,
        test.n,
        t0.elapsed().as_secs_f32()
    ))?;

    // --- visualize class samples ---
    nb.h2("Training examples by class")?;
    nb.p(&format!(
        "A few training images from each class (rows: {}). Planar CHW → RGB montage.",
        CLASSES.join(", ")
    ))?;
    let montage = class_montage(&train, config.samples_per_class);
    nb.image("CIFAR-10 class samples", &montage)?;

    // --- subsample ---
    let num_train = config.num_train.min(train.n);
    let num_test = config.num_test.min(test.n);
    nb.h2("Subsample")?;
    nb.p(&format!(
        "Using first {num_train} train and {num_test} test images (assignment defaults: 5000 / 500). \
         Pixels are flattened float32 vectors of length {IMG_CHW}."
    ))?;

    let x_train = images_f32(&train, num_train);
    let y_train = train.labels[..num_train].to_vec();
    let x_test = images_f32(&test, num_test);
    let y_test = test.labels[..num_test].to_vec();

    // --- train (memorize) + distances ---
    nb.h2("Distance matrix (L2)")?;
    nb.p(
        "k-NN training is just storing the data. We compute all pairwise L2 \
         distances with the vectorized identity \
         ‖a−b‖² = ‖a‖² + ‖b‖² − 2 a·b (no explicit loops over the feature axis in the API sense; \
         one fused multiply-add loop in Rust).",
    )?;

    let knn = Knn::train(x_train, y_train, IMG_CHW);
    let t0 = Instant::now();
    let dists = knn.l2_distances(&x_test);
    let dist_secs = t0.elapsed().as_secs_f32();
    nb.p(&format!(
        "Computed {num_test}×{num_train} distances in {dist_secs:.2}s."
    ))?;

    // Downsample for the heatmap so the HTML stays reasonable.
    let heat = downsample_dists(&dists, num_test, num_train, 100, 100);
    let mut plot = Plot::new();
    plot.add_trace(
        HeatMap::new_z(heat)
            .color_scale(ColorScale::Palette(ColorScalePalette::Greys))
            .reverse_scale(true),
    );
    plot.set_layout(
        Layout::new()
            .title(Title::from("L2 distances (test × train, downsampled)"))
            .height(500),
    );
    nb.plot(&plot)?;
    nb.p(
        "Darker ≈ nearer. Structured stripes often come from duplicate-ish images \
         or background-dominated samples.",
    )?;

    // --- predict k=1, k=5 ---
    nb.h2("Predict")?;
    for k in [1usize, 5] {
        let pred = knn.predict_labels(&dists, num_test, k);
        let acc = accuracy(&pred, &y_test);
        nb.p(&format!(
            "k = {k}: accuracy = {acc:.2}%  (assignment expects ~27% for k=1, a bit higher for k=5)."
        ))?;
    }

    // --- cross-validation ---
    nb.h2("Cross-validation over k")?;
    nb.p(&format!(
        "{}-fold CV on the training subsample; k ∈ {:?}.",
        config.num_folds, config.k_choices
    ))?;

    let t0 = Instant::now();
    let (means, stds) = cross_validate(
        &knn.x,
        &knn.y,
        knn.d,
        config.num_folds,
        &config.k_choices,
        |msg| {
            eprintln!("{msg}");
        },
    );
    nb.p(&format!(
        "Cross-validation finished in {:.1}s.",
        t0.elapsed().as_secs_f32()
    ))?;

    let mut plot = Plot::new();
    let xs: Vec<f64> = config.k_choices.iter().map(|&k| k as f64).collect();
    let ys: Vec<f64> = means.iter().map(|m| m * 100.0).collect();
    let err: Vec<f64> = stds.iter().map(|s| s * 100.0).collect();
    plot.add_trace(
        Scatter::new(xs.clone(), ys)
            .mode(Mode::LinesMarkers)
            .name("mean accuracy")
            .error_y(plotly::common::ErrorData::new(plotly::common::ErrorType::Data).array(err)),
    );
    plot.set_layout(
        Layout::new()
            .title(Title::from("CV accuracy vs k"))
            .x_axis(plotly::layout::Axis::new().title(Title::from("k")))
            .y_axis(plotly::layout::Axis::new().title(Title::from("accuracy %")))
            .height(420),
    );
    nb.plot(&plot)?;

    let best_i = means
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let best_k = config.k_choices[best_i];
    nb.p(&format!(
        "Best k by mean CV accuracy: {best_k} ({:.2}%).",
        means[best_i] * 100.0
    ))?;

    let pred = knn.predict_labels(&dists, num_test, best_k);
    let acc = accuracy(&pred, &y_test);
    nb.p(&format!(
        "Retrain is free (already memorized). Test accuracy at k={best_k}: {acc:.2}% \
         (assignment hopes for >28%)."
    ))?;

    nb.h2("Done")?;
    nb.p(&format!(
        "View with: `live-server --path . --open {}`",
        nb.path().display()
    ))?;
    eprintln!("done → {}", nb.path().display());
    Ok(())
}

// --- k-NN -------------------------------------------------------------------

struct Knn {
    x: Vec<f32>,
    y: Vec<u8>,
    n: usize,
    d: usize,
}

impl Knn {
    fn train(x: Vec<f32>, y: Vec<u8>, d: usize) -> Self {
        let n = y.len();
        assert_eq!(x.len(), n * d);
        Self { x, y, n, d }
    }

    /// Row-major distances: `dists[i * n_train + j] = L2(test_i, train_j)`.
    fn l2_distances(&self, x_test: &[f32]) -> Vec<f32> {
        let n_test = x_test.len() / self.d;
        let n_train = self.n;
        let d = self.d;

        let mut test_sq = vec![0f32; n_test];
        for i in 0..n_test {
            let mut s = 0f32;
            let base = i * d;
            for k in 0..d {
                let v = x_test[base + k];
                s += v * v;
            }
            test_sq[i] = s;
        }
        let mut train_sq = vec![0f32; n_train];
        for j in 0..n_train {
            let mut s = 0f32;
            let base = j * d;
            for k in 0..d {
                let v = self.x[base + k];
                s += v * v;
            }
            train_sq[j] = s;
        }

        let mut dists = vec![0f32; n_test * n_train];
        for i in 0..n_test {
            let tb = i * d;
            let row = i * n_train;
            for j in 0..n_train {
                let ub = j * d;
                let mut dot = 0f32;
                for k in 0..d {
                    dot += x_test[tb + k] * self.x[ub + k];
                }
                let v = test_sq[i] + train_sq[j] - 2.0 * dot;
                dists[row + j] = v.max(0.0).sqrt();
            }
        }
        dists
    }

    fn predict_labels(&self, dists: &[f32], n_test: usize, k: usize) -> Vec<u8> {
        let n_train = self.n;
        let k = k.min(n_train);
        let mut pred = vec![0u8; n_test];
        let mut idx: Vec<usize> = (0..n_train).collect();
        for i in 0..n_test {
            let row = i * n_train;
            idx.sort_by(|&a, &b| dists[row + a].total_cmp(&dists[row + b]));
            // Majority vote; ties → smaller label.
            let mut votes = [0u32; NUM_CLS];
            for &j in idx.iter().take(k) {
                votes[self.y[j] as usize] += 1;
            }
            let mut best_c = 0usize;
            let mut best_v = 0u32;
            for (c, &v) in votes.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_c = c;
                }
            }
            pred[i] = best_c as u8;
        }
        pred
    }
}

fn accuracy(pred: &[u8], y: &[u8]) -> f32 {
    let n = pred.len().max(1) as f32;
    let correct = pred.iter().zip(y).filter(|(a, b)| a == b).count() as f32;
    100.0 * correct / n
}

fn cross_validate(
    x: &[f32],
    y: &[u8],
    d: usize,
    num_folds: usize,
    k_choices: &[usize],
    log: impl Fn(&str),
) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let fold_size = n / num_folds;
    let mut accs: Vec<Vec<f64>> = k_choices.iter().map(|_| Vec::new()).collect();

    for fold in 0..num_folds {
        let val_lo = fold * fold_size;
        let val_hi = if fold + 1 == num_folds {
            n
        } else {
            (fold + 1) * fold_size
        };
        log(&format!(
            "  fold {}/{} val=[{val_lo}, {val_hi})",
            fold + 1,
            num_folds
        ));

        let mut x_tr = Vec::new();
        let mut y_tr = Vec::new();
        let mut x_va = Vec::new();
        let mut y_va = Vec::new();
        for i in 0..n {
            let row = &x[i * d..(i + 1) * d];
            if i >= val_lo && i < val_hi {
                x_va.extend_from_slice(row);
                y_va.push(y[i]);
            } else {
                x_tr.extend_from_slice(row);
                y_tr.push(y[i]);
            }
        }
        let model = Knn::train(x_tr, y_tr, d);
        let dists = model.l2_distances(&x_va);
        let n_va = y_va.len();
        for (ki, &k) in k_choices.iter().enumerate() {
            let pred = model.predict_labels(&dists, n_va, k);
            accs[ki].push(accuracy(&pred, &y_va) as f64 / 100.0);
        }
    }

    let mut means = Vec::with_capacity(k_choices.len());
    let mut stds = Vec::with_capacity(k_choices.len());
    for a in &accs {
        let m = a.iter().sum::<f64>() / a.len().max(1) as f64;
        let var = a.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / a.len().max(1) as f64;
        means.push(m);
        stds.push(var.sqrt());
    }
    (means, stds)
}

// --- data helpers -----------------------------------------------------------

fn images_f32(ds: &Cifar10Dataset, n: usize) -> Vec<f32> {
    ds.images[..n * IMG_CHW].iter().map(|&p| p as f32).collect()
}

/// CHW planar sample → RGB rows for PNG / display.
fn sample_rgb(ds: &Cifar10Dataset, idx: usize) -> Vec<u8> {
    let base = idx * IMG_CHW;
    let mut rgb = vec![0u8; IMG_W * IMG_H * 3];
    for y in 0..IMG_H {
        for x in 0..IMG_W {
            let i = y * IMG_W + x;
            let o = i * 3;
            rgb[o] = ds.images[base + i]; // R plane
            rgb[o + 1] = ds.images[base + IMG_W * IMG_H + i];
            rgb[o + 2] = ds.images[base + 2 * IMG_W * IMG_H + i];
        }
    }
    let _ = IMG_C;
    rgb
}

fn class_montage(ds: &Cifar10Dataset, per_class: usize) -> Vec<u8> {
    let pad = 2usize;
    let cell = IMG_W + pad;
    let w = per_class * cell + pad;
    let h = NUM_CLS * cell + pad;
    let mut rgb = vec![32u8; w * h * 3]; // dark grey background

    let mut counts = [0usize; NUM_CLS];
    for i in 0..ds.n {
        let c = ds.labels[i] as usize;
        if counts[c] >= per_class {
            continue;
        }
        let col = counts[c];
        counts[c] += 1;
        let x0 = pad + col * cell;
        let y0 = pad + c * cell;
        let img = sample_rgb(ds, i);
        for y in 0..IMG_H {
            for x in 0..IMG_W {
                let si = (y * IMG_W + x) * 3;
                let di = ((y0 + y) * w + (x0 + x)) * 3;
                rgb[di..di + 3].copy_from_slice(&img[si..si + 3]);
            }
        }
        if counts.iter().all(|&n| n >= per_class) {
            break;
        }
    }
    encode_png_rgb(w as u32, h as u32, &rgb)
}

fn downsample_dists(
    dists: &[f32],
    n_test: usize,
    n_train: usize,
    max_rows: usize,
    max_cols: usize,
) -> Vec<Vec<f64>> {
    let rows = n_test.min(max_rows);
    let cols = n_train.min(max_cols);
    let mut z = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        let src_i = i * n_test / rows;
        for j in 0..cols {
            let src_j = j * n_train / cols;
            row.push(dists[src_i * n_train + src_j] as f64);
        }
        z.push(row);
    }
    z
}

// --- minimal RGB8 PNG (uncompressed) ----------------------------------------

fn encode_png_rgb(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    assert_eq!(rgb.len(), (width * height * 3) as usize);
    let mut raw = Vec::with_capacity(((width * 3 + 1) * height) as usize);
    for y in 0..height as usize {
        raw.push(0); // filter: None
        let s = y * width as usize * 3;
        raw.extend_from_slice(&rgb[s..s + width as usize * 3]);
    }
    let mut out = Vec::new();
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]); // signature
    write_chunk(&mut out, b"IHDR", &{
        let mut d = Vec::new();
        d.extend_from_slice(&width.to_be_bytes());
        d.extend_from_slice(&height.to_be_bytes());
        d.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
        d
    });
    write_chunk(&mut out, b"IDAT", &zlib_store(&raw));
    write_chunk(&mut out, b"IEND", &[]);
    out
}

fn write_chunk(out: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(ty);
    out.extend_from_slice(data);
    let mut c = Crc32::new();
    c.update(ty);
    c.update(data);
    out.extend_from_slice(&c.finish().to_be_bytes());
}

/// zlib wrapper around a single uncompressed deflate block.
fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // zlib CM=8, FLEVEL=0
    let mut i = 0;
    while i < data.len() {
        let end = (i + 65535).min(data.len());
        let chunk = &data[i..end];
        let last = end == data.len();
        out.push(if last { 0x01 } else { 0x00 });
        let n = chunk.len() as u16;
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&(!n).to_le_bytes());
        out.extend_from_slice(chunk);
        i = end;
    }
    let mut adler = Adler32::new();
    adler.update(data);
    out.extend_from_slice(&adler.finish().to_be_bytes());
    out
}

struct Crc32 {
    n: u32,
}

impl Crc32 {
    fn new() -> Self {
        Self { n: 0xffff_ffff }
    }
    fn update(&mut self, data: &[u8]) {
        for &b in data {
            self.n ^= b as u32;
            for _ in 0..8 {
                let mask = (self.n & 1).wrapping_neg();
                self.n = (self.n >> 1) ^ (0xEDB88320 & mask);
            }
        }
    }
    fn finish(self) -> u32 {
        !self.n
    }
}

struct Adler32 {
    a: u32,
    b: u32,
}

impl Adler32 {
    fn new() -> Self {
        Self { a: 1, b: 0 }
    }
    fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.a = (self.a + byte as u32) % 65521;
            self.b = (self.b + self.a) % 65521;
        }
    }
    fn finish(self) -> u32 {
        (self.b << 16) | self.a
    }
}

// --- CLI --------------------------------------------------------------------

fn parse_args() -> Config {
    let mut quick = false;
    let mut out = PathBuf::from("out/cs231n-assignment1-q1-knn.html");
    let mut cache = PathBuf::from("data/cifar10");
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--quick" => quick = true,
            "--out" => {
                i += 1;
                out = PathBuf::from(args.get(i).expect("--out needs a path"));
            }
            "--cache" => {
                i += 1;
                cache = PathBuf::from(args.get(i).expect("--cache needs a path"));
            }
            "-h" | "--help" => {
                eprintln!("Usage: cs231n-assignment1-q1-knn [--quick] [--out PATH] [--cache DIR]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    if quick {
        Config {
            num_train: 500,
            num_test: 100,
            samples_per_class: 5,
            k_choices: vec![1, 3, 5, 10, 20],
            num_folds: 3,
            out,
            cache,
        }
    } else {
        Config {
            num_train: 5000,
            num_test: 500,
            samples_per_class: 7,
            k_choices: vec![1, 3, 5, 8, 10, 12, 15, 20, 50, 100],
            num_folds: 5,
            out,
            cache,
        }
    }
}
