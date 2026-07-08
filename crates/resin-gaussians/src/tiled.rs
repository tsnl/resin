//! Tiled forward renderer: bounded *work* per pixel (TODO 3.5).
//!
//! The dense renderer evaluates every gaussian at every pixel (`O(N·H·W)`
//! work); the chunked fold bounds memory but not work. Tiling restores
//! locality the way real 3DGS rasterizers do — and here it is **one pure
//! traced graph** built from the same combinators, so it stays fully
//! differentiable with zero custom kernels and zero adjoint code:
//!
//! 1. Preprocess + global depth `argsort` (as in the dense path).
//! 2. **Duplicate** each gaussian into the tiles its radius box overlaps —
//!    a fixed-capacity `[N·D]` instance array built from `iota` div/mod
//!    arithmetic and `gather_rows`.
//! 3. **Stable radix sort** of instances by tile id (only `⌈log₂(T+1)⌉`
//!    bits); depth order is preserved within each tile because the input
//!    was already depth-sorted and the sort is stable.
//! 4. Per-tile segment starts from a prefix sum of scatter-add tile counts;
//!    a capacity-`C` padded gather list `[T·C]` selects each tile's
//!    instances (overflow drops the *deepest* instances in that tile).
//! 5. Blend on `[T, C, th, tw]`: alpha from conics, transmittance via
//!    `cumprod_exclusive` along the capacity axis, weighted reductions,
//!    then a single constant-permutation gather assembles `[H, W, 3]`.
//!
//! Fixed capacities are the static-shape trade: `max_tiles_per_gaussian`
//! (D) truncates gaussians spanning more tiles, `tile_capacity` (C) drops
//! the deepest overflow per tile. With generous caps the result matches the
//! dense renderer up to the radius-box crop (the same crop real tiled
//! rasterizers apply). Per-frame adaptive caps (host readback + compile-
//! cache bucketing) are a follow-up.

use resin_dsl::sort::{argsort_f32, argsort_u32};
use resin_dsl::{cumsum, ElementType, IndexKeyElement, Tensor};

use crate::cloud::GaussianCloud;
use crate::linalg::{col, sc};
use crate::preprocess::preprocess_view;

#[derive(Debug, Clone, Copy)]
pub struct TiledConfig {
    /// Tile side in pixels; must divide both image dimensions.
    pub tile: usize,
    /// D: instance slots per gaussian (tiles beyond this are truncated,
    /// row-major over the gaussian's tile bounding box).
    pub max_tiles_per_gaussian: usize,
    /// C: instance slots per tile (the deepest overflow is dropped).
    pub tile_capacity: usize,
}

impl Default for TiledConfig {
    fn default() -> Self {
        Self {
            tile: 16,
            max_tiles_per_gaussian: 32,
            tile_capacity: 256,
        }
    }
}

/// Tiled render with the camera as graph inputs (`[4, 4]` view/proj
/// tensors). Same signature family as [`crate::render_view`]; returns an
/// `[H, W, 3]` image and is differentiable end-to-end.
pub fn render_tiled(
    cloud: &GaussianCloud<Tensor>,
    view: &Tensor,
    proj: &Tensor,
    width: usize,
    height: usize,
    cfg: &TiledConfig,
) -> Tensor {
    let (tile, d_cap, c_cap) = (cfg.tile, cfg.max_tiles_per_gaussian, cfg.tile_capacity);
    assert!(tile > 0 && d_cap > 0 && c_cap > 0, "tiled config must be positive");
    assert_eq!(width % tile, 0, "tile must divide width");
    assert_eq!(height % tile, 0, "tile must divide height");
    let (tiles_x, tiles_y) = (width / tile, height / tile);
    let t_count = tiles_x * tiles_y;

    let pre = preprocess_view(cloud, view, proj, width, height);
    let n = pre.depth.shape()[0];

    // --- 1. depth-sorted screen attributes --------------------------------
    let order = argsort_f32(&pre.depth);
    let px = pre.mean_px.gather_rows(&order);
    let py = pre.mean_py.gather_rows(&order);
    let c0 = pre.conic[0].gather_rows(&order);
    let c1 = pre.conic[1].gather_rows(&order);
    let c2 = pre.conic[2].gather_rows(&order);
    let opv = (pre.opacity * pre.valid.clone()).gather_rows(&order);
    let valid = pre.valid.gather_rows(&order);
    let color = pre.color.gather_rows(&order); // [N, 3]

    // Radius box in tile units (same 3σ-of-conic radius the culling uses).
    let radius = (c0.maximum(&c2).sqrt() * sc(3.0)).ceil();
    let tile_f = sc(tile as f32);
    let u = |v: u32, shape: &[usize]| Tensor::full_u32(shape, v);
    let x0 = ((px.clone() - radius.clone()) / tile_f.clone())
        .floor()
        .maximum(&sc(0.0))
        .cast(ElementType::U32)
        .minimum(&u(tiles_x as u32 - 1, &[n]));
    let x1 = ((px.clone() + radius.clone()) / tile_f.clone())
        .ceil()
        .maximum(&sc(1.0))
        .cast(ElementType::U32)
        .minimum(&u(tiles_x as u32, &[n]));
    let y0 = ((py.clone() - radius.clone()) / tile_f.clone())
        .floor()
        .maximum(&sc(0.0))
        .cast(ElementType::U32)
        .minimum(&u(tiles_y as u32 - 1, &[n]));
    let y1 = ((py.clone() + radius) / tile_f)
        .ceil()
        .maximum(&sc(1.0))
        .cast(ElementType::U32)
        .minimum(&u(tiles_y as u32, &[n]));
    let span_w = x1 - x0.clone();
    let span_h = y1 - y0.clone();
    // Culled gaussians produce no instances at all.
    let count = (span_w.clone() * span_h)
        .minimum(&u(d_cap as u32, &[n]))
        * valid.cast(ElementType::U32);

    // --- 2. instance expansion: [M] with M = N·D ---------------------------
    let m = n * d_cap;
    let inst = Tensor::iota(m);
    let g = inst.clone() / u(d_cap as u32, &[m]);
    let d = inst - g.clone() * u(d_cap as u32, &[m]);
    let per_inst = |t: &Tensor| t.gather_rows(&g);
    let span_w_i = per_inst(&span_w);
    let dy = d.clone() / span_w_i.clone();
    let dx = d.clone() - dy.clone() * span_w_i;
    let tile_id = (per_inst(&y0) + dy) * u(tiles_x as u32, &[m]) + (per_inst(&x0) + dx);
    let inst_ok = d.cmp_lt(&per_inst(&count)); // u32 0/1
    // Invalid instances go to a sentinel tile that sorts last and is never
    // gathered.
    let tile_key = inst_ok.select(&tile_id, &u(t_count as u32, &[m]));

    // --- 3. stable tile sort (depth order preserved within tiles) ----------
    let tile_bits = usize::BITS - t_count.leading_zeros(); // log2(T+1) rounded up
    let inst_order = argsort_u32(&tile_key, tile_bits);
    let sorted_g = g.gather_rows(&inst_order); // gaussian per sorted instance

    // --- 4. per-tile segments and the [T·C] gather list --------------------
    let ones = Tensor::full_u32(&[m], 1);
    let counts_all = ones.scatter_rows(&tile_key, t_count + 1, resin_dsl::ScatterOp::Add);
    let counts = counts_all.index(&[IndexKeyElement::Slice(0..t_count)]); // [T]
    let starts = cumsum(&counts, 0) - counts.clone();
    let counts_capped = counts.minimum(&u(c_cap as u32, &[t_count]));

    let tc = t_count * c_cap;
    let slot = Tensor::iota(tc);
    let tt = slot.clone() / u(c_cap as u32, &[tc]);
    let cc = slot - tt.clone() * u(c_cap as u32, &[tc]);
    let pos = starts.gather_rows(&tt) + cc.clone();
    let in_list = cc.cmp_lt(&counts_capped.gather_rows(&tt)); // u32 0/1
    // Out-of-range positions clamp inside gather_rows; their contribution is
    // zeroed by the mask below.
    let gsel = sorted_g.gather_rows(&pos); // [T·C] gaussian ids

    // --- 5. blend on [T, C, th, tw] ----------------------------------------
    let list_mask = in_list.cast(ElementType::F32);
    let attr = |t: &Tensor| t.gather_rows(&gsel).reshape(&[t_count, c_cap]); // [T, C]
    let shape4 = [t_count, c_cap, tile, tile];
    let per_i = |t: &Tensor| t.broadcast_to(&shape4, &[0, 1]);
    let per_px = |t: &Tensor| t.broadcast_to(&shape4, &[0, 2, 3]);

    let (gx, gy) = tile_pixel_grids(tiles_x, tiles_y, tile);
    let dx_p = per_px(&gx) - per_i(&attr(&px));
    let dy_p = per_px(&gy) - per_i(&attr(&py));
    let power = (dx_p.clone() * dx_p.clone() * per_i(&attr(&c0))
        + dy_p.clone() * dy_p.clone() * per_i(&attr(&c2)))
        * sc(-0.5)
        - dx_p * dy_p * per_i(&attr(&c1));
    let opacity_i = attr(&opv) * list_mask.reshape(&[t_count, c_cap]);
    let a0 = (per_i(&opacity_i) * power.exp()).minimum(&sc(0.99));
    let contributes = power.cmp_le(&sc(0.0)) * a0.cmp_ge(&sc(1.0 / 255.0));
    let alpha = a0 * contributes;

    let transmittance = resin_dsl::cumprod_exclusive(&(sc(1.0) - alpha.clone()), 1);
    let weight = alpha * transmittance; // [T, C, th, tw]

    // --- 6. assemble tiles into the image via a constant permutation -------
    let perm = pixel_permutation(width, height, tiles_x, tile);
    let mut image: Option<Tensor> = None;
    for ch in 0..3 {
        let color_ch = attr(&col(&color, ch));
        let tile_img = (weight.clone() * per_i(&color_ch))
            .sum_axes(&[1])
            .squeeze(&[1]) // [T, th, tw]
            .reshape(&[t_count * tile * tile]);
        let flat = tile_img.gather_rows(&perm).reshape(&[height, width]); // [H, W]
        let placed = flat.broadcast_to(&[height, width, 1], &[0, 1]).scatter_index(
            &[height, width, 3],
            &[
                IndexKeyElement::Slice(0..height),
                IndexKeyElement::Slice(0..width),
                IndexKeyElement::Single(ch),
            ],
        );
        image = Some(match image {
            None => placed,
            Some(acc) => acc + placed,
        });
    }
    image.expect("three channels")
}

/// Absolute pixel-center coordinate grids per tile: `[T, th, tw]` constants.
fn tile_pixel_grids(tiles_x: usize, tiles_y: usize, tile: usize) -> (Tensor, Tensor) {
    let t_count = tiles_x * tiles_y;
    let mut gx = Vec::with_capacity(t_count * tile * tile);
    let mut gy = Vec::with_capacity(t_count * tile * tile);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            for iy in 0..tile {
                for ix in 0..tile {
                    gx.push((tx * tile + ix) as f32 + 0.5);
                    gy.push((ty * tile + iy) as f32 + 0.5);
                }
            }
        }
    }
    (
        Tensor::constant_f32(&[t_count, tile, tile], &gx),
        Tensor::constant_f32(&[t_count, tile, tile], &gy),
    )
}

/// `perm[y·W + x]` = index of pixel `(x, y)` in the `[T, th, tw]` layout.
fn pixel_permutation(width: usize, height: usize, tiles_x: usize, tile: usize) -> Tensor {
    let mut perm = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let t = (y / tile) * tiles_x + x / tile;
            perm.push((t * tile * tile + (y % tile) * tile + (x % tile)) as u32);
        }
    }
    Tensor::constant_u32(&[width * height], &perm)
}
