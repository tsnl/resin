//! Tile grids for notebook figures.
//!
//! Stitch many small RGB images into one canvas, then embed with [`plot_image`].
//! One image avoids a Plotly subplot per tile (plotly-rs only types layout
//! axes 1–8). Grayscale tiles: expand with `R = G = B` first (see
//! [`gray_to_rgb`]).

use plotly::color::Rgb;
use plotly::common::HoverInfo;
use plotly::layout::{Axis, Layout, Margin};
use plotly::{Image, Plot};

/// Row-major RGB tiles → one canvas (nearest-neighbor `scale`, `gap` between tiles).
///
/// `tiles` is row-major over the grid: index `row * cols + col`. Each tile is
/// `tile_h * tile_w` pixels, also row-major.
pub fn mosaic(
    tiles: &[&[[u8; 3]]],
    tile_w: usize,
    tile_h: usize,
    cols: usize,
    scale: usize,
    gap: usize,
    bg: [u8; 3],
) -> Vec<Vec<Rgb>> {
    assert!(cols > 0 && scale > 0);
    let rows = tiles.len().div_ceil(cols);
    let cell_w = tile_w * scale;
    let cell_h = tile_h * scale;
    let w = cols * cell_w + cols.saturating_sub(1) * gap;
    let h = rows * cell_h + rows.saturating_sub(1) * gap;
    let bg = Rgb::new(bg[0], bg[1], bg[2]);
    let mut canvas = vec![vec![bg; w]; h];

    for (i, tile) in tiles.iter().enumerate() {
        assert_eq!(
            tile.len(),
            tile_w * tile_h,
            "tile {i}: expected {} pixels, got {}",
            tile_w * tile_h,
            tile.len()
        );
        let col = i % cols;
        let row = i / cols;
        let x0 = col * (cell_w + gap);
        let y0 = row * (cell_h + gap);
        for ty in 0..tile_h {
            for tx in 0..tile_w {
                let [r, g, b] = tile[ty * tile_w + tx];
                let px = Rgb::new(r, g, b);
                for dy in 0..scale {
                    for dx in 0..scale {
                        canvas[y0 + ty * scale + dy][x0 + tx * scale + dx] = px;
                    }
                }
            }
        }
    }
    canvas
}

/// Gray `u8` pixels → RGB with `R = G = B` (for feeding [`mosaic`]).
pub fn gray_to_rgb(pixels: &[u8]) -> Vec<[u8; 3]> {
    pixels.iter().map(|&g| [g, g, g]).collect()
}

/// Single Plotly image figure with hidden axes (no per-tile annotations).
pub fn plot_image(z: Vec<Vec<Rgb>>, title: &str) -> Plot {
    let h = z.len();
    let w = z.first().map(|row| row.len()).unwrap_or(0);
    let margin = 16;
    let hidden = || {
        Axis::new()
            .visible(false)
            .show_grid(false)
            .show_line(false)
            .show_tick_labels(false)
            .zero_line(false)
    };

    let mut plot = Plot::new();
    plot.set_layout(
        Layout::new()
            .title(title)
            .width(w + 2 * margin)
            .height(h + 2 * margin)
            .show_legend(false)
            .margin(
                Margin::new()
                    .left(margin)
                    .right(margin)
                    .top(margin)
                    .bottom(margin),
            )
            .x_axis(hidden())
            .y_axis(hidden()),
    );
    plot.add_trace(Image::new(z).hover_info(HoverInfo::Skip));
    plot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mosaic_layout_and_scale() {
        // Two 1×1 tiles side by side, 2× scale, 1 px gap → width 5, height 2.
        let a = [[255, 0, 0]];
        let b = [[0, 0, 255]];
        let tiles: &[&[[u8; 3]]] = &[&a, &b];
        let z = mosaic(tiles, 1, 1, 2, 2, 1, [0, 0, 0]);
        assert_eq!(z.len(), 2);
        assert_eq!(z[0].len(), 5);
        assert_eq!(z[0][0], Rgb::new(255, 0, 0));
        assert_eq!(z[0][1], Rgb::new(255, 0, 0));
        assert_eq!(z[0][2], Rgb::new(0, 0, 0)); // gap
        assert_eq!(z[0][3], Rgb::new(0, 0, 255));
        assert_eq!(z[1][4], Rgb::new(0, 0, 255));
    }

    #[test]
    fn gray_to_rgb_then_mosaic() {
        let t0 = gray_to_rgb(&[10]);
        let t1 = gray_to_rgb(&[20]);
        let tiles: &[&[[u8; 3]]] = &[&t0, &t1];
        let z = mosaic(tiles, 1, 1, 2, 1, 0, [0, 0, 0]);
        assert_eq!(z[0][0], Rgb::new(10, 10, 10));
        assert_eq!(z[0][1], Rgb::new(20, 20, 20));
    }
}
