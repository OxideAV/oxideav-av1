//! Frame-level deblocking driver.
//!
//! The AV1 spec actually walks every 4-sample edge and only filters
//! where transform-block boundaries fall. For the intra-only still-image
//! path we start with a simpler grid-based driver; full per-edge
//! tracking lands with the inter decoder.

use super::narrow::{
    apply_horizontal_edge4, apply_horizontal_edge4_16, apply_vertical_edge4,
    apply_vertical_edge4_16, Thresholds, Thresholds16,
};

/// A single u8 plane for the deblocking driver.
pub struct Plane<'a> {
    pub pix: &'a mut [u8],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

/// A single u16 plane for the deblocking driver.
pub struct Plane16<'a> {
    pub pix: &'a mut [u16],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

/// Edge grid — a sorted list of column/row offsets at which to run the
/// deblocker.
#[derive(Clone, Debug, Default)]
pub struct EdgeGrid {
    pub edge_xs: Vec<usize>,
    pub edge_ys: Vec<usize>,
}

/// Return a uniform grid of internal edges for a `w × h` plane whose
/// internal blocks are `cell_w × cell_h`.
pub fn uniform_grid(w: usize, h: usize, cell_w: usize, cell_h: usize) -> EdgeGrid {
    let mut g = EdgeGrid::default();
    let mut x = cell_w;
    while x < w {
        g.edge_xs.push(x);
        x += cell_w;
    }
    let mut y = cell_h;
    while y < h {
        g.edge_ys.push(y);
        y += cell_h;
    }
    g
}

/// Apply the 4-tap narrow filter to every internal edge in `grid`.
/// Edges within 2 samples of the plane border are skipped because the
/// narrow filter reads p1/q1 outside the immediate edge.
pub fn apply_frame_narrow(p: Plane<'_>, grid: &EdgeGrid, th: Thresholds) {
    // Work around borrow-checker: we mutate `p.pix` across sequential
    // loops, each only inspecting one row/column at a time.
    let Plane {
        pix,
        stride,
        width,
        height,
    } = p;
    for &x in &grid.edge_xs {
        if x < 2 || x + 2 > width {
            continue;
        }
        apply_vertical_edge4(pix, stride, x, height, th);
    }
    for &y in &grid.edge_ys {
        if y < 2 || y + 2 > height {
            continue;
        }
        apply_horizontal_edge4(pix, stride, y, width, th);
    }
}

/// 16-bit counterpart of [`apply_frame_narrow`].
pub fn apply_frame_narrow16(p: Plane16<'_>, grid: &EdgeGrid, th: Thresholds16) {
    let Plane16 {
        pix,
        stride,
        width,
        height,
    } = p;
    for &x in &grid.edge_xs {
        if x < 2 || x + 2 > width {
            continue;
        }
        apply_vertical_edge4_16(pix, stride, x, height, th);
    }
    for &y in &grid.edge_ys {
        if y < 2 || y + 2 > height {
            continue;
        }
        apply_horizontal_edge4_16(pix, stride, y, width, th);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_skips_zero_and_end() {
        let g = uniform_grid(16, 16, 4, 4);
        assert_eq!(g.edge_xs, vec![4, 8, 12]);
        assert_eq!(g.edge_ys, vec![4, 8, 12]);
    }

    #[test]
    fn narrow_identity_when_level_zero() {
        let mut pix = vec![100u8; 64];
        let th = super::super::mask::derive_thresholds(0, 0);
        let grid = uniform_grid(8, 8, 4, 4);
        let p = Plane {
            pix: &mut pix,
            stride: 8,
            width: 8,
            height: 8,
        };
        apply_frame_narrow(p, &grid, th);
        for v in &pix {
            assert_eq!(*v, 100);
        }
    }
}
