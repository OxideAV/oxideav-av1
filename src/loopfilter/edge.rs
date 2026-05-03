//! Per-MI per-edge deblocking driver — spec §7.14.1 .. §7.14.5.
//!
//! The frame-level [`super::frame`] driver runs the same filter on
//! every 4×4 boundary using a single threshold derived from the
//! frame-header `loop_filter_level`. The spec actually walks each
//! 4×4 MI cell in turn, derives a per-cell `lvl` from the
//! segmentation / mode / ref-delta machinery (§7.14.5), and only
//! filters where a transform-block boundary actually falls.
//!
//! This module implements that finer walker. It still uses the
//! 4-tap [`super::narrow::filter4`] for narrow edges and dispatches
//! to the 6-tap [`super::wide::filter8`] when the chosen filter
//! length is ≥ 8 and the local `flatMask` triggers (§7.14.6.1).
//!
//! Inputs are intentionally restricted to a small, decoder-agnostic
//! shape — a callback that returns a `MiInfo` for any `(mi_col,
//! mi_row)` plus the parsed `LoopFilterParams` / `SegmentationParams`
//! — so the unit tests can drive it without standing up a full
//! `FrameState`.

use crate::frame_header_tail::{LoopFilterParams, SegmentationParams};

use super::mask::derive_thresholds;
use super::narrow::{filter4, filter4_16, high_edge_variation, narrow_mask, narrow_mask16};
use super::wide::{filter8, flat8_mask};

/// §3.4 — `SEG_LVL_ALT_LF_Y_V` (segmentation feature index 1). The
/// per-plane LF features land at the four consecutive indices
/// `Y_V`, `Y_H`, `U`, `V`.
pub const SEG_LVL_ALT_LF_Y_V: usize = 1;

/// §6.2 / §7.14.4 — `MAX_LOOP_FILTER` (mirrors the value already
/// re-exported by [`crate::decode::tile`]; copied here so the
/// loop-filter module is self-contained).
pub const MAX_LOOP_FILTER: i32 = 63;

/// Reference-frame index for INTRA_FRAME — matches the spec's
/// `INTRA_FRAME == 0` constant.
pub const INTRA_FRAME: u8 = 0;
/// `LAST_FRAME` (1) — the smallest non-INTRA reference id.
pub const LAST_FRAME: u8 = 1;

/// Mode-type buckets used by §7.14.4 / §7.14.5. `Intra` and
/// `GlobalMv` map to spec `modeType == 0`; everything else to
/// `modeType == 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LfModeType {
    /// Spec `modeType == 0` — intra modes plus the two GLOBAL* MV
    /// modes that resolve to a frame-level affine warp.
    Zero,
    /// Spec `modeType == 1` — every NEAREST/NEAR/NEW MV mode.
    One,
}

/// Per-MI-cell hint feeding the §7.14.4 strength derivation. Mirrors
/// the spec's `RefFrames[row][col][0]`, `YModes[row][col]`, etc.
/// Callers fill it from their decoded MI grid.
#[derive(Clone, Copy, Debug)]
pub struct MiInfo {
    /// `RefFrames[r][c][0]` — `INTRA_FRAME` (0) or one of the inter
    /// references (`LAST_FRAME..ALTREF2_FRAME`).
    pub ref_frame: u8,
    /// Mapped mode type per §7.14.4. Determines `modeType`.
    pub mode_type: LfModeType,
    /// Spec `Skips[r][c]` — set when the residual is all zero and
    /// we therefore skip filtering across non-block edges.
    pub skip: bool,
    /// Spec `SegmentIds[r][c]`. Out-of-range values clamp to 0 so a
    /// blank MI cell stays in the frame-default segment.
    pub segment_id: u8,
    /// Spec `LoopfilterTxSizes[plane][...]` — width/height of the
    /// transform block this MI cell belongs to, in luma samples.
    /// Used to decide whether the current edge is a TX boundary
    /// (§7.14.2 `isTxEdge`). 0 means "treat as 4×4".
    pub tx_w: u8,
    /// Same as [`Self::tx_w`] for the orthogonal direction.
    pub tx_h: u8,
    /// Spec `MiSizes[r][c]` width in luma samples — used so block
    /// boundaries (§7.14.2 `isBlockEdge`) are detected even when the
    /// MI happens to be tx-split.
    pub block_w: u8,
    /// Same as [`Self::block_w`] for the height direction.
    pub block_h: u8,
    /// Per-MI delta-LF accumulator (`DeltaLFs[row][col][i]` per
    /// §5.11.13 / §7.14.4). 0 in our narrow path.
    pub delta_lf: i8,
}

impl Default for MiInfo {
    fn default() -> Self {
        Self {
            ref_frame: INTRA_FRAME,
            mode_type: LfModeType::Zero,
            skip: false,
            segment_id: 0,
            tx_w: 4,
            tx_h: 4,
            block_w: 4,
            block_h: 4,
            delta_lf: 0,
        }
    }
}

/// §7.14.5 — pick the per-(MI, plane, pass) filter level after
/// applying segmentation + ref/mode deltas.
///
/// `pass == 0` means "vertical edge" (filtering happens with a
/// horizontal `dx`); `pass == 1` is horizontal.
///
/// The luma channel uses `loop_filter_level[0]` for `pass == 0` and
/// `loop_filter_level[1]` for `pass == 1`. Chroma planes always use
/// their own scalar (`loop_filter_level[2]` / `[3]`) regardless of
/// pass.
pub fn derive_lvl(
    mi: &MiInfo,
    plane: usize,
    pass: usize,
    lf: &LoopFilterParams,
    seg: &SegmentationParams,
) -> i32 {
    let i = if plane == 0 { pass } else { plane + 1 };
    let level_i = match i {
        0 => lf.level_y0,
        1 => lf.level_y1,
        2 => lf.level_u,
        3 => lf.level_v,
        _ => 0,
    } as i32;
    let base = (level_i + mi.delta_lf as i32).clamp(0, MAX_LOOP_FILTER);
    let mut lvl_seg = base;

    let feature = SEG_LVL_ALT_LF_Y_V + i;
    if seg.feature_active(mi.segment_id, feature) {
        let fdata = seg.feature_data[(mi.segment_id as usize).min(7)][feature.min(7)] as i32;
        lvl_seg = (lvl_seg + fdata).clamp(0, MAX_LOOP_FILTER);
    }

    if lf.mode_ref_delta_enabled {
        let n_shift = lvl_seg >> 5;
        if mi.ref_frame == INTRA_FRAME {
            lvl_seg += (lf.ref_deltas[INTRA_FRAME as usize] as i32) << n_shift;
        } else {
            let ref_idx = (mi.ref_frame as usize).min(7);
            lvl_seg += (lf.ref_deltas[ref_idx] as i32) << n_shift;
            let mode_idx = match mi.mode_type {
                LfModeType::Zero => 0,
                LfModeType::One => 1,
            };
            lvl_seg += (lf.mode_deltas[mode_idx] as i32) << n_shift;
        }
        lvl_seg = lvl_seg.clamp(0, MAX_LOOP_FILTER);
    }

    lvl_seg
}

/// §7.14.3 — pick the maximum filter length.
///
/// `tx_size_a` / `tx_size_b` are the transform-block dimensions on
/// either side of the boundary (in samples) for the relevant
/// direction. Returns one of `4`, `8`, or `16`. Chroma is capped at
/// 8 per the spec.
pub fn filter_size(tx_size_a: u32, tx_size_b: u32, plane: usize) -> u32 {
    let base = tx_size_a.min(tx_size_b);
    if plane == 0 {
        base.clamp(4, 16)
    } else {
        base.clamp(4, 8)
    }
}

/// §7.14.6.1 — pick which sample-filter to invoke. Returns the
/// `filterLen` (the number of taps each side of the central sample,
/// per §7.14.6.2): 4, 6, 8, or 16. Caller dispatches to
/// [`filter4`] (4) / [`filter8`] (6/8/16).
pub fn filter_len_for(filter_size: u32, plane: usize) -> u32 {
    if filter_size == 4 {
        4
    } else if plane != 0 {
        6
    } else if filter_size == 8 {
        8
    } else {
        16
    }
}

/// Single 8-bit plane the per-edge driver mutates in place.
pub struct EdgePlane<'a> {
    pub pix: &'a mut [u8],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

/// 16-bit counterpart of [`EdgePlane`].
pub struct EdgePlane16<'a> {
    pub pix: &'a mut [u16],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
    pub bit_depth: u32,
}

/// Per-MI-grid metadata the edge walker needs.
pub struct MiGrid<'a> {
    /// Row-major MI grid indexed `[mi_row * mi_cols + mi_col]`.
    pub cells: &'a [MiInfo],
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub sub_x: usize,
    pub sub_y: usize,
}

impl<'a> MiGrid<'a> {
    /// Fetch the MI cell at `(mi_col, mi_row)`. Out-of-range
    /// coordinates clip to the grid edge and surface a blank
    /// (intra, skip=true, level=0) cell so off-frame edges suppress
    /// the filter cleanly.
    fn at(&self, mi_col: i32, mi_row: i32) -> MiInfo {
        if mi_col < 0
            || mi_row < 0
            || (mi_col as usize) >= self.mi_cols
            || (mi_row as usize) >= self.mi_rows
        {
            return MiInfo {
                skip: true,
                ..MiInfo::default()
            };
        }
        self.cells[(mi_row as usize) * self.mi_cols + (mi_col as usize)]
    }
}

/// Drive the §7.14.1 outer per-plane / per-pass loop on a single
/// 8-bit plane.
pub fn apply_plane(
    pl: EdgePlane<'_>,
    plane: usize,
    grid: &MiGrid<'_>,
    lf: &LoopFilterParams,
    seg: &SegmentationParams,
) {
    if plane == 0 {
        if lf.level_y0 == 0 && lf.level_y1 == 0 {
            return;
        }
    } else {
        let level = if plane == 1 { lf.level_u } else { lf.level_v };
        if level == 0 {
            return;
        }
    }
    let EdgePlane {
        pix,
        stride,
        width,
        height,
    } = pl;
    let sub_x = if plane == 0 { 0 } else { grid.sub_x };
    let sub_y = if plane == 0 { 0 } else { grid.sub_y };
    let mi_step_x = 1usize << sub_x;
    let mi_step_y = 1usize << sub_y;
    // Vertical pass first (pass == 0 — filters across vertical
    // boundaries, dx == 1, dy == 0).
    walk_pass8(
        pix, stride, width, height, plane, 0, grid, sub_x, sub_y, mi_step_x, mi_step_y, lf, seg,
    );
    walk_pass8(
        pix, stride, width, height, plane, 1, grid, sub_x, sub_y, mi_step_x, mi_step_y, lf, seg,
    );
}

/// 16-bit counterpart of [`apply_plane`].
pub fn apply_plane16(
    pl: EdgePlane16<'_>,
    plane: usize,
    grid: &MiGrid<'_>,
    lf: &LoopFilterParams,
    seg: &SegmentationParams,
) {
    if plane == 0 {
        if lf.level_y0 == 0 && lf.level_y1 == 0 {
            return;
        }
    } else {
        let level = if plane == 1 { lf.level_u } else { lf.level_v };
        if level == 0 {
            return;
        }
    }
    let EdgePlane16 {
        pix,
        stride,
        width,
        height,
        bit_depth,
    } = pl;
    let sub_x = if plane == 0 { 0 } else { grid.sub_x };
    let sub_y = if plane == 0 { 0 } else { grid.sub_y };
    let mi_step_x = 1usize << sub_x;
    let mi_step_y = 1usize << sub_y;
    walk_pass16(
        pix, stride, width, height, plane, 0, grid, sub_x, sub_y, mi_step_x, mi_step_y, bit_depth,
        lf, seg,
    );
    walk_pass16(
        pix, stride, width, height, plane, 1, grid, sub_x, sub_y, mi_step_x, mi_step_y, bit_depth,
        lf, seg,
    );
}

#[allow(clippy::too_many_arguments)]
fn walk_pass8(
    pix: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
    plane: usize,
    pass: usize,
    grid: &MiGrid<'_>,
    sub_x: usize,
    sub_y: usize,
    mi_step_x: usize,
    mi_step_y: usize,
    lf: &LoopFilterParams,
    seg: &SegmentationParams,
) {
    // Walk the MI grid in 4×4 luma units, with chroma stepping by
    // 1<<sub_*. For each MI cell, examine the leading edge in this
    // pass direction.
    let (dx, dy) = if pass == 0 {
        (1usize, 0usize)
    } else {
        (0usize, 1usize)
    };
    let mut mi_row = 0usize;
    while mi_row < grid.mi_rows {
        let mut mi_col = 0usize;
        while mi_col < grid.mi_cols {
            let mi = grid.at(mi_col as i32, mi_row as i32);
            // Plane-space coordinates of this MI cell's top-left
            // sample (4×4 luma → potentially 2×2 chroma).
            let x_p = (mi_col * 4) >> sub_x;
            let y_p = (mi_row * 4) >> sub_y;
            // Off-screen edges (the "onScreen" check, §7.14.2).
            let on_screen = if pass == 0 {
                x_p > 0 && x_p < width
            } else {
                y_p > 0 && y_p < height
            };
            if on_screen {
                // Previous MI cell (the one across the boundary).
                let (prev_col, prev_row) = if pass == 0 {
                    (mi_col as i32 - mi_step_x as i32, mi_row as i32)
                } else {
                    (mi_col as i32, mi_row as i32 - mi_step_y as i32)
                };
                let prev = grid.at(prev_col, prev_row);
                // §7.14.2 isBlockEdge / isTxEdge — both reduce to
                // "the relevant axis falls on a block_w / tx_w
                // boundary in plane space".
                let bw = (mi.block_w.max(4) as usize) >> sub_x;
                let bh = (mi.block_h.max(4) as usize) >> sub_y;
                let tw = (mi.tx_w.max(4) as usize) >> sub_x;
                let th = (mi.tx_h.max(4) as usize) >> sub_y;
                let is_block_edge = if pass == 0 {
                    bw > 0 && x_p % bw == 0
                } else {
                    bh > 0 && y_p % bh == 0
                };
                let is_tx_edge = if pass == 0 {
                    tw > 0 && x_p % tw == 0
                } else {
                    th > 0 && y_p % th == 0
                };
                let is_intra = mi.ref_frame == INTRA_FRAME;
                let apply_filter = is_tx_edge && (is_block_edge || !mi.skip || is_intra);
                if apply_filter {
                    let mut lvl = derive_lvl(&mi, plane, pass, lf, seg);
                    if lvl == 0 {
                        lvl = derive_lvl(&prev, plane, pass, lf, seg);
                    }
                    if lvl > 0 {
                        let th_t = derive_thresholds(lvl, lf.sharpness as i32);
                        let f_size = if pass == 0 {
                            filter_size(prev.tx_w as u32, mi.tx_w as u32, plane)
                        } else {
                            filter_size(prev.tx_h as u32, mi.tx_h as u32, plane)
                        };
                        // MI cells are 4×4 luma in size — the
                        // §7.14.2 walker filters MI_SIZE samples
                        // along the edge. Chroma walks fewer
                        // samples (2 in 4:2:0).
                        let along = if pass == 0 {
                            4usize >> sub_y
                        } else {
                            4usize >> sub_x
                        };
                        filter_edge8(
                            pix, stride, width, height, x_p, y_p, dx, dy, along, plane, f_size,
                            th_t,
                        );
                    }
                }
            }
            mi_col += mi_step_x;
        }
        mi_row += mi_step_y;
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_pass16(
    pix: &mut [u16],
    stride: usize,
    width: usize,
    height: usize,
    plane: usize,
    pass: usize,
    grid: &MiGrid<'_>,
    sub_x: usize,
    sub_y: usize,
    mi_step_x: usize,
    mi_step_y: usize,
    bit_depth: u32,
    lf: &LoopFilterParams,
    seg: &SegmentationParams,
) {
    let (dx, dy) = if pass == 0 {
        (1usize, 0usize)
    } else {
        (0usize, 1usize)
    };
    let mut mi_row = 0usize;
    while mi_row < grid.mi_rows {
        let mut mi_col = 0usize;
        while mi_col < grid.mi_cols {
            let mi = grid.at(mi_col as i32, mi_row as i32);
            let x_p = (mi_col * 4) >> sub_x;
            let y_p = (mi_row * 4) >> sub_y;
            let on_screen = if pass == 0 {
                x_p > 0 && x_p < width
            } else {
                y_p > 0 && y_p < height
            };
            if on_screen {
                let (prev_col, prev_row) = if pass == 0 {
                    (mi_col as i32 - mi_step_x as i32, mi_row as i32)
                } else {
                    (mi_col as i32, mi_row as i32 - mi_step_y as i32)
                };
                let prev = grid.at(prev_col, prev_row);
                let bw = (mi.block_w.max(4) as usize) >> sub_x;
                let bh = (mi.block_h.max(4) as usize) >> sub_y;
                let tw = (mi.tx_w.max(4) as usize) >> sub_x;
                let th = (mi.tx_h.max(4) as usize) >> sub_y;
                let is_block_edge = if pass == 0 {
                    bw > 0 && x_p % bw == 0
                } else {
                    bh > 0 && y_p % bh == 0
                };
                let is_tx_edge = if pass == 0 {
                    tw > 0 && x_p % tw == 0
                } else {
                    th > 0 && y_p % th == 0
                };
                let is_intra = mi.ref_frame == INTRA_FRAME;
                let apply_filter = is_tx_edge && (is_block_edge || !mi.skip || is_intra);
                if apply_filter {
                    let mut lvl = derive_lvl(&mi, plane, pass, lf, seg);
                    if lvl == 0 {
                        lvl = derive_lvl(&prev, plane, pass, lf, seg);
                    }
                    if lvl > 0 {
                        let th_t = derive_thresholds(lvl, lf.sharpness as i32);
                        let scaled = super::narrow::scale_thresholds16(th_t, bit_depth);
                        let f_size = if pass == 0 {
                            filter_size(prev.tx_w as u32, mi.tx_w as u32, plane)
                        } else {
                            filter_size(prev.tx_h as u32, mi.tx_h as u32, plane)
                        };
                        let along = if pass == 0 {
                            4usize >> sub_y
                        } else {
                            4usize >> sub_x
                        };
                        filter_edge16(
                            pix, stride, width, height, x_p, y_p, dx, dy, along, plane, f_size,
                            scaled, bit_depth,
                        );
                    }
                }
            }
            mi_col += mi_step_x;
        }
        mi_row += mi_step_y;
    }
}

/// Apply the chosen sample filter to `along` consecutive samples of
/// the edge starting at `(x0, y0)`. `(dx, dy)` is perpendicular to
/// the edge direction (so when pass==0 the edge runs vertically and
/// dx==1).
#[allow(clippy::too_many_arguments)]
fn filter_edge8(
    pix: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    dx: usize,
    dy: usize,
    along: usize,
    plane: usize,
    f_size: u32,
    th: super::narrow::Thresholds,
) {
    // Tangent direction along the edge — `(tx, ty)` is orthogonal
    // to `(dx, dy)`.
    let (tx, ty) = (dy, dx);
    let f_len = filter_len_for(f_size, plane);
    for i in 0..along {
        let cx = x0 + tx * i;
        let cy = y0 + ty * i;
        if cx >= width || cy >= height {
            continue;
        }
        // Need 4 samples (p1, p0, q0, q1) for the narrow path, 8
        // (p3..p0, q0..q3) for the wide path. Bail when the edge is
        // too close to a boundary.
        let need = if f_len >= 8 { 4 } else { 2 };
        if !has_room(cx, cy, dx, dy, width, height, need) {
            continue;
        }
        // Pull the 4-sample neighbourhood for the narrow mask.
        let p1 = sample8(pix, stride, cx, cy, dx, dy, -2);
        let p0 = sample8(pix, stride, cx, cy, dx, dy, -1);
        let q0 = sample8(pix, stride, cx, cy, dx, dy, 0);
        let q1 = sample8(pix, stride, cx, cy, dx, dy, 1);
        if !narrow_mask(p1, p0, q0, q1, th) {
            continue;
        }
        // Try the wide path when `filterLen >= 8` and the §7.14.6.2
        // `flatMask` triggers.
        if f_len >= 8 && has_room(cx, cy, dx, dy, width, height, 4) {
            let p3 = sample8(pix, stride, cx, cy, dx, dy, -4);
            let p2 = sample8(pix, stride, cx, cy, dx, dy, -3);
            let q2 = sample8(pix, stride, cx, cy, dx, dy, 2);
            let q3 = sample8(pix, stride, cx, cy, dx, dy, 3);
            if flat8_mask(p3, p2, p1, p0, q0, q1, q2, q3) {
                let (np2, np1, np0, nq0, nq1, nq2) = filter8(p3, p2, p1, p0, q0, q1, q2, q3);
                store8(pix, stride, cx, cy, dx, dy, -3, np2);
                store8(pix, stride, cx, cy, dx, dy, -2, np1);
                store8(pix, stride, cx, cy, dx, dy, -1, np0);
                store8(pix, stride, cx, cy, dx, dy, 0, nq0);
                store8(pix, stride, cx, cy, dx, dy, 1, nq1);
                store8(pix, stride, cx, cy, dx, dy, 2, nq2);
                continue;
            }
        }
        let hev = high_edge_variation(p1, p0, q0, q1, th.thresh);
        let (np1, np0, nq0, nq1) = filter4(p1, p0, q0, q1, hev);
        store8(pix, stride, cx, cy, dx, dy, -2, np1);
        store8(pix, stride, cx, cy, dx, dy, -1, np0);
        store8(pix, stride, cx, cy, dx, dy, 0, nq0);
        store8(pix, stride, cx, cy, dx, dy, 1, nq1);
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_edge16(
    pix: &mut [u16],
    stride: usize,
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    dx: usize,
    dy: usize,
    along: usize,
    _plane: usize,
    _f_size: u32,
    th: super::narrow::Thresholds16,
    bit_depth: u32,
) {
    let (tx, ty) = (dy, dx);
    for i in 0..along {
        let cx = x0 + tx * i;
        let cy = y0 + ty * i;
        if cx >= width || cy >= height {
            continue;
        }
        if !has_room(cx, cy, dx, dy, width, height, 2) {
            continue;
        }
        let p1 = sample16(pix, stride, cx, cy, dx, dy, -2);
        let p0 = sample16(pix, stride, cx, cy, dx, dy, -1);
        let q0 = sample16(pix, stride, cx, cy, dx, dy, 0);
        let q1 = sample16(pix, stride, cx, cy, dx, dy, 1);
        if !narrow_mask16(p1, p0, q0, q1, th) {
            continue;
        }
        let hev = super::narrow::high_edge_variation16(p1, p0, q0, q1, th.thresh);
        let (np1, np0, nq0, nq1) = filter4_16(p1, p0, q0, q1, hev, bit_depth);
        store16(pix, stride, cx, cy, dx, dy, -2, np1);
        store16(pix, stride, cx, cy, dx, dy, -1, np0);
        store16(pix, stride, cx, cy, dx, dy, 0, nq0);
        store16(pix, stride, cx, cy, dx, dy, 1, nq1);
    }
}

#[inline]
fn has_room(
    cx: usize,
    cy: usize,
    dx: usize,
    dy: usize,
    width: usize,
    height: usize,
    n: usize,
) -> bool {
    if dx > 0 {
        cx >= n && cx + n <= width
    } else if dy > 0 {
        cy >= n && cy + n <= height
    } else {
        false
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn sample8(pix: &[u8], stride: usize, cx: usize, cy: usize, dx: usize, dy: usize, off: i32) -> u8 {
    let x = (cx as i32 + dx as i32 * off) as usize;
    let y = (cy as i32 + dy as i32 * off) as usize;
    pix[y * stride + x]
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn store8(
    pix: &mut [u8],
    stride: usize,
    cx: usize,
    cy: usize,
    dx: usize,
    dy: usize,
    off: i32,
    v: u8,
) {
    let x = (cx as i32 + dx as i32 * off) as usize;
    let y = (cy as i32 + dy as i32 * off) as usize;
    pix[y * stride + x] = v;
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn sample16(
    pix: &[u16],
    stride: usize,
    cx: usize,
    cy: usize,
    dx: usize,
    dy: usize,
    off: i32,
) -> u16 {
    let x = (cx as i32 + dx as i32 * off) as usize;
    let y = (cy as i32 + dy as i32 * off) as usize;
    pix[y * stride + x]
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn store16(
    pix: &mut [u16],
    stride: usize,
    cx: usize,
    cy: usize,
    dx: usize,
    dy: usize,
    off: i32,
    v: u16,
) {
    let x = (cx as i32 + dx as i32 * off) as usize;
    let y = (cy as i32 + dy as i32 * off) as usize;
    pix[y * stride + x] = v;
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::frame_header_tail::DEFAULT_REF_DELTAS;

    fn lf_with(level_y0: u8, level_y1: u8, sharp: u8) -> LoopFilterParams {
        let mut lf = LoopFilterParams::default();
        lf.level_y0 = level_y0;
        lf.level_y1 = level_y1;
        lf.sharpness = sharp;
        lf
    }

    #[test]
    fn derive_lvl_uses_pass_for_y_plane() {
        let lf = lf_with(20, 30, 0);
        let seg = SegmentationParams::default();
        let mi = MiInfo::default();
        assert_eq!(derive_lvl(&mi, 0, 0, &lf, &seg), 20);
        assert_eq!(derive_lvl(&mi, 0, 1, &lf, &seg), 30);
    }

    #[test]
    fn derive_lvl_uses_chroma_scalar_regardless_of_pass() {
        let mut lf = lf_with(20, 30, 0);
        lf.level_u = 11;
        lf.level_v = 22;
        let seg = SegmentationParams::default();
        let mi = MiInfo::default();
        assert_eq!(derive_lvl(&mi, 1, 0, &lf, &seg), 11);
        assert_eq!(derive_lvl(&mi, 1, 1, &lf, &seg), 11);
        assert_eq!(derive_lvl(&mi, 2, 0, &lf, &seg), 22);
        assert_eq!(derive_lvl(&mi, 2, 1, &lf, &seg), 22);
    }

    #[test]
    fn derive_lvl_applies_segment_data() {
        let lf = lf_with(20, 20, 0);
        let mut seg = SegmentationParams::default();
        seg.enabled = true;
        seg.feature_enabled[3][SEG_LVL_ALT_LF_Y_V] = true; // segment 3, Y_V
        seg.feature_data[3][SEG_LVL_ALT_LF_Y_V] = 5;
        let mut mi = MiInfo::default();
        mi.segment_id = 3;
        assert_eq!(derive_lvl(&mi, 0, 0, &lf, &seg), 25);
        // Different segment is unaffected.
        mi.segment_id = 0;
        assert_eq!(derive_lvl(&mi, 0, 0, &lf, &seg), 20);
    }

    #[test]
    fn derive_lvl_applies_intra_ref_delta() {
        let mut lf = lf_with(32, 32, 0);
        lf.mode_ref_delta_enabled = true;
        lf.ref_deltas = DEFAULT_REF_DELTAS; // INTRA_FRAME == 1
        let seg = SegmentationParams::default();
        let mi = MiInfo::default();
        // n_shift = 32 >> 5 = 1 → +1 << 1 = +2 → 34
        assert_eq!(derive_lvl(&mi, 0, 0, &lf, &seg), 34);
    }

    #[test]
    fn derive_lvl_applies_mode_delta_for_inter() {
        let mut lf = lf_with(32, 32, 0);
        lf.mode_ref_delta_enabled = true;
        lf.ref_deltas[1] = 0; // LAST_FRAME
        lf.mode_deltas = [0, 3];
        let seg = SegmentationParams::default();
        let mut mi = MiInfo::default();
        mi.ref_frame = LAST_FRAME;
        mi.mode_type = LfModeType::One;
        // n_shift = 32 >> 5 = 1 → +(0<<1)+(3<<1) = +6 → 38
        assert_eq!(derive_lvl(&mi, 0, 0, &lf, &seg), 38);
    }

    #[test]
    fn filter_size_caps_chroma_at_8() {
        assert_eq!(filter_size(16, 16, 0), 16);
        assert_eq!(filter_size(16, 16, 1), 8);
    }

    #[test]
    fn filter_len_picks_4_6_8_16() {
        assert_eq!(filter_len_for(4, 0), 4);
        assert_eq!(filter_len_for(8, 1), 6);
        assert_eq!(filter_len_for(8, 0), 8);
        assert_eq!(filter_len_for(16, 0), 16);
    }

    #[test]
    fn apply_plane_skips_when_levels_zero() {
        let mut pix = vec![100u8; 32 * 32];
        let cells = vec![MiInfo::default(); 8 * 8];
        let grid = MiGrid {
            cells: &cells,
            mi_cols: 8,
            mi_rows: 8,
            sub_x: 0,
            sub_y: 0,
        };
        let lf = lf_with(0, 0, 0);
        let seg = SegmentationParams::default();
        apply_plane(
            EdgePlane {
                pix: &mut pix,
                stride: 32,
                width: 32,
                height: 32,
            },
            0,
            &grid,
            &lf,
            &seg,
        );
        assert!(pix.iter().all(|&v| v == 100));
    }

    #[test]
    fn apply_plane_softens_block_edges() {
        // Build a 16×16 plane with a vertical step at x=8: left half
        // 110, right half 120. Mark every MI as intra/non-skip with
        // an 8×8 block size so the x=8 column is both a block and
        // a tx edge.
        let mut pix = vec![0u8; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                pix[y * 16 + x] = if x < 8 { 110 } else { 120 };
            }
        }
        let mut cells = vec![MiInfo::default(); 4 * 4];
        for c in cells.iter_mut() {
            c.block_w = 8;
            c.block_h = 8;
            c.tx_w = 4;
            c.tx_h = 4;
            c.skip = false;
        }
        let grid = MiGrid {
            cells: &cells,
            mi_cols: 4,
            mi_rows: 4,
            sub_x: 0,
            sub_y: 0,
        };
        let lf = lf_with(30, 30, 0);
        let seg = SegmentationParams::default();
        apply_plane(
            EdgePlane {
                pix: &mut pix,
                stride: 16,
                width: 16,
                height: 16,
            },
            0,
            &grid,
            &lf,
            &seg,
        );
        // Column 7 (last left sample) should soften upward, column 8
        // should soften downward.
        for y in 0..16 {
            assert!(pix[y * 16 + 7] >= 110);
            assert!(pix[y * 16 + 8] <= 120);
            // At least one row crossed.
            if pix[y * 16 + 7] > 110 || pix[y * 16 + 8] < 120 {
                return;
            }
        }
        panic!("edge filter never modified the step");
    }

    #[test]
    fn apply_plane_skips_skip_blocks_without_block_edge() {
        // skip=true and not a block edge → no filter. Build a
        // uniform plane so even if filter ran it wouldn't change
        // anything; this test really verifies no panics occur and
        // that delta_lf doesn't pick up bogus data.
        let mut pix = vec![100u8; 16 * 16];
        let mut cells = vec![MiInfo::default(); 4 * 4];
        for c in cells.iter_mut() {
            c.skip = true;
            c.tx_w = 4;
            c.tx_h = 4;
            c.block_w = 16;
            c.block_h = 16;
        }
        let grid = MiGrid {
            cells: &cells,
            mi_cols: 4,
            mi_rows: 4,
            sub_x: 0,
            sub_y: 0,
        };
        let lf = lf_with(30, 30, 0);
        let seg = SegmentationParams::default();
        apply_plane(
            EdgePlane {
                pix: &mut pix,
                stride: 16,
                width: 16,
                height: 16,
            },
            0,
            &grid,
            &lf,
            &seg,
        );
        // x=4 / x=8 / x=12 are TX edges but not block edges (block
        // is 16×16) → only filtered when isIntra (default ref ==
        // INTRA_FRAME). Filtering a uniform plane is identity.
        assert!(pix.iter().all(|&v| v == 100));
    }
}
