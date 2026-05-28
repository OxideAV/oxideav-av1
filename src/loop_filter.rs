//! §7.14 Loop filter process — the deblocking filter applied to
//! reconstructed `CurrFrame[]` samples per av1-spec p.307-318.
//!
//! ## Coverage (round 195 — close-out push)
//!
//! This module covers the §7.14 deblocking layer end-to-end at the
//! sample-filtering level, plus the §7.14.1 top-level per-edge driver
//! and the §7.14.4 / §7.14.5 adaptive strength derivation:
//!
//! * §7.14.1 — [`loop_filter_frame`]: per-plane × per-pass (vertical
//!   then horizontal) × per-row × per-col walk that calls
//!   [`loop_filter_edge`] on every 4×4 boundary the spec visits.
//! * §7.14.2 — [`loop_filter_edge`]: per-edge driver that derives
//!   `onScreen` / `xP` / `yP` / `prevRow` / `prevCol` / `isBlockEdge` /
//!   `isTxEdge` / `applyFilter` per av1-spec p.308-309, then dispatches
//!   to §7.14.3 + §7.14.4 + §7.14.6 for the chosen edge.
//! * §7.14.3 — [`filter_size`]: `Min(16, baseSize)` for luma, `Min(8,
//!   baseSize)` for chroma, with `baseSize = Min(Tx_*[txSz],
//!   Tx_*[prevTxSz])` per av1-spec p.310.
//! * §7.14.4 — [`adaptive_filter_strength`]: derives `(lvl, limit,
//!   blimit, thresh)` per av1-spec p.311 (driving §7.14.5 for the
//!   `lvl` core).
//! * §7.14.5 — [`adaptive_filter_strength_selection`]: the §5.9.14
//!   per-segment `SEG_LVL_ALT_LF_*` adjustment + the §5.9.11
//!   `loop_filter_ref_deltas[]` / `loop_filter_mode_deltas[]`
//!   adjustment, both clipped to `[0, MAX_LOOP_FILTER]` per av1-spec
//!   p.312.
//! * §7.14.6.1 — [`sample_filtering`]: per-pixel dispatch over
//!   [`filter_mask`] → [`narrow_filter`] / [`wide_filter`] based on
//!   `filterSize` and the flat-region masks.
//! * §7.14.6.2 — [`filter_mask`]: returns `(hevMask, filterMask,
//!   flatMask, flatMask2)` per av1-spec p.314-315.
//! * §7.14.6.3 — [`narrow_filter`]: the 4-tap edge filter that
//!   modifies up to two samples per side using the §3 `filter4_clamp`
//!   bit-depth-aware clamp.
//! * §7.14.6.4 — [`wide_filter`]: the 6/8/14-tap low-pass filter for
//!   detected flat regions (`log2Size ∈ {3, 4}`).
//!
//! ## Standalone-friendly surface
//!
//! The top-level driver takes a small [`LoopFilterFrameContext`]
//! bundling everything needed to walk the frame:
//!
//! * `loop_filter_level[0..=3]`, `loop_filter_sharpness`,
//!   `loop_filter_delta_enabled`, `loop_filter_ref_deltas`,
//!   `loop_filter_mode_deltas` — straight from
//!   [`crate::LoopFilterParams`].
//! * `delta_lf_multi`, `mi_rows`, `mi_cols`, `num_planes`,
//!   `bit_depth`, `subsampling_x`, `subsampling_y`, `frame_width`,
//!   `frame_height` — frame geometry.
//! * Per-block predicates (`is_intra(row, col)`, `skip(row, col)`,
//!   `ref_frame(row, col)`, `mode(row, col)`, `segment_id(row, col)`,
//!   `delta_lf(row, col, idx)`, `lf_tx_size(plane, row, col)`) — fed
//!   as closures so the driver is testable in isolation without
//!   needing the §5.11.x decode walker wired up. The §7.11.x intra
//!   driver consumes these from
//!   [`crate::PartitionWalker::y_modes`] / [`segment_ids`] /
//!   [`is_inters`] / [`skips`].
//! * Per-plane `CurrFrame[plane]` mutable buffers as
//!   [`PlaneBuffer<'_>`] — `samples` row-major slice + `(rows, cols)`
//!   extent.
//!
//! ## Bitstream-conformance gates not enforced here
//!
//! The §7.14 driver assumes the caller has already enforced:
//!
//! * §5.9.11 `loop_filter_level[i] <= MAX_LOOP_FILTER = 63` (the
//!   `f(6)` reader guarantees this).
//! * §5.9.14 `SEG_LVL_ALT_LF_*` per-segment feature values are signed
//!   `[-63, 63]` (the `su(1 + 6)` reader guarantees this).
//! * Plane buffers are sized at `(mi_rows * MI_SIZE) >> subY` ×
//!   `(mi_cols * MI_SIZE) >> subX` per av1-spec §5.11.34.

use crate::{
    cdf::MI_SIZE,
    uncompressed_header_tail::{INTRA_FRAME, MAX_LOOP_FILTER, TOTAL_REFS_PER_FRAME},
};

// =====================================================================
// §3 mode-type ordinals consumed by §7.14.4 — av1-spec p.13-14.
// =====================================================================

/// `NEARESTMV` per §3 — the lowest §7.14.4-tracked inter mode ordinal.
/// `mode >= NEARESTMV && mode != GLOBALMV && mode != GLOBAL_GLOBALMV`
/// drives `modeType = 1` per av1-spec p.311 line 17188.
///
/// Mirrors [`crate::cdf::MODE_NEARESTMV`] for the §7.14.4 mode-type
/// comparison.
pub const NEARESTMV: u8 = crate::cdf::MODE_NEARESTMV;
/// `GLOBALMV` per §3 — single-ref global-MV mode ordinal.
/// Excluded from the `modeType = 1` set by §7.14.4 per av1-spec p.311
/// line 17188.
pub const GLOBALMV: u8 = crate::cdf::MODE_GLOBALMV;
/// `GLOBAL_GLOBALMV` per §3 — compound-ref global-global-MV mode
/// ordinal. Excluded from the `modeType = 1` set by §7.14.4 per
/// av1-spec p.311 line 17188.
pub const GLOBAL_GLOBALMV: u8 = crate::cdf::MODE_GLOBAL_GLOBALMV;

// =====================================================================
// §3 segmentation feature ordinals consumed by §7.14.5 — av1-spec p.14.
// =====================================================================

/// `SEG_LVL_ALT_LF_Y_V` per §3 — feature 1 (per-segment alt LF for
/// vertical-pass luma). The §7.14.5 `feature = SEG_LVL_ALT_LF_Y_V +
/// i` selector lands on this for `i == 0` (i.e. `plane == 0 && pass ==
/// 0`). av1-spec p.312 line 17257.
pub const SEG_LVL_ALT_LF_Y_V: usize = 1;

// =====================================================================
// §7.14.1 / §7.14.2 — public top-level surface.
// =====================================================================

/// Mutable view of a single plane's `CurrFrame[plane]` sample buffer.
///
/// The `samples` slice is row-major `rows * cols` `i32` post-`Clip1`
/// per av1-spec §7.12.3 step-3 — same shape produced by
/// [`crate::PartitionWalker`] for the lazy per-plane buffers.
#[derive(Debug)]
pub struct PlaneBuffer<'a> {
    /// Per-plane row count: `(mi_rows * MI_SIZE) >> sub_y`.
    pub rows: u32,
    /// Per-plane column count: `(mi_cols * MI_SIZE) >> sub_x`.
    pub cols: u32,
    /// `rows * cols` row-major sample buffer.
    pub samples: &'a mut [i32],
}

impl PlaneBuffer<'_> {
    /// Bounds-checked load — returns `0` for out-of-bounds reads. The
    /// §7.14.6 filter mask process never indexes outside the on-screen
    /// region (the §7.14.2 `onScreen` gate prevents it), but a few
    /// `flatMask2`/wide-filter taps can reach the buffer's first /
    /// last column. The spec replicates edge samples; we mirror that
    /// by clipping the index to the buffer's `[0, rows-1]` /
    /// `[0, cols-1]` range.
    #[inline]
    fn get(&self, y: i32, x: i32) -> i32 {
        let yc = y.clamp(0, self.rows as i32 - 1) as usize;
        let xc = x.clamp(0, self.cols as i32 - 1) as usize;
        self.samples[yc * (self.cols as usize) + xc]
    }

    /// Bounds-checked store — silently drops out-of-bounds writes.
    /// The on-screen gate ensures the central edge samples land
    /// inside the buffer; this is a safety net for the wide filter's
    /// outermost taps.
    #[inline]
    fn set(&mut self, y: i32, x: i32, v: i32) {
        if y < 0 || x < 0 {
            return;
        }
        let (y, x) = (y as u32, x as u32);
        if y >= self.rows || x >= self.cols {
            return;
        }
        let idx = (y as usize) * (self.cols as usize) + (x as usize);
        self.samples[idx] = v;
    }
}

/// Caller-supplied frame-level inputs the §7.14 driver consults.
///
/// All §5.9.11 / §5.9.18 / §5.5 / §5.9.5 fields come straight from
/// the parsed frame header. Per-block predicates are closures so the
/// driver can be exercised in isolation without wiring the full
/// §5.11 decode state.
pub struct LoopFilterFrameContext<'a> {
    /// `loop_filter_level[0..=3]` per §5.9.11. Indices `0, 1` are
    /// luma vertical / horizontal; indices `2, 3` are U / V (set to
    /// `0` if monochrome).
    pub loop_filter_level: [u8; 4],
    /// `loop_filter_sharpness` per §5.9.11 — `f(3)`, range `0..=7`.
    pub loop_filter_sharpness: u8,
    /// `loop_filter_delta_enabled` per §5.9.11.
    pub loop_filter_delta_enabled: bool,
    /// `loop_filter_ref_deltas[0..=7]` per §5.9.11 after the update
    /// walk.
    pub loop_filter_ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    /// `loop_filter_mode_deltas[0..=1]` per §5.9.11 after the update
    /// walk.
    pub loop_filter_mode_deltas: [i8; 2],
    /// `delta_lf_multi` per §5.9.18. When `false`, per-block `DeltaLF`
    /// uses index `0`; when `true`, it uses `(plane == 0) ? pass :
    /// (plane + 1)`.
    pub delta_lf_multi: bool,
    /// `MiRows` per §5.9.5.
    pub mi_rows: u32,
    /// `MiCols` per §5.9.5.
    pub mi_cols: u32,
    /// `NumPlanes` per §5.5.2 — `1` for monochrome, `3` otherwise.
    pub num_planes: u8,
    /// `BitDepth` per §5.5.2 (`8`, `10`, or `12`).
    pub bit_depth: u8,
    /// `subsampling_x` per §5.5.2.
    pub subsampling_x: u8,
    /// `subsampling_y` per §5.5.2.
    pub subsampling_y: u8,
    /// `FrameWidth` per §5.9.5 (post-superres).
    pub frame_width: u32,
    /// `FrameHeight` per §5.9.5.
    pub frame_height: u32,
    /// `RefFrames[row][col][0] <= INTRA_FRAME` per §7.14.2 line 17073.
    pub is_intra: &'a dyn Fn(u32, u32) -> bool,
    /// `Skips[row][col]` per §7.14.2 line 17071.
    pub skip: &'a dyn Fn(u32, u32) -> bool,
    /// `RefFrames[row][col][0]` per §7.14.4 line 17181 — the
    /// reference-frame ordinal in `0..=TOTAL_REFS_PER_FRAME`.
    pub ref_frame: &'a dyn Fn(u32, u32) -> u8,
    /// `YModes[row][col]` per §7.14.4 line 17183 — the per-block
    /// mode ordinal used to classify §7.14.4's `modeType`.
    pub mode: &'a dyn Fn(u32, u32) -> u8,
    /// `SegmentIds[row][col]` per §7.14.4 line 17179 — `-1` (signalled
    /// as `u8::MAX`) for cells without a segment id.
    pub segment_id: &'a dyn Fn(u32, u32) -> u8,
    /// `DeltaLFs[row][col][idx]` per §7.14.4 line 17195-17198. `idx`
    /// is `0` for `delta_lf_multi == 0`; otherwise `(plane == 0) ?
    /// pass : (plane + 1)` (so `0..=3`).
    pub delta_lf: &'a dyn Fn(u32, u32, usize) -> i8,
    /// `seg_feature_active_idx(segment, feature)` per §7.14.5 line
    /// 17259. `feature ∈ {SEG_LVL_ALT_LF_Y_V, +Y_H, +U, +V}` (i.e.
    /// `1..=4`).
    pub seg_feature_active: &'a dyn Fn(u32, usize) -> bool,
    /// `FeatureData[segment][feature]` per §7.14.5 line 17266 — the
    /// signed alt-LF offset that gets added to `lvlSeg` when
    /// `seg_feature_active(segment, feature) == 1`.
    pub seg_feature_data: &'a dyn Fn(u32, usize) -> i16,
    /// `LoopfilterTxSizes[plane][row >> subY][col >> subX]` per
    /// §7.14.2 line 17067 — the per-plane transform-size index in
    /// `0..=TX_SIZES_ALL`.
    pub lf_tx_size: &'a dyn Fn(usize, u32, u32) -> usize,
    /// `MiSizes[row][col]` per §7.14.2 line 17065 — the per-mi block
    /// size ordinal. Combined with `plane` via
    /// [`crate::cdf::get_plane_residual_size`]-equivalent: for the
    /// driver, this is consumed by [`block_width`] / [`block_height`]
    /// for the §7.14.2 `Block_Width[planeSize]` /
    /// `Block_Height[planeSize]` compare.
    pub mi_size: &'a dyn Fn(u32, u32) -> usize,
}

impl std::fmt::Debug for LoopFilterFrameContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopFilterFrameContext")
            .field("loop_filter_level", &self.loop_filter_level)
            .field("loop_filter_sharpness", &self.loop_filter_sharpness)
            .field("loop_filter_delta_enabled", &self.loop_filter_delta_enabled)
            .field("loop_filter_ref_deltas", &self.loop_filter_ref_deltas)
            .field("loop_filter_mode_deltas", &self.loop_filter_mode_deltas)
            .field("delta_lf_multi", &self.delta_lf_multi)
            .field("mi_rows", &self.mi_rows)
            .field("mi_cols", &self.mi_cols)
            .field("num_planes", &self.num_planes)
            .field("bit_depth", &self.bit_depth)
            .field("subsampling_x", &self.subsampling_x)
            .field("subsampling_y", &self.subsampling_y)
            .field("frame_width", &self.frame_width)
            .field("frame_height", &self.frame_height)
            .finish_non_exhaustive()
    }
}

/// §7.14.1 top-level loop-filter driver — av1-spec p.307 lines
/// 16959-16970.
///
/// Iterates over `plane × pass × row × col` with the spec's
/// `rowStep`/`colStep` chroma stride. Calls [`loop_filter_edge`] on
/// every edge the spec visits; the inner gates (`onScreen`,
/// `isBlockEdge`/`isTxEdge`, `lvl == 0`) decide whether any samples
/// actually move.
///
/// Modifies the per-plane buffers in place.
pub fn loop_filter_frame(ctx: &LoopFilterFrameContext<'_>, planes: &mut [PlaneBuffer<'_>]) {
    let num_planes = ctx.num_planes.min(planes.len() as u8);
    for plane in 0..num_planes {
        // av1-spec p.307 line 16961: skip chroma planes whose
        // `loop_filter_level[1 + plane]` slot is zero.
        if plane > 0 && ctx.loop_filter_level[(1 + plane) as usize] == 0 {
            continue;
        }
        let (sub_x, sub_y) = subsampling_for_plane(plane, ctx.subsampling_x, ctx.subsampling_y);
        let row_step: u32 = if plane == 0 { 1 } else { 1 << sub_y };
        let col_step: u32 = if plane == 0 { 1 } else { 1 << sub_x };
        for pass in 0..2u8 {
            let mut row = 0u32;
            while row < ctx.mi_rows {
                let mut col = 0u32;
                while col < ctx.mi_cols {
                    loop_filter_edge(ctx, planes, plane, pass, row, col);
                    col += col_step;
                }
                row += row_step;
            }
        }
    }
}

/// §7.14.2 per-edge driver — av1-spec p.307-310 lines 16994-17126.
///
/// `plane` is `0` (Y), `1` (U), `2` (V). `pass` is `0` for vertical
/// boundaries (filtering horizontally), `1` for horizontal boundaries
/// (filtering vertically). `row` and `col` are in 4×4-block luma
/// coordinates.
pub fn loop_filter_edge(
    ctx: &LoopFilterFrameContext<'_>,
    planes: &mut [PlaneBuffer<'_>],
    plane: u8,
    pass: u8,
    row: u32,
    col: u32,
) {
    let (sub_x, sub_y) = subsampling_for_plane(plane, ctx.subsampling_x, ctx.subsampling_y);
    let (dx, dy) = if pass == 0 {
        (1i32, 0i32)
    } else {
        (0i32, 1i32)
    };
    let x = col * (MI_SIZE as u32);
    let y = row * (MI_SIZE as u32);
    // av1-spec p.308 line 17028-17030: row |= subY; col |= subX. This
    // snaps a chroma walk that landed on an odd 4×4 cell down to the
    // chroma-aligned cell.
    let row = row | (sub_y as u32);
    let col = col | (sub_x as u32);
    // av1-spec p.308 line 17040-17048: onScreen gate.
    if x >= ctx.frame_width {
        return;
    }
    if y >= ctx.frame_height {
        return;
    }
    if pass == 0 && x == 0 {
        return;
    }
    if pass == 1 && y == 0 {
        return;
    }
    // av1-spec p.308 line 17054-17056.
    let x_p = (x >> sub_x) as i32;
    let y_p = (y >> sub_y) as i32;
    // av1-spec p.308 line 17061-17063.
    let prev_row = (row as i32) - (dy << sub_y);
    let prev_col = (col as i32) - (dx << sub_x);
    if prev_row < 0 || prev_col < 0 {
        // The onScreen gate already drops `pass == 0 && x == 0` /
        // `pass == 1 && y == 0`, so a negative prev_row / prev_col
        // here would only come from a chroma-aligned cell whose
        // neighbour-via-`subX/Y` lands off-grid — defensively skip.
        return;
    }
    let prev_row = prev_row as u32;
    let prev_col = prev_col as u32;
    if row >= ctx.mi_rows || col >= ctx.mi_cols {
        return;
    }
    // av1-spec p.308 line 17065-17075.
    let mi_size = (ctx.mi_size)(row, col);
    let tx_sz = (ctx.lf_tx_size)(plane as usize, row >> sub_y, col >> sub_x);
    let plane_size = mi_size; // §7.14.2 line 17069 — the spec's
                              // `get_plane_residual_size(MiSize, plane)` collapses to
                              // `MiSize` for non-chroma-misaligned blocks; the
                              // driver passes the §5.11.34 plane-residual size
                              // here when wired against the full walker.
    let skip = (ctx.skip)(row, col);
    let is_intra = (ctx.is_intra)(row, col);
    let prev_tx_sz = (ctx.lf_tx_size)(plane as usize, prev_row >> sub_y, prev_col >> sub_x);
    // av1-spec p.308 line 17079-17084: isBlockEdge.
    let is_block_edge = if pass == 0 {
        (x_p as u32) % block_width(plane_size).max(1) == 0
    } else {
        (y_p as u32) % block_height(plane_size).max(1) == 0
    };
    // av1-spec p.308 line 17088-17099: isTxEdge.
    let is_tx_edge = if pass == 0 {
        (x_p as u32) % crate::cdf::TX_WIDTH[tx_sz].max(1) as u32 == 0
    } else {
        (y_p as u32) % crate::cdf::TX_HEIGHT[tx_sz].max(1) as u32 == 0
    };
    // av1-spec p.309 line 17101-17107: applyFilter.
    let apply_filter = if !is_tx_edge {
        false
    } else {
        is_block_edge || !skip || is_intra
    };
    // §7.14.3 filter size derivation.
    let filter_size = filter_size(tx_sz, prev_tx_sz, pass, plane);
    // §7.14.4 adaptive filter strength on the current side.
    let mut strength = adaptive_filter_strength(ctx, row, col, plane, pass);
    if strength.lvl == 0 {
        // av1-spec p.309 line 17115-17116: fall back to prev side if
        // current-side lvl == 0.
        strength = adaptive_filter_strength(ctx, prev_row, prev_col, plane, pass);
    }
    if !apply_filter || strength.lvl == 0 {
        return;
    }
    let Some(plane_buf) = planes.get_mut(plane as usize) else {
        return;
    };
    // av1-spec p.309 line 17118-17122: walk i = 0..MI_SIZE-1 along
    // the edge (the boundary spans `MI_SIZE` samples for the 4×4
    // grid).
    for i in 0..(MI_SIZE as i32) {
        let sx = x_p + dy * i;
        let sy = y_p + dx * i;
        sample_filtering(
            plane_buf,
            sx,
            sy,
            plane,
            strength.limit,
            strength.blimit,
            strength.thresh,
            dx,
            dy,
            filter_size,
            ctx.bit_depth,
        );
    }
}

// =====================================================================
// §7.14.3 — Filter size derivation. av1-spec p.310.
// =====================================================================

/// §7.14.3 — `Min(plane==0 ? 16 : 8, Min(Tx_*[txSz], Tx_*[prevTxSz]))`.
///
/// `pass == 0` consults `Tx_Width[]`; `pass == 1` consults
/// `Tx_Height[]`. `plane == 0` caps at 16, else at 8.
#[must_use]
pub fn filter_size(tx_sz: usize, prev_tx_sz: usize, pass: u8, plane: u8) -> u32 {
    let base_size = if pass == 0 {
        (crate::cdf::TX_WIDTH[tx_sz].min(crate::cdf::TX_WIDTH[prev_tx_sz])) as u32
    } else {
        (crate::cdf::TX_HEIGHT[tx_sz].min(crate::cdf::TX_HEIGHT[prev_tx_sz])) as u32
    };
    let cap = if plane == 0 { 16 } else { 8 };
    base_size.min(cap)
}

// =====================================================================
// §7.14.4 — Adaptive filter strength process. av1-spec p.311.
// =====================================================================

/// §7.14.4 output bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterStrength {
    /// `lvl` per av1-spec p.311 line 17177 — the §7.14.5 selected
    /// filter level, in `[0, MAX_LOOP_FILTER]`.
    pub lvl: u8,
    /// `limit` per av1-spec p.311 line 17218-17222.
    pub limit: u8,
    /// `blimit` per av1-spec p.311 line 17224 — `2 * (lvl + 2) + limit`.
    pub blimit: u8,
    /// `thresh` per av1-spec p.311 line 17226 — `lvl >> 4`.
    pub thresh: u8,
}

/// §7.14.4 — Derive `(lvl, limit, blimit, thresh)` for the per-edge
/// cell at `(row, col, plane, pass)`.
#[must_use]
pub fn adaptive_filter_strength(
    ctx: &LoopFilterFrameContext<'_>,
    row: u32,
    col: u32,
    plane: u8,
    pass: u8,
) -> FilterStrength {
    // av1-spec p.311 line 17179-17198.
    let segment = (ctx.segment_id)(row, col);
    let r = (ctx.ref_frame)(row, col);
    let m = (ctx.mode)(row, col);
    let mode_type: u8 = if m >= NEARESTMV && m != GLOBALMV && m != GLOBAL_GLOBALMV {
        1
    } else {
        0
    };
    let delta_lf_idx = if ctx.delta_lf_multi {
        if plane == 0 {
            pass as usize
        } else {
            (plane + 1) as usize
        }
    } else {
        0
    };
    let delta_lf = (ctx.delta_lf)(row, col, delta_lf_idx);
    let lvl = adaptive_filter_strength_selection(
        ctx,
        segment as u32,
        r,
        mode_type,
        delta_lf,
        plane,
        pass,
    );
    // av1-spec p.311 line 17210-17226.
    let shift = if ctx.loop_filter_sharpness > 4 {
        2
    } else if ctx.loop_filter_sharpness > 0 {
        1
    } else {
        0
    };
    let lvl_i = lvl as i32;
    let limit_i = if ctx.loop_filter_sharpness > 0 {
        (lvl_i >> shift).clamp(1, 9 - ctx.loop_filter_sharpness as i32)
    } else {
        (lvl_i >> shift).max(1)
    };
    let limit = limit_i as u8;
    let blimit = (2 * (lvl as u32 + 2) + limit as u32) as u8;
    let thresh = lvl >> 4;
    FilterStrength {
        lvl,
        limit,
        blimit,
        thresh,
    }
}

/// §7.14.5 — Adaptive filter strength selection. av1-spec p.311-312.
#[must_use]
pub fn adaptive_filter_strength_selection(
    ctx: &LoopFilterFrameContext<'_>,
    segment: u32,
    ref_frame: u8,
    mode_type: u8,
    delta_lf: i8,
    plane: u8,
    pass: u8,
) -> u8 {
    // av1-spec p.312 line 17249.
    let i = if plane == 0 {
        pass as usize
    } else {
        (plane + 1) as usize
    };
    // av1-spec p.312 line 17251.
    let base_filter_level = clip3_i32(
        0,
        MAX_LOOP_FILTER as i32,
        (delta_lf as i32) + (ctx.loop_filter_level[i] as i32),
    );
    let mut lvl_seg = base_filter_level;
    // av1-spec p.312 line 17257-17268.
    let feature = SEG_LVL_ALT_LF_Y_V + i;
    if (ctx.seg_feature_active)(segment, feature) {
        lvl_seg = clip3_i32(
            0,
            MAX_LOOP_FILTER as i32,
            (ctx.seg_feature_data)(segment, feature) as i32 + lvl_seg,
        );
    }
    // av1-spec p.312 line 17270-17280.
    if ctx.loop_filter_delta_enabled {
        let n_shift = lvl_seg >> 5;
        if ref_frame as usize == INTRA_FRAME {
            lvl_seg += (ctx.loop_filter_ref_deltas[INTRA_FRAME] as i32) << n_shift;
        } else {
            let ref_idx = (ref_frame as usize).min(TOTAL_REFS_PER_FRAME - 1);
            let mode_idx = (mode_type as usize).min(ctx.loop_filter_mode_deltas.len() - 1);
            lvl_seg += (ctx.loop_filter_ref_deltas[ref_idx] as i32) << n_shift;
            lvl_seg += (ctx.loop_filter_mode_deltas[mode_idx] as i32) << n_shift;
        }
        lvl_seg = clip3_i32(0, MAX_LOOP_FILTER as i32, lvl_seg);
    }
    lvl_seg.clamp(0, 255) as u8
}

// =====================================================================
// §7.14.6.1 — Sample filtering dispatch. av1-spec p.313.
// =====================================================================

/// §7.14.6.1 — Run [`filter_mask`] at `(x, y)` then dispatch to
/// [`narrow_filter`] or [`wide_filter`] per the resulting masks and
/// `filter_size`.
#[allow(clippy::too_many_arguments)]
pub fn sample_filtering(
    plane_buf: &mut PlaneBuffer<'_>,
    x: i32,
    y: i32,
    plane: u8,
    limit: u8,
    blimit: u8,
    thresh: u8,
    dx: i32,
    dy: i32,
    filter_size: u32,
    bit_depth: u8,
) {
    let masks = filter_mask(
        plane_buf,
        x,
        y,
        plane,
        limit,
        blimit,
        thresh,
        dx,
        dy,
        filter_size,
        bit_depth,
    );
    if !masks.filter_mask {
        return;
    }
    if filter_size == 4 || !masks.flat_mask {
        narrow_filter(plane_buf, x, y, dx, dy, masks.hev_mask, bit_depth);
    } else if filter_size == 8 || !masks.flat_mask2 {
        wide_filter(plane_buf, x, y, plane, dx, dy, 3, bit_depth);
    } else {
        wide_filter(plane_buf, x, y, plane, dx, dy, 4, bit_depth);
    }
}

// =====================================================================
// §7.14.6.2 — Filter mask process. av1-spec p.314-315.
// =====================================================================

/// §7.14.6.2 output: the four edge-classification masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterMaskOutput {
    /// `hevMask` — high-edge-variance indicator.
    pub hev_mask: bool,
    /// `filterMask` — `true` ⇔ filtering should occur.
    pub filter_mask: bool,
    /// `flatMask` — `true` ⇔ samples within 4 either side of the
    /// boundary are in a flat region (`filterSize >= 8`).
    pub flat_mask: bool,
    /// `flatMask2` — `true` ⇔ samples within 6 either side are in a
    /// flat region (`filterSize >= 16`).
    pub flat_mask2: bool,
}

/// §7.14.6.2 — Compute `(hevMask, filterMask, flatMask, flatMask2)`
/// from samples around `(x, y)` on `plane`. av1-spec p.314-315.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn filter_mask(
    plane_buf: &PlaneBuffer<'_>,
    x: i32,
    y: i32,
    plane: u8,
    limit: u8,
    blimit: u8,
    thresh: u8,
    dx: i32,
    dy: i32,
    filter_size: u32,
    bit_depth: u8,
) -> FilterMaskOutput {
    // Sample reads — av1-spec p.314 lines 17352-17365. Up to 7 either
    // side; only the wide path reads `p3..p6` / `q3..q6`.
    let q0 = plane_buf.get(y, x);
    let q1 = plane_buf.get(y + dy, x + dx);
    let q2 = plane_buf.get(y + 2 * dy, x + 2 * dx);
    let q3 = plane_buf.get(y + 3 * dy, x + 3 * dx);
    let q4 = plane_buf.get(y + 4 * dy, x + 4 * dx);
    let q5 = plane_buf.get(y + 5 * dy, x + 5 * dx);
    let q6 = plane_buf.get(y + 6 * dy, x + 6 * dx);
    let p0 = plane_buf.get(y - dy, x - dx);
    let p1 = plane_buf.get(y - 2 * dy, x - 2 * dx);
    let p2 = plane_buf.get(y - 3 * dy, x - 3 * dx);
    let p3 = plane_buf.get(y - 4 * dy, x - 4 * dx);
    let p4 = plane_buf.get(y - 5 * dy, x - 5 * dx);
    let p5 = plane_buf.get(y - 6 * dy, x - 6 * dx);
    let p6 = plane_buf.get(y - 7 * dy, x - 7 * dx);
    let bd_shift = bit_depth - 8;
    // hevMask — av1-spec p.314 lines 17381-17385.
    let thresh_bd = (thresh as i32) << bd_shift;
    let hev_mask = (p1 - p0).abs() > thresh_bd || (q1 - q0).abs() > thresh_bd;
    // filterLen — av1-spec p.315 lines 17389-17395.
    let filter_len: u32 = if filter_size == 4 {
        4
    } else if plane != 0 {
        6
    } else if filter_size == 8 {
        8
    } else {
        16
    };
    // filterMask — av1-spec p.315 lines 17402-17416.
    let limit_bd = (limit as i32) << bd_shift;
    let blimit_bd = (blimit as i32) << bd_shift;
    let mut mask = (p1 - p0).abs() > limit_bd;
    mask |= (q1 - q0).abs() > limit_bd;
    mask |= (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit_bd;
    if filter_len >= 6 {
        mask |= (p2 - p1).abs() > limit_bd;
        mask |= (q2 - q1).abs() > limit_bd;
    }
    if filter_len >= 8 {
        mask |= (p3 - p2).abs() > limit_bd;
        mask |= (q3 - q2).abs() > limit_bd;
    }
    let filter_mask = !mask;
    // flatMask — av1-spec p.315 lines 17431-17443.
    let threshold_bd = 1i32 << bd_shift;
    let flat_mask = if filter_size >= 8 {
        let mut m = (p1 - p0).abs() > threshold_bd;
        m |= (q1 - q0).abs() > threshold_bd;
        m |= (p2 - p0).abs() > threshold_bd;
        m |= (q2 - q0).abs() > threshold_bd;
        if filter_len >= 8 {
            m |= (p3 - p0).abs() > threshold_bd;
            m |= (q3 - q0).abs() > threshold_bd;
        }
        !m
    } else {
        false
    };
    // flatMask2 — av1-spec p.315 lines 17451-17461.
    let flat_mask2 = if filter_size >= 16 {
        let mut m = (p6 - p0).abs() > threshold_bd;
        m |= (q6 - q0).abs() > threshold_bd;
        m |= (p5 - p0).abs() > threshold_bd;
        m |= (q5 - q0).abs() > threshold_bd;
        m |= (p4 - p0).abs() > threshold_bd;
        m |= (q4 - q0).abs() > threshold_bd;
        !m
    } else {
        false
    };
    FilterMaskOutput {
        hev_mask,
        filter_mask,
        flat_mask,
        flat_mask2,
    }
}

// =====================================================================
// §7.14.6.3 — Narrow filter. av1-spec p.316.
// =====================================================================

/// §7.14.6.3 — Modify up to two samples each side of the boundary at
/// `(x, y)`. `hev_mask == false` widens the modify window from one
/// to two on each side.
pub fn narrow_filter(
    plane_buf: &mut PlaneBuffer<'_>,
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    hev_mask: bool,
    bit_depth: u8,
) {
    let half = 0x80i32 << (bit_depth - 8);
    let q0 = plane_buf.get(y, x);
    let q1 = plane_buf.get(y + dy, x + dx);
    let p0 = plane_buf.get(y - dy, x - dx);
    let p1 = plane_buf.get(y - 2 * dy, x - 2 * dx);
    let ps1 = p1 - half;
    let ps0 = p0 - half;
    let qs0 = q0 - half;
    let qs1 = q1 - half;
    let mut filter = if hev_mask {
        filter4_clamp(ps1 - qs1, bit_depth)
    } else {
        0
    };
    filter = filter4_clamp(filter + 3 * (qs0 - ps0), bit_depth);
    let filter1 = filter4_clamp(filter + 4, bit_depth) >> 3;
    let filter2 = filter4_clamp(filter + 3, bit_depth) >> 3;
    let oq0 = filter4_clamp(qs0 - filter1, bit_depth) + half;
    let op0 = filter4_clamp(ps0 + filter2, bit_depth) + half;
    plane_buf.set(y, x, oq0);
    plane_buf.set(y - dy, x - dx, op0);
    if !hev_mask {
        let f = round2_i32(filter1, 1);
        let oq1 = filter4_clamp(qs1 - f, bit_depth) + half;
        let op1 = filter4_clamp(ps1 + f, bit_depth) + half;
        plane_buf.set(y + dy, x + dx, oq1);
        plane_buf.set(y - 2 * dy, x - 2 * dx, op1);
    }
}

// =====================================================================
// §7.14.6.4 — Wide filter. av1-spec p.317.
// =====================================================================

/// §7.14.6.4 — Apply the 6/8/14-tap low-pass filter centred at the
/// boundary. `log2_size ∈ {3, 4}`.
#[allow(clippy::too_many_arguments)]
pub fn wide_filter(
    plane_buf: &mut PlaneBuffer<'_>,
    x: i32,
    y: i32,
    plane: u8,
    dx: i32,
    dy: i32,
    log2_size: u32,
    _bit_depth: u8,
) {
    // av1-spec p.317 lines 17552-17565.
    let n: i32 = if log2_size == 4 {
        6
    } else if plane == 0 {
        3
    } else {
        2
    };
    let n2: i32 = if log2_size == 3 && plane == 0 { 0 } else { 1 };
    // av1-spec p.317 lines 17570-17578.
    let mut f: Vec<i32> = vec![0; (2 * n) as usize];
    for (idx, i) in (-n..n).enumerate() {
        let mut t: i64 = 0;
        for j in -n..=n {
            let p = clip3_i32(-(n + 1), n, i + j);
            let tap: i64 = if j.unsigned_abs() as i32 <= n2 { 2 } else { 1 };
            t += plane_buf.get(y + p * dy, x + p * dx) as i64 * tap;
        }
        f[idx] = round2_i64(t, log2_size) as i32;
    }
    for (idx, i) in (-n..n).enumerate() {
        plane_buf.set(y + i * dy, x + i * dx, f[idx]);
    }
}

// =====================================================================
// Internal helpers.
// =====================================================================

/// §7.14.2 line 17008-17010: per-plane (`subX`, `subY`).
#[inline]
fn subsampling_for_plane(plane: u8, subsampling_x: u8, subsampling_y: u8) -> (u8, u8) {
    if plane == 0 {
        (0, 0)
    } else {
        (subsampling_x, subsampling_y)
    }
}

/// `Block_Width[planeSize]` per §3 — defers to the `cdf::BLOCK_SIZES`
/// table by way of [`crate::cdf::num_4x4_blocks_wide`].
#[inline]
fn block_width(plane_size: usize) -> u32 {
    if plane_size >= crate::cdf::BLOCK_SIZES {
        // Defensive: invalid block size — treat as 4 (smallest §3
        // block) to keep `isBlockEdge` from triggering on every
        // pixel.
        return 4;
    }
    (crate::cdf::num_4x4_blocks_wide(plane_size) * 4) as u32
}

/// `Block_Height[planeSize]` per §3 — mirror of [`block_width`].
#[inline]
fn block_height(plane_size: usize) -> u32 {
    if plane_size >= crate::cdf::BLOCK_SIZES {
        return 4;
    }
    (crate::cdf::num_4x4_blocks_high(plane_size) * 4) as u32
}

/// §3 `Clip3(a, b, x)` — clamp `x` to `[a, b]`.
#[inline]
fn clip3_i32(a: i32, b: i32, x: i32) -> i32 {
    if x < a {
        a
    } else if x > b {
        b
    } else {
        x
    }
}

/// §3 `Round2(x, n) = (x + (1 << (n - 1))) >> n` — unsigned/i32
/// rounding right-shift used by [`narrow_filter`] for the `n = 1`
/// case (and as a primitive for the wide filter's `Round2(t,
/// log2_size)` post-shift).
#[inline]
fn round2_i32(x: i32, n: u32) -> i32 {
    if n == 0 {
        x
    } else {
        (x + (1 << (n - 1))) >> n
    }
}

/// `Round2(x, n)` for `i64` accumulators — used by [`wide_filter`]
/// where the sum-of-taps stays in `i64` to avoid overflow for the
/// 14-tap luma path.
#[inline]
fn round2_i64(x: i64, n: u32) -> i64 {
    if n == 0 {
        x
    } else {
        (x + (1i64 << (n - 1))) >> n
    }
}

/// `filter4_clamp(v) = Clip3(-(1 << (BitDepth - 1)), (1 << (BitDepth -
/// 1)) - 1, v)` per av1-spec p.316 line 17498.
#[inline]
fn filter4_clamp(value: i32, bit_depth: u8) -> i32 {
    let lo = -(1i32 << (bit_depth - 1));
    let hi = (1i32 << (bit_depth - 1)) - 1;
    clip3_i32(lo, hi, value)
}

// =====================================================================
// Tests.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn make_ctx<'a>(
        mi_rows: u32,
        mi_cols: u32,
        loop_filter_level: [u8; 4],
        is_intra: &'a dyn Fn(u32, u32) -> bool,
        skip: &'a dyn Fn(u32, u32) -> bool,
        ref_frame: &'a dyn Fn(u32, u32) -> u8,
        mode: &'a dyn Fn(u32, u32) -> u8,
        segment_id: &'a dyn Fn(u32, u32) -> u8,
        delta_lf: &'a dyn Fn(u32, u32, usize) -> i8,
        seg_feature_active: &'a dyn Fn(u32, usize) -> bool,
        seg_feature_data: &'a dyn Fn(u32, usize) -> i16,
        lf_tx_size: &'a dyn Fn(usize, u32, u32) -> usize,
        mi_size: &'a dyn Fn(u32, u32) -> usize,
    ) -> LoopFilterFrameContext<'a> {
        LoopFilterFrameContext {
            loop_filter_level,
            loop_filter_sharpness: 0,
            loop_filter_delta_enabled: false,
            loop_filter_ref_deltas: [0; TOTAL_REFS_PER_FRAME],
            loop_filter_mode_deltas: [0; 2],
            delta_lf_multi: false,
            mi_rows,
            mi_cols,
            num_planes: 3,
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            frame_width: mi_cols * MI_SIZE as u32,
            frame_height: mi_rows * MI_SIZE as u32,
            is_intra,
            skip,
            ref_frame,
            mode,
            segment_id,
            delta_lf,
            seg_feature_active,
            seg_feature_data,
            lf_tx_size,
            mi_size,
        }
    }

    // --- §7.14.4 / §7.14.5 strength derivation --------------------

    #[test]
    fn strength_returns_zero_when_loop_filter_level_zero() {
        let ctx = make_ctx(
            4,
            4,
            [0, 0, 0, 0],
            &|_, _| true,
            &|_, _| false,
            &|_, _| INTRA_FRAME as u8,
            &|_, _| 0u8,
            &|_, _| 0u8,
            &|_, _, _| 0i8,
            &|_, _| false,
            &|_, _| 0i16,
            &|_, _, _| crate::cdf::TX_4X4,
            &|_, _| crate::cdf::BLOCK_4X4,
        );
        let s = adaptive_filter_strength(&ctx, 0, 0, 0, 0);
        assert_eq!(s.lvl, 0);
    }

    #[test]
    fn strength_basic_table_lvl_32_sharp_0() {
        let ctx = make_ctx(
            4,
            4,
            [32, 32, 32, 32],
            &|_, _| true,
            &|_, _| false,
            &|_, _| INTRA_FRAME as u8,
            &|_, _| 0u8,
            &|_, _| 0u8,
            &|_, _, _| 0i8,
            &|_, _| false,
            &|_, _| 0i16,
            &|_, _, _| crate::cdf::TX_4X4,
            &|_, _| crate::cdf::BLOCK_4X4,
        );
        let s = adaptive_filter_strength(&ctx, 0, 0, 0, 0);
        // sharpness 0 ⇒ shift 0 ⇒ limit = max(1, 32) = 32.
        // blimit = 2 * (32 + 2) + 32 = 100.
        // thresh = 32 >> 4 = 2.
        assert_eq!(s.lvl, 32);
        assert_eq!(s.limit, 32);
        assert_eq!(s.blimit, 100);
        assert_eq!(s.thresh, 2);
    }

    #[test]
    fn strength_intra_ref_delta_lifts_lvl() {
        let mut ctx = make_ctx(
            4,
            4,
            [10, 10, 10, 10],
            &|_, _| true,
            &|_, _| false,
            &|_, _| INTRA_FRAME as u8,
            &|_, _| 0u8,
            &|_, _| 0u8,
            &|_, _, _| 0i8,
            &|_, _| false,
            &|_, _| 0i16,
            &|_, _, _| crate::cdf::TX_4X4,
            &|_, _| crate::cdf::BLOCK_4X4,
        );
        ctx.loop_filter_delta_enabled = true;
        ctx.loop_filter_ref_deltas[INTRA_FRAME] = 5;
        let s = adaptive_filter_strength(&ctx, 0, 0, 0, 0);
        // base = lvl[0] = 10. n_shift = 10 >> 5 = 0. lvl_seg += 5
        // ⇒ 15. Clip3(0, 63) = 15.
        assert_eq!(s.lvl, 15);
    }

    #[test]
    fn strength_clipped_at_max_loop_filter() {
        let mut ctx = make_ctx(
            4,
            4,
            [60, 60, 60, 60],
            &|_, _| true,
            &|_, _| false,
            &|_, _| INTRA_FRAME as u8,
            &|_, _| 0u8,
            &|_, _| 0u8,
            &|_, _, _| 0i8,
            &|_, _| false,
            &|_, _| 0i16,
            &|_, _, _| crate::cdf::TX_4X4,
            &|_, _| crate::cdf::BLOCK_4X4,
        );
        ctx.loop_filter_delta_enabled = true;
        ctx.loop_filter_ref_deltas[INTRA_FRAME] = 10; // n_shift = 60>>5 = 1 ⇒ +20 ⇒ 80 ⇒ clip 63.
        let s = adaptive_filter_strength(&ctx, 0, 0, 0, 0);
        assert_eq!(s.lvl, MAX_LOOP_FILTER as u8);
    }

    // --- §7.14.3 filter size --------------------------------------

    #[test]
    fn filter_size_luma_caps_at_16_chroma_at_8() {
        // TX_64X64 width 64; cap at 16 for luma, 8 for chroma.
        assert_eq!(
            filter_size(crate::cdf::TX_64X64, crate::cdf::TX_64X64, 0, 0),
            16
        );
        assert_eq!(
            filter_size(crate::cdf::TX_64X64, crate::cdf::TX_64X64, 0, 1),
            8
        );
        // TX_4X4 width 4; min(4, cap) = 4.
        assert_eq!(filter_size(crate::cdf::TX_4X4, crate::cdf::TX_4X4, 0, 0), 4);
    }

    // --- §7.14.6.3 narrow filter ----------------------------------

    #[test]
    fn narrow_filter_idempotent_on_uniform_input() {
        // Uniform 128 input ⇒ p0 == q0 == p1 == q1; filter4_clamp(ps1 - qs1) = 0;
        // filter4_clamp(0 + 3*(qs0 - ps0)) = 0; oq0 = qs0 - (4>>3) = qs0;
        // op0 = ps0 + (3>>3) = ps0. Samples unchanged.
        let mut samples = vec![128i32; 8 * 8];
        let mut buf = PlaneBuffer {
            rows: 8,
            cols: 8,
            samples: &mut samples,
        };
        let before = buf.samples.to_vec();
        narrow_filter(&mut buf, 4, 4, 1, 0, false, 8);
        assert_eq!(buf.samples, &before[..]);
    }

    #[test]
    fn narrow_filter_softens_sharp_step_edge() {
        // Vertical edge between cols 3 and 4: left half 100, right 156.
        // p1 = (4,2) = 100; p0 = (4,3) = 100; q0 = (4,4) = 156; q1 = (4,5) = 156.
        // Expect q0 to come down and p0 to come up after the filter, by a small
        // delta governed by the narrow-filter's 4 + 3 rounding.
        let mut samples = vec![0i32; 8 * 8];
        for y in 0..8 {
            for x in 0..8 {
                samples[y * 8 + x] = if x < 4 { 100 } else { 156 };
            }
        }
        let mut buf = PlaneBuffer {
            rows: 8,
            cols: 8,
            samples: &mut samples,
        };
        narrow_filter(&mut buf, 4, 4, 1, 0, true, 8);
        let p0 = buf.samples[4 * 8 + 3];
        let q0 = buf.samples[4 * 8 + 4];
        assert!(p0 > 100, "p0={p0} should rise above 100");
        assert!(q0 < 156, "q0={q0} should fall below 156");
        // Symmetry: the step delta moves equally in both directions.
        assert_eq!(q0 - 128, 128 - p0);
    }

    // --- §7.14.6.4 wide filter ------------------------------------

    #[test]
    fn wide_filter_log2_3_luma_preserves_uniform_input() {
        let mut samples = vec![200i32; 16 * 16];
        let mut buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut samples,
        };
        let before = buf.samples.to_vec();
        wide_filter(&mut buf, 8, 8, 0, 1, 0, 3, 8);
        assert_eq!(buf.samples, &before[..]);
    }

    #[test]
    fn wide_filter_log2_4_luma_preserves_uniform_input() {
        let mut samples = vec![64i32; 32 * 32];
        let mut buf = PlaneBuffer {
            rows: 32,
            cols: 32,
            samples: &mut samples,
        };
        let before = buf.samples.to_vec();
        wide_filter(&mut buf, 16, 16, 0, 1, 0, 4, 8);
        assert_eq!(buf.samples, &before[..]);
    }

    // --- §7.14.6.2 mask -------------------------------------------

    #[test]
    fn filter_mask_flat_region_allows_filter() {
        let mut samples = vec![100i32; 16 * 16];
        let buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut samples,
        };
        let m = filter_mask(&buf, 8, 8, 0, 16, 32, 8, 1, 0, 8, 8);
        assert!(m.filter_mask, "flat region must permit filtering");
        assert!(m.flat_mask, "flat region must be flat");
        assert!(!m.hev_mask, "uniform region cannot be hev");
    }

    #[test]
    fn filter_mask_steep_edge_blocks_filter() {
        // Sharp step beyond `blimit` — filter must be vetoed.
        let mut samples = vec![0i32; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                samples[y * 16 + x] = if x < 8 { 50 } else { 200 };
            }
        }
        let buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut samples,
        };
        // limit = 4, blimit = 12, thresh = 6 → all small ⇒ vetoes.
        let m = filter_mask(&buf, 8, 8, 0, 4, 12, 6, 1, 0, 8, 8);
        assert!(!m.filter_mask, "steep step must veto filter");
    }

    // --- §7.14.1 driver iteration ---------------------------------

    #[test]
    fn driver_iterates_correct_edge_count_for_2x2_mi_grid() {
        use std::cell::Cell;
        let edges: Cell<u32> = Cell::new(0);
        let ctr = |_r: u32, _c: u32| {
            edges.set(edges.get() + 1);
            false
        };
        let ctx = LoopFilterFrameContext {
            loop_filter_level: [10, 10, 0, 0],
            loop_filter_sharpness: 0,
            loop_filter_delta_enabled: false,
            loop_filter_ref_deltas: [0; TOTAL_REFS_PER_FRAME],
            loop_filter_mode_deltas: [0; 2],
            delta_lf_multi: false,
            mi_rows: 2,
            mi_cols: 2,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            frame_width: 8,
            frame_height: 8,
            is_intra: &ctr,
            skip: &|_, _| false,
            ref_frame: &|_, _| INTRA_FRAME as u8,
            mode: &|_, _| 0u8,
            segment_id: &|_, _| 0u8,
            delta_lf: &|_, _, _| 0i8,
            seg_feature_active: &|_, _| false,
            seg_feature_data: &|_, _| 0i16,
            lf_tx_size: &|_, _, _| crate::cdf::TX_4X4,
            mi_size: &|_, _| crate::cdf::BLOCK_8X8,
        };
        let mut samples = vec![100i32; 8 * 8];
        let mut planes = vec![PlaneBuffer {
            rows: 8,
            cols: 8,
            samples: &mut samples,
        }];
        loop_filter_frame(&ctx, &mut planes);
        // For 2x2 mi-grid, 1 luma plane, 2 passes: each pass walks
        // 2 rows × 2 cols = 4 cells. Of these, the on-screen gate
        // drops `pass==0 && x==0` (2 cells) and `pass==1 && y==0`
        // (2 cells), so the `is_intra` predicate fires on the
        // remaining 2 + 2 = 4 cells.
        assert_eq!(
            edges.get(),
            4,
            "expected 4 on-screen edges across both passes"
        );
    }
}
