//! §7.17 Loop restoration — the post-CDEF restoration pass that sits
//! **after** the §7.15 CDEF de-ringing pass per av1-spec p.327-335.
//!
//! ## Coverage (round 197 — close-out push)
//!
//! This module covers the §7.17 loop-restoration top-level driver and
//! the §7.17.4 Wiener arm end-to-end at the sample-filtering level:
//!
//! * §7.17 — [`loop_restoration_frame`]: top-level walk over `(y, x)`
//!   in `MI_SIZE` steps, dispatching per plane / per
//!   `FrameRestorationType` to the per-block driver.
//! * §7.17.1 — [`loop_restore_block`]: per-`MI_SIZE` block driver that
//!   derives `(unitRow, unitCol, x, y, w, h, StripeStartY, StripeEndY,
//!   PlaneEndX, PlaneEndY)` and dispatches by `rType` to the Wiener,
//!   self-guided, or no-op arm.
//! * §7.17.4 — [`wiener_filter`]: 7-tap horizontal × 7-tap vertical
//!   separable convolution with the §7.11.3.2 `(InterRound0,
//!   InterRound1)` rounding-shift schedule.
//! * §7.17.5 — [`wiener_coefficients`]: reconstruct the symmetric 7-tap
//!   filter (`filter[3] = 128 - 2 * Σ coeff[i]`; `filter[i] =
//!   filter[6 - i] = coeff[i]`).
//! * §7.17.6 — [`get_source_sample`]: per-sample fetch that snaps to
//!   `[0, PlaneEndX] × [0, PlaneEndY]` and routes to either the
//!   `UpscaledCdefFrame` (inside the current stripe) or the
//!   `UpscaledCurrFrame` (above / below the stripe, with the cropped
//!   `StripeStartY - 2` / `StripeEndY + 2` neighbour-line reach).
//!
//! ## Constant tables (av1-spec p.107 / p.332)
//!
//! * [`WIENER_TAPS_MID`] / [`WIENER_TAPS_MIN`] / [`WIENER_TAPS_MAX`] /
//!   [`WIENER_TAPS_K`] — §5.11.57 / §5.11.58 reference midpoint, range,
//!   and subexp `k` for the three transmitted Wiener coefficients.
//! * [`SGRPROJ_XQD_MID`] / [`SGRPROJ_XQD_MIN`] / [`SGRPROJ_XQD_MAX`] —
//!   §5.11.57 reference midpoint and range for the two transmitted
//!   self-guided projection weights.
//! * [`SGR_PARAMS`] — §7.17.3 `Sgr_Params[16][4]` per av1-spec p.332
//!   lines 18395-18400. Each `set` row is `(r0, eps0, r1, eps1)`.
//!
//! ## Self-guided projection arm
//!
//! The §7.17.2 self-guided projection arm is currently stubbed: a unit
//! whose `FrameRestorationType` is `RESTORE_SGRPROJ` falls through to a
//! straight `CdefFrame ↦ LrFrame` copy (the same effect as
//! `RESTORE_NONE`) and the driver returns `Ok(())`. The §7.17.3 box
//! filter (`A[]/B[]` integral-image accumulator + `Sgr_Params`-driven
//! eps scaling) lands in the next arc; constant tables already live
//! here so the next-arc patch is a body-only addition.
//!
//! ## Standalone-friendly surface
//!
//! Like the §7.14 / §7.15 drivers, the top-level driver takes a small
//! [`LoopRestorationFrameContext`] bundling the frame-level inputs plus
//! closures for the §5.11.x decode state the driver reads:
//!
//! * Per-unit `LrType[plane][unitRow][unitCol]` (§5.11.58).
//! * Per-unit `LrWiener[plane][unitRow][unitCol][pass][i]` (§5.11.58 —
//!   only consulted when `LrType == RESTORE_WIENER`).
//! * Per-plane subsampling, bit depth, MI dimensions, plane sizes, and
//!   the [`crate::LrParams`] schedule.
//!
//! ## Bitstream-conformance gates not enforced here
//!
//! * The caller is responsible for the §5.11.58 `Wiener_Taps_{Min,Max}`
//!   clipping (the bit-reader applies it).
//! * The caller has already populated `UpscaledCurrFrame[plane]` (pre-
//!   CDEF samples) and `UpscaledCdefFrame[plane]` (post-CDEF samples)
//!   per §7.15.
//! * Plane buffers are sized at `Round2(FrameHeight, subY) ×
//!   Round2(UpscaledWidth, subX)` per av1-spec §5.11.34.
//!
//! ## Out-of-scope for this arc
//!
//! * §7.17.2 / §7.17.3 self-guided projection filter body — see the
//!   "self-guided projection arm" note above.
//! * §7.18 output process — separate pass that runs after loop
//!   restoration.
//! * Cross-plane SIMD / cache-friendly batched filtering — the
//!   reference loop here mirrors the spec's per-sample formulation.

use crate::cdf::{MI_SIZE, MI_SIZE_LOG2};
use crate::loop_filter::PlaneBuffer;
use crate::uncompressed_header_tail::{FrameRestorationType, LrParams};

// =====================================================================
// §7.17 constant lookup tables — av1-spec p.107 / p.332.
// =====================================================================

/// `FILTER_BITS = 7` per av1-spec p.32 line 1312 — bit precision of the
/// Wiener filter coefficients.
pub const FILTER_BITS: u32 = 7;

/// `WIENER_COEFFS = 3` per av1-spec p.32 line 1314 — number of
/// transmitted Wiener filter coefficients (the symmetric 7-tap filter
/// has 4 independent values; the centre tap is derived).
pub const WIENER_COEFFS: usize = 3;

/// `SGRPROJ_PARAMS_BITS = 4` per av1-spec p.32 line 1316 — bit count of
/// the §5.11.58 `lr_sgr_set` selector.
pub const SGRPROJ_PARAMS_BITS: u32 = 4;

/// `SGRPROJ_PRJ_SUBEXP_K = 4` per av1-spec p.32 line 1319 — `k`
/// parameter of the §5.11.58 subexp decoder for self-guided projection
/// weight deltas.
pub const SGRPROJ_PRJ_SUBEXP_K: u32 = 4;

/// `SGRPROJ_PRJ_BITS = 7` per av1-spec p.32 line 1321 — precision of
/// the self-guided projection weights.
pub const SGRPROJ_PRJ_BITS: u32 = 7;

/// `SGRPROJ_RST_BITS = 4` per av1-spec p.32 line 1323 — extra
/// restoration precision bits the box-filter output carries above the
/// 8-bit sample range.
pub const SGRPROJ_RST_BITS: u32 = 4;

/// `SGRPROJ_MTABLE_BITS = 20` per av1-spec p.32 line 1326 — precision
/// of the §7.17.3 `m`-table division shift.
pub const SGRPROJ_MTABLE_BITS: u32 = 20;

/// `SGRPROJ_RECIP_BITS = 12` per av1-spec p.32 line 1328 — precision of
/// the `1/n` division-by-area reciprocal.
pub const SGRPROJ_RECIP_BITS: u32 = 12;

/// `SGRPROJ_SGR_BITS = 8` per av1-spec p.32 line 1330 — internal
/// precision of the box-filter core.
pub const SGRPROJ_SGR_BITS: u32 = 8;

/// `Wiener_Taps_Mid[3]` per av1-spec p.74 line 3884 — reference centre
/// of the three transmitted Wiener coefficients (used by §5.11.57's
/// frame-state reset).
pub const WIENER_TAPS_MID: [i32; WIENER_COEFFS] = [3, -7, 15];

/// `Wiener_Taps_Min[3]` per av1-spec p.107 line 6484 — minimum value of
/// the three transmitted Wiener coefficients.
pub const WIENER_TAPS_MIN: [i32; WIENER_COEFFS] = [-5, -23, -17];

/// `Wiener_Taps_Max[3]` per av1-spec p.107 line 6485 — maximum value of
/// the three transmitted Wiener coefficients.
pub const WIENER_TAPS_MAX: [i32; WIENER_COEFFS] = [10, 8, 46];

/// `Wiener_Taps_K[3]` per av1-spec p.107 line 6486 — subexp `k`
/// parameter for the per-tap §5.11.58 delta read.
pub const WIENER_TAPS_K: [u32; WIENER_COEFFS] = [1, 2, 3];

/// `Sgrproj_Xqd_Mid[2]` per av1-spec p.74 line 3886 — reference centre
/// of the two transmitted self-guided projection weights.
pub const SGRPROJ_XQD_MID: [i32; 2] = [-32, 31];

/// `Sgrproj_Xqd_Min[2]` per av1-spec p.107 line 6488 — minimum value of
/// the two transmitted self-guided projection weights.
pub const SGRPROJ_XQD_MIN: [i32; 2] = [-96, -32];

/// `Sgrproj_Xqd_Max[2]` per av1-spec p.107 line 6489 — maximum value of
/// the two transmitted self-guided projection weights.
pub const SGRPROJ_XQD_MAX: [i32; 2] = [31, 95];

/// `Sgr_Params[16][4]` per av1-spec p.332 lines 18395-18400 — the
/// `(1 << SGRPROJ_PARAMS_BITS)` self-guided parameter sets, each
/// `(r0, eps0, r1, eps1)`. `r0` / `r1` are the box-filter radii and
/// `eps0` / `eps1` are the variance-scaling thresholds.
pub const SGR_PARAMS: [[i32; 4]; 16] = [
    [2, 12, 1, 4],
    [2, 15, 1, 6],
    [2, 18, 1, 8],
    [2, 21, 1, 9],
    [2, 24, 1, 10],
    [2, 29, 1, 11],
    [2, 36, 1, 12],
    [2, 45, 1, 13],
    [2, 56, 1, 14],
    [2, 68, 1, 15],
    [0, 0, 1, 5],
    [0, 0, 1, 8],
    [0, 0, 1, 11],
    [0, 0, 1, 14],
    [2, 30, 0, 0],
    [2, 75, 0, 0],
];

// =====================================================================
// §7.17 standalone-driver surface — av1-spec p.327.
// =====================================================================

/// Caller-supplied frame-level inputs the §7.17 driver consults.
///
/// All §5.5 / §5.9.5 / §5.9.20 fields come straight from the parsed
/// frame / sequence header. Per-unit predicates are closures so the
/// driver can be exercised in isolation without wiring the full
/// §5.11.58 decode state.
pub struct LoopRestorationFrameContext<'a> {
    /// `MiRows` per §5.9.5.
    pub mi_rows: u32,
    /// `MiCols` per §5.9.5.
    pub mi_cols: u32,
    /// `NumPlanes` per §5.5.2 — `1` for monochrome, `3` otherwise.
    pub num_planes: u8,
    /// `BitDepth` per §5.5.2 (`8`, `10`, or `12`).
    pub bit_depth: u8,
    /// `subsampling_x` per §5.5.2 (`0` or `1`).
    pub subsampling_x: u8,
    /// `subsampling_y` per §5.5.2 (`0` or `1`).
    pub subsampling_y: u8,
    /// `FrameHeight` per §5.9.5.
    pub frame_height: u32,
    /// `UpscaledWidth` per §5.9.8 (pre-superres-downscale).
    pub upscaled_width: u32,
    /// `LrParams` per §5.9.20 — `FrameRestorationType[0..3]`, `UsesLr`,
    /// and `LoopRestorationSize[0..3]`. The §7.17 walker indexes the
    /// `loop_restoration_size[plane]` array.
    pub lr_params: &'a LrParams,
    /// `LrType[plane][unitRow][unitCol]` per §5.11.58 — `RESTORE_NONE`
    /// / `RESTORE_WIENER` / `RESTORE_SGRPROJ`.
    pub lr_type: &'a dyn Fn(u8, u32, u32) -> FrameRestorationType,
    /// `LrWiener[plane][unitRow][unitCol][pass][i]` per §5.11.58 —
    /// `pass ∈ {0, 1}` selects vertical / horizontal coefficient sets;
    /// `i ∈ {0, 1, 2}` selects one of the three transmitted Wiener
    /// coefficients. Only consulted when the per-unit `LrType` is
    /// `RESTORE_WIENER`.
    pub lr_wiener: &'a dyn Fn(u8, u32, u32, u8, usize) -> i32,
}

impl std::fmt::Debug for LoopRestorationFrameContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopRestorationFrameContext")
            .field("mi_rows", &self.mi_rows)
            .field("mi_cols", &self.mi_cols)
            .field("num_planes", &self.num_planes)
            .field("bit_depth", &self.bit_depth)
            .field("subsampling_x", &self.subsampling_x)
            .field("subsampling_y", &self.subsampling_y)
            .field("frame_height", &self.frame_height)
            .field("upscaled_width", &self.upscaled_width)
            .field("lr_params", &self.lr_params)
            .finish_non_exhaustive()
    }
}

// =====================================================================
// §7.17 top-level driver — av1-spec p.327 lines 18088-18106.
// =====================================================================

/// §7.17 top-level loop-restoration driver — av1-spec p.327 lines
/// 18088-18106.
///
/// Iterates the frame in `MI_SIZE`-sized blocks, dispatches each
/// `(plane, row, col)` triple whose `FrameRestorationType[plane] !=
/// RESTORE_NONE` to [`loop_restore_block`].
///
/// Reads pre-CDEF samples from `curr_planes` and post-CDEF samples from
/// `cdef_planes`; writes restored samples into `lr_planes`. The §7.17
/// "copy `LrFrame = UpscaledCdefFrame`" prelude is materialised by
/// copying `cdef_planes ↦ lr_planes` up-front; on `UsesLr == 0` the
/// driver returns immediately after the copy.
pub fn loop_restoration_frame(
    ctx: &LoopRestorationFrameContext<'_>,
    curr_planes: &[PlaneBuffer<'_>],
    cdef_planes: &[PlaneBuffer<'_>],
    lr_planes: &mut [PlaneBuffer<'_>],
) {
    let num_planes = ctx
        .num_planes
        .min(curr_planes.len() as u8)
        .min(cdef_planes.len() as u8)
        .min(lr_planes.len() as u8);
    // av1-spec p.327 line 18088: LrFrame = UpscaledCdefFrame.
    for plane in 0..num_planes {
        copy_plane(cdef_planes, lr_planes, plane);
    }
    // av1-spec p.327 line 18091: UsesLr == 0 ⇒ return.
    if !ctx.lr_params.uses_lr {
        return;
    }
    // av1-spec p.327 lines 18096-18106: walk in `MI_SIZE` steps over
    // `(y, x) ∈ [0, FrameHeight) × [0, UpscaledWidth)`.
    let mi_step = MI_SIZE as u32;
    let mut y = 0u32;
    while y < ctx.frame_height {
        let mut x = 0u32;
        while x < ctx.upscaled_width {
            for plane in 0..num_planes {
                let r_type = ctx.lr_params.frame_restoration_type[plane as usize];
                if r_type == FrameRestorationType::None {
                    continue;
                }
                let row = y >> (MI_SIZE_LOG2 as u32);
                let col = x >> (MI_SIZE_LOG2 as u32);
                loop_restore_block(ctx, curr_planes, cdef_planes, lr_planes, plane, row, col);
            }
            x += mi_step;
        }
        y += mi_step;
    }
}

/// Copy `src_planes[plane]` into `dst_planes[plane]` in full — used by
/// the §7.17 driver's `LrFrame = UpscaledCdefFrame` prelude. Both
/// buffers carry the same `(rows, cols)` shape; on a mismatch the copy
/// honours the smaller extent on each axis.
fn copy_plane(src_planes: &[PlaneBuffer<'_>], dst_planes: &mut [PlaneBuffer<'_>], plane: u8) {
    let p = plane as usize;
    let Some(src) = src_planes.get(p) else { return };
    let Some(dst) = dst_planes.get_mut(p) else {
        return;
    };
    let rows = src.rows.min(dst.rows) as usize;
    let cols = src.cols.min(dst.cols) as usize;
    let src_cols = src.cols as usize;
    let dst_cols = dst.cols as usize;
    for y in 0..rows {
        for x in 0..cols {
            dst.samples[y * dst_cols + x] = src.samples[y * src_cols + x];
        }
    }
}

// =====================================================================
// §7.17.1 loop-restore block driver — av1-spec p.328-329 lines
// 18113-18215.
// =====================================================================

/// Per-block derived geometry returned by [`derive_block_geometry`].
///
/// The fields map directly to the §7.17.1 named locals
/// `(unitRow, unitCol, x, y, w, h, StripeStartY, StripeEndY,
/// PlaneEndX, PlaneEndY)` and feed both the Wiener and (future)
/// self-guided arms.
#[derive(Debug, Clone, Copy)]
pub struct LrBlockGeometry {
    /// `unitRow` per av1-spec p.328 line 18168.
    pub unit_row: u32,
    /// `unitCol` per av1-spec p.328 line 18171.
    pub unit_col: u32,
    /// `x` per av1-spec p.329 line 18182.
    pub x: u32,
    /// `y` per av1-spec p.329 line 18184.
    pub y: u32,
    /// `w` per av1-spec p.329 line 18186.
    pub w: u32,
    /// `h` per av1-spec p.329 line 18195.
    pub h: u32,
    /// `StripeStartY` per av1-spec p.328 line 18144.
    pub stripe_start_y: i32,
    /// `StripeEndY` per av1-spec p.328 line 18147.
    pub stripe_end_y: i32,
    /// `PlaneEndX` per av1-spec p.329 line 18176.
    pub plane_end_x: i32,
    /// `PlaneEndY` per av1-spec p.329 line 18179.
    pub plane_end_y: i32,
}

/// §7.17.1 geometry derivation — av1-spec p.328-329 lines 18122-18198.
///
/// Computes `(unitRow, unitCol, x, y, w, h, StripeStartY, StripeEndY,
/// PlaneEndX, PlaneEndY)` from the input `(plane, row, col)` triple
/// and the [`LoopRestorationFrameContext`].
pub fn derive_block_geometry(
    ctx: &LoopRestorationFrameContext<'_>,
    plane: u8,
    row: u32,
    col: u32,
) -> LrBlockGeometry {
    // av1-spec p.328 lines 18122-18124: stripeNum = (lumaY + 8) / 64.
    let luma_y = (row as i32) * (MI_SIZE as i32);
    let stripe_num = (luma_y + 8) / 64;
    // av1-spec p.328 lines 18138-18142: subX / subY from the plane.
    let (sub_x, sub_y) = subsampling_for_plane(plane, ctx.subsampling_x, ctx.subsampling_y);
    // av1-spec p.328 line 18144: StripeStartY = (-8 + stripeNum * 64) >> subY.
    let stripe_start_y = (-8 + stripe_num * 64) >> sub_y;
    // av1-spec p.328 line 18147: StripeEndY = StripeStartY + (64 >> subY) - 1.
    let stripe_end_y = stripe_start_y + (64 >> sub_y) - 1;
    // av1-spec p.328 line 18155: unitSize = LoopRestorationSize[plane].
    let unit_size = ctx
        .lr_params
        .loop_restoration_size
        .get(plane as usize)
        .copied()
        .unwrap_or(0)
        .max(1);
    // av1-spec p.328 lines 18158-18162: unitRows, unitCols via
    // count_units_in_frame.
    let unit_rows = count_units_in_frame(unit_size, round2(ctx.frame_height, sub_y));
    let unit_cols = count_units_in_frame(unit_size, round2(ctx.upscaled_width, sub_x));
    // av1-spec p.328 lines 18168-18172: unitRow / unitCol pegged to
    // `unitRows - 1` / `unitCols - 1`.
    let unit_row_raw = ((row * (MI_SIZE as u32) + 8) >> sub_y) / unit_size;
    let unit_row = unit_row_raw.min(unit_rows.saturating_sub(1));
    let unit_col_raw = ((col * (MI_SIZE as u32)) >> sub_x) / unit_size;
    let unit_col = unit_col_raw.min(unit_cols.saturating_sub(1));
    // av1-spec p.329 lines 18176-18179: PlaneEndX / PlaneEndY.
    let plane_end_x = round2(ctx.upscaled_width, sub_x) as i32 - 1;
    let plane_end_y = round2(ctx.frame_height, sub_y) as i32 - 1;
    // av1-spec p.329 lines 18182-18195: x / y / w / h.
    let x = (col * (MI_SIZE as u32)) >> sub_x;
    let y = (row * (MI_SIZE as u32)) >> sub_y;
    let w = ((MI_SIZE as u32) >> sub_x).min((plane_end_x + 1 - x as i32).max(0) as u32);
    let h = ((MI_SIZE as u32) >> sub_y).min((plane_end_y + 1 - y as i32).max(0) as u32);
    LrBlockGeometry {
        unit_row,
        unit_col,
        x,
        y,
        w,
        h,
        stripe_start_y,
        stripe_end_y,
        plane_end_x,
        plane_end_y,
    }
}

/// §7.17.1 per-block driver — av1-spec p.328-329 lines 18113-18215.
///
/// Computes the block's `(unitRow, unitCol, x, y, w, h, ...)`
/// geometry, fetches `rType = LrType[plane][unitRow][unitCol]`, and
/// dispatches to the Wiener / self-guided / no-op arm.
pub fn loop_restore_block(
    ctx: &LoopRestorationFrameContext<'_>,
    curr_planes: &[PlaneBuffer<'_>],
    cdef_planes: &[PlaneBuffer<'_>],
    lr_planes: &mut [PlaneBuffer<'_>],
    plane: u8,
    row: u32,
    col: u32,
) {
    let geom = derive_block_geometry(ctx, plane, row, col);
    if geom.w == 0 || geom.h == 0 {
        return;
    }
    // av1-spec p.329 line 18205: rType = LrType[plane][unitRow][unitCol].
    let r_type = (ctx.lr_type)(plane, geom.unit_row, geom.unit_col);
    match r_type {
        FrameRestorationType::Wiener => {
            wiener_filter(ctx, curr_planes, cdef_planes, lr_planes, plane, &geom);
        }
        FrameRestorationType::SgrProj => {
            // §7.17.2 self-guided projection arm is stubbed: leave the
            // pre-copied `LrFrame = UpscaledCdefFrame` content alone.
            // The next-arc patch lands the box-filter body here.
        }
        FrameRestorationType::None | FrameRestorationType::Switchable => {
            // RESTORE_NONE — `LrFrame` already holds `UpscaledCdefFrame`
            // from the §7.17 prelude. Switchable is a frame-level
            // setting; per-unit `rType` materialises to one of the
            // three primitive types, so this arm is unreachable on a
            // conformant decode but kept exhaustive.
        }
    }
}

// =====================================================================
// §7.17.4 Wiener filter — av1-spec p.333-334 lines 18404-18484.
// =====================================================================

/// §7.17.4 Wiener filter — av1-spec p.333-334 lines 18404-18484.
///
/// Two-pass separable convolution: builds an `intermediate[h+6][w]`
/// array via the 7-tap horizontal filter, then writes
/// `LrFrame[plane][y + r][x + c]` via the 7-tap vertical filter.
/// `(InterRound0, InterRound1)` follow the §7.11.3.2 single-ref
/// schedule (`isCompound = 0`).
pub fn wiener_filter(
    ctx: &LoopRestorationFrameContext<'_>,
    curr_planes: &[PlaneBuffer<'_>],
    cdef_planes: &[PlaneBuffer<'_>],
    lr_planes: &mut [PlaneBuffer<'_>],
    plane: u8,
    geom: &LrBlockGeometry,
) {
    let Some(_curr) = curr_planes.get(plane as usize) else {
        return;
    };
    let Some(_cdef) = cdef_planes.get(plane as usize) else {
        return;
    };
    // av1-spec p.333 lines 18429-18433: vfilter = wiener_coefficients(
    // LrWiener[plane][unitRow][unitCol][0]); hfilter = wiener_coefficients(
    // LrWiener[plane][unitRow][unitCol][1]).
    let vcoeff: [i32; WIENER_COEFFS] = [
        (ctx.lr_wiener)(plane, geom.unit_row, geom.unit_col, 0, 0),
        (ctx.lr_wiener)(plane, geom.unit_row, geom.unit_col, 0, 1),
        (ctx.lr_wiener)(plane, geom.unit_row, geom.unit_col, 0, 2),
    ];
    let hcoeff: [i32; WIENER_COEFFS] = [
        (ctx.lr_wiener)(plane, geom.unit_row, geom.unit_col, 1, 0),
        (ctx.lr_wiener)(plane, geom.unit_row, geom.unit_col, 1, 1),
        (ctx.lr_wiener)(plane, geom.unit_row, geom.unit_col, 1, 2),
    ];
    let vfilter = wiener_coefficients(&vcoeff);
    let hfilter = wiener_coefficients(&hcoeff);
    // av1-spec p.333 line 18426: §7.11.3.2 isCompound = 0 schedule.
    let inter_round0 = inter_round0(ctx.bit_depth);
    let inter_round1 = inter_round1(ctx.bit_depth);
    let bit_depth = ctx.bit_depth as i32;
    let offset = 1i32 << (bit_depth + FILTER_BITS as i32 - inter_round0 as i32 - 1);
    let limit = (1i32 << (bit_depth + 1 + FILTER_BITS as i32 - inter_round0 as i32)) - 1;
    let h = geom.h as usize;
    let w = geom.w as usize;
    // av1-spec p.333 lines 18447-18455: horizontal pass into
    // `intermediate[h + 6][w]`.
    let mut intermediate = vec![0i32; (h + 6) * w];
    for r in 0..(h + 6) {
        for c in 0..w {
            let mut s = 0i32;
            for (t, &hcoef) in hfilter.iter().enumerate() {
                let sx = geom.x as i32 + c as i32 + t as i32 - 3;
                let sy = geom.y as i32 + r as i32 - 3;
                let src = get_source_sample(curr_planes, cdef_planes, plane, sx, sy, geom);
                s += hcoef * src;
            }
            let v = round2_i32(s, inter_round0);
            let clipped = clip3(-offset, limit - offset, v);
            intermediate[r * w + c] = clipped;
        }
    }
    // av1-spec p.334 lines 18476-18484: vertical pass writes
    // `LrFrame[plane][y + r][x + c]`.
    let max_sample = (1i32 << bit_depth) - 1;
    let Some(lr) = lr_planes.get_mut(plane as usize) else {
        return;
    };
    let lr_cols = lr.cols as i32;
    let lr_rows = lr.rows as i32;
    for r in 0..h {
        for c in 0..w {
            let mut s = 0i32;
            for (t, &vcoef) in vfilter.iter().enumerate() {
                s += vcoef * intermediate[(r + t) * w + c];
            }
            let v = round2_i32(s, inter_round1);
            let out = clip3(0, max_sample, v);
            let dst_y = geom.y as i32 + r as i32;
            let dst_x = geom.x as i32 + c as i32;
            if dst_y < 0 || dst_y >= lr_rows || dst_x < 0 || dst_x >= lr_cols {
                continue;
            }
            lr.samples[(dst_y as usize) * (lr.cols as usize) + dst_x as usize] = out;
        }
    }
}

// =====================================================================
// §7.17.5 Wiener coefficient process — av1-spec p.334 lines 18488-18508.
// =====================================================================

/// §7.17.5 Wiener coefficient process — av1-spec p.334 lines
/// 18488-18508.
///
/// Reconstructs the symmetric 7-tap Wiener filter from three
/// transmitted coefficients: `filter[i] = coeff[i]`, `filter[6 - i] =
/// coeff[i]`, `filter[3] = 128 - 2 * Σ coeff[i]`. The DC-gain
/// constraint forces `Σ filter == 128` — the §7.17.4 horizontal pass
/// then rounds by `InterRound0` and the vertical pass rounds by
/// `InterRound1`, with the two together giving unity overall gain.
///
/// Per the spec's `Note`, for chroma planes `coeff[0]` is always 0, so
/// `filter[0]` and `filter[6]` are 0 and the effective filter is a
/// 5-tap convolution.
pub fn wiener_coefficients(coeff: &[i32; WIENER_COEFFS]) -> [i32; 7] {
    let mut filter = [0i32; 7];
    filter[3] = 128;
    for i in 0..3usize {
        let c = coeff[i];
        filter[i] = c;
        filter[6 - i] = c;
        filter[3] -= 2 * c;
    }
    filter
}

// =====================================================================
// §7.17.6 Get source sample process — av1-spec p.335 lines 18516-18550.
// =====================================================================

/// §7.17.6 source-sample fetch — av1-spec p.335 lines 18516-18550.
///
/// Snaps `(x, y)` to `[0, PlaneEndX] × [0, PlaneEndY]` and routes to
/// either `UpscaledCurrFrame[plane]` (when `y` lands above
/// `StripeStartY` or below `StripeEndY`, with the cropped
/// `StripeStartY - 2` / `StripeEndY + 2` neighbour-line reach) or
/// `UpscaledCdefFrame[plane]` (when `y` lands inside the stripe).
pub fn get_source_sample(
    curr_planes: &[PlaneBuffer<'_>],
    cdef_planes: &[PlaneBuffer<'_>],
    plane: u8,
    x: i32,
    y: i32,
    geom: &LrBlockGeometry,
) -> i32 {
    // av1-spec p.335 lines 18538-18541: clamp x / y to plane extent.
    let x = x.clamp(0, geom.plane_end_x.max(0));
    let mut y = y.clamp(0, geom.plane_end_y.max(0));
    // av1-spec p.335 lines 18542-18549: stripe-aware fetch routing.
    let p = plane as usize;
    if y < geom.stripe_start_y {
        y = y.max(geom.stripe_start_y - 2);
        return sample_at(curr_planes, p, x, y);
    } else if y > geom.stripe_end_y {
        y = y.min(geom.stripe_end_y + 2);
        return sample_at(curr_planes, p, x, y);
    }
    sample_at(cdef_planes, p, x, y)
}

/// Bounds-checked sample fetch from a [`PlaneBuffer`] used by
/// [`get_source_sample`]. The §7.17.6 clamping already pegs `(x, y)`
/// inside `[0, PlaneEndX] × [0, PlaneEndY]` and the stripe-boundary
/// fetches into `UpscaledCurrFrame` stay within
/// `[StripeStartY - 2, StripeEndY + 2]` — but the planes the caller
/// passes in may be sized differently, so the fetch here clamps
/// defensively to the buffer's `[0, rows-1] × [0, cols-1]` extent.
fn sample_at(planes: &[PlaneBuffer<'_>], plane: usize, x: i32, y: i32) -> i32 {
    let Some(buf) = planes.get(plane) else {
        return 0;
    };
    if buf.rows == 0 || buf.cols == 0 {
        return 0;
    }
    let yc = y.clamp(0, buf.rows as i32 - 1) as usize;
    let xc = x.clamp(0, buf.cols as i32 - 1) as usize;
    buf.samples[yc * (buf.cols as usize) + xc]
}

// =====================================================================
// Helpers — §3 / §5.9.20 numeric primitives.
// =====================================================================

/// §5.9.20 `count_units_in_frame(unitSize, frameSize)` per av1-spec
/// p.107 line 6406 — `Max((frameSize + (unitSize >> 1)) / unitSize, 1)`.
pub fn count_units_in_frame(unit_size: u32, frame_size: u32) -> u32 {
    if unit_size == 0 {
        return 1;
    }
    ((frame_size + (unit_size >> 1)) / unit_size).max(1)
}

/// §3 `Round2(x, n)` — `(x + (1 << (n - 1))) >> n` for `n > 0`,
/// identity for `n == 0`.
#[inline]
fn round2(x: u32, n: u8) -> u32 {
    if n == 0 {
        x
    } else {
        (x + (1u32 << (n - 1))) >> n
    }
}

/// `Round2(x, n)` over `i32` — same shape as [`round2`] but signed.
#[inline]
fn round2_i32(x: i32, n: u32) -> i32 {
    if n == 0 {
        x
    } else {
        (x + (1i32 << (n - 1))) >> n
    }
}

/// §3 `Clip3(a, b, x)` — clamp `x` to `[a, b]`.
#[inline]
fn clip3(a: i32, b: i32, x: i32) -> i32 {
    if x < a {
        a
    } else if x > b {
        b
    } else {
        x
    }
}

/// `(subX, subY)` for `plane` per av1-spec p.328 lines 18138-18142.
#[inline]
fn subsampling_for_plane(plane: u8, sub_x: u8, sub_y: u8) -> (u8, u8) {
    if plane == 0 {
        (0, 0)
    } else {
        (sub_x, sub_y)
    }
}

/// §7.11.3.2 `InterRound0` for `isCompound = 0` per av1-spec p.259
/// lines 14423 / 14427 — `3` for `BitDepth ∈ {8, 10}`, `5` for `12`.
#[inline]
fn inter_round0(bit_depth: u8) -> u32 {
    if bit_depth == 12 {
        5
    } else {
        3
    }
}

/// §7.11.3.2 `InterRound1` for `isCompound = 0` per av1-spec p.259
/// lines 14425 / 14429 — `11` for `BitDepth ∈ {8, 10}`, `9` for `12`.
#[inline]
fn inter_round1(bit_depth: u8) -> u32 {
    if bit_depth == 12 {
        9
    } else {
        11
    }
}

// =====================================================================
// Tests.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uncompressed_header_tail::RESTORATION_TILESIZE_MAX;

    fn make_lr_params(uses_lr: bool, sizes: [u32; 3]) -> LrParams {
        LrParams {
            frame_restoration_type: [FrameRestorationType::None; 3],
            uses_lr,
            uses_chroma_lr: false,
            lr_unit_shift: 0,
            lr_uv_shift: 0,
            loop_restoration_size: sizes,
            short_circuited: false,
        }
    }

    fn make_plane(rows: u32, cols: u32, fill: i32) -> Vec<i32> {
        vec![fill; (rows * cols) as usize]
    }

    #[test]
    fn wiener_taps_tables_match_spec() {
        // av1-spec p.74 lines 3884-3886 + p.107 lines 6484-6489.
        assert_eq!(WIENER_TAPS_MID, [3, -7, 15]);
        assert_eq!(WIENER_TAPS_MIN, [-5, -23, -17]);
        assert_eq!(WIENER_TAPS_MAX, [10, 8, 46]);
        assert_eq!(WIENER_TAPS_K, [1, 2, 3]);
        assert_eq!(SGRPROJ_XQD_MID, [-32, 31]);
        assert_eq!(SGRPROJ_XQD_MIN, [-96, -32]);
        assert_eq!(SGRPROJ_XQD_MAX, [31, 95]);
    }

    #[test]
    fn sgr_params_table_matches_spec() {
        // av1-spec p.332 lines 18395-18400 — spot-check rows 0, 10, 14.
        assert_eq!(SGR_PARAMS[0], [2, 12, 1, 4]);
        assert_eq!(SGR_PARAMS[10], [0, 0, 1, 5]);
        assert_eq!(SGR_PARAMS[14], [2, 30, 0, 0]);
        assert_eq!(SGR_PARAMS.len(), 1 << SGRPROJ_PARAMS_BITS);
    }

    #[test]
    fn wiener_coefficients_unit_dc_gain() {
        // av1-spec p.334 line 18499: filter[3] = 128, filter[i] =
        // coeff[i], filter[6-i] = coeff[i], filter[3] -= 2 * Σ coeff.
        // The symmetric DC gain Σ filter == 128 across all inputs.
        for coeff in [
            [0, 0, 0],
            [3, -7, 15],    // Wiener_Taps_Mid
            [-5, -23, -17], // Wiener_Taps_Min
            [10, 8, 46],    // Wiener_Taps_Max
        ] {
            let f = wiener_coefficients(&coeff);
            assert_eq!(f.iter().sum::<i32>(), 128);
            // Symmetry: filter[i] == filter[6 - i].
            assert_eq!(f[0], f[6]);
            assert_eq!(f[1], f[5]);
            assert_eq!(f[2], f[4]);
            // Transmitted taps land in slots 0..=2.
            assert_eq!(f[0], coeff[0]);
            assert_eq!(f[1], coeff[1]);
            assert_eq!(f[2], coeff[2]);
        }
    }

    #[test]
    fn wiener_coefficients_zero_centre_is_128() {
        // All-zero transmitted taps ⇒ filter = [0, 0, 0, 128, 0, 0, 0].
        let f = wiener_coefficients(&[0, 0, 0]);
        assert_eq!(f, [0, 0, 0, 128, 0, 0, 0]);
    }

    #[test]
    fn count_units_in_frame_matches_spec() {
        // av1-spec p.107 line 6406:
        //   count_units_in_frame(unitSize, frameSize) =
        //     Max((frameSize + (unitSize >> 1)) / unitSize, 1)
        assert_eq!(count_units_in_frame(64, 256), 4);
        assert_eq!(count_units_in_frame(64, 100), 2); // (100 + 32) / 64 = 2
        assert_eq!(count_units_in_frame(64, 0), 1); // floor at 1
        assert_eq!(count_units_in_frame(256, 240), 1); // (240 + 128) / 256 = 1
    }

    #[test]
    fn loop_restoration_frame_no_op_when_uses_lr_false() {
        // UsesLr == 0 ⇒ the driver just copies UpscaledCdefFrame into
        // LrFrame and returns. Per-plane closures are never consulted.
        let mut curr = make_plane(16, 16, 50);
        let mut cdef = make_plane(16, 16, 200);
        let mut lr = make_plane(16, 16, 0);
        let curr_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut curr,
        };
        let cdef_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut cdef,
        };
        let mut lr_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut lr,
        };
        let params = make_lr_params(false, [64; 3]);
        let ctx = LoopRestorationFrameContext {
            mi_rows: 4,
            mi_cols: 4,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            frame_height: 16,
            upscaled_width: 16,
            lr_params: &params,
            lr_type: &|_, _, _| FrameRestorationType::None,
            lr_wiener: &|_, _, _, _, _| 0,
        };
        loop_restoration_frame(
            &ctx,
            std::slice::from_ref(&curr_buf),
            std::slice::from_ref(&cdef_buf),
            std::slice::from_mut(&mut lr_buf),
        );
        // LrFrame holds a copy of UpscaledCdefFrame.
        assert!(lr_buf.samples.iter().all(|&v| v == 200));
    }

    #[test]
    fn loop_restoration_frame_restore_none_keeps_cdef_copy() {
        // Per-plane FrameRestorationType == RESTORE_NONE ⇒ LrFrame
        // holds UpscaledCdefFrame, even when UsesLr == 1 (e.g. another
        // plane has loop restoration enabled).
        let mut curr = make_plane(16, 16, 50);
        let mut cdef = make_plane(16, 16, 175);
        let mut lr = make_plane(16, 16, 0);
        let curr_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut curr,
        };
        let cdef_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut cdef,
        };
        let mut lr_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut lr,
        };
        let mut params = make_lr_params(true, [64; 3]);
        params.frame_restoration_type[0] = FrameRestorationType::None;
        let ctx = LoopRestorationFrameContext {
            mi_rows: 4,
            mi_cols: 4,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            frame_height: 16,
            upscaled_width: 16,
            lr_params: &params,
            lr_type: &|_, _, _| FrameRestorationType::None,
            lr_wiener: &|_, _, _, _, _| 0,
        };
        loop_restoration_frame(
            &ctx,
            std::slice::from_ref(&curr_buf),
            std::slice::from_ref(&cdef_buf),
            std::slice::from_mut(&mut lr_buf),
        );
        assert!(lr_buf.samples.iter().all(|&v| v == 175));
    }

    #[test]
    fn wiener_filter_identity_filter_recovers_source() {
        // With coeff = [0, 0, 0], wiener_coefficients returns
        // [0, 0, 0, 128, 0, 0, 0] — i.e. the identity 7-tap filter
        // scaled by 128. Two-pass: horizontal then vertical, with
        // InterRound0=3 and InterRound1=11 for BitDepth=8. The combined
        // gain is 128 * 128 >> 14 = 1, so a uniform input recovers as
        // itself.
        let mut curr = make_plane(64, 64, 100);
        let mut cdef = make_plane(64, 64, 100);
        let mut lr = make_plane(64, 64, 0);
        let curr_buf = PlaneBuffer {
            rows: 64,
            cols: 64,
            samples: &mut curr,
        };
        let cdef_buf = PlaneBuffer {
            rows: 64,
            cols: 64,
            samples: &mut cdef,
        };
        let mut lr_buf = PlaneBuffer {
            rows: 64,
            cols: 64,
            samples: &mut lr,
        };
        let mut params = make_lr_params(true, [RESTORATION_TILESIZE_MAX; 3]);
        params.frame_restoration_type[0] = FrameRestorationType::Wiener;
        let ctx = LoopRestorationFrameContext {
            mi_rows: 16,
            mi_cols: 16,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            frame_height: 64,
            upscaled_width: 64,
            lr_params: &params,
            lr_type: &|_, _, _| FrameRestorationType::Wiener,
            lr_wiener: &|_, _, _, _, _| 0,
        };
        loop_restoration_frame(
            &ctx,
            std::slice::from_ref(&curr_buf),
            std::slice::from_ref(&cdef_buf),
            std::slice::from_mut(&mut lr_buf),
        );
        // The Wiener identity convolution over a uniform plane yields
        // the same uniform plane back. (Rounding constants cancel: a
        // uniform input means every horizontal sum is 128 * src; the
        // vertical pass scales that by 128 / 2^11 ≈ src after the
        // intermediate `>> 3` rounding.)
        for &v in lr_buf.samples.iter() {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn get_source_sample_inside_stripe_reads_cdef() {
        // y ∈ [StripeStartY, StripeEndY] ⇒ UpscaledCdefFrame.
        let mut curr = make_plane(16, 16, 11);
        let mut cdef = make_plane(16, 16, 99);
        let curr_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut curr,
        };
        let cdef_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut cdef,
        };
        let geom = LrBlockGeometry {
            unit_row: 0,
            unit_col: 0,
            x: 0,
            y: 4,
            w: 4,
            h: 4,
            stripe_start_y: 0,
            stripe_end_y: 8,
            plane_end_x: 15,
            plane_end_y: 15,
        };
        let v = get_source_sample(
            std::slice::from_ref(&curr_buf),
            std::slice::from_ref(&cdef_buf),
            0,
            3,
            5,
            &geom,
        );
        assert_eq!(v, 99);
    }

    #[test]
    fn get_source_sample_above_stripe_reads_curr_clamped() {
        // y < StripeStartY ⇒ UpscaledCurrFrame, with y pegged at
        // max(StripeStartY - 2, y).
        let mut curr = make_plane(16, 16, 33);
        let mut cdef = make_plane(16, 16, 77);
        let curr_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut curr,
        };
        let cdef_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut cdef,
        };
        let geom = LrBlockGeometry {
            unit_row: 0,
            unit_col: 0,
            x: 0,
            y: 8,
            w: 4,
            h: 4,
            stripe_start_y: 6,
            stripe_end_y: 14,
            plane_end_x: 15,
            plane_end_y: 15,
        };
        // y = 2 < stripe_start_y = 6; clamp pegs to max(6 - 2, 2) = 4.
        let v = get_source_sample(
            std::slice::from_ref(&curr_buf),
            std::slice::from_ref(&cdef_buf),
            0,
            3,
            2,
            &geom,
        );
        assert_eq!(v, 33);
    }

    #[test]
    fn get_source_sample_below_stripe_reads_curr() {
        // y > StripeEndY ⇒ UpscaledCurrFrame.
        let mut curr = make_plane(32, 16, 42);
        let mut cdef = make_plane(32, 16, 88);
        let curr_buf = PlaneBuffer {
            rows: 32,
            cols: 16,
            samples: &mut curr,
        };
        let cdef_buf = PlaneBuffer {
            rows: 32,
            cols: 16,
            samples: &mut cdef,
        };
        let geom = LrBlockGeometry {
            unit_row: 0,
            unit_col: 0,
            x: 0,
            y: 0,
            w: 4,
            h: 4,
            stripe_start_y: 0,
            stripe_end_y: 8,
            plane_end_x: 15,
            plane_end_y: 31,
        };
        // y = 12 > stripe_end_y = 8; clamp pegs to min(8 + 2, 12) = 10.
        let v = get_source_sample(
            std::slice::from_ref(&curr_buf),
            std::slice::from_ref(&cdef_buf),
            0,
            3,
            12,
            &geom,
        );
        assert_eq!(v, 42);
    }

    #[test]
    fn derive_block_geometry_luma_top_left() {
        let params = make_lr_params(true, [RESTORATION_TILESIZE_MAX; 3]);
        let ctx = LoopRestorationFrameContext {
            mi_rows: 16,
            mi_cols: 16,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            frame_height: 64,
            upscaled_width: 64,
            lr_params: &params,
            lr_type: &|_, _, _| FrameRestorationType::None,
            lr_wiener: &|_, _, _, _, _| 0,
        };
        let geom = derive_block_geometry(&ctx, 0, 0, 0);
        assert_eq!(geom.unit_row, 0);
        assert_eq!(geom.unit_col, 0);
        assert_eq!(geom.x, 0);
        assert_eq!(geom.y, 0);
        assert_eq!(geom.w, MI_SIZE as u32);
        assert_eq!(geom.h, MI_SIZE as u32);
        assert_eq!(geom.plane_end_x, 63);
        assert_eq!(geom.plane_end_y, 63);
        // stripeNum = (0 + 8) / 64 = 0 ⇒ stripe_start_y = -8.
        assert_eq!(geom.stripe_start_y, -8);
        // stripe_end_y = -8 + 64 - 1 = 55.
        assert_eq!(geom.stripe_end_y, 55);
    }

    #[test]
    fn restore_sgrproj_passes_cdef_through() {
        // Self-guided arm is stubbed for this arc; expect LrFrame to
        // hold UpscaledCdefFrame after the driver returns.
        let mut curr = make_plane(16, 16, 50);
        let mut cdef = make_plane(16, 16, 222);
        let mut lr = make_plane(16, 16, 0);
        let curr_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut curr,
        };
        let cdef_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut cdef,
        };
        let mut lr_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut lr,
        };
        let mut params = make_lr_params(true, [RESTORATION_TILESIZE_MAX; 3]);
        params.frame_restoration_type[0] = FrameRestorationType::SgrProj;
        let ctx = LoopRestorationFrameContext {
            mi_rows: 4,
            mi_cols: 4,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            frame_height: 16,
            upscaled_width: 16,
            lr_params: &params,
            lr_type: &|_, _, _| FrameRestorationType::SgrProj,
            lr_wiener: &|_, _, _, _, _| 0,
        };
        loop_restoration_frame(
            &ctx,
            std::slice::from_ref(&curr_buf),
            std::slice::from_ref(&cdef_buf),
            std::slice::from_mut(&mut lr_buf),
        );
        assert!(lr_buf.samples.iter().all(|&v| v == 222));
    }
}
