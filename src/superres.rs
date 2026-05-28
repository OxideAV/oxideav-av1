//! §7.16 Upscaling (superres) — the horizontal post-CDEF / pre-LR pass
//! that takes the encoded `FrameWidth`-wide downscaled frame and
//! produces an `UpscaledWidth`-wide frame per av1-spec p.325-326.
//!
//! ## Coverage (round 200 — close-out push)
//!
//! This module covers §7.16 in full at the sample-filtering level. With
//! it landed the per-frame post-processing pipeline (§7.14 LF →
//! §7.15 CDEF → §7.16 superres → §7.17 LR → §7.18.3 film grain) is now
//! complete:
//!
//! * §7.16 — [`upscale_frame`]: top-level per-plane driver. Iterates
//!   `plane ∈ [0, NumPlanes)`; for each plane derives
//!   `(subX, subY, downscaledPlaneW, upscaledPlaneW, planeH, stepX,
//!   err, initialSubpelX, miW, minX, maxX)` per av1-spec p.325 lines
//!   17979-17999, then dispatches every `(y, x)` to the 8-tap
//!   polyphase upscale sample build.
//! * §7.16 — [`upscale_plane`]: per-plane body (the inner half of
//!   the spec listing). Walks `y ∈ [0, planeH)` × `x ∈ [0,
//!   upscaledPlaneW)` and writes `outputFrame[plane][y][x] =
//!   Clip1(Round2(Σ px * Upscale_Filter[srcXSubpel][k], FILTER_BITS))`
//!   per av1-spec p.325 lines 18000-18013.
//! * §7.16 — [`upscale_sample`]: the per-output-pixel inner loop.
//!   Given `srcXPx`, `srcXSubpel`, and the source row slice, evaluates
//!   the 8-tap horizontal convolution against `Upscale_Filter[srcXSubpel]`
//!   with the §3 `Clip3(minX, maxX, srcXPx + (k - SUPERRES_FILTER_OFFSET))`
//!   left/right edge replication.
//! * §7.16 — [`UPSCALE_FILTER`]: the
//!   `Upscale_Filter[SUPERRES_FILTER_SHIFTS][SUPERRES_FILTER_TAPS]`
//!   table verbatim from av1-spec p.326 lines 18027-18060 — 64 sub-pixel
//!   phases × 8 taps, each tap a signed `i32` already scaled to the
//!   `FILTER_BITS = 7` precision (Σ taps per row = 128).
//!
//! ## §3 named constants surfaced — av1-spec p.32 lines 1395-1412
//!
//! * [`SUPERRES_FILTER_BITS = 6`] — bit count of the fractional position
//!   index into [`UPSCALE_FILTER`].
//! * [`SUPERRES_FILTER_SHIFTS = 64`] — number of sub-pixel filter
//!   phases (`1 << SUPERRES_FILTER_BITS`).
//! * [`SUPERRES_FILTER_TAPS = 8`] — taps per filter phase.
//! * [`SUPERRES_FILTER_OFFSET = 3`] — centre-tap offset that maps the
//!   `k`-th tap onto `srcXPx + (k - 3)`.
//! * [`SUPERRES_SCALE_BITS = 14`] — sub-pixel fractional bits for the
//!   `srcX` accumulator.
//! * [`SUPERRES_SCALE_MASK`] — `(1 << 14) - 1`, isolates the fractional
//!   part of `srcX` before the `>> SUPERRES_EXTRA_BITS` reduction.
//! * [`SUPERRES_EXTRA_BITS = 8`] — gap between `SUPERRES_SCALE_BITS`
//!   and [`SUPERRES_FILTER_BITS`], i.e. the right-shift that maps an
//!   `srcX & SUPERRES_SCALE_MASK` value into a filter-phase index.
//! * [`FILTER_BITS = 7`] — coefficient precision of [`UPSCALE_FILTER`].
//!
//! The [`SUPERRES_NUM`] / [`SUPERRES_DENOM_MIN`] / [`SUPERRES_DENOM_BITS`]
//! constants are re-exported from [`crate::frame_header`] because they
//! belong to §5.9.8 (`superres_params()` bitstream parse).
//!
//! ## Standalone-friendly surface
//!
//! Like the §7.14 / §7.15 / §7.17 drivers, [`upscale_frame`] takes a
//! small [`SuperresFrameContext`] bundling the §5.5.2 / §5.9.5 / §5.9.8
//! state plus `(MiCols, UseSuperres, NumPlanes, BitDepth)`. The input
//! / output plane buffers are [`PlaneBuffer<'_>`]s; the input is
//! `FrameWidth`-wide and the output is `UpscaledWidth`-wide (the
//! `downscaledPlaneW` / `upscaledPlaneW` derivation drops the chroma
//! subsampling factor on top per the spec).
//!
//! When `use_superres == 0` the input width equals the output width and
//! the driver short-circuits with a verbatim plane copy per the
//! av1-spec p.325 line 17968 "If use_superres is equal to 0, no
//! upscaling is required and this process returns inputFrame" branch.
//!
//! ## Bitstream-conformance gates not enforced here
//!
//! The §7.16 driver assumes the caller has already enforced:
//!
//! * The §5.9.8 `SuperresDenom ∈ [SUPERRES_DENOM_MIN,
//!   SUPERRES_DENOM_MIN + (1 << SUPERRES_DENOM_BITS) - 1]` range
//!   (`9..=16`). [`crate::parse_frame_header`] guarantees this.
//! * The av1-spec p.326 line 18063 conformance requirement that
//!   `upscaledPlaneW > downscaledPlaneW` whenever `use_superres == 1`.
//!   With `SUPERRES_NUM = 8` and `SuperresDenom ∈ 9..=16` the spec's
//!   `FrameWidth = (UpscaledWidth * 8 + SuperresDenom/2) / SuperresDenom`
//!   derivation always satisfies this; we surface a [`UpscaledNotLarger`]
//!   error for the caller that hand-builds a context anyway.
//! * Plane buffers are sized at
//!   input  = `Round2(FrameHeight, subY) × Round2(FrameWidth, subX)`,
//!   output = `Round2(FrameHeight, subY) × Round2(UpscaledWidth, subX)`.

use crate::loop_filter::PlaneBuffer;

// =====================================================================
// §3 constants — av1-spec p.32 lines 1395-1412.
// =====================================================================

/// `FILTER_BITS = 7` per av1-spec p.32 line 1312 — coefficient
/// precision of [`UPSCALE_FILTER`] (each row sums to `128 = 1 << 7`).
pub const FILTER_BITS: u32 = 7;

/// `SUPERRES_FILTER_BITS = 6` per av1-spec p.32 line 1395 — bit width
/// of the fractional position index into [`UPSCALE_FILTER`].
pub const SUPERRES_FILTER_BITS: u32 = 6;

/// `SUPERRES_FILTER_SHIFTS = 1 << SUPERRES_FILTER_BITS = 64` per
/// av1-spec p.32 line 1398 — number of sub-pixel filter phases.
pub const SUPERRES_FILTER_SHIFTS: usize = 1 << SUPERRES_FILTER_BITS;

/// `SUPERRES_FILTER_TAPS = 8` per av1-spec p.32 line 1401 — taps per
/// filter phase.
pub const SUPERRES_FILTER_TAPS: usize = 8;

/// `SUPERRES_FILTER_OFFSET = 3` per av1-spec p.32 line 1403 — centre
/// tap of the 8-tap kernel: the `k`-th tap reads from
/// `srcXPx + (k - SUPERRES_FILTER_OFFSET)`.
pub const SUPERRES_FILTER_OFFSET: i32 = 3;

/// `SUPERRES_SCALE_BITS = 14` per av1-spec p.32 line 1405 — fractional
/// bits of the `srcX` accumulator used to step through the source
/// `downscaledPlaneW`-wide row.
pub const SUPERRES_SCALE_BITS: u32 = 14;

/// `SUPERRES_SCALE_MASK = (1 << 14) - 1` per av1-spec p.32 line 1408 —
/// isolates the fractional part of `srcX`.
pub const SUPERRES_SCALE_MASK: i32 = (1 << SUPERRES_SCALE_BITS) - 1;

/// `SUPERRES_EXTRA_BITS = SUPERRES_SCALE_BITS - SUPERRES_FILTER_BITS = 8`
/// per av1-spec p.32 line 1410 — right-shift that maps a
/// `srcX & SUPERRES_SCALE_MASK` value into a filter-phase index in
/// `[0, SUPERRES_FILTER_SHIFTS)`.
pub const SUPERRES_EXTRA_BITS: u32 = SUPERRES_SCALE_BITS - SUPERRES_FILTER_BITS;

// =====================================================================
// `Upscale_Filter[SUPERRES_FILTER_SHIFTS][SUPERRES_FILTER_TAPS]`
// av1-spec p.326 lines 18027-18060 — verbatim.
// =====================================================================

/// `Upscale_Filter[64][8]` per av1-spec p.326 lines 18027-18060 — the
/// 8-tap polyphase upscaling filter for the §7.16 superres pass.
///
/// Indexed as `UPSCALE_FILTER[srcXSubpel][k]` where `srcXSubpel` is a
/// 6-bit sub-pixel phase index in `[0, 64)` and `k` is the tap index in
/// `[0, 8)`. Each row sums to `128` (= `1 << FILTER_BITS`), so the
/// per-output-sample `Round2(Σ px * tap, FILTER_BITS)` reduction
/// preserves average brightness.
#[rustfmt::skip]
pub const UPSCALE_FILTER: [[i32; SUPERRES_FILTER_TAPS]; SUPERRES_FILTER_SHIFTS] = [
    [ 0, 0,   0, 128,   0,   0, 0,  0],
    [ 0, 0,  -1, 128,   2,  -1, 0,  0],
    [ 0, 1,  -3, 127,   4,  -2, 1,  0],
    [ 0, 1,  -4, 127,   6,  -3, 1,  0],
    [ 0, 2,  -6, 126,   8,  -3, 1,  0],
    [ 0, 2,  -7, 125,  11,  -4, 1,  0],
    [-1, 2,  -8, 125,  13,  -5, 2,  0],
    [-1, 3,  -9, 124,  15,  -6, 2,  0],
    [-1, 3, -10, 123,  18,  -6, 2, -1],
    [-1, 3, -11, 122,  20,  -7, 3, -1],
    [-1, 4, -12, 121,  22,  -8, 3, -1],
    [-1, 4, -13, 120,  25,  -9, 3, -1],
    [-1, 4, -14, 118,  28,  -9, 3, -1],
    [-1, 4, -15, 117,  30, -10, 4, -1],
    [-1, 5, -16, 116,  32, -11, 4, -1],
    [-1, 5, -16, 114,  35, -12, 4, -1],
    [-1, 5, -17, 112,  38, -12, 4, -1],
    [-1, 5, -18, 111,  40, -13, 5, -1],
    [-1, 5, -18, 109,  43, -14, 5, -1],
    [-1, 6, -19, 107,  45, -14, 5, -1],
    [-1, 6, -19, 105,  48, -15, 5, -1],
    [-1, 6, -19, 103,  51, -16, 5, -1],
    [-1, 6, -20, 101,  53, -16, 6, -1],
    [-1, 6, -20,  99,  56, -17, 6, -1],
    [-1, 6, -20,  97,  58, -17, 6, -1],
    [-1, 6, -20,  95,  61, -18, 6, -1],
    [-2, 7, -20,  93,  64, -18, 6, -2],
    [-2, 7, -20,  91,  66, -19, 6, -1],
    [-2, 7, -20,  88,  69, -19, 6, -1],
    [-2, 7, -20,  86,  71, -19, 6, -1],
    [-2, 7, -20,  84,  74, -20, 7, -2],
    [-2, 7, -20,  81,  76, -20, 7, -1],
    [-2, 7, -20,  79,  79, -20, 7, -2],
    [-1, 7, -20,  76,  81, -20, 7, -2],
    [-2, 7, -20,  74,  84, -20, 7, -2],
    [-1, 6, -19,  71,  86, -20, 7, -2],
    [-1, 6, -19,  69,  88, -20, 7, -2],
    [-1, 6, -19,  66,  91, -20, 7, -2],
    [-2, 6, -18,  64,  93, -20, 7, -2],
    [-1, 6, -18,  61,  95, -20, 6, -1],
    [-1, 6, -17,  58,  97, -20, 6, -1],
    [-1, 6, -17,  56,  99, -20, 6, -1],
    [-1, 6, -16,  53, 101, -20, 6, -1],
    [-1, 5, -16,  51, 103, -19, 6, -1],
    [-1, 5, -15,  48, 105, -19, 6, -1],
    [-1, 5, -14,  45, 107, -19, 6, -1],
    [-1, 5, -14,  43, 109, -18, 5, -1],
    [-1, 5, -13,  40, 111, -18, 5, -1],
    [-1, 4, -12,  38, 112, -17, 5, -1],
    [-1, 4, -12,  35, 114, -16, 5, -1],
    [-1, 4, -11,  32, 116, -16, 5, -1],
    [-1, 4, -10,  30, 117, -15, 4, -1],
    [-1, 3,  -9,  28, 118, -14, 4, -1],
    [-1, 3,  -9,  25, 120, -13, 4, -1],
    [-1, 3,  -8,  22, 121, -12, 4, -1],
    [-1, 3,  -7,  20, 122, -11, 3, -1],
    [-1, 2,  -6,  18, 123, -10, 3, -1],
    [ 0, 2,  -6,  15, 124,  -9, 3, -1],
    [ 0, 2,  -5,  13, 125,  -8, 2, -1],
    [ 0, 1,  -4,  11, 125,  -7, 2,  0],
    [ 0, 1,  -3,   8, 126,  -6, 2,  0],
    [ 0, 1,  -3,   6, 127,  -4, 1,  0],
    [ 0, 1,  -2,   4, 127,  -3, 1,  0],
    [ 0, 0,  -1,   2, 128,  -1, 0,  0],
];

// =====================================================================
// Error / context types.
// =====================================================================

/// Error variants the §7.16 driver can return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuperresError {
    /// Caller supplied `use_superres == true` but `UpscaledWidth` was
    /// not strictly greater than `FrameWidth`, violating the av1-spec
    /// p.326 line 18063 conformance requirement that `upscaledPlaneW >
    /// downscaledPlaneW`.
    UpscaledNotLarger,
    /// A supplied input / output plane buffer had a row / column count
    /// that disagreed with the `(plane, subX, subY)` derivation from
    /// `(FrameWidth, UpscaledWidth, FrameHeight)`.
    PlaneShapeMismatch,
    /// `num_planes` exceeded the number of plane buffers supplied
    /// (either input or output side).
    PlaneCountMismatch,
}

/// Caller-supplied frame-level inputs the §7.16 driver consults.
///
/// Every field comes straight from the parsed frame header (§5.5.2 for
/// `subsampling_*` / `bit_depth` / `num_planes`; §5.9.5 for
/// `frame_height` / `mi_cols`; §5.9.8 for `use_superres` /
/// `frame_width` (post-superres downscale) / `upscaled_width`).
#[derive(Debug, Clone, Copy)]
pub struct SuperresFrameContext {
    /// `use_superres` per §5.9.8. When `false`, the driver returns a
    /// straight copy of the input planes.
    pub use_superres: bool,
    /// `FrameWidth` per §5.9.5 — the post-superres downscaled width
    /// (i.e. the width of the input plane buffers).
    pub frame_width: u32,
    /// `UpscaledWidth` per §5.9.8 — the pre-superres-downscale width
    /// (i.e. the width of the output plane buffers).
    pub upscaled_width: u32,
    /// `FrameHeight` per §5.9.5 — both input and output share this
    /// height (superres only upscales horizontally).
    pub frame_height: u32,
    /// `MiCols` per §5.9.5 — drives the `miW = MiCols >> subX` /
    /// `maxX = miW * MI_SIZE - 1` derivation that defines how far the
    /// `Clip3` edge replication reaches.
    pub mi_cols: u32,
    /// `NumPlanes` per §5.5.2 (`1` for monochrome, `3` otherwise).
    pub num_planes: u8,
    /// `BitDepth` per §5.5.2 (`8`, `10`, or `12`).
    pub bit_depth: u8,
    /// `subsampling_x` per §5.5.2 (`0` or `1`).
    pub subsampling_x: u8,
    /// `subsampling_y` per §5.5.2 (`0` or `1`).
    pub subsampling_y: u8,
}

// =====================================================================
// §7.16 driver — av1-spec p.325-326 lines 17963-18065.
// =====================================================================

/// §7.16 top-level upscaling driver — av1-spec p.325 lines 17963-17968.
///
/// Reads `FrameWidth`-wide (post-CDEF) samples from `input_planes` and
/// writes `UpscaledWidth`-wide samples into `output_planes`. The
/// per-plane buffer extents must match the `Round2(width, subX)` /
/// `Round2(FrameHeight, subY)` derivation.
///
/// On `use_superres == 0` the driver performs a verbatim per-plane copy
/// per the spec's "this process returns inputFrame" short-circuit.
///
/// `output_planes.len()` and `input_planes.len()` must both be at least
/// `ctx.num_planes`. Any excess buffers are ignored.
pub fn upscale_frame(
    ctx: &SuperresFrameContext,
    input_planes: &[PlaneBuffer<'_>],
    output_planes: &mut [PlaneBuffer<'_>],
) -> Result<(), SuperresError> {
    let num_planes = ctx.num_planes as usize;
    if input_planes.len() < num_planes || output_planes.len() < num_planes {
        return Err(SuperresError::PlaneCountMismatch);
    }

    // av1-spec p.325 line 17968 — use_superres == 0 ⇒ verbatim copy.
    // Also covers the FrameWidth == UpscaledWidth degenerate case.
    if !ctx.use_superres || ctx.frame_width == ctx.upscaled_width {
        for plane in 0..ctx.num_planes {
            copy_plane(input_planes, output_planes, plane)?;
        }
        return Ok(());
    }

    // av1-spec p.326 line 18063 — conformance: upscaledPlaneW must be
    // strictly greater than downscaledPlaneW. With the §5.9.8
    // FrameWidth derivation this is automatic, but we guard the
    // hand-built-context path.
    if ctx.upscaled_width <= ctx.frame_width {
        return Err(SuperresError::UpscaledNotLarger);
    }

    for plane in 0..ctx.num_planes {
        upscale_plane(
            ctx,
            plane,
            &input_planes[plane as usize],
            &mut output_planes[plane as usize],
        )?;
    }
    Ok(())
}

/// §7.16 per-plane body — av1-spec p.325 lines 17979-18013.
///
/// Derives `(subX, subY, downscaledPlaneW, upscaledPlaneW, planeH,
/// stepX, err, initialSubpelX, miW, minX, maxX)` per the spec, then
/// invokes [`upscale_sample`] for each `(y, x) ∈ [0, planeH) × [0,
/// upscaledPlaneW)`.
pub fn upscale_plane(
    ctx: &SuperresFrameContext,
    plane: u8,
    input: &PlaneBuffer<'_>,
    output: &mut PlaneBuffer<'_>,
) -> Result<(), SuperresError> {
    // av1-spec p.325 lines 17979-17985 — (subX, subY) for `plane`.
    let (sub_x, sub_y) = subsampling_for_plane(plane, ctx.subsampling_x, ctx.subsampling_y);

    // av1-spec p.325 lines 17987-17989 — plane-relative widths.
    let downscaled_plane_w = round2(ctx.frame_width, sub_x);
    let upscaled_plane_w = round2(ctx.upscaled_width, sub_x);
    let plane_h = round2(ctx.frame_height, sub_y);

    if input.rows != plane_h
        || input.cols != downscaled_plane_w
        || output.rows != plane_h
        || output.cols != upscaled_plane_w
    {
        return Err(SuperresError::PlaneShapeMismatch);
    }

    if upscaled_plane_w == 0 || plane_h == 0 {
        return Ok(());
    }
    if upscaled_plane_w <= downscaled_plane_w {
        return Err(SuperresError::UpscaledNotLarger);
    }

    // av1-spec p.325 line 17990 — stepX accumulator step.
    let stepx_num: i64 =
        (i64::from(downscaled_plane_w) << SUPERRES_SCALE_BITS) + i64::from(upscaled_plane_w / 2);
    let step_x: i32 = (stepx_num / i64::from(upscaled_plane_w)) as i32;

    // av1-spec p.325 line 17991 — err residual.
    let err: i32 = (i64::from(upscaled_plane_w) * i64::from(step_x)
        - (i64::from(downscaled_plane_w) << SUPERRES_SCALE_BITS)) as i32;

    // av1-spec p.325 lines 17992-17995 — initialSubpelX seed.
    let diff: i64 = i64::from(upscaled_plane_w) - i64::from(downscaled_plane_w);
    let num: i64 = -(diff << (SUPERRES_SCALE_BITS - 1)) + i64::from(upscaled_plane_w / 2);
    // The spec division rounds toward zero (signed integer division).
    let division: i32 = (num / i64::from(upscaled_plane_w)) as i32;
    let mut initial_subpel_x: i32 = division + (1 << (SUPERRES_EXTRA_BITS - 1)) - err / 2;
    initial_subpel_x &= SUPERRES_SCALE_MASK;

    // av1-spec p.325 lines 17997-17999 — clamp window for srcXPx.
    let mi_w = ctx.mi_cols >> sub_x;
    let min_x: i32 = 0;
    let max_x: i32 = (mi_w as i32) * (crate::cdf::MI_SIZE as i32) - 1;

    let bit_depth = ctx.bit_depth;
    let in_cols = input.cols as usize;
    let out_cols = output.cols as usize;

    for y in 0..plane_h {
        let in_row_off = (y as usize) * in_cols;
        let in_row = &input.samples[in_row_off..in_row_off + in_cols];
        let out_row_off = (y as usize) * out_cols;
        for x in 0..upscaled_plane_w {
            // av1-spec p.325 lines 18002-18004 — srcX derivation.
            let src_x: i32 =
                -(1i32 << SUPERRES_SCALE_BITS) + initial_subpel_x + (x as i32) * step_x;
            let src_x_px: i32 = src_x >> SUPERRES_SCALE_BITS;
            let src_x_subpel: i32 = (src_x & SUPERRES_SCALE_MASK) >> SUPERRES_EXTRA_BITS;
            let sample = upscale_sample(
                in_row,
                src_x_px,
                src_x_subpel as usize,
                min_x,
                max_x,
                bit_depth,
            );
            output.samples[out_row_off + (x as usize)] = sample;
        }
    }
    Ok(())
}

/// §7.16 per-output-sample inner loop — av1-spec p.325 lines 18005-18011.
///
/// Performs the 8-tap polyphase convolution against
/// `UPSCALE_FILTER[srcXSubpel]`, with the §3 `Clip3(minX, maxX, srcXPx
/// + (k - SUPERRES_FILTER_OFFSET))` left/right edge replication.
/// Returns the post-`Round2(., FILTER_BITS)` post-`Clip1` sample.
///
/// `in_row` is the source `downscaledPlaneW`-wide row that the spec
/// addresses as `frame[plane][y][.]`; the caller is responsible for
/// passing the row for the matching `y`.
pub fn upscale_sample(
    in_row: &[i32],
    src_x_px: i32,
    src_x_subpel: usize,
    min_x: i32,
    max_x: i32,
    bit_depth: u8,
) -> i32 {
    let taps = &UPSCALE_FILTER[src_x_subpel & (SUPERRES_FILTER_SHIFTS - 1)];
    let mut sum: i32 = 0;
    for (k, &tap) in taps.iter().enumerate() {
        // av1-spec p.325 line 18007 — Clip3(minX, maxX, ...).
        let sample_x = clip3(min_x, max_x, src_x_px + (k as i32 - SUPERRES_FILTER_OFFSET));
        // Defensive clip into the row range — `max_x` follows the
        // `miW * MI_SIZE` derivation and can exceed the actual
        // downscaledPlaneW (which always satisfies `downscaledPlaneW <=
        // miW * MI_SIZE`); the spec is silent on that case but the
        // reference reads `frame[plane][y][sampleX]` so we replicate
        // the rightmost real sample.
        let idx = sample_x.clamp(0, in_row.len() as i32 - 1) as usize;
        sum += in_row[idx] * tap;
    }
    clip1(bit_depth, round2_i32(sum, FILTER_BITS))
}

// =====================================================================
// Helpers — §3 numeric primitives.
// =====================================================================

/// §3 `Round2(x, n)` — `(x + (1 << (n - 1))) >> n` for `n > 0`,
/// identity for `n == 0`. `u32` flavour for the §5.9.8
/// `Round2(width, subX)` plane-extent derivation.
#[inline]
fn round2(x: u32, n: u8) -> u32 {
    if n == 0 {
        x
    } else {
        (x + (1u32 << (n - 1))) >> n
    }
}

/// `Round2(x, n)` over `i32` — used by the §7.16 final
/// `Round2(sum, FILTER_BITS)` reduction.
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

/// §3 `Clip1(x)` — clamp to the bit-depth range `[0, (1 << bit_depth) - 1]`.
#[inline]
fn clip1(bit_depth: u8, x: i32) -> i32 {
    let hi = (1i32 << bit_depth) - 1;
    clip3(0, hi, x)
}

/// `(subX, subY)` for `plane` — `(0, 0)` for luma per av1-spec p.325
/// lines 17980-17986.
#[inline]
fn subsampling_for_plane(plane: u8, sub_x: u8, sub_y: u8) -> (u8, u8) {
    if plane == 0 {
        (0, 0)
    } else {
        (sub_x, sub_y)
    }
}

/// Verbatim plane copy used by the `use_superres == 0` short-circuit.
fn copy_plane(
    input_planes: &[PlaneBuffer<'_>],
    output_planes: &mut [PlaneBuffer<'_>],
    plane: u8,
) -> Result<(), SuperresError> {
    let p = plane as usize;
    let src = &input_planes[p];
    let dst = &mut output_planes[p];
    if src.rows != dst.rows || src.cols != dst.cols {
        return Err(SuperresError::PlaneShapeMismatch);
    }
    dst.samples.copy_from_slice(src.samples);
    Ok(())
}

// =====================================================================
// Tests.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_plane(rows: u32, cols: u32, fill: i32) -> Vec<i32> {
        vec![fill; (rows * cols) as usize]
    }

    #[test]
    fn upscale_filter_table_sums_to_128() {
        // Each row of Upscale_Filter sums to 128 = (1 << FILTER_BITS),
        // so the `Round2(sum, FILTER_BITS)` reduction preserves average
        // brightness. av1-spec p.326 lines 18027-18060.
        for (i, row) in UPSCALE_FILTER.iter().enumerate() {
            let sum: i32 = row.iter().sum();
            assert_eq!(sum, 128, "row {i} sums to {sum}, expected 128");
        }
    }

    #[test]
    fn upscale_filter_table_geometry() {
        assert_eq!(UPSCALE_FILTER.len(), SUPERRES_FILTER_SHIFTS);
        assert_eq!(UPSCALE_FILTER[0].len(), SUPERRES_FILTER_TAPS);
        // Phase 0 is the identity kernel: `[0,0,0,128,0,0,0,0]`.
        assert_eq!(
            UPSCALE_FILTER[0],
            [0, 0, 0, 128, 0, 0, 0, 0],
            "phase 0 of Upscale_Filter must be identity"
        );
    }

    #[test]
    fn upscale_constants_match_spec() {
        // av1-spec p.32 lines 1395-1412 — sanity check the named-constant
        // table relations.
        assert_eq!(SUPERRES_FILTER_SHIFTS, 1 << SUPERRES_FILTER_BITS);
        assert_eq!(
            SUPERRES_EXTRA_BITS,
            SUPERRES_SCALE_BITS - SUPERRES_FILTER_BITS
        );
        assert_eq!(SUPERRES_SCALE_MASK, (1 << SUPERRES_SCALE_BITS) - 1);
    }

    #[test]
    fn use_superres_false_copies_planes_verbatim() {
        // av1-spec p.325 line 17968 — use_superres == 0 short-circuits
        // to a verbatim copy.
        let mut src = make_plane(16, 16, 137);
        let mut dst = make_plane(16, 16, 0);
        let src_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut src,
        };
        let mut dst_buf = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut dst,
        };
        let ctx = SuperresFrameContext {
            use_superres: false,
            frame_width: 16,
            upscaled_width: 16,
            frame_height: 16,
            mi_cols: 4,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
        };
        upscale_frame(
            &ctx,
            std::slice::from_ref(&src_buf),
            std::slice::from_mut(&mut dst_buf),
        )
        .unwrap();
        assert!(dst_buf.samples.iter().all(|&v| v == 137));
    }

    #[test]
    fn equal_widths_copies_planes_verbatim() {
        // Belt-and-braces: even with use_superres == true, when
        // FrameWidth == UpscaledWidth the spec's stepX collapses to a
        // copy. We short-circuit explicitly so the FILTER_OFFSET drift
        // doesn't bleed through.
        let mut src = make_plane(8, 24, 99);
        let mut dst = make_plane(8, 24, 0);
        let src_buf = PlaneBuffer {
            rows: 8,
            cols: 24,
            samples: &mut src,
        };
        let mut dst_buf = PlaneBuffer {
            rows: 8,
            cols: 24,
            samples: &mut dst,
        };
        let ctx = SuperresFrameContext {
            use_superres: true,
            frame_width: 24,
            upscaled_width: 24,
            frame_height: 8,
            mi_cols: 6,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
        };
        upscale_frame(
            &ctx,
            std::slice::from_ref(&src_buf),
            std::slice::from_mut(&mut dst_buf),
        )
        .unwrap();
        assert!(dst_buf.samples.iter().all(|&v| v == 99));
    }

    #[test]
    fn flat_input_yields_flat_output() {
        // Convolving a constant input with a unity-sum filter returns
        // the same constant (modulo Round2 rounding, which is exact
        // here because sum = 128*v + (1 << (FILTER_BITS - 1)) shifts
        // back to v cleanly when the rounding bit aligns). This
        // exercises every (y, x) of the inner loop including the
        // Clip3 edge taps.
        let frame_w = 8u32;
        let upscaled_w = 16u32;
        let frame_h = 4u32;
        let mut src = make_plane(frame_h, frame_w, 80);
        let mut dst = make_plane(frame_h, upscaled_w, 0);
        let src_buf = PlaneBuffer {
            rows: frame_h,
            cols: frame_w,
            samples: &mut src,
        };
        let mut dst_buf = PlaneBuffer {
            rows: frame_h,
            cols: upscaled_w,
            samples: &mut dst,
        };
        let ctx = SuperresFrameContext {
            use_superres: true,
            frame_width: frame_w,
            upscaled_width: upscaled_w,
            frame_height: frame_h,
            mi_cols: 4, // 4 * MI_SIZE = 16 ≥ upscaled width
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
        };
        upscale_frame(
            &ctx,
            std::slice::from_ref(&src_buf),
            std::slice::from_mut(&mut dst_buf),
        )
        .unwrap();
        // Σ taps = 128, sample = 80, Round2(80 * 128, 7) = Round2(10240, 7).
        // 10240 + 64 = 10304; 10304 >> 7 = 80. Exact.
        for (i, &v) in dst_buf.samples.iter().enumerate() {
            assert_eq!(v, 80, "pixel {i} = {v}, expected 80");
        }
    }

    #[test]
    fn upscale_clips_to_bit_depth() {
        // 8-bit samples saturate at 255 — even when the kernel's
        // negative-tap geometry would push the convolution outside the
        // [0, 255] envelope on a synthetic input, Clip1 brings it back.
        // We construct a deliberately spiky 8-wide row of 0/255
        // alternations and check none of the upscaled samples exceeds
        // 255 nor falls below 0.
        let row: [i32; 8] = [255, 0, 255, 0, 255, 0, 255, 0];
        let upscaled_w = 24u32;
        let mut src = Vec::with_capacity(8);
        src.extend_from_slice(&row);
        let mut dst = vec![0i32; upscaled_w as usize];
        let src_buf = PlaneBuffer {
            rows: 1,
            cols: 8,
            samples: &mut src,
        };
        let mut dst_buf = PlaneBuffer {
            rows: 1,
            cols: upscaled_w,
            samples: &mut dst,
        };
        let ctx = SuperresFrameContext {
            use_superres: true,
            frame_width: 8,
            upscaled_width: upscaled_w,
            frame_height: 1,
            mi_cols: 6, // 6 * 4 = 24 ≥ upscaled width
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
        };
        upscale_frame(
            &ctx,
            std::slice::from_ref(&src_buf),
            std::slice::from_mut(&mut dst_buf),
        )
        .unwrap();
        for &v in dst_buf.samples.iter() {
            assert!((0..=255).contains(&v), "sample {v} escaped 8-bit Clip1");
        }
    }

    #[test]
    fn upscaled_smaller_than_downscaled_is_rejected() {
        // av1-spec p.326 line 18063 — upscaledPlaneW > downscaledPlaneW
        // is a bitstream-conformance requirement when use_superres == 1.
        let mut src = make_plane(4, 16, 0);
        let mut dst = make_plane(4, 8, 0);
        let src_buf = PlaneBuffer {
            rows: 4,
            cols: 16,
            samples: &mut src,
        };
        let mut dst_buf = PlaneBuffer {
            rows: 4,
            cols: 8,
            samples: &mut dst,
        };
        let ctx = SuperresFrameContext {
            use_superres: true,
            frame_width: 16,
            upscaled_width: 8,
            frame_height: 4,
            mi_cols: 4,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
        };
        let r = upscale_frame(
            &ctx,
            std::slice::from_ref(&src_buf),
            std::slice::from_mut(&mut dst_buf),
        );
        assert_eq!(r, Err(SuperresError::UpscaledNotLarger));
    }

    #[test]
    fn upscale_sample_phase_zero_is_passthrough() {
        // Phase 0 is the identity kernel; for any srcXPx well inside the
        // row the per-sample helper returns the source sample exactly.
        let row: Vec<i32> = (0..16).collect();
        for src_x_px in 3..13 {
            let v = upscale_sample(&row, src_x_px, 0, 0, 15, 8);
            assert_eq!(v, src_x_px, "phase 0 at px {src_x_px} returned {v}");
        }
    }

    #[test]
    fn upscale_sample_edge_replicates_via_clip3() {
        // When srcXPx + (k - 3) walks off the left edge, the spec uses
        // Clip3(minX, maxX, .) to replicate the boundary sample. With
        // srcXPx = 0 and minX = 0, k ∈ {0, 1, 2} all clip to in_row[0].
        let row = vec![10, 0, 0, 0, 0, 0, 0, 0];
        let v = upscale_sample(&row, 0, 0, 0, 7, 8);
        // Phase 0 is `[0,0,0,128,0,0,0,0]` — the centre tap reads
        // row[0 + (3 - 3)] = row[0] = 10.
        assert_eq!(v, 10);
    }

    #[test]
    fn upscale_plane_geometry_mismatch_is_rejected() {
        // Caller passed an output buffer that's narrower than the
        // ctx-derived upscaled width — must error out, not panic on
        // index.
        let mut src = make_plane(4, 8, 0);
        let mut dst = make_plane(4, 10, 0);
        let src_buf = PlaneBuffer {
            rows: 4,
            cols: 8,
            samples: &mut src,
        };
        let mut dst_buf = PlaneBuffer {
            rows: 4,
            cols: 10,
            samples: &mut dst,
        };
        let ctx = SuperresFrameContext {
            use_superres: true,
            frame_width: 8,
            upscaled_width: 16,
            frame_height: 4,
            mi_cols: 4,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
        };
        let r = upscale_frame(
            &ctx,
            std::slice::from_ref(&src_buf),
            std::slice::from_mut(&mut dst_buf),
        );
        assert_eq!(r, Err(SuperresError::PlaneShapeMismatch));
    }

    #[test]
    fn chroma_subsampling_halves_widths() {
        // For 4:2:0, the chroma plane's downscaledPlaneW =
        // Round2(FrameWidth, 1) and upscaledPlaneW = Round2(UpscaledWidth, 1).
        // The same flat-input invariant holds.
        let frame_w = 16u32;
        let upscaled_w = 24u32;
        let frame_h = 8u32;
        let chroma_in_w = frame_w.div_ceil(2); // = 8
        let chroma_out_w = upscaled_w.div_ceil(2); // = 12
        let chroma_h = frame_h.div_ceil(2); // = 4

        let mut y_in = make_plane(frame_h, frame_w, 64);
        let mut u_in = make_plane(chroma_h, chroma_in_w, 128);
        let mut v_in = make_plane(chroma_h, chroma_in_w, 192);
        let mut y_out = make_plane(frame_h, upscaled_w, 0);
        let mut u_out = make_plane(chroma_h, chroma_out_w, 0);
        let mut v_out = make_plane(chroma_h, chroma_out_w, 0);

        let inputs = [
            PlaneBuffer {
                rows: frame_h,
                cols: frame_w,
                samples: &mut y_in,
            },
            PlaneBuffer {
                rows: chroma_h,
                cols: chroma_in_w,
                samples: &mut u_in,
            },
            PlaneBuffer {
                rows: chroma_h,
                cols: chroma_in_w,
                samples: &mut v_in,
            },
        ];
        let mut outputs = [
            PlaneBuffer {
                rows: frame_h,
                cols: upscaled_w,
                samples: &mut y_out,
            },
            PlaneBuffer {
                rows: chroma_h,
                cols: chroma_out_w,
                samples: &mut u_out,
            },
            PlaneBuffer {
                rows: chroma_h,
                cols: chroma_out_w,
                samples: &mut v_out,
            },
        ];

        let ctx = SuperresFrameContext {
            use_superres: true,
            frame_width: frame_w,
            upscaled_width: upscaled_w,
            frame_height: frame_h,
            mi_cols: 6, // 6 * 4 = 24, halved for chroma = 12 cols max
            num_planes: 3,
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
        };

        upscale_frame(&ctx, &inputs, &mut outputs).unwrap();
        assert!(outputs[0].samples.iter().all(|&v| v == 64));
        assert!(outputs[1].samples.iter().all(|&v| v == 128));
        assert!(outputs[2].samples.iter().all(|&v| v == 192));
    }
}
