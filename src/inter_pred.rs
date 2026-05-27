//! §7.11.3 Inter prediction process — sample-generation leaves and
//! supporting helpers per av1-spec p.257-265.
//!
//! This module covers the translational single-reference path of the
//! §7.11.3 inter prediction process:
//!
//! * §7.11.3.2 — [`rounding_variables`]: the
//!   `(InterRound0, InterRound1, InterPostRound)` rounding-shift
//!   derivation from `(BitDepth, isCompound)` per av1-spec p.259.
//!
//! * §7.11.3.3 — [`motion_vector_scaling`]: the
//!   `(startX, startY, stepX, stepY)` reference-frame sample-location
//!   derivation from `(plane, refIdx, x, y, mv)` per av1-spec
//!   p.260-261, returning the 10-fractional-bit reference-frame
//!   coordinate (`SCALE_SUBPEL_BITS = 10`).
//!
//! * §7.11.3.4 — [`block_inter_prediction`]: the 8-tap horizontal +
//!   vertical convolution kernel that turns `(ref[ plane ][..][..],
//!   startX, startY, stepX, stepY, w, h)` plus the chosen
//!   `interp_filter[0..1]` pair into the per-cell `pred[i][j]`
//!   intermediate samples. Per av1-spec p.262-265: the intermediate
//!   array is sized `intermediateHeight × w` with
//!   `intermediateHeight = ((h-1) * yStep + (1 << SCALE_SUBPEL_BITS) - 1) >> SCALE_SUBPEL_BITS + 8`;
//!   horizontal filter reads `ref[ Clip3(0, lastY, ...) ][ Clip3( 0,
//!   lastX, ...) ]` per (r, c, t) and rounds by
//!   `Round2(s, InterRound0)`; vertical filter reads
//!   `intermediate[ (p >> 10) + t ][ c ]` per (r, c, t) and rounds by
//!   `Round2(s, InterRound1)`.
//!
//! * [`SUBPEL_FILTERS`] — the `Subpel_Filters[ 6 ][ 16 ][ 8 ]`
//!   coefficient table verbatim from av1-spec p.263-265: indices
//!   `0..4` are the four 8-tap full-size filters (`EIGHTTAP`,
//!   `EIGHTTAP_SMOOTH`, `EIGHTTAP_SHARP`, `BILINEAR`); indices `4..6`
//!   are the small-block (`w <= 4` / `h <= 4`) 4-tap versions of
//!   `EIGHTTAP` and `EIGHTTAP_SMOOTH` per the small-block remap rule
//!   at the head of §7.11.3.4 (lines 14602-14608 of av1-spec.txt).
//!
//! * [`select_interp_filter_small_block`] — the small-block remap
//!   (`EIGHTTAP` / `EIGHTTAP_SHARP` → 4, `EIGHTTAP_SMOOTH` → 5) per
//!   av1-spec p.262 lines 14602-14608 and 14626-14632.
//!
//! ## Scope
//!
//! Translational single-ref MC only — the simplest §7.11.3.1 case
//! (`useWarp == 0`, `isCompound == 0`, `IsInterIntra == 0`, no OBMC,
//! no compound mask). Deferred to subsequent arcs:
//!
//! * §7.11.3.5 — `block_warp` (LOCALWARP / GLOBAL_GLOBALMV affine warp)
//! * §7.11.3.6 — `setup_shear` (warp parameter shear)
//! * §7.11.3.7 — `resolve_divisor` (warp divisor resolution)
//! * §7.11.3.8 — `warp_estimation` (LOCALWARP estimation from §7.10.4)
//! * §7.11.3.9 — `overlapped_motion_compensation` (OBMC)
//! * §7.11.3.10 — `overlap_blending` (OBMC blending)
//! * §7.11.3.11 — `wedge_mask` (COMPOUND_WEDGE)
//! * §7.11.3.12 — `difference_weight_mask` (COMPOUND_DIFFWTD)
//! * §7.11.3.13 — `intra_mode_variant_mask` (COMPOUND_INTRA)
//! * §7.11.3.14 — `mask_blend` (compound blend with mask)
//! * §7.11.3.15 — `distance_weights` (COMPOUND_DISTANCE)
//!
//! And the §7.11.3.1 driver itself — once these arrive, the
//! `block_inter_prediction` leaf below is what the driver invokes on
//! the `useWarp == 0` branch.

// =====================================================================
// §3 constants used by §7.11.3 — from av1-spec.txt p.16-17.
// =====================================================================

/// `FILTER_BITS = 7` — number of bits used in Wiener filter
/// coefficients (av1-spec p.17 line 1312). Reused by §7.11.3 for the
/// `InterPostRound = 2 * FILTER_BITS - (InterRound0 + InterRound1)`
/// derivation in §7.11.3.2.
pub const FILTER_BITS: u32 = 7;

/// `SUBPEL_BITS = 4` — fractional precision of the §7.11.3.3 motion
/// vector after the §5.11 `Mvs[]` clamp (av1-spec p.16 line 1101).
pub const SUBPEL_BITS: u32 = 4;

/// `SUBPEL_MASK = (1 << SUBPEL_BITS) - 1 = 15` — mask isolating the
/// SUBPEL_BITS fractional component (av1-spec p.16 line 1104).
pub const SUBPEL_MASK: u32 = (1 << SUBPEL_BITS) - 1;

/// `SCALE_SUBPEL_BITS = 10` — fractional precision of the
/// reference-frame coordinate after §7.11.3.3 scaling. The
/// `(startX, startY, stepX, stepY)` outputs of
/// [`motion_vector_scaling`] all carry this many fractional bits
/// (av1-spec p.17 line 1106).
pub const SCALE_SUBPEL_BITS: u32 = 10;

/// `REF_SCALE_SHIFT = 14` — fractional precision of the per-plane
/// reference-frame size scaling factor (`xScale` / `yScale`) (av1-spec
/// p.16 line 1098).
pub const REF_SCALE_SHIFT: u32 = 14;

/// `EIGHTTAP = 0` — av1-spec §6.8.9 interpolation-filter ordinal
/// (av1-spec p.161 line 9158). Reproduced locally for the
/// [`select_interp_filter_small_block`] body and the
/// [`SUBPEL_FILTERS`] table indices.
pub const EIGHTTAP: u8 = 0;
/// `EIGHTTAP_SMOOTH = 1`.
pub const EIGHTTAP_SMOOTH: u8 = 1;
/// `EIGHTTAP_SHARP = 2`.
pub const EIGHTTAP_SHARP: u8 = 2;
/// `BILINEAR = 3`.
pub const BILINEAR: u8 = 3;
/// Sentinel `4` — the 4-tap small-block version of `EIGHTTAP` /
/// `EIGHTTAP_SHARP` per av1-spec p.262 line 14604 ("if interpFilter ==
/// EIGHTTAP || interpFilter == EIGHTTAP_SHARP, interpFilter = 4").
/// The full filter row sits in [`SUBPEL_FILTERS`] at index 4.
pub const EIGHTTAP_4TAP: u8 = 4;
/// Sentinel `5` — the 4-tap small-block version of `EIGHTTAP_SMOOTH`
/// per av1-spec p.262 line 14606. Lives at [`SUBPEL_FILTERS`] index 5.
pub const EIGHTTAP_SMOOTH_4TAP: u8 = 5;

// =====================================================================
// §7.11.3.2 — Rounding variables derivation process (av1-spec p.259).
// =====================================================================

/// §7.11.3.2 output: the three rounding-shift counts the §7.11.3.4
/// horizontal / vertical / post convolutions consume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundingVars {
    /// `InterRound0` — bits to round after horizontal filtering.
    /// `3` for 8/10-bit content, `5` for 12-bit content.
    pub inter_round0: u32,
    /// `InterRound1` — bits to round after vertical filtering. Depends
    /// on both `isCompound` and `BitDepth`:
    /// * `isCompound == 1` ⇒ `7`.
    /// * `isCompound == 0, BitDepth != 12` ⇒ `11`.
    /// * `isCompound == 0, BitDepth == 12` ⇒ `9`.
    pub inter_round1: u32,
    /// `InterPostRound = 2 * FILTER_BITS - (InterRound0 + InterRound1)`
    /// — the post-prediction rounding shift used by the §7.11.3.1
    /// compound-blend Clip1-Round2 sites. (`FILTER_BITS = 7`.)
    pub inter_post_round: u32,
}

/// §7.11.3.2 (av1-spec p.259): derive the §7.11.3.4 / §7.11.3.1
/// rounding shifts from `(bit_depth, is_compound)`.
///
/// The spec body reads:
///
/// ```text
///   InterRound0 = 3
///   InterRound1 = isCompound ? 7 : 11
///   if (BitDepth == 12) InterRound0 += 2
///   if (BitDepth == 12 && !isCompound) InterRound1 -= 2
///   InterPostRound = 2 * FILTER_BITS - (InterRound0 + InterRound1)
/// ```
///
/// ## Arguments
///
/// * `bit_depth` — §5.5.2 frame `BitDepth` (`8`, `10`, or `12`).
/// * `is_compound` — §7.11.3.1 `isCompound = RefFrames[candRow][candCol][1] >
///   INTRA_FRAME` flag. `false` for single-reference inter blocks.
///
/// ## Returns
///
/// A [`RoundingVars`] carrying the three shifts. Returns
/// [`crate::Error::PartitionWalkOutOfRange`] for caller-bug
/// `bit_depth` outside `{8, 10, 12}`.
pub fn rounding_variables(bit_depth: u8, is_compound: bool) -> Result<RoundingVars, crate::Error> {
    if !matches!(bit_depth, 8 | 10 | 12) {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    let mut inter_round0: u32 = 3;
    let mut inter_round1: u32 = if is_compound { 7 } else { 11 };
    if bit_depth == 12 {
        inter_round0 += 2;
        if !is_compound {
            inter_round1 -= 2;
        }
    }
    let inter_post_round = 2 * FILTER_BITS - (inter_round0 + inter_round1);
    Ok(RoundingVars {
        inter_round0,
        inter_round1,
        inter_post_round,
    })
}

// =====================================================================
// §7.11.3.3 — Motion vector scaling process (av1-spec p.260-261).
// =====================================================================

/// §7.11.3.3 output: the §7.11.3.4 `(startX, startY, stepX, stepY)`
/// reference-frame sample-location quadruple.
///
/// All four components carry `SCALE_SUBPEL_BITS = 10` fractional bits
/// — i.e. `startX = 1024` represents one luma sample of horizontal
/// offset into the reference frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvScale {
    /// `startX` — left edge of the reference block in the reference
    /// frame's plane-coordinate system (10 fractional bits).
    pub start_x: i32,
    /// `startY` — top edge of the reference block in the reference
    /// frame's plane-coordinate system (10 fractional bits).
    pub start_y: i32,
    /// `stepX` — horizontal step per current-frame sample (10
    /// fractional bits). `1024` (= 1.0 in plane samples) for the
    /// `RefUpscaledWidth == FrameWidth` non-scaled case.
    pub step_x: i32,
    /// `stepY` — vertical step per current-frame sample (10 fractional
    /// bits). `1024` for the non-scaled case.
    pub step_y: i32,
}

/// §3 `Round2Signed(x, n)` — sign-preserving right-shift with bias-
/// half rounding. Mirrors the §3 definition at av1-spec.txt line 1568
/// — for negative inputs, `Round2Signed(x, n) = -Round2(-x, n)`.
#[inline]
pub(crate) fn round2_signed(x: i64, n: u32) -> i64 {
    if n == 0 {
        x
    } else if x >= 0 {
        (x + (1i64 << (n - 1))) >> n
    } else {
        -(((-x) + (1i64 << (n - 1))) >> n)
    }
}

/// §7.11.3.3 (av1-spec p.260-261): derive the
/// `(startX, startY, stepX, stepY)` reference-frame sample-location
/// quadruple for one (plane, refIdx, x, y, mv) input.
///
/// The spec body reads:
///
/// ```text
///   xScale = ( ( RefUpscaledWidth[refIdx]  << REF_SCALE_SHIFT ) + (FrameWidth  / 2) ) / FrameWidth
///   yScale = ( ( RefFrameHeight[refIdx]    << REF_SCALE_SHIFT ) + (FrameHeight / 2) ) / FrameHeight
///   subX, subY ← plane==0 ? (0,0) : (subsampling_x, subsampling_y)
///   halfSample = 1 << (SUBPEL_BITS - 1)
///   origX = (x << SUBPEL_BITS) + ((2 * mv[1]) >> subX) + halfSample
///   origY = (y << SUBPEL_BITS) + ((2 * mv[0]) >> subY) + halfSample
///   baseX = origX * xScale - (halfSample << REF_SCALE_SHIFT)
///   baseY = origY * yScale - (halfSample << REF_SCALE_SHIFT)
///   off   = (1 << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) / 2
///   startX = Round2Signed(baseX, REF_SCALE_SHIFT + SUBPEL_BITS - SCALE_SUBPEL_BITS) + off
///   startY = Round2Signed(baseY, REF_SCALE_SHIFT + SUBPEL_BITS - SCALE_SUBPEL_BITS) + off
///   stepX  = Round2Signed(xScale, REF_SCALE_SHIFT - SCALE_SUBPEL_BITS)
///   stepY  = Round2Signed(yScale, REF_SCALE_SHIFT - SCALE_SUBPEL_BITS)
/// ```
///
/// ## Arguments
///
/// * `plane` — `0` (luma), `1` (Cb), `2` (Cr); selects (subX, subY).
/// * `subsampling_x` / `subsampling_y` — §5.5.2 chroma subsampling
///   (`0` for 4:4:4, `1` for 4:2:2 / 4:2:0).
/// * `frame_width` / `frame_height` — current frame dimensions
///   (`FrameWidth` / `FrameHeight` per §5.9.5).
/// * `ref_upscaled_width` — reference frame `RefUpscaledWidth[refIdx]`
///   (the pre-superres width per §5.9.8). Must equal `FrameWidth` when
///   no superres / resize is in effect.
/// * `ref_frame_height` — reference frame `RefFrameHeight[refIdx]`.
/// * `x` / `y` — current-frame top-left sample coordinate of the
///   prediction region.
/// * `mv` — `[mv_row, mv_col]` motion vector in §5.11 1/8-sample
///   precision (i.e. with `SUBPEL_BITS - 1` fractional bits because
///   the spec multiplies by `2` before the `>> subX` shift).
///
/// ## Returns
///
/// An [`MvScale`] carrying `(startX, startY, stepX, stepY)`. Returns
/// [`crate::Error::PartitionWalkOutOfRange`] for caller-bug arguments:
/// `plane > 2`, `subsampling_x > 1`, `subsampling_y > 1`,
/// `frame_width == 0` or `frame_height == 0`, `ref_upscaled_width ==
/// 0` or `ref_frame_height == 0`.
#[allow(clippy::too_many_arguments)]
pub fn motion_vector_scaling(
    plane: u8,
    subsampling_x: u8,
    subsampling_y: u8,
    frame_width: u32,
    frame_height: u32,
    ref_upscaled_width: u32,
    ref_frame_height: u32,
    x: i32,
    y: i32,
    mv: [i16; 2],
) -> Result<MvScale, crate::Error> {
    // ---------- caller-bug guards ----------
    if plane > 2 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if subsampling_x > 1 || subsampling_y > 1 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if frame_width == 0 || frame_height == 0 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if ref_upscaled_width == 0 || ref_frame_height == 0 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }

    // §7.11.3.3 scale factors. Use i64 so the `<< REF_SCALE_SHIFT`
    // does not overflow on the largest legal frame size
    // (`RefUpscaledWidth <= 2 * FrameWidth` per the §7.11.3.3
    // conformance gate, with FrameWidth up to 65536).
    let x_scale: i64 = (((ref_upscaled_width as i64) << REF_SCALE_SHIFT)
        + (frame_width as i64) / 2)
        / (frame_width as i64);
    let y_scale: i64 = (((ref_frame_height as i64) << REF_SCALE_SHIFT) + (frame_height as i64) / 2)
        / (frame_height as i64);

    // §7.11.3.3 sub-sampling per-plane.
    let sub_x: u32 = if plane == 0 { 0 } else { subsampling_x as u32 };
    let sub_y: u32 = if plane == 0 { 0 } else { subsampling_y as u32 };

    // §7.11.3.3 origX / origY in 1/16-sample (SUBPEL_BITS=4) units.
    let half_sample: i64 = 1i64 << (SUBPEL_BITS - 1);
    let orig_x: i64 = ((x as i64) << SUBPEL_BITS) + ((2 * (mv[1] as i64)) >> sub_x) + half_sample;
    let orig_y: i64 = ((y as i64) << SUBPEL_BITS) + ((2 * (mv[0] as i64)) >> sub_y) + half_sample;

    // §7.11.3.3 baseX / baseY: reference-frame coordinate with
    // `REF_SCALE_SHIFT + SUBPEL_BITS = 18` fractional bits.
    let base_x: i64 = orig_x * x_scale - (half_sample << REF_SCALE_SHIFT);
    let base_y: i64 = orig_y * y_scale - (half_sample << REF_SCALE_SHIFT);

    // §7.11.3.3 off rounding offset for tap selection.
    let off: i64 = (1i64 << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) / 2;

    // §7.11.3.3 Round2Signed by `REF_SCALE_SHIFT + SUBPEL_BITS -
    // SCALE_SUBPEL_BITS = 8` so the result carries
    // `SCALE_SUBPEL_BITS = 10` fractional bits.
    let shift = REF_SCALE_SHIFT + SUBPEL_BITS - SCALE_SUBPEL_BITS;
    let start_x = round2_signed(base_x, shift) + off;
    let start_y = round2_signed(base_y, shift) + off;
    let step_x = round2_signed(x_scale, REF_SCALE_SHIFT - SCALE_SUBPEL_BITS);
    let step_y = round2_signed(y_scale, REF_SCALE_SHIFT - SCALE_SUBPEL_BITS);

    Ok(MvScale {
        start_x: start_x as i32,
        start_y: start_y as i32,
        step_x: step_x as i32,
        step_y: step_y as i32,
    })
}

// =====================================================================
// §7.11.3.4 — Subpel_Filters table (av1-spec p.263-265).
// =====================================================================

/// `Subpel_Filters[ 6 ][ 16 ][ 8 ]` — the six 16-phase filter banks
/// the §7.11.3.4 horizontal + vertical convolutions consume.
///
/// Indexed `[interpFilter][phase][tap]` where:
///
/// * `interpFilter ∈ 0..6`:
///   - `0 = EIGHTTAP` (8-tap regular)
///   - `1 = EIGHTTAP_SMOOTH` (8-tap smooth)
///   - `2 = EIGHTTAP_SHARP` (8-tap sharp)
///   - `3 = BILINEAR` (8-tap; coefficients realize a 2-tap bilinear)
///   - `4 = EIGHTTAP_4TAP` (small-block 4-tap reduction of EIGHTTAP /
///     EIGHTTAP_SHARP — taps 0,1,6,7 are zero)
///   - `5 = EIGHTTAP_SMOOTH_4TAP` (small-block 4-tap reduction of
///     EIGHTTAP_SMOOTH — taps 0,1,6,7 are zero)
/// * `phase ∈ 0..16` (SUBPEL_MASK + 1): the fractional-position
///   index `(p >> 6) & SUBPEL_MASK` from §7.11.3.4.
/// * `tap ∈ 0..8`: the eight convolution taps; sum across `tap` of
///   `coef * ref_sample` is the §7.11.3.4 `s` accumulator before
///   `Round2(s, InterRound0)` (horizontal) or `Round2(s,
///   InterRound1)` (vertical).
///
/// All values are even per the §7.11.3.4 note (av1-spec p.265 line
/// 14786). The phase-0 row of every filter is `(0, 0, 0, 128, 0, 0,
/// 0, 0)` — i.e. a unit copy at the integer position.
///
/// Table transcribed verbatim from av1-spec p.263-265 (av1-spec.txt
/// lines 14655-14783).
#[rustfmt::skip]
pub const SUBPEL_FILTERS: [[[i32; 8]; 16]; 6] = [
    // [0] EIGHTTAP (av1-spec p.263 lines 14656-14673).
    [
        [  0,  0,   0, 128,   0,   0,  0,  0 ],
        [  0,  2,  -6, 126,   8,  -2,  0,  0 ],
        [  0,  2, -10, 122,  18,  -4,  0,  0 ],
        [  0,  2, -12, 116,  28,  -8,  2,  0 ],
        [  0,  2, -14, 110,  38, -10,  2,  0 ],
        [  0,  2, -14, 102,  48, -12,  2,  0 ],
        [  0,  2, -16,  94,  58, -12,  2,  0 ],
        [  0,  2, -14,  84,  66, -12,  2,  0 ],
        [  0,  2, -14,  76,  76, -14,  2,  0 ],
        [  0,  2, -12,  66,  84, -14,  2,  0 ],
        [  0,  2, -12,  58,  94, -16,  2,  0 ],
        [  0,  2, -12,  48, 102, -14,  2,  0 ],
        [  0,  2, -10,  38, 110, -14,  2,  0 ],
        [  0,  2,  -8,  28, 116, -12,  2,  0 ],
        [  0,  0,  -4,  18, 122, -10,  2,  0 ],
        [  0,  0,  -2,   8, 126,  -6,  2,  0 ],
    ],
    // [1] EIGHTTAP_SMOOTH (av1-spec p.263 lines 14675-14692).
    [
        [  0,  0,   0, 128,   0,   0,  0,  0 ],
        [  0,  2,  28,  62,  34,   2,  0,  0 ],
        [  0,  0,  26,  62,  36,   4,  0,  0 ],
        [  0,  0,  22,  62,  40,   4,  0,  0 ],
        [  0,  0,  20,  60,  42,   6,  0,  0 ],
        [  0,  0,  18,  58,  44,   8,  0,  0 ],
        [  0,  0,  16,  56,  46,  10,  0,  0 ],
        [  0, -2,  16,  54,  48,  12,  0,  0 ],
        [  0, -2,  14,  52,  52,  14, -2,  0 ],
        [  0,  0,  12,  48,  54,  16, -2,  0 ],
        [  0,  0,  10,  46,  56,  16,  0,  0 ],
        [  0,  0,   8,  44,  58,  18,  0,  0 ],
        [  0,  0,   6,  42,  60,  20,  0,  0 ],
        [  0,  0,   4,  40,  62,  22,  0,  0 ],
        [  0,  0,   4,  36,  62,  26,  0,  0 ],
        [  0,  0,   2,  34,  62,  28,  2,  0 ],
    ],
    // [2] EIGHTTAP_SHARP (av1-spec p.264 lines 14693-14718).
    [
        [  0,  0,   0, 128,   0,   0,  0,  0 ],
        [ -2,  2,  -6, 126,   8,  -2,  2,  0 ],
        [ -2,  6, -12, 124,  16,  -6,  4, -2 ],
        [ -2,  8, -18, 120,  26, -10,  6, -2 ],
        [ -4, 10, -22, 116,  38, -14,  6, -2 ],
        [ -4, 10, -22, 108,  48, -18,  8, -2 ],
        [ -4, 10, -24, 100,  60, -20,  8, -2 ],
        [ -4, 10, -24,  90,  70, -22, 10, -2 ],
        [ -4, 12, -24,  80,  80, -24, 12, -4 ],
        [ -2, 10, -22,  70,  90, -24, 10, -4 ],
        [ -2,  8, -20,  60, 100, -24, 10, -4 ],
        [ -2,  8, -18,  48, 108, -22, 10, -4 ],
        [ -2,  6, -14,  38, 116, -22, 10, -4 ],
        [ -2,  6, -10,  26, 120, -18,  8, -2 ],
        [ -2,  4,  -6,  16, 124, -12,  6, -2 ],
        [  0,  2,  -2,   8, 126,  -6,  2, -2 ],
    ],
    // [3] BILINEAR (av1-spec p.264 lines 14719-14736).
    [
        [  0,  0,   0, 128,   0,   0,  0,  0 ],
        [  0,  0,   0, 120,   8,   0,  0,  0 ],
        [  0,  0,   0, 112,  16,   0,  0,  0 ],
        [  0,  0,   0, 104,  24,   0,  0,  0 ],
        [  0,  0,   0,  96,  32,   0,  0,  0 ],
        [  0,  0,   0,  88,  40,   0,  0,  0 ],
        [  0,  0,   0,  80,  48,   0,  0,  0 ],
        [  0,  0,   0,  72,  56,   0,  0,  0 ],
        [  0,  0,   0,  64,  64,   0,  0,  0 ],
        [  0,  0,   0,  56,  72,   0,  0,  0 ],
        [  0,  0,   0,  48,  80,   0,  0,  0 ],
        [  0,  0,   0,  40,  88,   0,  0,  0 ],
        [  0,  0,   0,  32,  96,   0,  0,  0 ],
        [  0,  0,   0,  24, 104,   0,  0,  0 ],
        [  0,  0,   0,  16, 112,   0,  0,  0 ],
        [  0,  0,   0,   8, 120,   0,  0,  0 ],
    ],
    // [4] EIGHTTAP_4TAP (small-block reduction of EIGHTTAP — av1-spec
    // p.264-265 lines 14737-14754).
    [
        [  0,  0,   0, 128,   0,   0,  0,  0 ],
        [  0,  0,  -4, 126,   8,  -2,  0,  0 ],
        [  0,  0,  -8, 122,  18,  -4,  0,  0 ],
        [  0,  0, -10, 116,  28,  -6,  0,  0 ],
        [  0,  0, -12, 110,  38,  -8,  0,  0 ],
        [  0,  0, -12, 102,  48, -10,  0,  0 ],
        [  0,  0, -14,  94,  58, -10,  0,  0 ],
        [  0,  0, -12,  84,  66, -10,  0,  0 ],
        [  0,  0, -12,  76,  76, -12,  0,  0 ],
        [  0,  0, -10,  66,  84, -12,  0,  0 ],
        [  0,  0, -10,  58,  94, -14,  0,  0 ],
        [  0,  0, -10,  48, 102, -12,  0,  0 ],
        [  0,  0,  -8,  38, 110, -12,  0,  0 ],
        [  0,  0,  -6,  28, 116, -10,  0,  0 ],
        [  0,  0,  -4,  18, 122,  -8,  0,  0 ],
        [  0,  0,  -2,   8, 126,  -4,  0,  0 ],
    ],
    // [5] EIGHTTAP_SMOOTH_4TAP (small-block reduction of
    // EIGHTTAP_SMOOTH — av1-spec p.265 lines 14755-14781).
    [
        [  0,  0,   0, 128,   0,   0,  0,  0 ],
        [  0,  0,  30,  62,  34,   2,  0,  0 ],
        [  0,  0,  26,  62,  36,   4,  0,  0 ],
        [  0,  0,  22,  62,  40,   4,  0,  0 ],
        [  0,  0,  20,  60,  42,   6,  0,  0 ],
        [  0,  0,  18,  58,  44,   8,  0,  0 ],
        [  0,  0,  16,  56,  46,  10,  0,  0 ],
        [  0,  0,  14,  54,  48,  12,  0,  0 ],
        [  0,  0,  12,  52,  52,  12,  0,  0 ],
        [  0,  0,  12,  48,  54,  14,  0,  0 ],
        [  0,  0,  10,  46,  56,  16,  0,  0 ],
        [  0,  0,   8,  44,  58,  18,  0,  0 ],
        [  0,  0,   6,  42,  60,  20,  0,  0 ],
        [  0,  0,   4,  40,  62,  22,  0,  0 ],
        [  0,  0,   4,  36,  62,  26,  0,  0 ],
        [  0,  0,   2,  34,  62,  30,  0,  0 ],
    ],
];

/// §7.11.3.4 small-block filter remap (av1-spec p.262 lines
/// 14602-14608 / 14626-14632). On the horizontal pass, applied when
/// `w <= 4`; on the vertical pass, applied when `h <= 4`. The
/// `EIGHTTAP_4TAP` / `EIGHTTAP_SMOOTH_4TAP` substitutions select the
/// 4-tap small-block rows of [`SUBPEL_FILTERS`].
///
/// `interp_filter` values outside `0..=3` pass through unchanged —
/// the spec's two-branch `if` reads as "leave the input alone
/// otherwise" (no `SWITCHABLE` value reaches §7.11.3.4 because the
/// §5.11 reader resolves switchable to a concrete filter).
#[inline]
pub fn select_interp_filter_small_block(interp_filter: u8) -> u8 {
    match interp_filter {
        EIGHTTAP | EIGHTTAP_SHARP => EIGHTTAP_4TAP,
        EIGHTTAP_SMOOTH => EIGHTTAP_SMOOTH_4TAP,
        // BILINEAR + the two 4-tap rows already (idx 4, 5) pass
        // through. Out-of-range values likewise pass through; the
        // §7.11.3.4 caller validates.
        other => other,
    }
}

// =====================================================================
// §7.11.3.4 — Block inter prediction process (av1-spec p.261-265).
// =====================================================================

/// §3 `Round2(x, n)` integer form per av1-spec.txt line 1583. Used by
/// the §7.11.3.4 horizontal / vertical convolutions.
#[inline]
fn round2(x: i64, n: u32) -> i64 {
    if n == 0 {
        x
    } else {
        (x + (1i64 << (n - 1))) >> n
    }
}

/// §3 `Clip3(low, high, v)` for `i32` operands (av1-spec p.18 line
/// 1568). Used by the §7.11.3.4 reference-sample fetch's `Clip3(0,
/// lastY, ...)` / `Clip3(0, lastX, ...)` boundary clamp.
#[inline]
fn clip3_i32(low: i32, high: i32, v: i32) -> i32 {
    if v < low {
        low
    } else if v > high {
        high
    } else {
        v
    }
}

/// §7.11.3.4 (av1-spec p.261-265): the translational 8-tap horizontal
/// + vertical convolution kernel.
///
/// On the spec's translational `useWarp == 0` branch (the only
/// branch landed in this round), the §7.11.3.1 driver calls this
/// helper with the §7.11.3.3 `(startX, startY, stepX, stepY)`
/// quadruple plus the per-ref-list `interp_filter[0..1]` and the
/// reference plane's sample grid.
///
/// The kernel walks the spec's two ordered steps:
///
/// * Step 1 — horizontal filter populates `intermediate[r][c]` for
///   `r in 0..intermediateHeight`, `c in 0..w` per av1-spec p.262
///   lines 14609-14619, with `Round2(s, InterRound0)`.
/// * Step 2 — vertical filter populates `pred[r][c]` for
///   `r in 0..h`, `c in 0..w` per av1-spec p.262 lines 14633-14642,
///   with `Round2(s, InterRound1)`.
///
/// The §7.11.3.1 outer `Clip1( pred[0][i][j] )` is *not* applied
/// here: the spec applies it at the call site after the
/// compound-blend decision (see av1-spec p.258 lines 14402-14412).
/// Callers wanting the single-ref final samples can apply
/// `Clip1(pred[i][j])` (= `Clip3(0, (1 << BitDepth) - 1, pred[i][j])`)
/// to the returned buffer; see [`clip1_single_ref`].
///
/// ## Arguments
///
/// * `plane` — `0` (luma), `1` (Cb), `2` (Cr). Determines subsampling
///   for the `(lastX, lastY)` clamp.
/// * `subsampling_x` / `subsampling_y` — §5.5.2 chroma subsampling.
/// * `ref_plane` — the reference-frame sample grid for the chosen
///   `plane`, row-major (`ref_plane[ry * ref_stride + rx]`). Length
///   must be `>= ref_height * ref_stride`.
/// * `ref_stride` — row stride in samples of `ref_plane`. Must be
///   `>= ref_width`.
/// * `ref_width` / `ref_height` — reference plane's per-plane width
///   / height in samples. `(lastX, lastY) = (ref_width - 1,
///   ref_height - 1)` per av1-spec p.262 lines 14575-14577.
/// * `start_x` / `start_y` / `step_x` / `step_y` — §7.11.3.3 outputs.
/// * `w` / `h` — prediction region width / height in current-frame
///   samples.
/// * `interp_filter_x` — `InterpFilters[candRow][candCol][1]` per
///   av1-spec p.262 line 14601 (horizontal pass).
/// * `interp_filter_y` — `InterpFilters[candRow][candCol][0]` per
///   av1-spec p.262 line 14625 (vertical pass).
/// * `inter_round0` — `RoundingVars::inter_round0` (horizontal shift).
/// * `inter_round1` — `RoundingVars::inter_round1` (vertical shift).
/// * `pred` — output buffer, row-major; length must be `>= h * w`.
///   The helper writes `pred[r * w + c]` for `r in 0..h`, `c in 0..w`.
///
/// ## Returns
///
/// `Ok(())` on success. Returns
/// [`crate::Error::PartitionWalkOutOfRange`] for caller-bug arguments:
/// `plane > 2`, `subsampling_{x,y} > 1`, `ref_width == 0`,
/// `ref_height == 0`, `ref_stride < ref_width`, `ref_plane.len() <
/// ref_height * ref_stride`, `w == 0`, `h == 0`, `w > 256`, `h > 256`,
/// `step_x <= 0`, `step_y <= 0`, `interp_filter_x >= 6`,
/// `interp_filter_y >= 6`, `pred.len() < h * w`.
#[allow(clippy::too_many_arguments)]
pub fn block_inter_prediction(
    plane: u8,
    subsampling_x: u8,
    subsampling_y: u8,
    ref_plane: &[u16],
    ref_stride: usize,
    ref_width: u32,
    ref_height: u32,
    start_x: i32,
    start_y: i32,
    step_x: i32,
    step_y: i32,
    w: usize,
    h: usize,
    interp_filter_x: u8,
    interp_filter_y: u8,
    inter_round0: u32,
    inter_round1: u32,
    pred: &mut [i32],
) -> Result<(), crate::Error> {
    // ---------- caller-bug guards ----------
    if plane > 2 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if subsampling_x > 1 || subsampling_y > 1 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if ref_width == 0 || ref_height == 0 || ref_stride < ref_width as usize {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if ref_plane.len() < (ref_height as usize) * ref_stride {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if w == 0 || h == 0 || w > 256 || h > 256 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if step_x <= 0 || step_y <= 0 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    let n_filters = SUBPEL_FILTERS.len() as u8;
    if interp_filter_x >= n_filters || interp_filter_y >= n_filters {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if pred.len() < h * w {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }

    // §7.11.3.4 (av1-spec p.262 lines 14575-14577).
    //
    //   lastX = ((RefUpscaledWidth + subX) >> subX) - 1
    //   lastY = ((RefFrameHeight   + subY) >> subY) - 1
    //
    // The caller passes `ref_width` / `ref_height` already
    // sub-sampled to the per-plane grid (i.e. the per-plane
    // RefUpscaledWidth >> subX value), so the formula degenerates to
    // `last{X,Y} = ref_{width,height} - 1`. The (subX, subY)
    // arguments are retained for the spec-level argument shape; the
    // sub-sampling factor itself is already folded into ref_width /
    // ref_height by the caller per §7.11.3.4 lines 14569-14577.
    let _ = (subsampling_x, subsampling_y);
    let last_x: i32 = (ref_width as i32) - 1;
    let last_y: i32 = (ref_height as i32) - 1;

    // §7.11.3.4 (av1-spec p.262 line 14581).
    //
    //   intermediateHeight = (((h-1) * yStep + (1 << SCALE_SUBPEL_BITS) - 1)
    //                          >> SCALE_SUBPEL_BITS) + 8
    let intermediate_height: usize =
        ((((h as i64) - 1) * (step_y as i64) + (1i64 << SCALE_SUBPEL_BITS) - 1)
            >> SCALE_SUBPEL_BITS) as usize
            + 8;

    // §7.11.3.4 horizontal filter small-block remap (lines 14602-14608).
    let h_filter = if w <= 4 {
        select_interp_filter_small_block(interp_filter_x) as usize
    } else {
        interp_filter_x as usize
    };

    // §7.11.3.4 vertical filter small-block remap (lines 14626-14632).
    let v_filter = if h <= 4 {
        select_interp_filter_small_block(interp_filter_y) as usize
    } else {
        interp_filter_y as usize
    };

    // §7.11.3.4 intermediate buffer. The post-Round2 value fits in
    // ~14 bits for 8-bit content (sum of |coef| at one phase is at
    // most 384, times max sample 255 = 97920, Round2 by 3 = 12240
    // — well within i32). We use i32 for headroom across bit depths.
    let mut intermediate: Vec<i32> = vec![0i32; intermediate_height * w];

    // ---------- §7.11.3.4 step 1: horizontal filter ----------
    //
    //   for r in 0..intermediateHeight:
    //     for c in 0..w:
    //       s = 0
    //       p = x + xStep * c
    //       for t in 0..8:
    //         s += Subpel_Filters[hFilter][(p >> 6) & SUBPEL_MASK][t]
    //              * ref[Clip3(0, lastY, (y >> 10) + r - 3)]
    //                   [Clip3(0, lastX, (p >> 10) + t - 3)]
    //       intermediate[r][c] = Round2(s, InterRound0)
    //
    // (av1-spec p.262 lines 14609-14619.)
    let y_base = (start_y >> SCALE_SUBPEL_BITS) - 3;
    for r in 0..intermediate_height {
        let ry = clip3_i32(0, last_y, y_base + (r as i32)) as usize;
        let row_base = ry * ref_stride;
        for c in 0..w {
            let p: i32 = start_x + step_x * (c as i32);
            // §7.11.3.4 phase index `(p >> 6) & SUBPEL_MASK` —
            // `>> 6` reduces SCALE_SUBPEL_BITS=10 to SUBPEL_BITS=4
            // (10 - 4 = 6) and the mask isolates the 4-bit phase.
            let phase = ((p >> (SCALE_SUBPEL_BITS - SUBPEL_BITS)) as u32 & SUBPEL_MASK) as usize;
            let p_int = p >> SCALE_SUBPEL_BITS;
            let coeffs = &SUBPEL_FILTERS[h_filter][phase];
            let mut s: i64 = 0;
            for (t, coef) in coeffs.iter().enumerate() {
                let rx = clip3_i32(0, last_x, p_int + (t as i32) - 3) as usize;
                let sample = ref_plane[row_base + rx] as i64;
                s += (*coef as i64) * sample;
            }
            intermediate[r * w + c] = round2(s, inter_round0) as i32;
        }
    }

    // ---------- §7.11.3.4 step 2: vertical filter ----------
    //
    //   for r in 0..h:
    //     for c in 0..w:
    //       s = 0
    //       p = (y & 1023) + yStep * r
    //       for t in 0..8:
    //         s += Subpel_Filters[vFilter][(p >> 6) & SUBPEL_MASK][t]
    //              * intermediate[(p >> 10) + t][c]
    //       pred[r][c] = Round2(s, InterRound1)
    //
    // (av1-spec p.262 lines 14633-14642.)
    let y_frac_mask: i32 = (1i32 << SCALE_SUBPEL_BITS) - 1;
    for r in 0..h {
        let p: i32 = (start_y & y_frac_mask) + step_y * (r as i32);
        let phase = ((p >> (SCALE_SUBPEL_BITS - SUBPEL_BITS)) as u32 & SUBPEL_MASK) as usize;
        let p_int = (p >> SCALE_SUBPEL_BITS) as usize;
        let coeffs = &SUBPEL_FILTERS[v_filter][phase];
        for c in 0..w {
            let mut s: i64 = 0;
            for (t, coef) in coeffs.iter().enumerate() {
                let interp_row = p_int + t;
                // The intermediate-height derivation above guarantees
                // `p_int + t < intermediate_height` for every legal
                // (r, t) — `(h-1) * yStep < intermediateHeight - 7`
                // after applying the `+8` margin from line 14582.
                let v = intermediate[interp_row * w + c] as i64;
                s += (*coef as i64) * v;
            }
            pred[r * w + c] = round2(s, inter_round1) as i32;
        }
    }

    Ok(())
}

/// §7.11.3.1 single-ref post-prediction `Clip1` (av1-spec p.258 line
/// 14402): on the non-compound, non-interintra path, the spec writes
/// `CurrFrame[plane][y+i][x+j] = Clip1( preds[0][i][j] )`. This helper
/// applies that clip to the [`block_inter_prediction`] output for
/// callers who want sample-domain predictions.
///
/// `bit_depth` must be one of `{8, 10, 12}`; pred values are clipped
/// in-place to `[0, (1 << bit_depth) - 1]`.
pub fn clip1_single_ref(bit_depth: u8, pred: &[i32], out: &mut [u16]) -> Result<(), crate::Error> {
    if !matches!(bit_depth, 8 | 10 | 12) {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if out.len() < pred.len() {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    let max = (1i32 << bit_depth) - 1;
    for (dst, src) in out.iter_mut().zip(pred.iter().copied()) {
        *dst = clip3_i32(0, max, src) as u16;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- §7.11.3.2 rounding_variables ----------

    /// §7.11.3.2 spec table — `(bit_depth, is_compound) → (R0, R1,
    /// InterPostRound)`. Hand-derived from the av1-spec p.259 body:
    /// * `(8, false)  → (3, 11, 0)`  — `2*7 - (3 + 11) = 0`.
    /// * `(8, true)   → (3,  7, 4)`  — `2*7 - (3 +  7) = 4`.
    /// * `(10, false) → (3, 11, 0)`  — same as 8-bit, no BD-12 adjust.
    /// * `(10, true)  → (3,  7, 4)`.
    /// * `(12, false) → (5,  9, 0)`  — `R0 += 2`, `R1 -= 2`; `2*7 -
    ///   (5 + 9) = 0`.
    /// * `(12, true)  → (5,  7, 2)`  — `R0 += 2` only; `2*7 - (5 + 7) = 2`.
    #[test]
    fn rounding_variables_matches_spec_table() {
        let cases: &[(u8, bool, u32, u32, u32)] = &[
            (8, false, 3, 11, 0),
            (8, true, 3, 7, 4),
            (10, false, 3, 11, 0),
            (10, true, 3, 7, 4),
            (12, false, 5, 9, 0),
            (12, true, 5, 7, 2),
        ];
        for &(bd, ic, r0, r1, pr) in cases {
            let got = rounding_variables(bd, ic).unwrap();
            assert_eq!(
                (got.inter_round0, got.inter_round1, got.inter_post_round),
                (r0, r1, pr),
                "BD={bd} compound={ic}",
            );
        }
    }

    /// §7.11.3.2: `bit_depth` outside `{8, 10, 12}` is a caller bug.
    #[test]
    fn rounding_variables_rejects_invalid_bit_depth() {
        assert!(rounding_variables(7, false).is_err());
        assert!(rounding_variables(9, false).is_err());
        assert!(rounding_variables(16, false).is_err());
    }

    // ---------- §7.11.3.3 motion_vector_scaling ----------

    /// §7.11.3.3 identity case: `RefUpscaledWidth == FrameWidth`,
    /// `RefFrameHeight == FrameHeight`, `plane == 0`, zero MV — the
    /// `(x, y)` input should map to itself in 10-bit fractional units:
    /// `startX = (x << 10)`, `startY = (y << 10)`, `stepX = stepY = 1024`.
    ///
    /// Derivation:
    /// * `xScale = (W << 14 + W/2) / W = 1 << 14 = 16384` (the rounded
    ///   division is exact for our inputs).
    /// * `halfSample = 8` (1/16 sample).
    /// * `origX = (x << 4) + 0 + 8 = x*16 + 8`.
    /// * `baseX = (x*16 + 8) * 16384 - 8 * 16384 = x * 16 * 16384 = x << 18`.
    /// * `Round2Signed(baseX, 8) = x << 10`.
    /// * `off = (1 << 6) / 2 = 32`.
    /// * `startX = (x << 10) + 32`. Same for Y.
    /// * `stepX = Round2Signed(16384, 4) = 1024`.
    #[test]
    fn motion_vector_scaling_identity_yields_unit_scale() {
        let s = motion_vector_scaling(
            /*plane*/ 0,
            /*ssx*/ 0,
            /*ssy*/ 0,
            /*fw*/ 640,
            /*fh*/ 360,
            /*ruw*/ 640,
            /*rfh*/ 360,
            /*x*/ 16,
            /*y*/ 8,
            /*mv*/ [0, 0],
        )
        .unwrap();
        assert_eq!(s.step_x, 1024);
        assert_eq!(s.step_y, 1024);
        assert_eq!(s.start_x, (16 << 10) + 32);
        assert_eq!(s.start_y, (8 << 10) + 32);
    }

    /// §7.11.3.3 with a non-zero integer-sample MV: a `(+8, 0)` row
    /// MV (= 1 full luma sample down in 1/8-sample units; 2 * 8 = 16,
    /// which adds 16 sixteenths = one full sample) should shift the
    /// reference Y coordinate by `+1 << 10`.
    #[test]
    fn motion_vector_scaling_integer_mv_shifts_origin() {
        let base = motion_vector_scaling(0, 0, 0, 640, 360, 640, 360, 0, 0, [0, 0]).unwrap();
        let shifted = motion_vector_scaling(0, 0, 0, 640, 360, 640, 360, 0, 0, [8, 0]).unwrap();
        assert_eq!(shifted.start_y - base.start_y, 1 << SCALE_SUBPEL_BITS);
        assert_eq!(shifted.start_x, base.start_x);
    }

    /// §7.11.3.3 rejects caller-bug arguments.
    #[test]
    fn motion_vector_scaling_rejects_invalid_inputs() {
        assert!(motion_vector_scaling(3, 0, 0, 8, 8, 8, 8, 0, 0, [0, 0]).is_err());
        assert!(motion_vector_scaling(0, 2, 0, 8, 8, 8, 8, 0, 0, [0, 0]).is_err());
        assert!(motion_vector_scaling(0, 0, 2, 8, 8, 8, 8, 0, 0, [0, 0]).is_err());
        assert!(motion_vector_scaling(0, 0, 0, 0, 8, 8, 8, 0, 0, [0, 0]).is_err());
        assert!(motion_vector_scaling(0, 0, 0, 8, 0, 8, 8, 0, 0, [0, 0]).is_err());
        assert!(motion_vector_scaling(0, 0, 0, 8, 8, 0, 8, 0, 0, [0, 0]).is_err());
        assert!(motion_vector_scaling(0, 0, 0, 8, 8, 8, 0, 0, 0, [0, 0]).is_err());
    }

    // ---------- §7.11.3.4 SUBPEL_FILTERS table shape ----------

    /// av1-spec p.265 line 14786: every Subpel_Filters value is even.
    /// All 6 × 16 × 8 = 768 entries satisfy this.
    #[test]
    fn subpel_filters_all_even() {
        for (f, phases) in SUBPEL_FILTERS.iter().enumerate() {
            for (p, row) in phases.iter().enumerate() {
                for (t, &v) in row.iter().enumerate() {
                    assert_eq!(v & 1, 0, "filter {f} phase {p} tap {t} = {v} (odd)");
                }
            }
        }
    }

    /// av1-spec p.262: the phase-0 row of every filter is the unit
    /// row `(0, 0, 0, 128, 0, 0, 0, 0)` so an integer-aligned tap
    /// reproduces the reference sample exactly. (`128 = 1 <<
    /// FILTER_BITS`.)
    #[test]
    fn subpel_filters_phase_zero_is_unit_copy() {
        let unit = [0, 0, 0, 128, 0, 0, 0, 0];
        for (f, phases) in SUBPEL_FILTERS.iter().enumerate() {
            assert_eq!(phases[0], unit, "filter {f} phase 0 not unit");
        }
    }

    /// av1-spec p.265 note (line 14786): the §7.11.3.4 8-tap
    /// coefficients sum to `1 << FILTER_BITS = 128` at every phase
    /// — sample-conservation property. Verified for all 96 phase
    /// rows across all six filter banks.
    #[test]
    fn subpel_filters_taps_sum_to_filter_bits_unity() {
        for (f, phases) in SUBPEL_FILTERS.iter().enumerate() {
            for (p, row) in phases.iter().enumerate() {
                let sum: i32 = row.iter().sum();
                assert_eq!(sum, 128, "filter {f} phase {p} taps sum = {sum} != 128");
            }
        }
    }

    /// av1-spec p.264 lines 14737-14781: filters 4 and 5 (small-block
    /// reductions) zero out taps 0, 1, 6, 7 at every phase — the
    /// 4-tap structure is taps 2..6.
    #[test]
    fn small_block_filters_have_4tap_structure() {
        for f in [EIGHTTAP_4TAP as usize, EIGHTTAP_SMOOTH_4TAP as usize] {
            for (p, row) in SUBPEL_FILTERS[f].iter().enumerate() {
                for &t in &[0usize, 1, 6, 7] {
                    assert_eq!(
                        row[t], 0,
                        "filter {f} phase {p} tap {t} = {} (must be 0)",
                        row[t],
                    );
                }
            }
        }
    }

    /// §7.11.3.4 small-block remap per av1-spec p.262.
    #[test]
    fn select_interp_filter_small_block_remap() {
        assert_eq!(select_interp_filter_small_block(EIGHTTAP), EIGHTTAP_4TAP);
        assert_eq!(
            select_interp_filter_small_block(EIGHTTAP_SHARP),
            EIGHTTAP_4TAP,
        );
        assert_eq!(
            select_interp_filter_small_block(EIGHTTAP_SMOOTH),
            EIGHTTAP_SMOOTH_4TAP,
        );
        assert_eq!(select_interp_filter_small_block(BILINEAR), BILINEAR);
        assert_eq!(
            select_interp_filter_small_block(EIGHTTAP_4TAP),
            EIGHTTAP_4TAP,
        );
    }

    // ---------- §7.11.3.4 block_inter_prediction ----------

    /// §7.11.3.4 identity case: zero motion vector with an integer-
    /// aligned start position should reproduce a copy of the
    /// corresponding reference-frame region.
    ///
    /// `start_y = 4 << 10 = 4096` (4 luma samples down, integer-
    /// aligned — phase 0), same for X. The kernel's phase-0 unit
    /// copy means horizontal & vertical filters are both pass-
    /// throughs; the output `Round2(128 * sample, 11)` reduces to
    /// `Round2((128 * 128 * sample), 14) = sample` for 8-bit content
    /// (the `128` factors and the `2^14` rounding shift cancel).
    #[test]
    fn block_inter_prediction_zero_mv_copies_reference() {
        let ref_w: usize = 16;
        let ref_h: usize = 16;
        let stride = ref_w;
        // Build a reference plane with sample[r][c] = (r * 16 + c).
        let mut refp = vec![0u16; ref_h * stride];
        for r in 0..ref_h {
            for c in 0..ref_w {
                refp[r * stride + c] = (r * 16 + c) as u16;
            }
        }

        // 4x4 region starting at (2, 2) in plane coords, zero MV.
        let s = motion_vector_scaling(
            0,
            0,
            0,
            ref_w as u32,
            ref_h as u32,
            ref_w as u32,
            ref_h as u32,
            2,
            2,
            [0, 0],
        )
        .unwrap();
        let rv = rounding_variables(8, false).unwrap();
        let w = 4;
        let h = 4;
        let mut pred = vec![0i32; h * w];
        block_inter_prediction(
            0,
            0,
            0,
            &refp,
            stride,
            ref_w as u32,
            ref_h as u32,
            s.start_x,
            s.start_y,
            s.step_x,
            s.step_y,
            w,
            h,
            EIGHTTAP,
            EIGHTTAP,
            rv.inter_round0,
            rv.inter_round1,
            &mut pred,
        )
        .unwrap();

        // The `+ off` term (`off = 32`) lives entirely in the
        // fractional bits (10) and rounds out — the phase index
        // `(start >> 6) & 15` of `((2 << 10) + 32) >> 6 = 33 & 15 =
        // 1` instead of 0, which violates the integer-aligned
        // assumption. So apply the spec-level integer-pixel test
        // differently: assert the copy holds at the *centre* of the
        // half-sample grid — i.e. compare against the same kernel
        // applied to a different MV. Instead, use the unit-row
        // property: a deliberate `phase == 0` setup needs `start_x
        // % 1024 == 0` AND `off == 0`. The §7.11.3.3 `off` is a
        // fixed bias so we cannot get phase 0 from MV scaling alone.
        //
        // We therefore validate via the unit-row direct injection
        // below — the phase-0 row of every filter is `(0, 0, 0,
        // 128, 0, 0, 0, 0)`, so applying the kernel with
        // hand-crafted `start_x` / `start_y` that align to integer
        // sample positions and produce phase 0 yields a copy.
        let _ = pred; // discard the off-biased output

        // Hand-crafted integer-aligned start (phase 0 horizontal,
        // phase 0 vertical).
        let mut pred2 = vec![0i32; h * w];
        block_inter_prediction(
            0,
            0,
            0,
            &refp,
            stride,
            ref_w as u32,
            ref_h as u32,
            /*start_x*/ 2 << 10,
            /*start_y*/ 2 << 10,
            /*step_x*/ 1 << 10,
            /*step_y*/ 1 << 10,
            w,
            h,
            EIGHTTAP,
            EIGHTTAP,
            rv.inter_round0,
            rv.inter_round1,
            &mut pred2,
        )
        .unwrap();

        // With both filters at phase 0, the horizontal pass writes
        // `Round2(128 * ref, 3) = 16 * ref`, then vertical writes
        // `Round2(128 * 16 * ref, 11) = ref` (Round2(2048*ref, 11) =
        // ref).
        for r in 0..h {
            for c in 0..w {
                let expected = refp[(2 + r) * stride + (2 + c)] as i32;
                assert_eq!(
                    pred2[r * w + c],
                    expected,
                    "pred2[{r}][{c}] = {} expected {}",
                    pred2[r * w + c],
                    expected,
                );
            }
        }
    }

    /// §7.11.3.4 boundary-clamp: a `start_x = -4 << 10` puts the first
    /// tap at `Clip3(0, lastX, -7)` = `0`, so the kernel clamps to
    /// the left edge. The output for the left-edge column of a
    /// constant-row reference should match the constant.
    #[test]
    fn block_inter_prediction_clamps_at_negative_start() {
        let ref_w: usize = 8;
        let ref_h: usize = 8;
        let stride = ref_w;
        let mut refp = vec![0u16; ref_h * stride];
        for r in 0..ref_h {
            for c in 0..ref_w {
                refp[r * stride + c] = 42; // constant
            }
        }
        let rv = rounding_variables(8, false).unwrap();
        let w = 4;
        let h = 4;
        let mut pred = vec![0i32; h * w];
        // Use the small-block 4-tap filter for w=4.
        block_inter_prediction(
            0,
            0,
            0,
            &refp,
            stride,
            ref_w as u32,
            ref_h as u32,
            /*start_x*/ -4i32 << SCALE_SUBPEL_BITS,
            /*start_y*/ 0,
            /*step_x*/ 1 << SCALE_SUBPEL_BITS,
            /*step_y*/ 1 << SCALE_SUBPEL_BITS,
            w,
            h,
            EIGHTTAP,
            EIGHTTAP,
            rv.inter_round0,
            rv.inter_round1,
            &mut pred,
        )
        .unwrap();
        // Output of a constant reference under any phase / clamp =
        // the constant (sample-conservation property — taps sum to
        // 128, Round2 by R0=3 → 128*42/8 = 672, then 128*672/2048 =
        // 42).
        for r in 0..h {
            for c in 0..w {
                assert_eq!(pred[r * w + c], 42);
            }
        }
    }

    /// §7.11.3.4 caller-bug rejection.
    #[test]
    fn block_inter_prediction_rejects_invalid_inputs() {
        let refp = vec![0u16; 64];
        let mut pred = vec![0i32; 16];
        // plane > 2
        assert!(block_inter_prediction(
            3, 0, 0, &refp, 8, 8, 8, 0, 0, 1024, 1024, 4, 4, 0, 0, 3, 11, &mut pred
        )
        .is_err());
        // subsampling_x > 1
        assert!(block_inter_prediction(
            0, 2, 0, &refp, 8, 8, 8, 0, 0, 1024, 1024, 4, 4, 0, 0, 3, 11, &mut pred
        )
        .is_err());
        // ref_width == 0
        assert!(block_inter_prediction(
            0, 0, 0, &refp, 8, 0, 8, 0, 0, 1024, 1024, 4, 4, 0, 0, 3, 11, &mut pred
        )
        .is_err());
        // step_x == 0
        assert!(block_inter_prediction(
            0, 0, 0, &refp, 8, 8, 8, 0, 0, 0, 1024, 4, 4, 0, 0, 3, 11, &mut pred
        )
        .is_err());
        // interp_filter_x out of range
        assert!(block_inter_prediction(
            0, 0, 0, &refp, 8, 8, 8, 0, 0, 1024, 1024, 4, 4, 6, 0, 3, 11, &mut pred
        )
        .is_err());
        // pred too small
        let mut tiny = vec![0i32; 8];
        assert!(block_inter_prediction(
            0, 0, 0, &refp, 8, 8, 8, 0, 0, 1024, 1024, 4, 4, 0, 0, 3, 11, &mut tiny
        )
        .is_err());
    }

    /// `clip1_single_ref` bounds.
    #[test]
    fn clip1_single_ref_clamps_to_bit_depth() {
        let pred = [-3i32, 0, 255, 256, 1023, 4096, -1];
        let mut out8 = [0u16; 7];
        clip1_single_ref(8, &pred, &mut out8).unwrap();
        assert_eq!(out8, [0, 0, 255, 255, 255, 255, 0]);
        let mut out10 = [0u16; 7];
        clip1_single_ref(10, &pred, &mut out10).unwrap();
        assert_eq!(out10, [0, 0, 255, 256, 1023, 1023, 0]);
        let mut out12 = [0u16; 7];
        clip1_single_ref(12, &pred, &mut out12).unwrap();
        assert_eq!(out12, [0, 0, 255, 256, 1023, 4095, 0]);
        // Invalid bit depth.
        let mut out = [0u16; 7];
        assert!(clip1_single_ref(9, &pred, &mut out).is_err());
    }
}
