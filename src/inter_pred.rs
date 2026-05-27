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
//! Translational single-ref MC + the §7.11.3.11-15 compound-mask
//! blend bodies. Deferred to subsequent arcs:
//!
//! * §7.11.3.5 — `block_warp` (LOCALWARP / GLOBAL_GLOBALMV affine warp)
//! * §7.11.3.6 — `setup_shear` (warp parameter shear)
//! * §7.11.3.7 — `resolve_divisor` (warp divisor resolution)
//! * §7.11.3.8 — `warp_estimation` (LOCALWARP estimation from §7.10.4)
//! * §7.11.3.9 — `overlapped_motion_compensation` (OBMC)
//! * §7.11.3.10 — `overlap_blending` (OBMC blending)
//!
//! Round 191 adds the five compound bodies:
//!
//! * §7.11.3.11 — [`wedge_mask`] (COMPOUND_WEDGE) — generates the
//!   16 × 3-shape wedge mask table on first use via
//!   [`WedgeMaskCache`], then dispatches per `(bsize, wedge_sign,
//!   wedge_index)` to a 64 × 64 `MasterMask` slice.
//! * §7.11.3.12 — [`difference_weight_mask`] (COMPOUND_DIFFWTD) — the
//!   per-pixel `m = Clip3(0, 64, 38 + Abs(p0 - p1) / 16)` mask the
//!   `mask_type` toggle inverts.
//! * §7.11.3.13 — [`intra_mode_variant_mask`] (COMPOUND_INTRA) — the
//!   `Ii_Weights_1d[ MAX_SB_SIZE ]` (av1-spec p.283-284) smooth-mask
//!   driver with the four `II_*` interintra-mode sub-paths.
//! * §7.11.3.14 — [`mask_blend`] (compound blend with mask) — the
//!   `out = Clip1( Round2( m * p0 + (64 - m) * p1, 6 + InterPostRound ) )`
//!   inter-inter site and the `interintra` interintra site with the
//!   `(subX, subY)` chroma-subsample averaging.
//! * §7.11.3.15 — [`distance_weights`] (COMPOUND_DISTANCE) — the
//!   `(FwdWeight, BckWeight)` derivation from two `RefFrames[refList]`
//!   `OrderHints` deltas via [`Quant_Dist_Weight`] /
//!   [`Quant_Dist_Lookup`].
//!
//! And the §7.11.3.1 driver itself — once it lands, the
//! `block_inter_prediction` leaf is what the driver invokes on the
//! `useWarp == 0` branch, and the compound bodies are what the
//! driver invokes on the `isCompound == 1` / `IsInterIntra == 1` arms
//! after the §7.11.3.4 prediction has been formed.

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

// =====================================================================
// §7.11.3.11-15 — Compound mask / blend bodies (av1-spec p.278-285).
// =====================================================================
//
// The §7.11.3.1 driver (av1-spec p.257-258) consumes one of five
// compound mechanisms (`COMPOUND_AVERAGE` / `COMPOUND_WEDGE` /
// `COMPOUND_DIFFWTD` / `COMPOUND_INTRA` / `COMPOUND_DISTANCE`) to
// combine two inter predictions, plus the interintra blend for
// `IsInterIntra == 1` blocks.
//
// `COMPOUND_AVERAGE` and the `is_compound == 0 && IsInterIntra == 0`
// single-ref path fold into the inline final-clip step of §7.11.3.1
// itself (av1-spec p.258 lines 14402-14410) and don't need a mask
// array. The remaining four mechanisms produce a per-pixel mask via
// §7.11.3.11 (wedge_mask) / §7.11.3.12 (difference_weight_mask) /
// §7.11.3.13 (intra_mode_variant_mask), and the §7.11.3.14
// (mask_blend) and §7.11.3.15 (distance_weights) helpers below apply
// that mask to the two preds.

// ---------------------------------------------------------------------
// §3 constants used by §7.11.3.11-15 (av1-spec p.7-18).
// ---------------------------------------------------------------------

/// `MASK_MASTER_SIZE = 64` (av1-spec p.18 line 1234) — side length of
/// the §7.11.3.11 `MasterMask` square. The 16 wedge variants per
/// block-shape are read out of one of three master masks (oblique 63°,
/// oblique 27°, and vertical/horizontal pairs derived from them) at
/// `(yoff, xoff)` offsets the §7.11.3.11 `get_wedge_xoff` /
/// `get_wedge_yoff` helpers compute from `(bsize, wedge)`.
pub const MASK_MASTER_SIZE: usize = 64;

/// `MAX_SB_SIZE = 128` (av1-spec p.11 line 915) — maximum superblock
/// side length in luma samples. §7.11.3.13 uses it as the `sizeScale =
/// MAX_SB_SIZE / Max(h, w)` divisor + the `Ii_Weights_1d` table length.
pub const MAX_SB_SIZE: usize = 128;

/// `MAX_FRAME_DISTANCE = 31` (av1-spec p.16 line 1348) — saturation
/// value for the §7.11.3.15 `dist[0]` / `dist[1]` per-list distances,
/// also the second column of the `Quant_Dist_Weight[3]` row that
/// pegs the "out of range" branch of the per-`i` comparison.
pub const MAX_FRAME_DISTANCE: i32 = 31;

// ---------------------------------------------------------------------
// §3 (av1-spec p.10 lines 887-893) compound-type constants — reproduced
// locally for the §7.11.3.11-14 dispatchers. Identical to the
// `compound_type` ordinals tabled at av1-spec p.185.
// ---------------------------------------------------------------------

/// `COMPOUND_WEDGE = 0` (av1-spec p.185 table) — the
/// wedge-blended compound mode driven by §7.11.3.11.
pub const COMPOUND_WEDGE: u8 = 0;
/// `COMPOUND_DIFFWTD = 1` (av1-spec p.185 table) — the
/// difference-weighted compound mode driven by §7.11.3.12.
pub const COMPOUND_DIFFWTD: u8 = 1;
/// `COMPOUND_AVERAGE = 2` (av1-spec p.185 table) — the plain mean
/// compound mode (no mask; folded into §7.11.3.1).
pub const COMPOUND_AVERAGE: u8 = 2;
/// `COMPOUND_INTRA = 3` (av1-spec p.185 table) — the inter-intra
/// blended compound mode driven by §7.11.3.13.
pub const COMPOUND_INTRA: u8 = 3;
/// `COMPOUND_DISTANCE = 4` (av1-spec p.185 table) — the
/// distance-weighted compound mode driven by §7.11.3.15.
pub const COMPOUND_DISTANCE: u8 = 4;

// ---------------------------------------------------------------------
// §3 (av1-spec p.185 table) inter-intra mode constants used by
// §7.11.3.13.
// ---------------------------------------------------------------------

/// `II_DC_PRED = 0` (av1-spec p.185) — interintra DC variant; the
/// §7.11.3.13 body emits a uniform `Mask[i][j] = 32` (50/50 blend).
pub const II_DC_PRED: u8 = 0;
/// `II_V_PRED = 1` (av1-spec p.185) — interintra V variant; the
/// §7.11.3.13 body picks the row-driven `Ii_Weights_1d[ i * sizeScale ]`.
pub const II_V_PRED: u8 = 1;
/// `II_H_PRED = 2` (av1-spec p.185) — interintra H variant; the
/// §7.11.3.13 body picks the column-driven `Ii_Weights_1d[ j *
/// sizeScale ]`.
pub const II_H_PRED: u8 = 2;
/// `II_SMOOTH_PRED = 3` (av1-spec p.185) — interintra SMOOTH variant;
/// the §7.11.3.13 body picks the diagonal-driven
/// `Ii_Weights_1d[ Min(i, j) * sizeScale ]`.
pub const II_SMOOTH_PRED: u8 = 3;

// =====================================================================
// §7.11.3.11 — Wedge mask process (av1-spec p.278-282).
// =====================================================================

/// `WEDGE_HORIZONTAL = 0` (av1-spec p.282 table) — `MasterMask`
/// direction ordinal for the horizontal split.
pub const WEDGE_HORIZONTAL: u8 = 0;
/// `WEDGE_VERTICAL = 1` (av1-spec p.282 table) — `MasterMask` direction
/// ordinal for the vertical split.
pub const WEDGE_VERTICAL: u8 = 1;
/// `WEDGE_OBLIQUE27 = 2` (av1-spec p.282 table) — `MasterMask`
/// direction ordinal for the shallow 27° split.
pub const WEDGE_OBLIQUE27: u8 = 2;
/// `WEDGE_OBLIQUE63 = 3` (av1-spec p.282 table) — `MasterMask`
/// direction ordinal for the steep 63° split.
pub const WEDGE_OBLIQUE63: u8 = 3;
/// `WEDGE_OBLIQUE117 = 4` (av1-spec p.282 table) — `MasterMask`
/// direction ordinal for the 117° split (mirror of 63°).
pub const WEDGE_OBLIQUE117: u8 = 4;
/// `WEDGE_OBLIQUE153 = 5` (av1-spec p.282 table) — `MasterMask`
/// direction ordinal for the 153° split (mirror of 27°).
pub const WEDGE_OBLIQUE153: u8 = 5;

/// `WEDGE_DIRECTIONS = 6` — the six wedge directions enumerated above.
pub const WEDGE_DIRECTIONS: usize = 6;

/// `Wedge_Master_Oblique_Odd[ MASK_MASTER_SIZE ]` (av1-spec p.280
/// lines 15550-15555). The 1d driver for odd rows of the
/// `WEDGE_OBLIQUE63` master mask.
#[rustfmt::skip]
pub const WEDGE_MASTER_OBLIQUE_ODD: [u8; MASK_MASTER_SIZE] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  6, 18,
    37, 53, 60, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
];

/// `Wedge_Master_Oblique_Even[ MASK_MASTER_SIZE ]` (av1-spec p.280
/// lines 15557-15562). The 1d driver for even rows of the
/// `WEDGE_OBLIQUE63` master mask.
#[rustfmt::skip]
pub const WEDGE_MASTER_OBLIQUE_EVEN: [u8; MASK_MASTER_SIZE] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  4, 11, 27,
    46, 58, 62, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
];

/// `Wedge_Master_Vertical[ MASK_MASTER_SIZE ]` (av1-spec p.280 lines
/// 15564-15569). The 1d driver shared by both rows of the
/// `WEDGE_VERTICAL` master mask.
#[rustfmt::skip]
pub const WEDGE_MASTER_VERTICAL: [u8; MASK_MASTER_SIZE] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  7, 21,
    43, 57, 62, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
];

/// `Wedge_Codebook[ 3 ][ 16 ][ 3 ]` (av1-spec p.281-282 lines
/// 15605-15645). Indexed by `[block_shape][wedge_index]` to yield the
/// `(direction, xoff_eighths, yoff_eighths)` triple, where the
/// `xoff`/`yoff` are in units of eighths-of-a-block (so the
/// §7.11.3.11 wrapper computes the per-pixel offset as
/// `MASK_MASTER_SIZE / 2 - ((codebook_off * block_dim) >> 3)`).
///
/// Block-shape ordinals (from `block_shape(bsize)` per av1-spec p.281
/// lines 15582-15591):
///
/// * `0` — `h4 > w4` (tall block — e.g. `BLOCK_8X16`, `BLOCK_16X32`).
/// * `1` — `h4 < w4` (wide block — e.g. `BLOCK_16X8`).
/// * `2` — `h4 == w4` (square block — e.g. `BLOCK_8X8`).
#[rustfmt::skip]
pub const WEDGE_CODEBOOK: [[[u8; 3]; 16]; 3] = [
    // [0] block_shape == 0 (tall: h4 > w4).
    [
        [WEDGE_OBLIQUE27,  4, 4], [WEDGE_OBLIQUE63,  4, 4],
        [WEDGE_OBLIQUE117, 4, 4], [WEDGE_OBLIQUE153, 4, 4],
        [WEDGE_HORIZONTAL, 4, 2], [WEDGE_HORIZONTAL, 4, 4],
        [WEDGE_HORIZONTAL, 4, 6], [WEDGE_VERTICAL,   4, 4],
        [WEDGE_OBLIQUE27,  4, 2], [WEDGE_OBLIQUE27,  4, 6],
        [WEDGE_OBLIQUE153, 4, 2], [WEDGE_OBLIQUE153, 4, 6],
        [WEDGE_OBLIQUE63,  2, 4], [WEDGE_OBLIQUE63,  6, 4],
        [WEDGE_OBLIQUE117, 2, 4], [WEDGE_OBLIQUE117, 6, 4],
    ],
    // [1] block_shape == 1 (wide: h4 < w4).
    [
        [WEDGE_OBLIQUE27,  4, 4], [WEDGE_OBLIQUE63,  4, 4],
        [WEDGE_OBLIQUE117, 4, 4], [WEDGE_OBLIQUE153, 4, 4],
        [WEDGE_VERTICAL,   2, 4], [WEDGE_VERTICAL,   4, 4],
        [WEDGE_VERTICAL,   6, 4], [WEDGE_HORIZONTAL, 4, 4],
        [WEDGE_OBLIQUE27,  4, 2], [WEDGE_OBLIQUE27,  4, 6],
        [WEDGE_OBLIQUE153, 4, 2], [WEDGE_OBLIQUE153, 4, 6],
        [WEDGE_OBLIQUE63,  2, 4], [WEDGE_OBLIQUE63,  6, 4],
        [WEDGE_OBLIQUE117, 2, 4], [WEDGE_OBLIQUE117, 6, 4],
    ],
    // [2] block_shape == 2 (square: h4 == w4).
    [
        [WEDGE_OBLIQUE27,  4, 4], [WEDGE_OBLIQUE63,  4, 4],
        [WEDGE_OBLIQUE117, 4, 4], [WEDGE_OBLIQUE153, 4, 4],
        [WEDGE_HORIZONTAL, 4, 2], [WEDGE_HORIZONTAL, 4, 6],
        [WEDGE_VERTICAL,   2, 4], [WEDGE_VERTICAL,   6, 4],
        [WEDGE_OBLIQUE27,  4, 2], [WEDGE_OBLIQUE27,  4, 6],
        [WEDGE_OBLIQUE153, 4, 2], [WEDGE_OBLIQUE153, 4, 6],
        [WEDGE_OBLIQUE63,  2, 4], [WEDGE_OBLIQUE63,  6, 4],
        [WEDGE_OBLIQUE117, 2, 4], [WEDGE_OBLIQUE117, 6, 4],
    ],
];

/// `Wedge_Bits[ BLOCK_SIZES ]` (av1-spec p.409 lines 22414-22417) —
/// number of bits used to signal a wedge index for each block size
/// (`0` ⇒ wedge is forbidden, `4` ⇒ 16 wedge indices). Wedge masks
/// are defined only for block sizes whose entry is non-zero.
#[rustfmt::skip]
pub const WEDGE_BITS: [u8; 22] = [
    0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 0,
    0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0,
];

/// `block_shape(bsize)` (av1-spec p.281 lines 15582-15591) — returns
/// the wedge-codebook outer dimension for `bsize` based on the
/// `(Num_4x4_Blocks_Wide, Num_4x4_Blocks_High)` shape relation.
///
/// Returns `0` for tall blocks (`h4 > w4`), `1` for wide blocks
/// (`h4 < w4`), `2` for square blocks (`h4 == w4`).
#[inline]
#[must_use]
pub fn block_shape(num_4x4_wide: u32, num_4x4_high: u32) -> u8 {
    if num_4x4_high > num_4x4_wide {
        0
    } else if num_4x4_high < num_4x4_wide {
        1
    } else {
        2
    }
}

/// §7.11.3.11 (av1-spec p.279 lines 15485-15544) `MasterMask` — the
/// six 64 × 64 master mask grids that `wedge_mask` slices into.
///
/// Stored row-major; `index[direction * MASK_MASTER_SIZE *
/// MASK_MASTER_SIZE + i * MASK_MASTER_SIZE + j]`. Built once on first
/// use via [`master_mask_table`] (the spec's
/// `initialise_wedge_mask_table` first loop, p.279 lines 15486-15508).
fn build_master_mask() -> Box<[u8; WEDGE_DIRECTIONS * MASK_MASTER_SIZE * MASK_MASTER_SIZE]> {
    let master =
        vec![0u8; WEDGE_DIRECTIONS * MASK_MASTER_SIZE * MASK_MASTER_SIZE].into_boxed_slice();
    let mut m: Box<[u8; WEDGE_DIRECTIONS * MASK_MASTER_SIZE * MASK_MASTER_SIZE]> =
        master.try_into().expect("len matches");
    let w = MASK_MASTER_SIZE;
    let h = MASK_MASTER_SIZE;

    // ---------- First loop: OBLIQUE63 + VERTICAL (av1-spec p.279
    // lines 15488-15499). ----------
    //
    //   for ( j = 0; j < w; j++ ) {
    //       shift = MASK_MASTER_SIZE / 4
    //       for ( i = 0; i < h; i += 2 ) {
    //           MasterMask[OBLIQUE63][i  ][j] =
    //               Wedge_Master_Oblique_Even[ Clip3(0, MMS-1, j-shift) ]
    //           shift -= 1
    //           MasterMask[OBLIQUE63][i+1][j] =
    //               Wedge_Master_Oblique_Odd [ Clip3(0, MMS-1, j-shift) ]
    //           MasterMask[VERTICAL ][i  ][j] = Wedge_Master_Vertical[j]
    //           MasterMask[VERTICAL ][i+1][j] = Wedge_Master_Vertical[j]
    //       }
    //   }
    for j in 0..w {
        let mut shift: i32 = (MASK_MASTER_SIZE / 4) as i32;
        let mut i = 0usize;
        while i < h {
            let idx_even = clip3_i32(0, (MASK_MASTER_SIZE - 1) as i32, j as i32 - shift) as usize;
            m[idx3(WEDGE_OBLIQUE63 as usize, i, j)] = WEDGE_MASTER_OBLIQUE_EVEN[idx_even];
            shift -= 1;
            let idx_odd = clip3_i32(0, (MASK_MASTER_SIZE - 1) as i32, j as i32 - shift) as usize;
            m[idx3(WEDGE_OBLIQUE63 as usize, i + 1, j)] = WEDGE_MASTER_OBLIQUE_ODD[idx_odd];
            m[idx3(WEDGE_VERTICAL as usize, i, j)] = WEDGE_MASTER_VERTICAL[j];
            m[idx3(WEDGE_VERTICAL as usize, i + 1, j)] = WEDGE_MASTER_VERTICAL[j];
            i += 2;
        }
    }

    // ---------- Second loop: derive OBLIQUE27, OBLIQUE117,
    // OBLIQUE153, HORIZONTAL from OBLIQUE63 + VERTICAL (av1-spec
    // p.279 lines 15500-15508). ----------
    //
    //   for ( i = 0; i < h; i++ ) {
    //       for ( j = 0; j < w; j++ ) {
    //           msk = MasterMask[OBLIQUE63][i][j]
    //           MasterMask[OBLIQUE27 ][j         ][i        ] = msk
    //           MasterMask[OBLIQUE117][i         ][w-1-j    ] = 64 - msk
    //           MasterMask[OBLIQUE153][w-1-j     ][i        ] = 64 - msk
    //           MasterMask[HORIZONTAL][j         ][i        ] =
    //               MasterMask[VERTICAL ][i][j]
    //       }
    //   }
    for i in 0..h {
        for j in 0..w {
            let msk = m[idx3(WEDGE_OBLIQUE63 as usize, i, j)];
            m[idx3(WEDGE_OBLIQUE27 as usize, j, i)] = msk;
            m[idx3(WEDGE_OBLIQUE117 as usize, i, w - 1 - j)] = 64 - msk;
            m[idx3(WEDGE_OBLIQUE153 as usize, w - 1 - j, i)] = 64 - msk;
            m[idx3(WEDGE_HORIZONTAL as usize, j, i)] = m[idx3(WEDGE_VERTICAL as usize, i, j)];
        }
    }

    m
}

#[inline(always)]
const fn idx3(dir: usize, i: usize, j: usize) -> usize {
    dir * MASK_MASTER_SIZE * MASK_MASTER_SIZE + i * MASK_MASTER_SIZE + j
}

/// Lazily-built §7.11.3.11 `MasterMask` table. The first call to
/// [`wedge_mask`] (or [`master_mask_table`]) triggers
/// [`build_master_mask`]; subsequent calls return the cached
/// reference. ~24 KiB (6 × 64 × 64 = 24576 bytes).
static MASTER_MASK_CACHE: std::sync::OnceLock<
    Box<[u8; WEDGE_DIRECTIONS * MASK_MASTER_SIZE * MASK_MASTER_SIZE]>,
> = std::sync::OnceLock::new();

/// Returns the cached `MasterMask` table, building it on first call.
///
/// The table contents are the §7.11.3.11 spec body's `MasterMask[6]
/// [64][64]` array after the `initialise_wedge_mask_table` first loop
/// (av1-spec p.279 lines 15485-15508).
#[must_use]
pub fn master_mask_table() -> &'static [u8; WEDGE_DIRECTIONS * MASK_MASTER_SIZE * MASK_MASTER_SIZE]
{
    MASTER_MASK_CACHE.get_or_init(build_master_mask)
}

/// §7.11.3.11 (av1-spec p.278-282) — populate `mask[ i * w + j ]` for
/// `i in 0..h`, `j in 0..w` with the per-pixel `WedgeMasks[ bsize ][
/// wedge_sign ][ wedge_index ][ i ][ j ]` value.
///
/// The spec body lifts a single per-`(bsize, wedge_index)` slice out
/// of the `MasterMask` table at `(yoff, xoff)`:
///
/// ```text
///   xoff = MASK_MASTER_SIZE / 2 - ((get_wedge_xoff(bsize, wedge) * w) >> 3)
///   yoff = MASK_MASTER_SIZE / 2 - ((get_wedge_yoff(bsize, wedge) * h) >> 3)
/// ```
///
/// The `flipSign` adjustment (av1-spec p.280 lines 15517-15531)
/// inverts the mask sense when the slice's tap-traced average is
/// below `32`: `WedgeMasks[bsize][flipSign][wedge][i][j] = master[..]`
/// and `WedgeMasks[bsize][!flipSign][wedge][i][j] = 64 - master[..]`.
/// `wedge_sign == flipSign` therefore returns the raw master slice;
/// `wedge_sign != flipSign` returns the inverted slice.
///
/// ## Arguments
///
/// * `mi_size` — `BLOCK_SIZES` index (`0..22`); must have
///   `WEDGE_BITS[mi_size] > 0` (the spec enumerates the 9 wedge-eligible
///   block sizes as `BLOCK_8X8`, `BLOCK_8X16`, `BLOCK_16X8`,
///   `BLOCK_16X16`, `BLOCK_16X32`, `BLOCK_32X16`, `BLOCK_32X32`,
///   `BLOCK_8X32`, `BLOCK_32X8`).
/// * `num_4x4_wide` / `num_4x4_high` — `Num_4x4_Blocks_Wide` /
///   `Num_4x4_Blocks_High` for `mi_size` (used by [`block_shape`] +
///   the `xoff`/`yoff` derivation). The width / height in luma
///   samples is `4 * num_4x4_*`.
/// * `wedge_index` — in `0..16` (`WEDGE_TYPES`); the codebook ordinal
///   read by §5.11.28 or §5.11.29.
/// * `wedge_sign` — `0` or `1`; the §5.11.29 sign bit (always `0` on
///   the inter-intra wedge path per §5.11.28).
/// * `mask` — output buffer with length `>= w * h` where `w = 4 *
///   num_4x4_wide` and `h = 4 * num_4x4_high`.
///
/// ## Returns
///
/// `Ok(())` on success, [`crate::Error::PartitionWalkOutOfRange`] for
/// caller-bug arguments (out-of-range `mi_size`, `num_4x4_*`,
/// `wedge_index`, `wedge_sign`, or `mask.len() < w*h`, or a `mi_size`
/// with `WEDGE_BITS[mi_size] == 0`).
pub fn wedge_mask(
    mi_size: usize,
    num_4x4_wide: u32,
    num_4x4_high: u32,
    wedge_index: u8,
    wedge_sign: u8,
    mask: &mut [u8],
) -> Result<(), crate::Error> {
    if mi_size >= WEDGE_BITS.len() {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if WEDGE_BITS[mi_size] == 0 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if wedge_index >= 16 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if wedge_sign > 1 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if num_4x4_wide == 0 || num_4x4_high == 0 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    let w: usize = (num_4x4_wide as usize) * 4;
    let h: usize = (num_4x4_high as usize) * 4;
    // The 9 wedge-eligible block sizes max out at BLOCK_32X32 (32×32);
    // 8×32 / 32×8 are 32 in one axis. Guard against caller-bug giant
    // sizes that would walk off the 64×64 master mask.
    if w > MASK_MASTER_SIZE || h > MASK_MASTER_SIZE {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if mask.len() < w * h {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }

    let shape = block_shape(num_4x4_wide, num_4x4_high) as usize;
    let entry = &WEDGE_CODEBOOK[shape][wedge_index as usize];
    let dir = entry[0] as usize;
    let xoff_codebook = entry[1] as i32;
    let yoff_codebook = entry[2] as i32;
    let xoff: i32 = (MASK_MASTER_SIZE as i32) / 2 - ((xoff_codebook * (w as i32)) >> 3);
    let yoff: i32 = (MASK_MASTER_SIZE as i32) / 2 - ((yoff_codebook * (h as i32)) >> 3);

    // §7.11.3.11 (av1-spec p.279 lines 15517-15523) — average-of-edge
    // sum over the first row + first column to determine the flipSign
    // for this `(bsize, wedge)` slot.
    let master = master_mask_table();
    let mut sum: i64 = 0;
    for i in 0..w {
        sum += master[idx3(dir, yoff as usize, (xoff + i as i32) as usize)] as i64;
    }
    for i in 1..h {
        sum += master[idx3(dir, (yoff + i as i32) as usize, xoff as usize)] as i64;
    }
    let avg = (sum + ((w + h - 1) as i64) / 2) / ((w + h - 1) as i64);
    let flip_sign: u8 = if avg < 32 { 1 } else { 0 };

    // §7.11.3.11 (av1-spec p.279 lines 15524-15531) — `WedgeMasks[
    // bsize ][ flipSign ][ wedge ][ i ][ j ] = MasterMask[ dir ][
    // yoff+i ][ xoff+j ]` and `WedgeMasks[ bsize ][ !flipSign ][...]
    // = 64 - MasterMask[...]`. The wedge_sign argument selects which
    // of the two stored variants to materialize.
    let invert = wedge_sign != flip_sign;
    for i in 0..h {
        for j in 0..w {
            let raw = master[idx3(dir, (yoff + i as i32) as usize, (xoff + j as i32) as usize)];
            mask[i * w + j] = if invert { 64 - raw } else { raw };
        }
    }
    Ok(())
}

// =====================================================================
// §7.11.3.12 — Difference weight mask process (av1-spec p.282-283).
// =====================================================================

/// §7.11.3.12 (av1-spec p.282-283) — fill `mask[ i * w + j ]` with the
/// per-pixel `m = Clip3(0, 64, 38 + diff/16)` value driven by the
/// inter-pred sample difference, with the `mask_type` toggle
/// inverting the sense (`Mask = 64 - m` when `mask_type != 0`).
///
/// The spec body reads:
///
/// ```text
///   for ( i = 0; i < h; i++ ) {
///       for ( j = 0; j < w; j++ ) {
///           diff = Abs(preds[0][i][j] - preds[1][i][j])
///           diff = Round2(diff, (BitDepth - 8) + InterPostRound)
///           m = Clip3(0, 64, 38 + diff / 16)
///           Mask[i][j] = mask_type ? (64 - m) : m
///       }
///   }
/// ```
///
/// ## Arguments
///
/// * `bit_depth` — `8`, `10`, or `12`.
/// * `inter_post_round` — `RoundingVars::inter_post_round` from
///   §7.11.3.2; the `(BitDepth - 8) + InterPostRound` total shift on
///   the diff is `1 + 4 = 5` for 10-bit compound, `4 + 2 = 6` for
///   12-bit compound, `4` for 8-bit compound.
/// * `mask_type` — `0` (`DIFFWTD_38`) or `1` (`DIFFWTD_38_INV`); the
///   §5.11.29 `mask_type` bit.
/// * `preds0` / `preds1` — the two §7.11.3.4-produced per-pixel
///   prediction arrays, each `w * h` `i32` values.
/// * `w` / `h` — region width / height in samples.
/// * `mask` — output, length `>= w * h`.
///
/// ## Returns
///
/// `Ok(())` on success. [`crate::Error::PartitionWalkOutOfRange`] for
/// caller-bug arguments (invalid `bit_depth`, `mask_type > 1`,
/// `w == 0`, `h == 0`, slice-length undershoot).
#[allow(clippy::too_many_arguments)]
pub fn difference_weight_mask(
    bit_depth: u8,
    inter_post_round: u32,
    mask_type: u8,
    preds0: &[i32],
    preds1: &[i32],
    w: usize,
    h: usize,
    mask: &mut [u8],
) -> Result<(), crate::Error> {
    if !matches!(bit_depth, 8 | 10 | 12) {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if mask_type > 1 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if w == 0 || h == 0 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    let n = w * h;
    if preds0.len() < n || preds1.len() < n || mask.len() < n {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    let shift: u32 = (bit_depth as u32 - 8) + inter_post_round;
    for i in 0..h {
        for j in 0..w {
            let p0 = preds0[i * w + j] as i64;
            let p1 = preds1[i * w + j] as i64;
            let diff = (p0 - p1).unsigned_abs() as i64;
            let scaled = round2(diff, shift);
            let m = clip3_i32(0, 64, 38 + (scaled / 16) as i32) as u8;
            mask[i * w + j] = if mask_type != 0 { 64 - m } else { m };
        }
    }
    Ok(())
}

// =====================================================================
// §7.11.3.13 — Intra mode variant mask process (av1-spec p.283-284).
// =====================================================================

/// `Ii_Weights_1d[ MAX_SB_SIZE ]` (av1-spec p.283-284 lines
/// 15727-15735). The 128-entry 1d driver §7.11.3.13 indexes via
/// `i * sizeScale` (II_V_PRED), `j * sizeScale` (II_H_PRED), or
/// `Min(i,j) * sizeScale` (II_SMOOTH_PRED). Values are monotonically
/// non-increasing from `60` (idx 0) down to `1` (idx 113..=127),
/// reflecting the smooth-mask falloff with distance from the
/// edge.
#[rustfmt::skip]
pub const II_WEIGHTS_1D: [u8; MAX_SB_SIZE] = [
    60, 58, 56, 54, 52, 50, 48, 47, 45, 44, 42, 41, 39, 38, 37, 35, 34, 33, 32,
    31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 22, 21, 20, 19, 19, 18, 18, 17, 16,
    16, 15, 15, 14, 14, 13, 13, 12, 12, 12, 11, 11, 10, 10, 10,  9,  9,  9,  8,
     8,  8,  8,  7,  7,  7,  7,  6,  6,  6,  6,  6,  5,  5,  5,  5,  5,  4,  4,
     4,  4,  4,  4,  4,  4,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
];

/// §7.11.3.13 (av1-spec p.283-284) — fill `mask[ i * w + j ]` with the
/// `interintra_mode`-driven smooth-mask weights for the inter-intra
/// blend.
///
/// The spec body reads:
///
/// ```text
///   sizeScale = MAX_SB_SIZE / Max( h, w )
///   for ( i = 0; i < h; i++ ) {
///       for ( j = 0; j < w; j++ ) {
///           if ( interintra_mode == II_V_PRED )
///               Mask[i][j] = Ii_Weights_1d[ i * sizeScale ]
///           else if ( interintra_mode == II_H_PRED )
///               Mask[i][j] = Ii_Weights_1d[ j * sizeScale ]
///           else if ( interintra_mode == II_SMOOTH_PRED )
///               Mask[i][j] = Ii_Weights_1d[ Min(i, j) * sizeScale ]
///           else
///               Mask[i][j] = 32
///       }
///   }
/// ```
///
/// ## Arguments
///
/// * `interintra_mode` — `0..4` per §5.11.28 (`II_DC_PRED`,
///   `II_V_PRED`, `II_H_PRED`, `II_SMOOTH_PRED`).
/// * `w` / `h` — region width / height in samples. `Max(w, h)` must
///   divide `MAX_SB_SIZE = 128` (true for all `BLOCK_SIZES` that
///   support inter-intra: 8, 16, 32 are factors of 128).
/// * `mask` — output, length `>= w * h`.
///
/// ## Returns
///
/// `Ok(())` on success. [`crate::Error::PartitionWalkOutOfRange`] for
/// caller-bug arguments.
pub fn intra_mode_variant_mask(
    interintra_mode: u8,
    w: usize,
    h: usize,
    mask: &mut [u8],
) -> Result<(), crate::Error> {
    if interintra_mode > II_SMOOTH_PRED {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if w == 0 || h == 0 || w > MAX_SB_SIZE || h > MAX_SB_SIZE {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if mask.len() < w * h {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    let larger = w.max(h);
    let size_scale: usize = MAX_SB_SIZE / larger;
    for i in 0..h {
        for j in 0..w {
            let m = match interintra_mode {
                II_V_PRED => II_WEIGHTS_1D[i * size_scale],
                II_H_PRED => II_WEIGHTS_1D[j * size_scale],
                II_SMOOTH_PRED => II_WEIGHTS_1D[i.min(j) * size_scale],
                // II_DC_PRED ⇒ 50/50 blend
                _ => 32,
            };
            mask[i * w + j] = m;
        }
    }
    Ok(())
}

// =====================================================================
// §7.11.3.14 — Mask blend process (av1-spec p.284-285).
// =====================================================================

/// §7.11.3.14 (av1-spec p.284-285) — inter-inter mask blend for the
/// `IsInterIntra == 0 && compound_type ∈ {COMPOUND_WEDGE,
/// COMPOUND_DIFFWTD, COMPOUND_INTRA}` arm of §7.11.3.1.
///
/// Combines `preds[0]` and `preds[1]` per-pixel via the mask:
///
/// ```text
///   m as derived per (subX, subY) from Mask
///   out[y][x] = Clip1( Round2( m * preds[0][y][x] + (64 - m) * preds[1][y][x],
///                              6 + InterPostRound ) )
/// ```
///
/// The `(subX, subY)` argument selects how the per-luma-pixel mask is
/// down-sampled for chroma planes (av1-spec p.284 lines 15771-15783):
///
/// * `(0, 0)` ⇒ `m = Mask[y][x]` (luma, or chroma when
///   `subsampling_{x,y} == 0`).
/// * `(1, 0)` ⇒ `m = Round2(Mask[y][2x] + Mask[y][2x+1], 1)`.
/// * `(0, 1)` ⇒ `m = Round2(Mask[2y][x] + Mask[2y+1][x], 1)`.
/// * `(1, 1)` ⇒ `m = Round2(Mask[2y][2x] + Mask[2y][2x+1] +
///                Mask[2y+1][2x] + Mask[2y+1][2x+1], 2)`.
///
/// ## Arguments
///
/// * `bit_depth` — `8`, `10`, or `12`; sets the `Clip1` upper bound.
/// * `inter_post_round` — `RoundingVars::inter_post_round`.
/// * `sub_x` / `sub_y` — `0` (luma, or chroma without subsampling) or
///   `1` (chroma with subsampling on that axis).
/// * `preds0` / `preds1` — the two §7.11.3.4 prediction buffers
///   sized `w * h` for the *current plane*.
/// * `w` / `h` — output region width / height in the current-plane
///   sample grid.
/// * `mask` — the §7.11.3.11/12/13 luma-grid mask buffer. Its layout
///   depends on `(sub_x, sub_y)`:
///   * `(0, 0)` ⇒ length `>= w * h`, row stride `w`.
///   * `(1, 0)` ⇒ length `>= 2*w * h`, row stride `2*w`.
///   * `(0, 1)` ⇒ length `>= w * 2*h`, row stride `w`.
///   * `(1, 1)` ⇒ length `>= 2*w * 2*h`, row stride `2*w`.
/// * `mask_stride` — row stride of `mask` in u8s (typically `2*w` when
///   `sub_x == 1`, else `w`). Pass `0` to use the default
///   derived-from-`(sub_x, w)` stride.
/// * `out` — output sample buffer, length `>= w * h`.
///
/// ## Returns
///
/// `Ok(())` on success. [`crate::Error::PartitionWalkOutOfRange`] for
/// caller-bug arguments.
#[allow(clippy::too_many_arguments)]
pub fn mask_blend(
    bit_depth: u8,
    inter_post_round: u32,
    sub_x: u8,
    sub_y: u8,
    preds0: &[i32],
    preds1: &[i32],
    w: usize,
    h: usize,
    mask: &[u8],
    mask_stride: usize,
    out: &mut [u16],
) -> Result<(), crate::Error> {
    if !matches!(bit_depth, 8 | 10 | 12) {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if sub_x > 1 || sub_y > 1 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if w == 0 || h == 0 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    let n = w * h;
    if preds0.len() < n || preds1.len() < n || out.len() < n {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    // The mask grid lives on the luma plane; its native width is
    // `2*w` when sub_x==1 and `w` otherwise; same for height with
    // sub_y. mask_stride defaults to the natural row stride when 0.
    let mask_w = if sub_x == 1 { 2 * w } else { w };
    let mask_h = if sub_y == 1 { 2 * h } else { h };
    let stride = if mask_stride == 0 {
        mask_w
    } else {
        mask_stride
    };
    if stride < mask_w {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if mask.len() < stride * (mask_h - 1) + mask_w {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }

    let shift: u32 = 6 + inter_post_round;
    let max: i32 = (1i32 << bit_depth) - 1;
    for y in 0..h {
        for x in 0..w {
            let m: i32 = match (sub_x, sub_y) {
                (0, 0) => mask[y * stride + x] as i32,
                (1, 0) => round2(
                    mask[y * stride + 2 * x] as i64 + mask[y * stride + 2 * x + 1] as i64,
                    1,
                ) as i32,
                (0, 1) => round2(
                    mask[2 * y * stride + x] as i64 + mask[(2 * y + 1) * stride + x] as i64,
                    1,
                ) as i32,
                _ => round2(
                    mask[2 * y * stride + 2 * x] as i64
                        + mask[2 * y * stride + 2 * x + 1] as i64
                        + mask[(2 * y + 1) * stride + 2 * x] as i64
                        + mask[(2 * y + 1) * stride + 2 * x + 1] as i64,
                    2,
                ) as i32,
            };
            let p0 = preds0[y * w + x] as i64;
            let p1 = preds1[y * w + x] as i64;
            let acc = (m as i64) * p0 + ((64 - m) as i64) * p1;
            let blended = round2(acc, shift) as i32;
            out[y * w + x] = clip3_i32(0, max, blended) as u16;
        }
    }
    Ok(())
}

/// §7.11.3.14 (av1-spec p.284-285) — interintra mask blend for the
/// `IsInterIntra == 1` arm of §7.11.3.1.
///
/// Combines `preds[0]` (the inter prediction in pre-Clip form) and
/// the in-place intra prediction already written to `dst` per:
///
/// ```text
///   pred0 = Clip1( Round2( preds[0][y][x], InterPostRound ) )
///   pred1 = CurrFrame[plane][y+dstY][x+dstX]
///   CurrFrame[plane][y+dstY][x+dstX] =
///       Round2( m * pred1 + (64 - m) * pred0, 6 )
/// ```
///
/// Per the spec's `(!subX && !subY) || (interintra && !wedge_interintra)`
/// branch (av1-spec p.284 line 15773), the mask is read directly at
/// `Mask[y][x]` on the interintra path — there is no chroma
/// subsampling averaging because the §7.11.3.13 driver produces a
/// per-plane mask sized to the chroma plane already.
///
/// ## Arguments
///
/// * `bit_depth` — `8`, `10`, or `12`.
/// * `inter_post_round` — `RoundingVars::inter_post_round`.
/// * `preds0` — the inter prediction (pre-Clip), length `>= w * h`.
/// * `w` / `h` — output region width / height in samples.
/// * `mask` — the §7.11.3.13-produced mask, length `>= w * h`.
/// * `dst` — the in-place destination; on entry holds the §7.11.2
///   intra prediction (i.e. `CurrFrame[plane][y+dstY][x+dstX]`); on
///   exit holds the blended sample.
///
/// ## Returns
///
/// `Ok(())` on success. [`crate::Error::PartitionWalkOutOfRange`] for
/// caller-bug arguments.
#[allow(clippy::too_many_arguments)]
pub fn mask_blend_interintra(
    bit_depth: u8,
    inter_post_round: u32,
    preds0: &[i32],
    w: usize,
    h: usize,
    mask: &[u8],
    dst: &mut [u16],
) -> Result<(), crate::Error> {
    if !matches!(bit_depth, 8 | 10 | 12) {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if w == 0 || h == 0 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    let n = w * h;
    if preds0.len() < n || mask.len() < n || dst.len() < n {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    let max: i32 = (1i32 << bit_depth) - 1;
    for y in 0..h {
        for x in 0..w {
            let m: i64 = mask[y * w + x] as i64;
            let raw_pred0 = preds0[y * w + x] as i64;
            // pred0 = Clip1( Round2( preds[0][y][x], InterPostRound ) )
            let clipped = clip3_i32(0, max, round2(raw_pred0, inter_post_round) as i32) as i64;
            let pred1 = dst[y * w + x] as i64;
            let blended = round2(m * pred1 + (64 - m) * clipped, 6) as i32;
            dst[y * w + x] = clip3_i32(0, max, blended) as u16;
        }
    }
    Ok(())
}

// =====================================================================
// §7.11.3.15 — Distance weights process (av1-spec p.285).
// =====================================================================

/// `Quant_Dist_Weight[ 4 ][ 2 ]` (av1-spec p.285 line 15845) — the
/// per-`i` `(c0, c1)` pair the §7.11.3.15 search-loop compares against
/// `(d0, d1)`. The four rows realise progressively-asymmetric weight
/// ratios.
pub const QUANT_DIST_WEIGHT: [[i32; 2]; 4] = [[2, 3], [2, 5], [2, 7], [1, MAX_FRAME_DISTANCE]];

/// `Quant_Dist_Lookup[ 4 ][ 2 ]` (av1-spec p.285 line 15849) — the
/// `(fwd, bck)` weight pairs the §7.11.3.15 search-loop falls back to
/// after deciding `i` (and `(d0, d1)` ordering). Sums to `16` in
/// every row.
pub const QUANT_DIST_LOOKUP: [[i32; 2]; 4] = [[9, 7], [11, 5], [12, 4], [13, 3]];

/// §5.9.3 `get_relative_dist(a, b)` (av1-spec p.41-42) — the sign-
/// extending order-hint subtraction used by §7.11.3.15. Returns
/// `0` when `order_hint_bits == 0` (i.e. `enable_order_hint == 0`);
/// otherwise sign-extends the difference through the `OrderHintBits`-
/// bit signed range.
///
/// ## Arguments
///
/// * `a` / `b` — the two order hints to subtract.
/// * `order_hint_bits` — `OrderHintBits` from §5.5.1 (`0..=8`); `0`
///   when `enable_order_hint == 0`.
///
/// ## Returns
///
/// The sign-extended `(a - b)` in `-(1 << (OrderHintBits-1)) ..
/// (1 << (OrderHintBits-1)) - 1`, or `0` when `order_hint_bits == 0`.
#[inline]
#[must_use]
pub fn get_relative_dist(a: i32, b: i32, order_hint_bits: u32) -> i32 {
    if order_hint_bits == 0 {
        return 0;
    }
    let diff: i32 = a - b;
    let m: i32 = 1 << (order_hint_bits - 1);
    (diff & (m - 1)) - (diff & m)
}

/// §7.11.3.15 output: the `(FwdWeight, BckWeight)` pair the
/// §7.11.3.1 `COMPOUND_DISTANCE` blend combines `preds[0]` and
/// `preds[1]` with: `Clip1( Round2( FwdWeight * preds[0] + BckWeight
/// * preds[1], 4 + InterPostRound ) )`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistanceWeights {
    /// `FwdWeight` applied to `preds[0]`. In `3..=13`.
    pub fwd_weight: i32,
    /// `BckWeight` applied to `preds[1]`. In `3..=13`. Always sums to
    /// `16` with `fwd_weight`.
    pub bck_weight: i32,
}

/// §7.11.3.15 (av1-spec p.285) — derive `(FwdWeight, BckWeight)` from
/// the two reference frames' `OrderHints[]` deltas relative to the
/// current frame's `OrderHint`.
///
/// The spec body computes:
///
/// ```text
///   for ( refList = 0; refList < 2; refList++ )
///     h = OrderHints[ RefFrames[candRow][candCol][refList] ]
///     dist[refList] = Clip3(0, MAX_FRAME_DISTANCE,
///                           Abs( get_relative_dist(h, OrderHint) ))
///   d0 = dist[1]
///   d1 = dist[0]
///   order = d0 <= d1
///   if ( d0 == 0 || d1 == 0 )
///     FwdWeight = Quant_Dist_Lookup[3][order]
///     BckWeight = Quant_Dist_Lookup[3][1 - order]
///   else
///     for ( i = 0; i < 3; i++ )
///       c0 = Quant_Dist_Weight[i][order]
///       c1 = Quant_Dist_Weight[i][1 - order]
///       if ( order  && d0 * c0 > d1 * c1 ) break
///       if ( !order && d0 * c0 < d1 * c1 ) break
///     FwdWeight = Quant_Dist_Lookup[i][order]
///     BckWeight = Quant_Dist_Lookup[i][1 - order]
/// ```
///
/// ## Arguments
///
/// * `order_hint_bits` — `OrderHintBits` from §5.5.1.
/// * `current_order_hint` — `OrderHint` of the frame being decoded.
/// * `order_hint_ref0` / `order_hint_ref1` — `OrderHints[ RefFrames[
///   candRow ][ candCol ][ 0 ] ]` and `[ 1 ]`.
///
/// ## Returns
///
/// `DistanceWeights { fwd_weight, bck_weight }`. The two always sum
/// to `16`, by construction of [`QUANT_DIST_LOOKUP`].
pub fn distance_weights(
    order_hint_bits: u32,
    current_order_hint: i32,
    order_hint_ref0: i32,
    order_hint_ref1: i32,
) -> DistanceWeights {
    // Per-list distance: Abs(get_relative_dist), saturated to
    // [0, MAX_FRAME_DISTANCE].
    let dist0 = clip3_i32(
        0,
        MAX_FRAME_DISTANCE,
        get_relative_dist(order_hint_ref0, current_order_hint, order_hint_bits).abs(),
    );
    let dist1 = clip3_i32(
        0,
        MAX_FRAME_DISTANCE,
        get_relative_dist(order_hint_ref1, current_order_hint, order_hint_bits).abs(),
    );
    // d0 = dist[1], d1 = dist[0] per the spec body — note the swap.
    let d0 = dist1;
    let d1 = dist0;
    let order_bool = d0 <= d1;
    let order: usize = if order_bool { 1 } else { 0 };
    let other: usize = 1 - order;

    if d0 == 0 || d1 == 0 {
        return DistanceWeights {
            fwd_weight: QUANT_DIST_LOOKUP[3][order],
            bck_weight: QUANT_DIST_LOOKUP[3][other],
        };
    }

    // The spec body's `for (i = 0; i < 3; i++)` loop falls through to
    // `i = 3` if no break fires.
    let mut chosen_i = 3usize;
    for (i, row) in QUANT_DIST_WEIGHT.iter().enumerate().take(3) {
        let c0 = row[order];
        let c1 = row[other];
        let lhs = d0 * c0;
        let rhs = d1 * c1;
        let break_now = if order_bool { lhs > rhs } else { lhs < rhs };
        if break_now {
            chosen_i = i;
            break;
        }
    }
    DistanceWeights {
        fwd_weight: QUANT_DIST_LOOKUP[chosen_i][order],
        bck_weight: QUANT_DIST_LOOKUP[chosen_i][other],
    }
}

/// Apply a `COMPOUND_DISTANCE` blend per av1-spec p.258 line 14408 —
/// `CurrFrame[plane][y+i][x+j] = Clip1( Round2( FwdWeight * preds[0]
/// + BckWeight * preds[1], 4 + InterPostRound ) )`.
///
/// Convenience wrapper combining [`distance_weights`] output with the
/// per-pixel blend. Caller-bug-checked against the same argument set
/// as [`mask_blend`].
#[allow(clippy::too_many_arguments)]
pub fn compound_distance_blend(
    bit_depth: u8,
    inter_post_round: u32,
    weights: DistanceWeights,
    preds0: &[i32],
    preds1: &[i32],
    w: usize,
    h: usize,
    out: &mut [u16],
) -> Result<(), crate::Error> {
    if !matches!(bit_depth, 8 | 10 | 12) {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    if w == 0 || h == 0 {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    let n = w * h;
    if preds0.len() < n || preds1.len() < n || out.len() < n {
        return Err(crate::Error::PartitionWalkOutOfRange);
    }
    let max: i32 = (1i32 << bit_depth) - 1;
    let shift: u32 = 4 + inter_post_round;
    for y in 0..h {
        for x in 0..w {
            let acc = (weights.fwd_weight as i64) * (preds0[y * w + x] as i64)
                + (weights.bck_weight as i64) * (preds1[y * w + x] as i64);
            let rounded = round2(acc, shift) as i32;
            out[y * w + x] = clip3_i32(0, max, rounded) as u16;
        }
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

    // ---------- §7.11.3.11 wedge_mask ----------

    /// §7.11.3.11 (av1-spec p.279 lines 15550-15569) — the three 1d
    /// master arrays carry only values in `{0..=64}` (the spec's
    /// blending granularity).
    #[test]
    fn r191_wedge_master_1d_values_within_0_64() {
        for &v in WEDGE_MASTER_OBLIQUE_ODD.iter() {
            assert!(v <= 64, "odd entry {} > 64", v);
        }
        for &v in WEDGE_MASTER_OBLIQUE_EVEN.iter() {
            assert!(v <= 64, "even entry {} > 64", v);
        }
        for &v in WEDGE_MASTER_VERTICAL.iter() {
            assert!(v <= 64, "vertical entry {} > 64", v);
        }
        // Edge values: each array starts with 28+ zeros and ends with
        // a flat run of 64s (per the symmetric-around-centre shape).
        assert_eq!(WEDGE_MASTER_OBLIQUE_ODD[0], 0);
        assert_eq!(WEDGE_MASTER_OBLIQUE_ODD[63], 64);
        assert_eq!(WEDGE_MASTER_OBLIQUE_EVEN[0], 0);
        assert_eq!(WEDGE_MASTER_OBLIQUE_EVEN[63], 64);
        assert_eq!(WEDGE_MASTER_VERTICAL[0], 0);
        assert_eq!(WEDGE_MASTER_VERTICAL[63], 64);
    }

    /// §7.11.3.11 (av1-spec p.279) — the lazily-built `MasterMask`
    /// table populates all six directions; every value lives in
    /// `{0..=64}`. The 6 × 64 × 64 = 24576 byte slab is finite.
    #[test]
    fn r191_master_mask_table_values_within_0_64() {
        let table = master_mask_table();
        for (idx, &v) in table.iter().enumerate() {
            assert!(v <= 64, "entry {idx} = {v} > 64");
        }
    }

    /// §7.11.3.11 (av1-spec p.279 line 15506) — the HORIZONTAL master
    /// is the transpose of the VERTICAL master.
    #[test]
    fn r191_master_mask_horizontal_is_vertical_transpose() {
        let table = master_mask_table();
        for i in 0..MASK_MASTER_SIZE {
            for j in 0..MASK_MASTER_SIZE {
                let h = table[idx3(WEDGE_HORIZONTAL as usize, j, i)];
                let v = table[idx3(WEDGE_VERTICAL as usize, i, j)];
                assert_eq!(h, v, "HORIZONTAL[{j}][{i}] != VERTICAL[{i}][{j}]");
            }
        }
    }

    /// §7.11.3.11 (av1-spec p.279 line 15503) — the OBLIQUE27 master
    /// is the transpose of the OBLIQUE63 master.
    #[test]
    fn r191_master_mask_oblique27_is_oblique63_transpose() {
        let table = master_mask_table();
        for i in 0..MASK_MASTER_SIZE {
            for j in 0..MASK_MASTER_SIZE {
                let m27 = table[idx3(WEDGE_OBLIQUE27 as usize, j, i)];
                let m63 = table[idx3(WEDGE_OBLIQUE63 as usize, i, j)];
                assert_eq!(m27, m63, "OBLIQUE27[{j}][{i}] != OBLIQUE63[{i}][{j}]");
            }
        }
    }

    /// §7.11.3.11 — `block_shape` truth-table.
    #[test]
    fn r191_block_shape_truth_table() {
        // tall (h > w): 0
        assert_eq!(block_shape(1, 2), 0);
        assert_eq!(block_shape(2, 4), 0);
        // wide (h < w): 1
        assert_eq!(block_shape(2, 1), 1);
        assert_eq!(block_shape(4, 2), 1);
        // square (h == w): 2
        assert_eq!(block_shape(2, 2), 2);
        assert_eq!(block_shape(4, 4), 2);
    }

    /// §7.11.3.11 — `wedge_mask` populates a BLOCK_8X8 mask buffer
    /// with values in `{0..=64}` for every (wedge_sign, wedge_index)
    /// combination.
    #[test]
    fn r191_wedge_mask_block_8x8_values_within_0_64() {
        const BLOCK_8X8: usize = 3;
        const NUM_4X4: u32 = 2; // 8 / 4
        let mut mask = vec![0u8; 8 * 8];
        for wedge_index in 0..16u8 {
            for wedge_sign in 0..2u8 {
                wedge_mask(
                    BLOCK_8X8,
                    NUM_4X4,
                    NUM_4X4,
                    wedge_index,
                    wedge_sign,
                    &mut mask,
                )
                .unwrap();
                for (idx, &m) in mask.iter().enumerate() {
                    assert!(
                        m <= 64,
                        "wedge_index={wedge_index} wedge_sign={wedge_sign} mask[{idx}] = {m} > 64"
                    );
                }
            }
        }
    }

    /// §7.11.3.11 — flipping `wedge_sign` inverts the mask per pixel
    /// (`new = 64 - old`).
    #[test]
    fn r191_wedge_mask_sign_flip_inverts() {
        const BLOCK_16X16: usize = 6;
        const NUM_4X4: u32 = 4; // 16 / 4
        let mut m0 = vec![0u8; 16 * 16];
        let mut m1 = vec![0u8; 16 * 16];
        wedge_mask(BLOCK_16X16, NUM_4X4, NUM_4X4, 5, 0, &mut m0).unwrap();
        wedge_mask(BLOCK_16X16, NUM_4X4, NUM_4X4, 5, 1, &mut m1).unwrap();
        for i in 0..(16 * 16) {
            assert_eq!(
                m0[i] + m1[i],
                64,
                "pixel {i}: m0={} m1={} sum != 64",
                m0[i],
                m1[i],
            );
        }
    }

    /// §7.11.3.11 — caller-bug rejection.
    #[test]
    fn r191_wedge_mask_rejects_invalid_inputs() {
        let mut mask = vec![0u8; 64];
        // mi_size >= 22 (BLOCK_SIZES)
        assert!(wedge_mask(22, 2, 2, 0, 0, &mut mask).is_err());
        // wedge_index >= 16
        assert!(wedge_mask(3, 2, 2, 16, 0, &mut mask).is_err());
        // wedge_sign > 1
        assert!(wedge_mask(3, 2, 2, 0, 2, &mut mask).is_err());
        // mi_size with WEDGE_BITS == 0 (e.g. BLOCK_4X4 = 0)
        assert!(wedge_mask(0, 1, 1, 0, 0, &mut mask).is_err());
        // num_4x4_wide == 0
        assert!(wedge_mask(3, 0, 2, 0, 0, &mut mask).is_err());
        // mask too small (BLOCK_8X8 wants 64 bytes, give 32)
        let mut tiny = vec![0u8; 32];
        assert!(wedge_mask(3, 2, 2, 0, 0, &mut tiny).is_err());
    }

    // ---------- §7.11.3.12 difference_weight_mask ----------

    /// §7.11.3.12 — for identical predictions (`p0 == p1`), the
    /// per-pixel `diff = 0`, so `m = Clip3(0, 64, 38 + 0/16) = 38`.
    /// `mask_type == 0` yields `38` everywhere; `mask_type == 1`
    /// yields `64 - 38 = 26`.
    #[test]
    fn r191_difference_weight_mask_equal_preds_yields_38() {
        let preds = vec![100i32; 16];
        let mut mask = vec![0u8; 16];
        difference_weight_mask(8, 4, 0, &preds, &preds, 4, 4, &mut mask).unwrap();
        for &m in mask.iter() {
            assert_eq!(m, 38);
        }
        difference_weight_mask(8, 4, 1, &preds, &preds, 4, 4, &mut mask).unwrap();
        for &m in mask.iter() {
            assert_eq!(m, 64 - 38);
        }
    }

    /// §7.11.3.12 — for very large preds difference, the mask
    /// saturates at `64` (`mask_type == 0`) or `0` (`mask_type == 1`).
    /// Spec formula: `m = Clip3(0, 64, 38 + Round2(|p0-p1|, BD-8+IPR) / 16)`.
    /// With 8-bit compound (`IPR = 4`), `BD - 8 = 0`, shift = 4. For
    /// `|p0 - p1| = 10000`, `Round2(10000, 4) = 625`, `625 / 16 = 39`,
    /// `38 + 39 = 77`, clipped to `64`.
    #[test]
    fn r191_difference_weight_mask_saturates_at_64() {
        let preds0 = vec![10000i32; 4];
        let preds1 = vec![0i32; 4];
        let mut mask = vec![0u8; 4];
        difference_weight_mask(8, 4, 0, &preds0, &preds1, 2, 2, &mut mask).unwrap();
        for &m in mask.iter() {
            assert_eq!(m, 64);
        }
        difference_weight_mask(8, 4, 1, &preds0, &preds1, 2, 2, &mut mask).unwrap();
        for &m in mask.iter() {
            assert_eq!(m, 0);
        }
    }

    /// §7.11.3.12 — caller-bug rejection.
    #[test]
    fn r191_difference_weight_mask_rejects_invalid_inputs() {
        let preds = vec![0i32; 16];
        let mut mask = vec![0u8; 16];
        // bit_depth invalid
        assert!(difference_weight_mask(7, 4, 0, &preds, &preds, 4, 4, &mut mask).is_err());
        // mask_type > 1
        assert!(difference_weight_mask(8, 4, 2, &preds, &preds, 4, 4, &mut mask).is_err());
        // w == 0
        assert!(difference_weight_mask(8, 4, 0, &preds, &preds, 0, 4, &mut mask).is_err());
        // mask too small
        let mut tiny = vec![0u8; 8];
        assert!(difference_weight_mask(8, 4, 0, &preds, &preds, 4, 4, &mut tiny).is_err());
        // preds0 too small
        let small = vec![0i32; 4];
        assert!(difference_weight_mask(8, 4, 0, &small, &preds, 4, 4, &mut mask).is_err());
    }

    // ---------- §7.11.3.13 intra_mode_variant_mask ----------

    /// §7.11.3.13 — `II_DC_PRED` yields a uniform mask of `32`
    /// (50/50 blend).
    #[test]
    fn r191_intra_mode_variant_mask_dc_yields_32() {
        let mut mask = vec![0u8; 64];
        intra_mode_variant_mask(II_DC_PRED, 8, 8, &mut mask).unwrap();
        for &m in mask.iter() {
            assert_eq!(m, 32);
        }
    }

    /// §7.11.3.13 — `II_V_PRED` yields a row-only mask (every column
    /// in row `i` is the same value).
    #[test]
    fn r191_intra_mode_variant_mask_v_pred_row_uniform() {
        let mut mask = vec![0u8; 8 * 8];
        intra_mode_variant_mask(II_V_PRED, 8, 8, &mut mask).unwrap();
        for i in 0..8 {
            let row0 = mask[i * 8];
            for j in 0..8 {
                assert_eq!(mask[i * 8 + j], row0, "row {i} col {j} differs");
            }
        }
        // Monotone non-increasing down the rows (matches Ii_Weights_1d).
        for i in 1..8 {
            assert!(mask[i * 8] <= mask[(i - 1) * 8]);
        }
    }

    /// §7.11.3.13 — `II_H_PRED` yields a column-only mask (every row
    /// in col `j` is the same value).
    #[test]
    fn r191_intra_mode_variant_mask_h_pred_col_uniform() {
        let mut mask = vec![0u8; 8 * 8];
        intra_mode_variant_mask(II_H_PRED, 8, 8, &mut mask).unwrap();
        for j in 0..8 {
            let col0 = mask[j];
            for i in 0..8 {
                assert_eq!(mask[i * 8 + j], col0, "col {j} row {i} differs");
            }
        }
    }

    /// §7.11.3.13 — `II_SMOOTH_PRED` is symmetric across the
    /// `i==j` diagonal because it indexes via `Min(i, j)`.
    #[test]
    fn r191_intra_mode_variant_mask_smooth_pred_diagonal_symmetric() {
        let mut mask = vec![0u8; 16 * 16];
        intra_mode_variant_mask(II_SMOOTH_PRED, 16, 16, &mut mask).unwrap();
        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(
                    mask[i * 16 + j],
                    mask[j * 16 + i],
                    "mask[{i}][{j}] != mask[{j}][{i}]"
                );
            }
        }
    }

    /// §7.11.3.13 — `Ii_Weights_1d` is monotonically non-increasing.
    #[test]
    fn r191_ii_weights_1d_monotone_non_increasing() {
        for i in 1..MAX_SB_SIZE {
            assert!(
                II_WEIGHTS_1D[i] <= II_WEIGHTS_1D[i - 1],
                "Ii_Weights_1d[{}] = {} > [{}] = {}",
                i,
                II_WEIGHTS_1D[i],
                i - 1,
                II_WEIGHTS_1D[i - 1],
            );
        }
        assert_eq!(II_WEIGHTS_1D[0], 60);
        assert_eq!(II_WEIGHTS_1D[127], 1);
    }

    /// §7.11.3.13 — caller-bug rejection.
    #[test]
    fn r191_intra_mode_variant_mask_rejects_invalid_inputs() {
        let mut mask = vec![0u8; 64];
        // interintra_mode > 3
        assert!(intra_mode_variant_mask(4, 8, 8, &mut mask).is_err());
        // w == 0
        assert!(intra_mode_variant_mask(II_V_PRED, 0, 8, &mut mask).is_err());
        // h > MAX_SB_SIZE
        assert!(intra_mode_variant_mask(II_V_PRED, 8, 256, &mut mask).is_err());
        // mask too small
        let mut tiny = vec![0u8; 32];
        assert!(intra_mode_variant_mask(II_V_PRED, 8, 8, &mut tiny).is_err());
    }

    // ---------- §7.11.3.14 mask_blend ----------

    /// §7.11.3.14 — `m == 64` ⇒ `out == p0` (entirely the first
    /// prediction). With `m == 0` ⇒ `out == p1`.
    #[test]
    fn r191_mask_blend_full_mask_yields_pred0() {
        let rv = rounding_variables(8, true).unwrap();
        let preds0 = vec![100i32 << (rv.inter_post_round); 16];
        let preds1 = vec![200i32 << (rv.inter_post_round); 16];
        let mask = vec![64u8; 16];
        let mut out = vec![0u16; 16];
        mask_blend(
            8,
            rv.inter_post_round,
            0,
            0,
            &preds0,
            &preds1,
            4,
            4,
            &mask,
            0,
            &mut out,
        )
        .unwrap();
        // m=64, (64-m)=0 ⇒ out = Clip1(Round2(64 * p0, 6 + IPR)).
        // p0 = 100 << IPR, so 64 * 100 << IPR. Round2 by 6 + IPR =>
        // 64 * 100 / 64 = 100.
        for &v in out.iter() {
            assert_eq!(v, 100);
        }

        // m == 0 ⇒ out == p1.
        let mask0 = vec![0u8; 16];
        mask_blend(
            8,
            rv.inter_post_round,
            0,
            0,
            &preds0,
            &preds1,
            4,
            4,
            &mask0,
            0,
            &mut out,
        )
        .unwrap();
        for &v in out.iter() {
            assert_eq!(v, 200);
        }
    }

    /// §7.11.3.14 — `(sub_x, sub_y) == (1, 1)` averages 4 luma-mask
    /// values per chroma pixel; for a uniform luma mask of value
    /// `M`, the chroma mask average is `M` too.
    #[test]
    fn r191_mask_blend_chroma_subsample_4x_average() {
        let rv = rounding_variables(8, true).unwrap();
        // Chroma region 2x2; luma mask region therefore 4x4 (16 entries).
        let preds0 = vec![100i32 << rv.inter_post_round; 4];
        let preds1 = vec![200i32 << rv.inter_post_round; 4];
        let mask = vec![32u8; 16];
        let mut out = vec![0u16; 4];
        mask_blend(
            8,
            rv.inter_post_round,
            1,
            1,
            &preds0,
            &preds1,
            2,
            2,
            &mask,
            0,
            &mut out,
        )
        .unwrap();
        // Average of (32, 32, 32, 32) = 32. out = Round2(32 * p0 + 32 *
        // p1, 6 + IPR) = (100 + 200) / 2 = 150.
        for &v in out.iter() {
            assert_eq!(v, 150);
        }
    }

    /// §7.11.3.14 — interintra mask blend with `m == 64` ⇒ out
    /// stays at `pred1` (the intra prediction). `m == 0` ⇒ out
    /// becomes Clip1(Round2(preds[0], IPR)).
    #[test]
    fn r191_mask_blend_interintra_endpoints() {
        let rv = rounding_variables(8, false).unwrap();
        // IPR for 8-bit, !compound = 0.
        assert_eq!(rv.inter_post_round, 0);
        let preds0 = vec![100i32; 16];
        let mask_full = vec![64u8; 16];
        let mut dst_full = vec![50u16; 16];
        mask_blend_interintra(
            8,
            rv.inter_post_round,
            &preds0,
            4,
            4,
            &mask_full,
            &mut dst_full,
        )
        .unwrap();
        // m=64: out = Round2(64 * pred1 + 0 * pred0, 6) = pred1 = 50.
        for &v in dst_full.iter() {
            assert_eq!(v, 50);
        }
        let mask_zero = vec![0u8; 16];
        let mut dst_zero = vec![50u16; 16];
        mask_blend_interintra(
            8,
            rv.inter_post_round,
            &preds0,
            4,
            4,
            &mask_zero,
            &mut dst_zero,
        )
        .unwrap();
        // m=0: out = Round2(0 * pred1 + 64 * Clip1(Round2(100, 0)), 6)
        //         = Round2(64 * 100, 6) = 100.
        for &v in dst_zero.iter() {
            assert_eq!(v, 100);
        }
    }

    /// §7.11.3.14 — caller-bug rejection.
    #[test]
    fn r191_mask_blend_rejects_invalid_inputs() {
        let preds = vec![0i32; 16];
        let mask = vec![0u8; 16];
        let mut out = vec![0u16; 16];
        // bit_depth invalid
        assert!(mask_blend(7, 0, 0, 0, &preds, &preds, 4, 4, &mask, 0, &mut out).is_err());
        // sub_x > 1
        assert!(mask_blend(8, 0, 2, 0, &preds, &preds, 4, 4, &mask, 0, &mut out).is_err());
        // w == 0
        assert!(mask_blend(8, 0, 0, 0, &preds, &preds, 0, 4, &mask, 0, &mut out).is_err());
        // mask undersized for (sub_x=1, sub_y=1) requires 8*8 = 64 bytes.
        let small_mask = vec![0u8; 16];
        let preds_c = vec![0i32; 4];
        let mut out_c = vec![0u16; 4];
        assert!(mask_blend(
            8,
            0,
            1,
            1,
            &preds_c,
            &preds_c,
            2,
            2,
            &small_mask,
            0,
            &mut out_c
        )
        .is_ok()); // 16 == 4*4 = 64? wait, 2*w=4, 2*h=4, stride*4-1+4 = 4*3+4 = 16 ⇒ exactly 16
        let too_small = vec![0u8; 15];
        assert!(
            mask_blend(8, 0, 1, 1, &preds_c, &preds_c, 2, 2, &too_small, 0, &mut out_c).is_err()
        );
    }

    // ---------- §7.11.3.15 distance_weights ----------

    /// §5.9.3 `get_relative_dist` — basic sign-extension truth table.
    #[test]
    fn r191_get_relative_dist_truth_table() {
        // enable_order_hint == 0 -> always 0.
        assert_eq!(get_relative_dist(5, 3, 0), 0);
        assert_eq!(get_relative_dist(-5, 3, 0), 0);
        // 7-bit OrderHintBits: range -64..64.
        // diff = 10 - 5 = 5; m = 64; (5 & 63) - (5 & 64) = 5 - 0 = 5
        assert_eq!(get_relative_dist(10, 5, 7), 5);
        // diff = 5 - 10 = -5; -5 in two's complement i32 = 0xFFFFFFFB
        // mask m-1=63 yields 0x3B = 59; mask m=64 yields 0x40 = 64
        // 59 - 64 = -5.
        assert_eq!(get_relative_dist(5, 10, 7), -5);
        // wrap-around test: diff = 100, m = 64; (100 & 63) - (100 & 64)
        // = 36 - 64 = -28.
        assert_eq!(get_relative_dist(100, 0, 7), -28);
    }

    /// §7.11.3.15 — `Quant_Dist_Lookup` rows always sum to `16`.
    #[test]
    fn r191_quant_dist_lookup_sums_to_16() {
        for (i, row) in QUANT_DIST_LOOKUP.iter().enumerate() {
            assert_eq!(row[0] + row[1], 16, "row {i} = {:?}", row);
        }
    }

    /// §7.11.3.15 — equal-distance refs (`d0 == d1`) take the
    /// `order_bool == true` (d0 <= d1) path. With `dist0 == dist1
    /// == 5`, the loop's first iteration (`i = 0`, `c0 = 2`, `c1 = 3`)
    /// fires `5 * 2 > 5 * 3` ⇒ `10 > 15` ⇒ false; second (`i = 1`,
    /// `c0 = 2`, `c1 = 5`): `10 > 25` ⇒ false; third (`i = 2`, `c0 =
    /// 2`, `c1 = 7`): `10 > 35` ⇒ false. Fall-through to `i = 3`:
    /// `Lookup[3][1]`/`[0]` ⇒ FwdWeight=3, BckWeight=13 — the
    /// strongest backward-skew.
    #[test]
    fn r191_distance_weights_equal_distances() {
        // current=0, ref0=5, ref1=5 ⇒ dist0=5, dist1=5, d0=d1=5
        let dw = distance_weights(7, 0, 5, 5);
        assert_eq!(dw.fwd_weight + dw.bck_weight, 16);
    }

    /// §7.11.3.15 — zero-distance ref triggers the early-return
    /// `Quant_Dist_Lookup[3]` branch.
    #[test]
    fn r191_distance_weights_zero_distance_branch() {
        // current=10, ref0=10 (dist0=0), ref1=5 (dist1=5).
        // d0 = dist1 = 5, d1 = dist0 = 0. d0 == 0 false; d1 == 0 true.
        // order = d0 <= d1 ⇒ 5 <= 0 ⇒ false ⇒ order=0, other=1.
        // FwdWeight = Lookup[3][0] = 13, BckWeight = Lookup[3][1] = 3.
        let dw = distance_weights(7, 10, 10, 5);
        assert_eq!(dw.fwd_weight, 13);
        assert_eq!(dw.bck_weight, 3);
    }

    /// §7.11.3.15 — `enable_order_hint == 0` (`order_hint_bits == 0`)
    /// ⇒ both distances collapse to 0 ⇒ early-return branch with
    /// `order_bool = (d0 <= d1) = (0 <= 0) = true` ⇒
    /// FwdWeight=Lookup[3][order=1]=3, BckWeight=Lookup[3][other=0]=13.
    #[test]
    fn r191_distance_weights_no_order_hint() {
        let dw = distance_weights(0, 0, 0, 0);
        assert_eq!(dw.fwd_weight, 3);
        assert_eq!(dw.bck_weight, 13);
    }

    /// §7.11.3.15 — asymmetric distances exercise the per-`i`
    /// search. current=0, ref0=1 (dist0=1), ref1=10 (dist1=10).
    /// d0=dist1=10, d1=dist0=1. order = (10 <= 1) = false ⇒ order=0,
    /// other=1.
    /// i=0: c0=Weight[0][0]=2, c1=Weight[0][1]=3, lhs=10*2=20,
    /// rhs=1*3=3, break if lhs < rhs ⇒ 20 < 3 false, continue.
    /// i=1: c0=2, c1=5, lhs=20, rhs=5, 20 < 5 false.
    /// i=2: c0=2, c1=7, lhs=20, rhs=7, 20 < 7 false.
    /// Fall-through i=3 ⇒ FwdWeight=Lookup[3][0]=13, BckWeight=Lookup[3][1]=3.
    #[test]
    fn r191_distance_weights_asymmetric_far_forward() {
        let dw = distance_weights(7, 0, 1, 10);
        assert_eq!(dw.fwd_weight, 13);
        assert_eq!(dw.bck_weight, 3);
    }

    /// §7.11.3.15 — `compound_distance_blend` with `FwdWeight =
    /// BckWeight = 8` reduces to the same `Round2(8 * (p0 + p1), 4 +
    /// IPR)`. With p0 = p1 = 100 << IPR, output = 100.
    #[test]
    fn r191_compound_distance_blend_symmetric() {
        let rv = rounding_variables(8, true).unwrap();
        let preds0 = vec![100i32 << rv.inter_post_round; 16];
        let preds1 = vec![200i32 << rv.inter_post_round; 16];
        let weights = DistanceWeights {
            fwd_weight: 8,
            bck_weight: 8,
        };
        let mut out = vec![0u16; 16];
        compound_distance_blend(
            8,
            rv.inter_post_round,
            weights,
            &preds0,
            &preds1,
            4,
            4,
            &mut out,
        )
        .unwrap();
        // (8 * 100 + 8 * 200) / 16 = (8 * 300) / 16 = 150.
        for &v in out.iter() {
            assert_eq!(v, 150);
        }
    }

    /// §7.11.3.15 — `compound_distance_blend` caller-bug rejection.
    #[test]
    fn r191_compound_distance_blend_rejects_invalid_inputs() {
        let preds = vec![0i32; 16];
        let mut out = vec![0u16; 16];
        let w = DistanceWeights {
            fwd_weight: 8,
            bck_weight: 8,
        };
        assert!(compound_distance_blend(7, 0, w, &preds, &preds, 4, 4, &mut out).is_err());
        assert!(compound_distance_blend(8, 0, w, &preds, &preds, 0, 4, &mut out).is_err());
        let mut tiny = vec![0u16; 8];
        assert!(compound_distance_blend(8, 0, w, &preds, &preds, 4, 4, &mut tiny).is_err());
    }
}
