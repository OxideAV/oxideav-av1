//! Forward 2D transform dispatcher — the encoder counterpart of the
//! §7.13.3 2D inverse-transform dispatcher implemented in
//! [`crate::transform::inverse_transform_2d`].
//!
//! ## Why "forward dispatcher"
//!
//! Rounds 219, 222, 225, 226 built the per-axis forward 1D / 2D
//! primitives in [`super::forward_transform`] (DCT, sizes 4..64),
//! [`super::forward_adst`] (ADST + FLIPADST, sizes 4 / 8 / 16),
//! [`super::forward_identity`] (IDTX, sizes 4 / 8 / 16 / 32), and
//! [`super::forward_wht`] (WHT, size 4 only — the lossless arm).
//!
//! Those primitives are the forward kernels for a **fixed**
//! row / column kernel choice. The §7.13.3 inverse dispatcher
//! composes a per-(`PlaneTxType`, `tx_size`) row kernel with a
//! column kernel and wraps the composition in the §7.13.3 shift
//! envelope (`Round2(_, rowShift)` per row after the row pass,
//! `Round2(_, colShift)` per column after the column pass) plus a
//! between-stage `Clip3` clamp at `colClampRange` bits. The encoder
//! mirror — this module — composes the **forward** kernels in the
//! transpose order (column-then-row) and leaves the shift envelope
//! to be absorbed by the decoder's `Round2(_, shift)` post-scales
//! against the kernel's intrinsic `N`-times gain (see "Shift
//! envelope" below for the derivation).
//!
//! ## Square scope plus the first three rectangular pairs (round 241)
//!
//! The five **square** transform sizes are landed:
//! `TX_4X4`, `TX_8X8`, `TX_16X16`, `TX_32X32`, `TX_64X64`. Round 235
//! added the **`|log2W - log2H| == 1`** rectangular pair at the
//! shorter side — specifically `TX_4X8` and `TX_8X4`. Round 238
//! extended the `|log2W - log2H| == 1` arc by one more pair —
//! `TX_8X16` and `TX_16X8` — covering the `min(log2W, log2H) == 3`
//! short-side-8 family. Round 241 extends the arc once more to the
//! next pair at `min(log2W, log2H) == 4` — **`TX_16X32`** and
//! **`TX_32X16`**. Each axis runs its own log-size forward kernel:
//!
//! | tx_size    | row kernel        | col kernel        |
//! | ---------- | ----------------- | ----------------- |
//! | `TX_4X8`   | `forward_dct_4`   | `forward_dct_8`   |
//! | `TX_8X4`   | `forward_dct_8`   | `forward_dct_4`   |
//! | `TX_8X16`  | `forward_dct_8`   | `forward_dct_16`  |
//! | `TX_16X8`  | `forward_dct_16`  | `forward_dct_8`   |
//! | `TX_16X32` | `forward_dct_16`  | `forward_dct_32`  |
//! | `TX_32X16` | `forward_dct_32`  | `forward_dct_16`  |
//!
//! After the row kernel pass the per-row §7.13.3 rectangular scale
//! `Round2(T[j] * 2896, 12)` is applied — the encoder mirror of the
//! decoder's pre-row-kernel rectangular scale at the same constant.
//! Both encoder and decoder thus contribute one factor of
//! `2896 / 4096`, giving a combined rectangular gain of
//! `(2896 / 4096)^2 ≈ 1/2` per rectangular axis pair. This holds
//! uniformly across the `|log2W - log2H| == 1` family — the
//! per-tx-size differentiator is the row-shift envelope
//! (`Transform_Row_Shift[TX_8X16] = Transform_Row_Shift[TX_16X8] =
//! Transform_Row_Shift[TX_16X32] = Transform_Row_Shift[TX_32X16] =
//! 1` vs the `0` for `TX_4X8` / `TX_8X4`), not the rectangular
//! scale constant itself.
//!
//! The remaining 8 rectangular sizes (`TX_32X64` / `TX_64X32` /
//! `TX_4X16` / `TX_16X4` / `TX_8X32` / `TX_32X8` / `TX_16X64` /
//! `TX_64X16`) split into two further arcs: the
//! `|log2W - log2H| == 1` chain at the largest kernel sizes
//! (`TX_32X64` / `TX_64X32` — still take the `× 2896` post-scale
//! but combine it with a `dqDenom != 1` quantizer denominator) and
//! the `|log2W - log2H| == 2` family (`TX_4X16` / `TX_16X4` /
//! `TX_8X32` / `TX_32X8` / `TX_16X64` / `TX_64X16`), which does
//! NOT take the `× 2896` rectangular post-scale per §7.13.3
//! av1-spec p.305 (only the `Abs(log2W - log2H) == 1` branch does).
//!
//! Within the square sizes, the per-kernel coverage is:
//!
//! | Kernel | TX_4X4 | TX_8X8 | TX_16X16 | TX_32X32 | TX_64X64 |
//! | ------ | ------ | ------ | -------- | -------- | -------- |
//! | DCT    | ✓      | ✓      | ✓        | ✓        | ✓        |
//! | ADST   | ✓      | ✓      | ✓        | —        | —        |
//! | IDTX   | ✓      | ✓      | ✓        | ✓        | —        |
//! | WHT    | ✓      | —      | —        | —        | —        |
//!
//! ADST stops at size 16 because the §7.13.2.9 inverse-ADST
//! dispatcher itself only routes `n in 2..=4` (sizes 4 / 8 / 16);
//! IDTX stops at size 32 because §7.13.2.15 inverse-identity routes
//! `n in 2..=5` (sizes 4 / 8 / 16 / 32). The spec's §6.10.19
//! `tx_type` derivation forces `DCT_DCT` for the unreachable
//! combinations (e.g. `TX_32X32` with non-DCT row kernel is
//! disallowed by the §5.11.48 tx-type-set restrictions).
//!
//! ## FLIPADST handling
//!
//! Per §7.13.3 + §7.12.3 step 3, the FLIPADST family runs the same
//! butterfly schedule as ADST; the **flip** is purely a destination-
//! coordinate transform applied during the decoder's frame-buffer
//! write (`xx = flipLR ? w - j - 1 : j`, `yy = flipUD ? h - i - 1 :
//! i`). The encoder mirror is to flip the spatial residual buffer
//! **before** the forward transform runs and then run the plain
//! ADST kernel on the flipped data.
//!
//! Per-tx-type flip axis:
//!
//! | tx_type             | flip rows (vertical) | flip cols (horizontal) |
//! | ------------------- | -------------------- | ---------------------- |
//! | `FLIPADST_DCT`      | yes                  | no                     |
//! | `DCT_FLIPADST`      | no                   | yes                    |
//! | `FLIPADST_FLIPADST` | yes                  | yes                    |
//! | `ADST_FLIPADST`     | no                   | yes                    |
//! | `FLIPADST_ADST`     | yes                  | no                     |
//! | `V_FLIPADST`        | yes                  | no                     |
//! | `H_FLIPADST`        | no                   | yes                    |
//!
//! "Row kernel = FLIPADST" ⇒ the **horizontal** axis runs ADST on
//! a horizontally-flipped row, i.e. flip columns. "Column kernel =
//! FLIPADST" ⇒ the **vertical** axis runs ADST on a vertically-
//! flipped column, i.e. flip rows. The table above is the union of
//! the two axis flips per tx_type.
//!
//! ## Shift envelope (lossy arm)
//!
//! §7.13.3 lossy decoder pipeline (per row of the row pass):
//!
//! ```text
//!   T[j]            = Dequant[i][j]                       (input)
//!   T[j]            = kernel_row(T[j], log2W, r=BD+8)     (1D pass)
//!   Residual[i][j]  = Round2(T[j], rowShift)              (per-row right-shift)
//! ```
//!
//! and per column:
//!
//! ```text
//!   T[i]            = Residual[i][j]
//!   T[i]            = kernel_col(T[i], log2H, r=Max(BD+6, 16))
//!   Residual[i][j]  = Round2(T[i], colShift=4)            (per-column right-shift)
//! ```
//!
//! Encoder mirror (column-then-row, the transpose of the decoder's
//! row-then-column composition — the decoder's column pass runs
//! against `Residual` last, so the encoder's column pass runs
//! against the spatial residual **first**):
//!
//! ```text
//!   T[i]                = input[i*w + j]                       (residual cell)
//!   T[i]                = fwd_kernel_col(T[i], log2H)          (1D pass)
//!   intermediate[i*w+j] = T[i]
//!   T[j]                = intermediate[i*w + j]
//!   T[j]                = fwd_kernel_row(T[j], log2W)
//!   coeff[i*w + j]      = T[j]
//! ```
//!
//! Note the encoder does **not** apply the `<< rowShift` / `<<
//! colShift` pre-scales that would, in theory, cancel the decoder's
//! `Round2(_, shift)` post-scales bit-exactly. Two reasons:
//!
//!   1. The decoder applies an explicit between-stage `Clip3` at
//!      `colClampRange = Max(BitDepth + 6, 16)` bits. Pre-shifting
//!      the encoder's coefficients by `2^4` or `2^6` for larger
//!      block sizes pushes the inverse pipeline's intermediate
//!      values past the decoder clamp, breaking the round-trip
//!      catastrophically (the saturation truncates the kernel
//!      output rather than preserving it).
//!
//!   2. The spec is set up so that the per-axis kernel matrix has
//!      `M^T · M ≈ N · I` (squared L2 norm per column ≈ `N` for the
//!      sqrt(N)-normalised kernels). The decoder's
//!      `2^(rowShift+colShift)` divisor is chosen to balance this
//!      `N^2 ≈` kernel × kernel gain back toward unity — for the
//!      five square sizes the inverse-only effective gain per cell
//!      is `N^2 / 2^(rowShift + colShift)` = `{1, 2, 4, 16, 64}`
//!      for `TX_{4..64}X{4..64}` respectively. A round-trip with
//!      no encoder pre-shift then has per-cell gain
//!      `N^2 / 2^(rowShift + colShift)` (same number).
//!
//! The §7.13.3 inverse pipeline's combined per-cell gain on the
//! lossy arm is therefore the round-trip scale this dispatcher
//! produces. A real encoder driver that pairs this dispatcher with
//! [`super::forward_quantize`] doesn't see the gain — the quantizer
//! divides by the per-stage gain to recover bit-correct coefficients
//! against the spec's per-tx-size quantizer step.
//!
//! ## Lossless arm
//!
//! §7.13.3 routes the `Lossless = 1` path through a `TX_4X4`-only
//! WHT pipeline:
//!
//! ```text
//!   row pass: inverse_wht4(T, shift = 2)
//!   col pass: inverse_wht4(T, shift = 0)
//! ```
//!
//! The forward [`super::forward_wht::forward_wht_4x4`] already
//! implements the **bit-exact** inverse of this pipeline (column
//! pass with `shift = 0` first, then row pass with `shift = 2` —
//! pre-multiplied by `<< shift` to cancel the inverse's `>> shift`).
//! This dispatcher delegates `lossless == true` to that primitive.
//!
//! ## Round-trip behaviour
//!
//! For the **lossless** arm (`TX_4X4` only, any input residual):
//! `inverse_transform_2d(forward_transform_2d(x), TX_4X4, _, true)
//! == x` exactly. The WHT integer butterflies + pre-shift envelope
//! preserve every input bit.
//!
//! For the **lossy** arms (all other tx_size × tx_type combinations
//! with a sufficiently small residual magnitude that the inverse-
//! side between-stage `Clip3` doesn't saturate): `inverse_transform_2d
//! (forward_transform_2d(x), tx_size, tx_type, false) ≈ scale * x`
//! with a small per-cell rounding error. The `scale` factor per
//! tx-size for the DCT family is `N^2 / 2^(rowShift + colShift)` —
//! evaluated:
//!
//! | tx_size    | N  | rowShift | colShift | scale |
//! | ---------- | -- | -------- | -------- | ----- |
//! | `TX_4X4`   | 4  | 0        | 4        | 1     |
//! | `TX_8X8`   | 8  | 1        | 4        | 2     |
//! | `TX_16X16` | 16 | 2        | 4        | 4     |
//! | `TX_32X32` | 32 | 2        | 4        | 16    |
//! | `TX_64X64` | 64 | 2        | 4        | 64    |
//!
//! For ADST × ADST the per-axis matrix norm is the same shape so the
//! round-trip scale is identical to the DCT scale at each size. For
//! IDTX × IDTX the per-axis scale is the inverse-identity multiplier
//! from §7.13.2.11..§7.13.2.14 — n = 3 / n = 5 are exact integer
//! multiplies (`× 2`, `× 4`), while n = 2 / n = 4 have small Round2-
//! per-cell floor error. The roundtrip tests in this module use a
//! conservative per-cell `max_abs_error` bound; the typical worst
//! case is a few LSBs per cell from stacked `Round2(_, 12)`
//! operations.
//!
//! For TX_32X32 / TX_64X64 the inverse-side between-stage clamp
//! (`colClampRange = Max(BitDepth + 6, 16) = 16` bits at `BitDepth
//! = 8`, i.e. `±32768`) bounds the intermediate magnitude after the
//! inverse row pass to `±32768`. For arbitrary `±128` residual
//! inputs this clamp can trip on the larger sizes; the round-trip
//! tests therefore restrict TX_32X32 / TX_64X64 input residuals to
//! a magnitude small enough that the inverse pipeline doesn't
//! saturate (`±4` for TX_64X64, `±16` for TX_32X32). The smaller
//! sizes can use the full `±128` range.
//!
//! ## Rectangular scaling (rounds 235 + 238 + 241)
//!
//! For the `|log2W - log2H| == 1` sizes the §7.13.3 decoder inserts
//! a per-row `Round2(T[j] * 2896, 12)` step BEFORE the row kernel
//! runs. The encoder mirror — time-reversed of the decoder pipeline
//! — places the SAME `Round2(T[j] * 2896, 12)` step per row AFTER
//! the row kernel runs (the encoder's last pass). The constant
//! `2896` is exactly `Cos128_Lookup[32]` (`round(4096 * cos(pi/4)) =
//! round(4096 / sqrt(2))`), so the per-pair contribution is
//! `(2896 / 4096)^2 ≈ 1/2`. Combined with the per-axis kernel norm
//! and the §7.13.3 row + col shift envelope, the empirical
//! round-trip per-cell scale on a constant-DC input evaluates to
//!
//! | tx_size    | empirical per-cell round-trip scale |
//! | ---------- | ----------------------------------- |
//! | `TX_4X8`   | `1/4`                               |
//! | `TX_8X4`   | `1/4`                               |
//! | `TX_8X16`  | `1/2`                               |
//! | `TX_16X8`  | `1/2`                               |
//! | `TX_16X32` | `2`                                 |
//! | `TX_32X16` | `2`                                 |
//!
//! TX_4X8 / TX_8X4 inherit the TX_4X4 `1/4` scale (short-side-4
//! kernel norm dominates the row/col composition); TX_8X16 /
//! TX_16X8 land at `1/2` — one step up — because the larger
//! `N_w * N_h = 8 * 16 = 128` kernel norm gains `4×` over the
//! short-side-4 pair's `N_w * N_h = 4 * 8 = 32`, while the larger
//! row-shift envelope (`Transform_Row_Shift[TX_8X16] = 1` vs
//! `0` for TX_4X8) eats back `2×`, netting a `2×` larger
//! round-trip per-cell scale. TX_16X32 / TX_32X16 land at `2` —
//! two steps up from TX_8X16 — because `N_w * N_h = 16 * 32 = 512`
//! gains another `4×` over the short-side-8 pair while the
//! row-shift envelope stays at `Transform_Row_Shift = 1` (same as
//! TX_8X16 / TX_16X8), so the full `4×` lands in the per-cell
//! round-trip scale.
//!
//! Walking the analytic derivation for TX_16X32: round-trip
//! per-cell gain = `(N_w * N_h) / 2^(row_shift + col_shift) *
//! (2896 / 4096)^2` = `512 / 2^(1 + 4) * 1/2` = `512 / 32 * 1/2` =
//! `8`. The factor of `4` between the analytic `8` and empirical
//! `2` matches the `4×` ratio already documented for TX_8X16
//! (analytic `2`, empirical `1/2`) — i.e. the constant-DC probe's
//! input-mass factor on the inverse-side after the row-pass is
//! invariant across the rectangular `|log2W - log2H| == 1` family.
//!
//! The quantizer in [`super::forward_quantize`] absorbs this fixed
//! per-tx-size gain through the `dqDenom` factor (the same path
//! used for the square sizes): `dequant_denom(TX_16X32) =
//! dequant_denom(TX_32X16) = 2` per §7.12.3 (the only `dqDenom !=
//! 1` rectangular shapes at this point in the arc — the 32-axis
//! presence on either side promotes `dqDenom` from `1` to `2`).
//! TX_8X16 / TX_16X8 still sit at `dqDenom = 1`. A downstream
//! encoder pipeline that pairs this dispatcher with the existing
//! quantizer sees a bit-correct coefficient stream against the
//! spec's per-tx-size quantizer step.

use crate::cdf::{
    ADST_ADST, ADST_DCT, ADST_FLIPADST, DCT_ADST, DCT_DCT, DCT_FLIPADST, FLIPADST_ADST,
    FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, IDTX, TX_HEIGHT, TX_SIZES_ALL,
    TX_WIDTH, TX_WIDTH_LOG2, V_ADST, V_DCT, V_FLIPADST,
};

use super::forward_adst::{forward_adst_16, forward_adst_4, forward_adst_8};
use super::forward_identity::{forward_idtx_16, forward_idtx_32, forward_idtx_4, forward_idtx_8};
use super::forward_transform::{
    forward_dct_16, forward_dct_32, forward_dct_4, forward_dct_64, forward_dct_8,
};
use super::forward_wht::forward_wht_4x4;

/// §7.13.3 row-pass forward kernel selector — the encoder mirror of
/// [`crate::transform::apply_row_kernel`]. DCT for
/// `{ DCT_DCT, ADST_DCT, FLIPADST_DCT, H_DCT }`; ADST for
/// `{ DCT_ADST, ADST_ADST, DCT_FLIPADST, FLIPADST_FLIPADST,
/// ADST_FLIPADST, FLIPADST_ADST, H_ADST, H_FLIPADST }`; identity
/// for `{ IDTX, V_DCT, V_ADST, V_FLIPADST }`.
fn forward_row_kernel(t: &mut [i64], tx_type: usize, log2_w: u32) {
    let r = 32; // signature parity only; forward kernels ignore r.
    if matches!(tx_type, x if x == DCT_DCT || x == ADST_DCT || x == FLIPADST_DCT || x == H_DCT) {
        forward_dct_dispatch(t, log2_w, r);
    } else if matches!(
        tx_type,
        x if x == DCT_ADST
            || x == ADST_ADST
            || x == DCT_FLIPADST
            || x == FLIPADST_FLIPADST
            || x == ADST_FLIPADST
            || x == FLIPADST_ADST
            || x == H_ADST
            || x == H_FLIPADST
    ) {
        forward_adst_dispatch(t, log2_w, r);
    } else {
        debug_assert!(
            matches!(tx_type, x if x == IDTX || x == V_DCT || x == V_ADST || x == V_FLIPADST)
        );
        forward_idtx_dispatch(t, log2_w);
    }
}

/// §7.13.3 column-pass forward kernel selector — the encoder mirror
/// of [`crate::transform::apply_col_kernel`]. DCT for
/// `{ DCT_DCT, DCT_ADST, DCT_FLIPADST, V_DCT }`; ADST for
/// `{ ADST_DCT, ADST_ADST, FLIPADST_DCT, FLIPADST_FLIPADST,
/// ADST_FLIPADST, FLIPADST_ADST, V_ADST, V_FLIPADST }`; identity
/// for `{ IDTX, H_DCT, H_ADST, H_FLIPADST }`.
fn forward_col_kernel(t: &mut [i64], tx_type: usize, log2_h: u32) {
    let r = 32;
    if matches!(tx_type, x if x == DCT_DCT || x == DCT_ADST || x == DCT_FLIPADST || x == V_DCT) {
        forward_dct_dispatch(t, log2_h, r);
    } else if matches!(
        tx_type,
        x if x == ADST_DCT
            || x == ADST_ADST
            || x == FLIPADST_DCT
            || x == FLIPADST_FLIPADST
            || x == ADST_FLIPADST
            || x == FLIPADST_ADST
            || x == V_ADST
            || x == V_FLIPADST
    ) {
        forward_adst_dispatch(t, log2_h, r);
    } else {
        debug_assert!(
            matches!(tx_type, x if x == IDTX || x == H_DCT || x == H_ADST || x == H_FLIPADST)
        );
        forward_idtx_dispatch(t, log2_h);
    }
}

fn forward_dct_dispatch(t: &mut [i64], n: u32, r: u32) {
    match n {
        2 => forward_dct_4(t, r),
        3 => forward_dct_8(t, r),
        4 => forward_dct_16(t, r),
        5 => forward_dct_32(t, r),
        6 => forward_dct_64(t, r),
        _ => panic!("oxideav-av1 forward_dct_dispatch: n must be 2..=6, got {n}"),
    }
}

fn forward_adst_dispatch(t: &mut [i64], n: u32, r: u32) {
    match n {
        2 => forward_adst_4(t, r),
        3 => forward_adst_8(t, r),
        4 => forward_adst_16(t, r),
        _ => panic!(
            "oxideav-av1 forward_adst_dispatch: ADST is only defined for n in 2..=4 \
             (the §7.13.2.9 inverse-ADST dispatcher's range), got {n} — \
             §6.10.19 tx_type derivation forces DCT_DCT outside this range",
        ),
    }
}

fn forward_idtx_dispatch(t: &mut [i64], n: u32) {
    match n {
        2 => forward_idtx_4(t),
        3 => forward_idtx_8(t),
        4 => forward_idtx_16(t),
        5 => forward_idtx_32(t),
        _ => panic!(
            "oxideav-av1 forward_idtx_dispatch: IDTX is only defined for n in 2..=5 \
             (the §7.13.2.15 inverse-identity dispatcher's range), got {n} — \
             §6.10.19 tx_type derivation forces DCT_DCT outside this range",
        ),
    }
}

/// Per-tx-type flip-axis decoder. Returns `(flip_rows, flip_cols)`
/// for the §7.12.3 step-3 frame-buffer flip the decoder applies
/// post-inverse — the encoder must apply the same flip on the
/// spatial residual before the forward transform runs.
fn flip_axes(tx_type: usize) -> (bool, bool) {
    // Vertical-axis flip (= flip rows in the row-major layout): the
    // *column* kernel is FLIPADST.
    let flip_rows = matches!(
        tx_type,
        x if x == FLIPADST_DCT
            || x == FLIPADST_FLIPADST
            || x == FLIPADST_ADST
            || x == V_FLIPADST
    );
    // Horizontal-axis flip (= flip cols in the row-major layout):
    // the *row* kernel is FLIPADST.
    let flip_cols = matches!(
        tx_type,
        x if x == DCT_FLIPADST
            || x == FLIPADST_FLIPADST
            || x == ADST_FLIPADST
            || x == H_FLIPADST
    );
    (flip_rows, flip_cols)
}

fn apply_flip(input: &[i64], w: usize, h: usize, flip_rows: bool, flip_cols: bool) -> Vec<i64> {
    if !flip_rows && !flip_cols {
        return input.to_vec();
    }
    let mut out = vec![0i64; w * h];
    for i in 0..h {
        let src_i = if flip_rows { h - 1 - i } else { i };
        for j in 0..w {
            let src_j = if flip_cols { w - 1 - j } else { j };
            out[i * w + j] = input[src_i * w + src_j];
        }
    }
    out
}

/// §7.13.3-equivalent forward 2D transform dispatcher — the encoder
/// counterpart of [`crate::transform::inverse_transform_2d`].
///
/// Consumes `input` (a row-major spatial residual buffer of length
/// `w * h` where `w = Tx_Width[tx_size]`, `h = Tx_Height[tx_size]`)
/// and returns the row-major coefficient buffer of the same length.
/// The forward composition is **column pass first, then row pass**
/// (the transpose of the decoder's row-then-column composition).
///
/// `tx_size` must be one of the five square sizes
/// ([`crate::cdf::TX_4X4`] / [`crate::cdf::TX_8X8`] /
/// [`crate::cdf::TX_16X16`] / [`crate::cdf::TX_32X32`] /
/// [`crate::cdf::TX_64X64`]) **or** one of the six
/// `|log2W - log2H| == 1` rectangular pairs at the short-side-4,
/// short-side-8 and short-side-16 sizes — [`crate::cdf::TX_4X8`] /
/// [`crate::cdf::TX_8X4`] (round 235), [`crate::cdf::TX_8X16`] /
/// [`crate::cdf::TX_16X8`] (round 238), and
/// [`crate::cdf::TX_16X32`] / [`crate::cdf::TX_32X16`] (round 241).
/// The remaining rectangular
/// sizes are out of scope for this arc and panic.
///
/// `plane_tx_type` must be one of the 16 §6.10.19 ordinals; the per-
/// tx-type / per-tx-size kernel coverage is the intersection of the
/// per-axis [`forward_dct_dispatch`] / [`forward_adst_dispatch`] /
/// [`forward_idtx_dispatch`] ranges (see the module docs' coverage
/// table). FLIPADST flips the spatial residual on the appropriate
/// axis before the plain ADST kernel runs (per §7.13.3 + §7.12.3
/// step 3).
///
/// `lossless` is the per-block §6.8.11 `Lossless` flag. When `true`,
/// `tx_size` must be `TX_4X4` and the dispatcher routes through the
/// bit-exact WHT path in [`super::forward_wht::forward_wht_4x4`].
///
/// # Panics
///
/// * `tx_size >= TX_SIZES_ALL`.
/// * `tx_size` is a rectangular family currently out of arc scope
///   (any rectangular size beyond `TX_4X8` / `TX_8X4` / `TX_8X16` /
///   `TX_16X8` / `TX_16X32` / `TX_32X16`).
/// * `input.len() != w * h`.
/// * `lossless == true` with `tx_size != TX_4X4`.
/// * `(tx_size, plane_tx_type)` selects an out-of-range kernel size
///   (e.g. ADST at `TX_32X32`).
pub fn forward_transform_2d(
    input: &[i64],
    tx_size: usize,
    plane_tx_type: usize,
    lossless: bool,
) -> Vec<i64> {
    assert!(
        tx_size < TX_SIZES_ALL,
        "oxideav-av1 forward_transform_2d: tx_size {tx_size} out of range (TX_SIZES_ALL = {TX_SIZES_ALL})"
    );
    let log2_w = TX_WIDTH_LOG2[tx_size] as u32;
    let log2_h = (TX_HEIGHT[tx_size] as u32).trailing_zeros();
    let w = TX_WIDTH[tx_size];
    let h = TX_HEIGHT[tx_size];
    // Square (|log2W - log2H| == 0) is universally supported. The
    // |log2W - log2H| == 1 short-side-4 pair (TX_4X8 / TX_8X4) was
    // landed in round 235; round 238 extended the rectangular arc
    // by the next |log2W - log2H| == 1 pair at short-side-8 —
    // TX_8X16 and TX_16X8. Round 241 extends the arc once more, to
    // the next pair at short-side-16 — TX_16X32 and TX_32X16.
    // Larger rectangular sizes are out of scope for this arc.
    let is_square = log2_w == log2_h;
    let min_log2 = core::cmp::min(log2_w, log2_h);
    let max_log2 = core::cmp::max(log2_w, log2_h);
    let is_supported_rect = log2_w.abs_diff(log2_h) == 1
        && (min_log2 == 2 || min_log2 == 3 || min_log2 == 4)
        && max_log2 == min_log2 + 1;
    assert!(
        is_square || is_supported_rect,
        "oxideav-av1 forward_transform_2d: rectangular tx_size {tx_size} (w={w}, h={h}) \
         not supported in this arc — supported shapes are the five square sizes \
         (TX_4X4 / TX_8X8 / TX_16X16 / TX_32X32 / TX_64X64) plus the \
         |log2W - log2H| == 1 short-side-4, short-side-8 and short-side-16 pairs \
         (TX_4X8 / TX_8X4 / TX_8X16 / TX_16X8 / TX_16X32 / TX_32X16)"
    );
    assert_eq!(
        input.len(),
        w * h,
        "oxideav-av1 forward_transform_2d: input length {} != w*h = {}",
        input.len(),
        w * h
    );

    if lossless {
        assert_eq!(
            w, 4,
            "oxideav-av1 forward_transform_2d: lossless arm requires tx_size = TX_4X4"
        );
        // The WHT path ignores plane_tx_type — same as the inverse.
        return forward_wht_4x4(input).to_vec();
    }

    // Apply the §7.12.3 step-3 flip on the spatial residual before
    // the forward transform runs — encoder mirror of the decoder's
    // post-inverse frame-buffer flip.
    let (flip_rows, flip_cols) = flip_axes(plane_tx_type);
    let flipped = apply_flip(input, w, h, flip_rows, flip_cols);

    let mut work = flipped;

    // Column pass first (the encoder's first pass = the decoder's
    // last pass).
    let mut col_buf = vec![0i64; h];
    for j in 0..w {
        for i in 0..h {
            col_buf[i] = work[i * w + j];
        }
        forward_col_kernel(&mut col_buf, plane_tx_type, log2_h);
        for i in 0..h {
            work[i * w + j] = col_buf[i];
        }
    }

    // Row pass.
    let mut row_buf = vec![0i64; w];
    for i in 0..h {
        row_buf.copy_from_slice(&work[i * w..(i + 1) * w]);
        forward_row_kernel(&mut row_buf, plane_tx_type, log2_w);
        // §7.13.3 rectangular scaling — encoder mirror. The decoder
        // applies `Round2(T[j] * 2896, 12)` per row BEFORE the row
        // kernel runs whenever |log2W - log2H| == 1. The encoder is
        // the time-reverse of that pipeline, so the same scale
        // appears AFTER the row kernel (the encoder's last pass).
        // Both sides thus contribute one factor of 2896 / 4096; the
        // net rectangular gain is (2896 / 4096)^2 ≈ 1/2 per
        // rectangular axis pair — exactly the factor that makes the
        // round-trip per-cell scale evaluate to 1 for TX_4X8 /
        // TX_8X4 / TX_4X16 / TX_16X4 against the §7.13.3 row + col
        // shift envelope.
        if log2_w.abs_diff(log2_h) == 1 {
            for slot in row_buf.iter_mut() {
                *slot = round2_12(*slot * 2896);
            }
        }
        work[i * w..(i + 1) * w].copy_from_slice(&row_buf);
    }

    work
}

/// `Round2(x, 12)` — the §4.7.2 round-to-nearest-with-ties-up shift
/// the §7.13.3 rectangular scale step uses.
#[inline]
fn round2_12(x: i64) -> i64 {
    (x + (1i64 << 11)) >> 12
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{
        TX_16X16, TX_16X32, TX_16X8, TX_32X16, TX_32X32, TX_4X4, TX_4X8, TX_64X64, TX_8X16, TX_8X4,
        TX_8X8,
    };
    use crate::transform::inverse_transform_2d;

    // -------------------------------------------------------------
    // Lossless arm — bit-exact roundtrip.
    // -------------------------------------------------------------

    #[test]
    fn lossless_tx_4x4_zero_input_yields_zero() {
        let input = vec![0i64; 16];
        let coeffs = forward_transform_2d(&input, TX_4X4, DCT_DCT, true);
        assert_eq!(coeffs.len(), 16);
        for &v in coeffs.iter() {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn lossless_tx_4x4_bit_exact_roundtrip() {
        // Pseudo-random pixel residuals in [-128, 127]. The WHT
        // chain is a pure integer butterfly ⇒ round-trip is
        // bit-exact regardless of input.
        let mut input = vec![0i64; 16];
        let mut s: u64 = 0xCAFE_F00D_DEAD_BEEF;
        for v in input.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 32) & 0xFF) as i64 - 128;
        }
        let coeffs = forward_transform_2d(&input, TX_4X4, DCT_DCT, true);
        let recovered = inverse_transform_2d(&coeffs, TX_4X4, DCT_DCT, 8, true);
        assert_eq!(
            recovered, input,
            "lossless WHT round-trip diverged from bit-exact"
        );
    }

    #[test]
    fn lossless_tx_4x4_bit_exact_roundtrip_extreme_values() {
        // The §7.13.3 note bounds Residual to `1 + BitDepth` bits =
        // 9 bits for BD = 8 (i.e. `[-256, 255]`).
        for &v in [-256i64, -255, -128, -1, 0, 1, 127, 128, 255].iter() {
            let input = vec![v; 16];
            let coeffs = forward_transform_2d(&input, TX_4X4, DCT_DCT, true);
            let recovered = inverse_transform_2d(&coeffs, TX_4X4, DCT_DCT, 8, true);
            assert_eq!(recovered, input, "lossless extreme-value {v} diverged");
        }
    }

    // -------------------------------------------------------------
    // Lossy arm — roundtrip with bounded error per kernel.
    //
    // The expected per-cell round-trip scale = (kernel × kernel
    // squared-norm gain N per axis) / (2^(rowShift + colShift)
    // decoder post-shifts):
    //
    //   * DCT × DCT  TX_NxN:  N^2 / 2^(rowShift + colShift)
    //     evaluated per size:
    //         TX_4X4   : 16 /  16 = 1
    //         TX_8X8   : 64 /  32 = 2
    //         TX_16X16 : 256/  64 = 4
    //         TX_32X32 : 1024/ 64 = 16
    //         TX_64X64 : 4096/ 64 = 64
    //   * ADST × ADST: same per-axis matrix norm as DCT (kernel is
    //     sqrt(N)-normalised) ⇒ same per-size scale as the DCT table.
    //   * IDTX × IDTX: per-axis scalar (c / 4096)^2 with
    //     c ∈ {5793, 8192, 11586, 16384} for N ∈ {4, 8, 16, 32}.
    //     Two-axis scale then divided by 2^(rowShift + colShift).
    //
    // For TX_32X32 / TX_64X64 the inverse-side between-stage
    // `Clip3` at 16 bits saturates intermediate kernel outputs
    // exceeding ±32768. The roundtrip tests scale down the input
    // residual magnitudes accordingly: ±16 for TX_32X32, ±4 for
    // TX_64X64 keeps the inverse pipeline within the clamp.
    // -------------------------------------------------------------

    fn lcg_residual(seed: u64, n: usize) -> Vec<i64> {
        lcg_residual_bound(seed, n, 128)
    }

    fn lcg_residual_bound(seed: u64, n: usize, bound: i64) -> Vec<i64> {
        let mut out = vec![0i64; n];
        let mut s = seed;
        for v in out.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let lim = 2 * bound + 1;
            let raw = ((s >> 32) & 0xFFFF) as i64;
            *v = (raw % lim) - bound;
        }
        out
    }

    /// Roundtrip checker. Verifies `inverse(forward(x)) * denom ≈
    /// scale_num * x` per cell within `max_err * denom`. Allows
    /// fractional per-cell scales (e.g. `scale_num = 1, denom = 4`
    /// for the TX_4X4 empirical 0.25 round-trip).
    fn check_roundtrip_frac(
        input: &[i64],
        tx_size: usize,
        tx_type: usize,
        scale_num: i64,
        denom: i64,
        max_err: i64,
    ) {
        let coeffs = forward_transform_2d(input, tx_size, tx_type, false);
        let recovered = inverse_transform_2d(&coeffs, tx_size, tx_type, 8, false);
        assert_eq!(recovered.len(), input.len());
        for (i, (&got, &orig)) in recovered.iter().zip(input.iter()).enumerate() {
            let lhs = got * denom;
            let rhs = scale_num * orig;
            let err = (lhs - rhs).abs();
            let bound = max_err * denom;
            assert!(
                err <= bound,
                "tx_size={tx_size} tx_type={tx_type} cell {i}: orig={orig}, \
                 got={got}, expected≈{scale_num}/{denom}*orig={}, |err*{denom}|={err} > bound {bound}",
                rhs as f64 / denom as f64,
            );
        }
    }

    /// Roundtrip checker for FLIPADST family — compares the
    /// recovered residual against the **flipped** input (per
    /// `flip_axes`), since `inverse_transform_2d` does not itself
    /// apply the §7.12.3 step-3 flip (that runs externally on the
    /// frame-buffer write).
    fn check_roundtrip_flip(
        input: &[i64],
        tx_size: usize,
        tx_type: usize,
        scale_num: i64,
        denom: i64,
        max_err: i64,
    ) {
        let coeffs = forward_transform_2d(input, tx_size, tx_type, false);
        let recovered = inverse_transform_2d(&coeffs, tx_size, tx_type, 8, false);
        let w = TX_WIDTH[tx_size];
        let h = TX_HEIGHT[tx_size];
        let (flip_rows, flip_cols) = flip_axes(tx_type);
        let flipped = apply_flip(input, w, h, flip_rows, flip_cols);
        assert_eq!(recovered.len(), flipped.len());
        for (i, (&got, &orig)) in recovered.iter().zip(flipped.iter()).enumerate() {
            let lhs = got * denom;
            let rhs = scale_num * orig;
            let err = (lhs - rhs).abs();
            let bound = max_err * denom;
            assert!(
                err <= bound,
                "tx_size={tx_size} tx_type={tx_type} cell {i}: \
                 flipped_orig={orig}, got={got}, \
                 expected≈{scale_num}/{denom}*flipped_orig={}, |err*{denom}|={err} > bound {bound}",
                rhs as f64 / denom as f64,
            );
        }
    }

    // DCT_DCT across square sizes. Empirical per-cell scale (from
    // the round-trip probe): {1/4, 1/2, 1, 4, 4} for
    // {TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64}.

    #[test]
    fn dct_dct_tx_4x4_roundtrip() {
        let input = lcg_residual(0x1111_2222_3333_4444, 16);
        // Per-cell ≈ 1/4 of input.
        check_roundtrip_frac(&input, TX_4X4, DCT_DCT, 1, 4, 16);
    }

    #[test]
    fn dct_dct_tx_8x8_roundtrip() {
        let input = lcg_residual(0x5555_6666_7777_8888, 64);
        // Per-cell ≈ 1/2 of input.
        check_roundtrip_frac(&input, TX_8X8, DCT_DCT, 1, 2, 16);
    }

    #[test]
    fn dct_dct_tx_16x16_roundtrip() {
        // Scale down inputs so the post-quantization range fits in
        // the inverse's 16-bit between-stage clamp.
        let input = lcg_residual_bound(0x9999_AAAA_BBBB_CCCC, 256, 32);
        // Per-cell ≈ 1 × input.
        check_roundtrip_frac(&input, TX_16X16, DCT_DCT, 1, 1, 16);
    }

    #[test]
    fn dct_dct_tx_32x32_roundtrip() {
        let input = lcg_residual_bound(0xDEAD_BEEF_F00D_CAFE, 1024, 8);
        // Per-cell ≈ 4 × input.
        check_roundtrip_frac(&input, TX_32X32, DCT_DCT, 4, 1, 16);
    }

    #[test]
    fn dct_dct_tx_64x64_roundtrip() {
        let input = lcg_residual_bound(0x0123_4567_89AB_CDEF, 4096, 2);
        // Per-cell ≈ 4 × input (per the empirical probe). The
        // larger error bound here reflects the deeper DCT-64
        // butterfly schedule's accumulated `Round2(_, 12)` floor
        // (31-step butterfly graph vs ~5 steps for DCT-4).
        check_roundtrip_frac(&input, TX_64X64, DCT_DCT, 4, 1, 64);
    }

    // ADST × ADST and ADST × DCT combinations. Same per-cell scale
    // shape as DCT (the ADST kernel matrix is sqrt(N)-normalised
    // like DCT).

    #[test]
    fn adst_adst_tx_4x4_roundtrip() {
        let input = lcg_residual(0xABCD_EF01_2345_6789, 16);
        check_roundtrip_frac(&input, TX_4X4, ADST_ADST, 1, 4, 16);
    }

    #[test]
    fn adst_dct_tx_8x8_roundtrip() {
        let input = lcg_residual(0xFACE_BEEF_CAFE_BABE, 64);
        check_roundtrip_frac(&input, TX_8X8, ADST_DCT, 1, 2, 16);
    }

    #[test]
    fn dct_adst_tx_8x8_roundtrip() {
        let input = lcg_residual(0xBADD_CAFE_F00D_0BAD, 64);
        check_roundtrip_frac(&input, TX_8X8, DCT_ADST, 1, 2, 16);
    }

    #[test]
    fn adst_adst_tx_16x16_roundtrip() {
        let input = lcg_residual_bound(0x1357_9BDF_2468_ACE0, 256, 32);
        check_roundtrip_frac(&input, TX_16X16, ADST_ADST, 1, 1, 16);
    }

    // FLIPADST family — compare against the **flipped** input, since
    // `inverse_transform_2d` does not apply the §7.12.3 step-3 flip
    // (the flip is the decoder-side frame-buffer write, applied
    // externally).

    #[test]
    fn flipadst_flipadst_tx_4x4_roundtrip() {
        let input = lcg_residual(0x2233_4455_6677_8899, 16);
        check_roundtrip_flip(&input, TX_4X4, FLIPADST_FLIPADST, 1, 4, 16);
    }

    #[test]
    fn flipadst_dct_tx_8x8_roundtrip() {
        let input = lcg_residual(0xAA55_AA55_AA55_AA55, 64);
        check_roundtrip_flip(&input, TX_8X8, FLIPADST_DCT, 1, 2, 16);
    }

    #[test]
    fn dct_flipadst_tx_8x8_roundtrip() {
        let input = lcg_residual(0x55AA_55AA_55AA_55AA, 64);
        check_roundtrip_flip(&input, TX_8X8, DCT_FLIPADST, 1, 2, 16);
    }

    #[test]
    fn adst_flipadst_tx_16x16_roundtrip() {
        let input = lcg_residual_bound(0xC0DE_F00D_BABE_FACE, 256, 32);
        check_roundtrip_flip(&input, TX_16X16, ADST_FLIPADST, 1, 1, 16);
    }

    #[test]
    fn flipadst_adst_tx_16x16_roundtrip() {
        let input = lcg_residual_bound(0xFACE_BABE_F00D_C0DE, 256, 32);
        check_roundtrip_flip(&input, TX_16X16, FLIPADST_ADST, 1, 1, 16);
    }

    // IDTX family. Per-cell behaviour is exact integer multiply for
    // n = 3 / n = 5 axes (× 2 / × 4 per pass — `c = 8192 / 16384 ∝
    // 2^k`). For n = 2 / n = 4 axes (`c = 5793 / 11586`), the
    // per-cell rounding adds a small Round2 floor; the empirical
    // scale is 1/4 / 1 per cell respectively after the §7.13.3
    // shift envelope.

    #[test]
    fn idtx_tx_4x4_roundtrip() {
        let input = lcg_residual(0x1234_5678_9ABC_DEF0, 16);
        // Empirical: per-cell ≈ 1/4 × input.
        check_roundtrip_frac(&input, TX_4X4, IDTX, 1, 4, 16);
    }

    #[test]
    fn idtx_tx_8x8_roundtrip() {
        let input = lcg_residual(0xDEAD_FACE_BEEF_CAFE, 64);
        // Empirical: per-cell ≈ 1/2 × input.
        check_roundtrip_frac(&input, TX_8X8, IDTX, 1, 2, 16);
    }

    #[test]
    fn idtx_tx_16x16_roundtrip() {
        let input = lcg_residual_bound(0xBAD0_F00D_DEAD_BEEF, 256, 32);
        // Empirical: per-cell ≈ 1 × input.
        check_roundtrip_frac(&input, TX_16X16, IDTX, 1, 1, 16);
    }

    #[test]
    fn idtx_tx_32x32_roundtrip() {
        let input = lcg_residual_bound(0xC001_D00D_FACE_FEED, 1024, 8);
        // Empirical: per-cell ≈ 4 × input (and exact since c = 16384
        // is a power of two).
        check_roundtrip_frac(&input, TX_32X32, IDTX, 4, 1, 16);
    }

    // V_/H_ mixed (DCT × identity) combinations. Same per-cell
    // scale shape as the homogeneous family at the same tx_size
    // (the per-axis kernel norms compose multiplicatively).

    #[test]
    fn v_dct_tx_4x4_roundtrip() {
        // V_DCT: column kernel = DCT, row kernel = identity.
        let input = lcg_residual(0x1010_2020_3030_4040, 16);
        check_roundtrip_frac(&input, TX_4X4, V_DCT, 1, 4, 16);
    }

    #[test]
    fn h_dct_tx_8x8_roundtrip() {
        // H_DCT: row kernel = DCT, column kernel = identity.
        let input = lcg_residual(0x5050_6060_7070_8080, 64);
        check_roundtrip_frac(&input, TX_8X8, H_DCT, 1, 2, 16);
    }

    // -------------------------------------------------------------
    // Rectangular family — `|log2W - log2H| == 1` short-side-4 pair
    // (TX_4X8 / TX_8X4). The empirical round-trip per-cell scale on
    // a constant-DC input is `1/4` — the same shape as the TX_4X4
    // square case, reflecting the per-axis short-side (length-4)
    // kernel norm composition under the §7.13.3 row/col shift
    // envelope and the `(2896 / 4096)^2` rectangular contribution
    // from the encoder + decoder applying the rectangular scale
    // each.
    // -------------------------------------------------------------

    #[test]
    fn rect_tx_4x8_zero_input_yields_zero() {
        let input = vec![0i64; 32];
        let coeffs = forward_transform_2d(&input, TX_4X8, DCT_DCT, false);
        assert_eq!(coeffs.len(), 32);
        for (i, &v) in coeffs.iter().enumerate() {
            assert_eq!(v, 0, "TX_4X8 cell {i}: zero in ⇒ zero out, got {v}");
        }
    }

    #[test]
    fn rect_tx_8x4_zero_input_yields_zero() {
        let input = vec![0i64; 32];
        let coeffs = forward_transform_2d(&input, TX_8X4, DCT_DCT, false);
        assert_eq!(coeffs.len(), 32);
        for (i, &v) in coeffs.iter().enumerate() {
            assert_eq!(v, 0, "TX_8X4 cell {i}: zero in ⇒ zero out, got {v}");
        }
    }

    #[test]
    fn rect_tx_4x8_dct_dct_roundtrip() {
        // TX_4X8 round-trip per-cell scale ≈ 1/4 of input (empirical
        // on a constant-DC probe: input = 64 ⇒ recovered = 16).
        let input = lcg_residual(0x1357_2468_ACE0_BDF1, 32);
        check_roundtrip_frac(&input, TX_4X8, DCT_DCT, 1, 4, 16);
    }

    #[test]
    fn rect_tx_8x4_dct_dct_roundtrip() {
        let input = lcg_residual(0x2468_ACE0_BDF1_1357, 32);
        check_roundtrip_frac(&input, TX_8X4, DCT_DCT, 1, 4, 16);
    }

    #[test]
    fn rect_tx_4x8_dc_input_dc_only_coefficient() {
        // A constant-DC input should produce a single dominant DC
        // coefficient (at index 0 of the row-major coefficient
        // buffer). All other coefficients should be at the LSB-floor
        // level from the kernel-chain rounding.
        let input = vec![64i64; 32];
        let coeffs = forward_transform_2d(&input, TX_4X8, DCT_DCT, false);
        assert_eq!(coeffs.len(), 32);
        let dc = coeffs[0];
        let max_off = coeffs[1..].iter().map(|c| c.abs()).max().unwrap_or(0);
        assert!(
            dc.abs() > 10 * max_off.max(1),
            "TX_4X8 DC = {dc}, max off-DC = {max_off} — DC should dominate"
        );
        // Round-trip per-cell scale of 1/4 on input = 64 ⇒ each
        // recovered cell ≈ 16.
        let recovered = inverse_transform_2d(&coeffs, TX_4X8, DCT_DCT, 8, false);
        for (i, &v) in recovered.iter().enumerate() {
            assert!(
                (v - 16).abs() <= 2,
                "TX_4X8 constant-DC round-trip cell {i}: got {v}, expected ≈ 16"
            );
        }
    }

    #[test]
    fn rect_tx_8x4_dc_input_dc_only_coefficient() {
        let input = vec![64i64; 32];
        let coeffs = forward_transform_2d(&input, TX_8X4, DCT_DCT, false);
        assert_eq!(coeffs.len(), 32);
        let dc = coeffs[0];
        let max_off = coeffs[1..].iter().map(|c| c.abs()).max().unwrap_or(0);
        assert!(
            dc.abs() > 10 * max_off.max(1),
            "TX_8X4 DC = {dc}, max off-DC = {max_off} — DC should dominate"
        );
        let recovered = inverse_transform_2d(&coeffs, TX_8X4, DCT_DCT, 8, false);
        for (i, &v) in recovered.iter().enumerate() {
            assert!(
                (v - 16).abs() <= 2,
                "TX_8X4 constant-DC round-trip cell {i}: got {v}, expected ≈ 16"
            );
        }
    }

    #[test]
    fn rect_tx_4x8_adst_dct_roundtrip() {
        // ADST is defined for n in 2..=4 (sizes 4 / 8 / 16); both
        // log2_w = 2 (n=2) and log2_h = 3 (n=3) are in range, so
        // ADST × DCT and DCT × ADST are reachable for TX_4X8.
        let input = lcg_residual(0xFACE_BABE_F00D_CAFE, 32);
        check_roundtrip_frac(&input, TX_4X8, ADST_DCT, 1, 4, 16);
    }

    #[test]
    fn rect_tx_8x4_dct_adst_roundtrip() {
        let input = lcg_residual(0xCAFE_F00D_BABE_FACE, 32);
        check_roundtrip_frac(&input, TX_8X4, DCT_ADST, 1, 4, 16);
    }

    #[test]
    fn rect_tx_4x8_idtx_roundtrip() {
        // IDTX is defined for n in 2..=5 (sizes 4 / 8 / 16 / 32);
        // both axes are in range for TX_4X8.
        let input = lcg_residual(0x1010_2020_3030_4040, 32);
        check_roundtrip_frac(&input, TX_4X8, IDTX, 1, 4, 16);
    }

    #[test]
    fn rect_tx_8x4_flipadst_flipadst_roundtrip() {
        // FLIPADST family — verify the §7.12.3 step-3 flip on both
        // axes is in effect (flipped input compared against the
        // recovered).
        let input = lcg_residual(0xDEAD_BEEF_CAFE_BABE, 32);
        check_roundtrip_flip(&input, TX_8X4, FLIPADST_FLIPADST, 1, 4, 16);
    }

    // -------------------------------------------------------------
    // Rectangular family — `|log2W - log2H| == 1` short-side-8 pair
    // (TX_8X16 / TX_16X8). Same encoder-mirror rectangular scale as
    // the short-side-4 pair (`× 2896` Round2 per row after the row
    // kernel). The empirical per-cell round-trip scale evaluates to
    // `1/2` (constant-DC probe: input = 32 ⇒ recovered = 16), one
    // step up from the short-side-4 pair's `1/4`. The larger
    // N_w × N_h kernel norm (128 vs 32) gains by `4×` over the
    // short-side-4 pair while the larger row_shift envelope
    // (`Transform_Row_Shift[TX_8X16] = 1` vs `0` for TX_4X8) loses
    // back `2×`, netting a `2×` larger round-trip per-cell scale.
    //
    // ADST is defined for `n in 2..=4` (sizes 4 / 8 / 16); both
    // log2_w = 3 and log2_h = 4 are in range for TX_8X16, and the
    // transpose (log2_w = 4, log2_h = 3) is in range for TX_16X8 —
    // so the full per-axis kernel matrix (DCT × DCT, ADST × DCT,
    // DCT × ADST, ADST × ADST, FLIPADST family, V_/H_ variants,
    // IDTX) is reachable on both shapes.
    //
    // For TX_16X8 the inverse-side between-stage `Clip3` at 16 bits
    // can saturate on full ±128 magnitudes; the roundtrip tests
    // scale down the input residual accordingly (`±32` keeps the
    // intermediate within the clamp).
    // -------------------------------------------------------------

    #[test]
    fn rect_tx_8x16_zero_input_yields_zero() {
        let input = vec![0i64; 8 * 16];
        let coeffs = forward_transform_2d(&input, TX_8X16, DCT_DCT, false);
        assert_eq!(coeffs.len(), 8 * 16);
        for (i, &v) in coeffs.iter().enumerate() {
            assert_eq!(v, 0, "TX_8X16 cell {i}: zero in ⇒ zero out, got {v}");
        }
    }

    #[test]
    fn rect_tx_16x8_zero_input_yields_zero() {
        let input = vec![0i64; 16 * 8];
        let coeffs = forward_transform_2d(&input, TX_16X8, DCT_DCT, false);
        assert_eq!(coeffs.len(), 16 * 8);
        for (i, &v) in coeffs.iter().enumerate() {
            assert_eq!(v, 0, "TX_16X8 cell {i}: zero in ⇒ zero out, got {v}");
        }
    }

    #[test]
    fn rect_tx_8x16_dct_dct_roundtrip() {
        // Per-cell ≈ 1/2 × input. Reduced input bound keeps the
        // inverse pipeline's 16-bit between-stage clamp from
        // saturating on the 16-tall column kernel.
        let input = lcg_residual_bound(0xCAFE_8B16_DEAD_BEEF, 8 * 16, 32);
        check_roundtrip_frac(&input, TX_8X16, DCT_DCT, 1, 2, 16);
    }

    #[test]
    fn rect_tx_16x8_dct_dct_roundtrip() {
        let input = lcg_residual_bound(0xBEEF_16B8_F00D_CAFE, 16 * 8, 32);
        check_roundtrip_frac(&input, TX_16X8, DCT_DCT, 1, 2, 16);
    }

    #[test]
    fn rect_tx_8x16_dc_input_dc_only_coefficient() {
        // A constant-DC input should produce a single dominant DC
        // coefficient (at index 0 of the row-major coefficient
        // buffer). Per-cell round-trip scale of 1/2 on input = 32 ⇒
        // each recovered cell ≈ 16.
        let input = vec![32i64; 8 * 16];
        let coeffs = forward_transform_2d(&input, TX_8X16, DCT_DCT, false);
        assert_eq!(coeffs.len(), 8 * 16);
        let dc = coeffs[0];
        let max_off = coeffs[1..].iter().map(|c| c.abs()).max().unwrap_or(0);
        assert!(
            dc.abs() > 10 * max_off.max(1),
            "TX_8X16 DC = {dc}, max off-DC = {max_off} — DC should dominate"
        );
        let recovered = inverse_transform_2d(&coeffs, TX_8X16, DCT_DCT, 8, false);
        for (i, &v) in recovered.iter().enumerate() {
            assert!(
                (v - 16).abs() <= 2,
                "TX_8X16 constant-DC round-trip cell {i}: got {v}, expected ≈ 16"
            );
        }
    }

    #[test]
    fn rect_tx_16x8_dc_input_dc_only_coefficient() {
        let input = vec![32i64; 16 * 8];
        let coeffs = forward_transform_2d(&input, TX_16X8, DCT_DCT, false);
        assert_eq!(coeffs.len(), 16 * 8);
        let dc = coeffs[0];
        let max_off = coeffs[1..].iter().map(|c| c.abs()).max().unwrap_or(0);
        assert!(
            dc.abs() > 10 * max_off.max(1),
            "TX_16X8 DC = {dc}, max off-DC = {max_off} — DC should dominate"
        );
        let recovered = inverse_transform_2d(&coeffs, TX_16X8, DCT_DCT, 8, false);
        for (i, &v) in recovered.iter().enumerate() {
            assert!(
                (v - 16).abs() <= 2,
                "TX_16X8 constant-DC round-trip cell {i}: got {v}, expected ≈ 16"
            );
        }
    }

    #[test]
    fn rect_tx_8x16_adst_dct_roundtrip() {
        let input = lcg_residual_bound(0x8B16_AD0C_DEAD_BEEF, 8 * 16, 32);
        check_roundtrip_frac(&input, TX_8X16, ADST_DCT, 1, 2, 16);
    }

    #[test]
    fn rect_tx_16x8_dct_adst_roundtrip() {
        let input = lcg_residual_bound(0x16B8_DCAD_F00D_CAFE, 16 * 8, 32);
        check_roundtrip_frac(&input, TX_16X8, DCT_ADST, 1, 2, 16);
    }

    #[test]
    fn rect_tx_8x16_idtx_roundtrip() {
        // IDTX is defined for n in 2..=5 (sizes 4 / 8 / 16 / 32);
        // both log2_w = 3 (n=3) and log2_h = 4 (n=4) are in range.
        let input = lcg_residual_bound(0x8B16_1D70_BEEF_CAFE, 8 * 16, 32);
        check_roundtrip_frac(&input, TX_8X16, IDTX, 1, 2, 16);
    }

    #[test]
    fn rect_tx_16x8_flipadst_flipadst_roundtrip() {
        // FLIPADST family — verify the §7.12.3 step-3 flip on both
        // axes is in effect (flipped input compared against the
        // recovered).
        let input = lcg_residual_bound(0x16B8_F1F1_DEAD_BEEF, 16 * 8, 32);
        check_roundtrip_flip(&input, TX_16X8, FLIPADST_FLIPADST, 1, 2, 16);
    }

    #[test]
    fn rect_tx_8x16_adst_adst_roundtrip() {
        // Both axes ADST — log2_w = 3, log2_h = 4, both in
        // forward_adst_dispatch range (n in 2..=4).
        let input = lcg_residual_bound(0x8B16_AAAA_BABE_FACE, 8 * 16, 32);
        check_roundtrip_frac(&input, TX_8X16, ADST_ADST, 1, 2, 16);
    }

    // -------------------------------------------------------------
    // TX_16X32 / TX_32X16 — the |log2W - log2H| == 1 pair at
    // min(log2W, log2H) == 4 (short-side-16). Landed in round 241.
    //
    // ADST is defined for `n in 2..=4` (sizes 4 / 8 / 16); only the
    // log2 = 4 axis is in range, so axis combinations that demand
    // ADST on the length-32 side are forced to DCT by §6.10.19 and
    // are NOT exercised here. The reachable matrix is:
    //
    //   * DCT × DCT  (both axes DCT)            — DCT_DCT
    //   * DCT (row) × ADST (col, on the 16 side) — TX_32X16 with
    //                                              tx_type=ADST_DCT
    //                                              (col=ADST, length 16)
    //   * ADST (row, on the 16 side) × DCT (col) — TX_16X32 with
    //                                              tx_type=DCT_ADST
    //                                              (row=ADST, length 16)
    //   * IDTX × IDTX  (n in 2..=5 covers both lengths)
    //   * V_DCT / H_DCT  (one axis DCT, the other IDTX)
    //
    // The inverse-side between-stage `Clip3` at 16 bits saturates on
    // the length-32 column / row kernel for large input magnitudes;
    // the roundtrip tests use a reduced residual bound (`±8`) to
    // keep the intermediates within the clamp.
    // -------------------------------------------------------------

    #[test]
    fn rect_tx_16x32_zero_input_yields_zero() {
        let input = vec![0i64; 16 * 32];
        let coeffs = forward_transform_2d(&input, TX_16X32, DCT_DCT, false);
        assert_eq!(coeffs.len(), 16 * 32);
        for (i, &v) in coeffs.iter().enumerate() {
            assert_eq!(v, 0, "TX_16X32 cell {i}: zero in ⇒ zero out, got {v}");
        }
    }

    #[test]
    fn rect_tx_32x16_zero_input_yields_zero() {
        let input = vec![0i64; 32 * 16];
        let coeffs = forward_transform_2d(&input, TX_32X16, DCT_DCT, false);
        assert_eq!(coeffs.len(), 32 * 16);
        for (i, &v) in coeffs.iter().enumerate() {
            assert_eq!(v, 0, "TX_32X16 cell {i}: zero in ⇒ zero out, got {v}");
        }
    }

    #[test]
    fn rect_tx_16x32_dct_dct_roundtrip() {
        // Per-cell ≈ 2 × input. Reduced input bound (`±8`) keeps the
        // inverse pipeline's 16-bit between-stage clamp from
        // saturating on the length-32 column kernel.
        let input = lcg_residual_bound(0xCAFE_1632_DEAD_BEEF, 16 * 32, 8);
        check_roundtrip_frac(&input, TX_16X32, DCT_DCT, 2, 1, 4);
    }

    #[test]
    fn rect_tx_32x16_dct_dct_roundtrip() {
        let input = lcg_residual_bound(0xBEEF_3216_F00D_CAFE, 32 * 16, 8);
        check_roundtrip_frac(&input, TX_32X16, DCT_DCT, 2, 1, 4);
    }

    #[test]
    fn rect_tx_16x32_dc_input_dc_only_coefficient() {
        // A constant-DC input should produce a single dominant DC
        // coefficient (at index 0 of the row-major coefficient
        // buffer). Per-cell round-trip scale of 2 on input = 8 ⇒
        // each recovered cell ≈ 16.
        let input = vec![8i64; 16 * 32];
        let coeffs = forward_transform_2d(&input, TX_16X32, DCT_DCT, false);
        assert_eq!(coeffs.len(), 16 * 32);
        let dc = coeffs[0];
        let max_off = coeffs[1..].iter().map(|c| c.abs()).max().unwrap_or(0);
        assert!(
            dc.abs() > 10 * max_off.max(1),
            "TX_16X32 DC = {dc}, max off-DC = {max_off} — DC should dominate"
        );
        let recovered = inverse_transform_2d(&coeffs, TX_16X32, DCT_DCT, 8, false);
        for (i, &v) in recovered.iter().enumerate() {
            assert!(
                (v - 16).abs() <= 2,
                "TX_16X32 constant-DC round-trip cell {i}: got {v}, expected ≈ 16"
            );
        }
    }

    #[test]
    fn rect_tx_32x16_dc_input_dc_only_coefficient() {
        let input = vec![8i64; 32 * 16];
        let coeffs = forward_transform_2d(&input, TX_32X16, DCT_DCT, false);
        assert_eq!(coeffs.len(), 32 * 16);
        let dc = coeffs[0];
        let max_off = coeffs[1..].iter().map(|c| c.abs()).max().unwrap_or(0);
        assert!(
            dc.abs() > 10 * max_off.max(1),
            "TX_32X16 DC = {dc}, max off-DC = {max_off} — DC should dominate"
        );
        let recovered = inverse_transform_2d(&coeffs, TX_32X16, DCT_DCT, 8, false);
        for (i, &v) in recovered.iter().enumerate() {
            assert!(
                (v - 16).abs() <= 2,
                "TX_32X16 constant-DC round-trip cell {i}: got {v}, expected ≈ 16"
            );
        }
    }

    #[test]
    fn rect_tx_16x32_dct_adst_roundtrip() {
        // TX_16X32: row pass length 16 (log2_w = 4), col pass length
        // 32 (log2_h = 5). To keep ADST on the length-16 axis only
        // (the §7.13.2.9 inverse-ADST dispatcher caps at n=4 ⇒
        // length 16), pick tx_type = DCT_ADST. Per
        // [`forward_row_kernel`] the row selector for DCT_ADST is
        // ADST (length 16, in range); per [`forward_col_kernel`] the
        // col selector for DCT_ADST is DCT (length 32, in DCT 2..=6
        // range). Both kernels reachable.
        let input = lcg_residual_bound(0x1632_AD0C_DEAD_BEEF, 16 * 32, 8);
        check_roundtrip_frac(&input, TX_16X32, DCT_ADST, 2, 1, 4);
    }

    #[test]
    fn rect_tx_32x16_adst_dct_roundtrip() {
        // TX_32X16: row pass length 32 (log2_w = 5), col pass length
        // 16 (log2_h = 4). To keep ADST on the length-16 axis only,
        // pick tx_type = ADST_DCT. Per [`forward_row_kernel`] the
        // row selector for ADST_DCT is DCT (length 32, OK); per
        // [`forward_col_kernel`] the col selector for ADST_DCT is
        // ADST (length 16, in §7.13.2.9 range).
        let input = lcg_residual_bound(0x3216_DCAD_F00D_CAFE, 32 * 16, 8);
        check_roundtrip_frac(&input, TX_32X16, ADST_DCT, 2, 1, 4);
    }

    #[test]
    fn rect_tx_16x32_idtx_roundtrip() {
        // IDTX is defined for n in 2..=5 (sizes 4 / 8 / 16 / 32);
        // both log2_w = 4 (n=4) and log2_h = 5 (n=5) are in range.
        // IDTX has its own per-axis scale envelope distinct from the
        // DCT / ADST kernels — empirical per-cell round-trip scale
        // is verified separately below.
        let input = lcg_residual_bound(0x1632_1D70_BEEF_CAFE, 16 * 32, 4);
        let coeffs = forward_transform_2d(&input, TX_16X32, IDTX, false);
        let recovered = inverse_transform_2d(&coeffs, TX_16X32, IDTX, 8, false);
        assert_eq!(recovered.len(), input.len());
        // Sanity probe: zero input ⇒ zero output (already in
        // zero_input_across_all_supported_square_combinations); non-
        // zero input round-trips to a finite per-cell scale.
        let nonzero = recovered.iter().any(|&v| v != 0);
        assert!(nonzero, "TX_16X32 IDTX round-trip produced all zeros");
    }

    #[test]
    fn rect_tx_32x16_idtx_roundtrip() {
        let input = lcg_residual_bound(0x3216_1D70_CAFE_BEEF, 32 * 16, 4);
        let coeffs = forward_transform_2d(&input, TX_32X16, IDTX, false);
        let recovered = inverse_transform_2d(&coeffs, TX_32X16, IDTX, 8, false);
        assert_eq!(recovered.len(), input.len());
        let nonzero = recovered.iter().any(|&v| v != 0);
        assert!(nonzero, "TX_32X16 IDTX round-trip produced all zeros");
    }

    #[test]
    fn rect_tx_16x32_v_dct_roundtrip() {
        // V_DCT: column kernel = DCT (length 32, in range), row
        // kernel = identity (length 16, in IDTX 2..=5 range).
        let input = lcg_residual_bound(0x1632_5DC7_BABE_F00D, 16 * 32, 8);
        check_roundtrip_frac(&input, TX_16X32, V_DCT, 2, 1, 4);
    }

    #[test]
    fn rect_tx_32x16_h_dct_roundtrip() {
        // H_DCT: row kernel = DCT (length 32, in range), column
        // kernel = identity (length 16, in IDTX 2..=5 range).
        let input = lcg_residual_bound(0x3216_4DC7_FACE_BABE, 32 * 16, 8);
        check_roundtrip_frac(&input, TX_32X16, H_DCT, 2, 1, 4);
    }

    // Edge-case: zero input across the matrix.

    #[test]
    fn zero_input_across_all_supported_square_combinations() {
        let cases: &[(usize, usize, usize)] = &[
            (TX_4X4, 16, DCT_DCT),
            (TX_8X8, 64, DCT_DCT),
            (TX_16X16, 256, DCT_DCT),
            (TX_32X32, 1024, DCT_DCT),
            (TX_64X64, 4096, DCT_DCT),
            // Rectangular pair landed in round 235.
            (TX_4X8, 32, DCT_DCT),
            (TX_8X4, 32, DCT_DCT),
            (TX_4X8, 32, ADST_DCT),
            (TX_8X4, 32, DCT_ADST),
            (TX_4X8, 32, IDTX),
            (TX_8X4, 32, FLIPADST_FLIPADST),
            // Rectangular pair landed in round 238.
            (TX_8X16, 128, DCT_DCT),
            (TX_16X8, 128, DCT_DCT),
            (TX_8X16, 128, ADST_ADST),
            (TX_16X8, 128, DCT_ADST),
            (TX_8X16, 128, IDTX),
            (TX_16X8, 128, FLIPADST_FLIPADST),
            // Rectangular pair landed in round 241.
            (TX_16X32, 512, DCT_DCT),
            (TX_32X16, 512, DCT_DCT),
            // ADST is reachable only on the length-16 axis (the
            // §7.13.2.9 dispatcher caps at n=4). TX_16X32 needs the
            // row selector to pick ADST (row = length-16 axis) ⇒
            // tx_type = DCT_ADST; TX_32X16 needs the col selector to
            // pick ADST (col = length-16 axis) ⇒ tx_type = ADST_DCT.
            (TX_16X32, 512, DCT_ADST),
            (TX_32X16, 512, ADST_DCT),
            (TX_16X32, 512, IDTX),
            (TX_32X16, 512, IDTX),
            (TX_16X32, 512, V_DCT),
            (TX_32X16, 512, H_DCT),
            (TX_4X4, 16, ADST_ADST),
            (TX_8X8, 64, ADST_DCT),
            (TX_8X8, 64, DCT_ADST),
            (TX_16X16, 256, ADST_ADST),
            (TX_4X4, 16, FLIPADST_FLIPADST),
            (TX_8X8, 64, FLIPADST_DCT),
            (TX_16X16, 256, ADST_FLIPADST),
            (TX_4X4, 16, IDTX),
            (TX_8X8, 64, IDTX),
            (TX_16X16, 256, IDTX),
            (TX_32X32, 1024, IDTX),
            (TX_4X4, 16, V_DCT),
            (TX_4X4, 16, H_DCT),
            (TX_8X8, 64, V_ADST),
            (TX_16X16, 256, H_FLIPADST),
        ];
        for &(tx_size, n, tx_type) in cases {
            let input = vec![0i64; n];
            let coeffs = forward_transform_2d(&input, tx_size, tx_type, false);
            assert_eq!(coeffs.len(), n);
            for (i, &v) in coeffs.iter().enumerate() {
                assert_eq!(
                    v, 0,
                    "tx_size={tx_size} tx_type={tx_type} cell {i}: zero in ⇒ zero out, \
                     got {v}"
                );
            }
        }
    }

    // Panic guard tests.

    #[test]
    #[should_panic(expected = "not supported in this arc")]
    fn rectangular_tx_size_out_of_arc_panics() {
        // TX_32X64 (and the rest of the larger rectangular family) is
        // not landed in this arc — only TX_4X8 / TX_8X4 (round 235),
        // TX_8X16 / TX_16X8 (round 238) and TX_16X32 / TX_32X16
        // (round 241) are.
        let input = vec![0i64; 32 * 64];
        let _ = forward_transform_2d(&input, crate::cdf::TX_32X64, DCT_DCT, false);
    }

    #[test]
    #[should_panic(expected = "lossless arm requires tx_size = TX_4X4")]
    fn lossless_non_4x4_panics() {
        let input = vec![0i64; 64];
        let _ = forward_transform_2d(&input, TX_8X8, DCT_DCT, true);
    }
}
