//! Forward quantization primitive — the encoder counterpart of the
//! §7.12.3 step-1 dequantization loop in
//! [`crate::cdf::dequantize_step1`].
//!
//! ## Why "forward"
//!
//! The §7.12.3 step-1 dequant maps **`Quant[]` levels → `Dequant[][]`
//! reconstructed coefficients** by multiplying each `Quant[i*w+j]` by a
//! position-dependent quantizer `q2` (with optional §9.5.3
//! `Quantizer_Matrix` deviation) and dividing by `dqDenom`. The
//! encoder needs the inverse direction — **coefficients (post-forward
//! transform) → `Quant[]` levels** — so the §5.11.39 coefficient
//! writers (round 212-215) have an integer-rounded `Quant[]` to consume
//! from a pixel-space encoder.
//!
//! ## Derivation (clean-room)
//!
//! Restating the §7.12.3 step-1 body (av1-spec p.294-295) at the
//! `(i, j)` position with `i < th && j < tw`:
//!
//! ```text
//!   q   = (i == 0 && j == 0) ? get_dc_quant(plane) : get_ac_quant(plane)
//!   q2  = qm_active ? Round2(q * Quantizer_Matrix[...], 5) : q
//!   dq  = Quant[i*w+j] * q2
//!   dq2 = sign(dq) * (|dq| & 0xFF_FFFF) / dqDenom
//!   Dequant[i][j] = Clip3(-(1<<(7+BitDepth)), (1<<(7+BitDepth))-1, dq2)
//! ```
//!
//! For typical encoder `Quant` magnitudes (`< 2^16`) and the spec's
//! `q2 < 2^8` regime the product `|Quant * q2|` stays under `2^24`, so
//! the spec's `& 0xFF_FFFF` mask is a no-op and the body simplifies to
//!
//! ```text
//!   Dequant[i][j] = sign(Quant) * (|Quant| * q2 / dqDenom)
//! ```
//!
//! with the divide round-toward-zero (C's `/` on signed values, but
//! `|Quant|` and `q2 / dqDenom` are non-negative so the divide is a
//! plain unsigned floor). To invert, given a target post-clip
//! `Dequant[i][j]` and a `q2`, we solve for the integer `Quant[i*w+j]`
//! such that re-running the spec body reproduces the target.
//!
//! The mathematically exact solution
//!
//! ```text
//!   Quant = round(Dequant * dqDenom / q2)
//! ```
//!
//! using round-half-away-from-zero produces the nearest integer to the
//! analytic ratio. Because the spec's divide truncates rather than
//! rounds, the dequant of `Quant` may differ from the target by up to
//! `(q2 / dqDenom) - 1` cells. For test inputs that are exact
//! multiples of `q2 / dqDenom` this is a zero-error roundtrip; for
//! arbitrary inputs it bounds the per-coefficient error by one
//! quantization step.
//!
//! ## Scope (arc 14 / round 220)
//!
//! Single function [`forward_quantize`]: takes a `coeffs` buffer
//! (output of a forward transform) plus the §7.12.2 / §7.12.3
//! quantizer state ([`crate::cdf::QuantizerParams`] + per-block
//! `plane` / `segment_id` / `plane_tx_type` / `seg_qm_level`) and
//! returns the dense `Tx_Width[tx] * Tx_Height[tx]` `Quant[]` array.
//! Cells outside the `(th, tw)` active region — where `tw = Min(32,
//! Tx_Width)` and `th = Min(32, Tx_Height)` — are forced to zero
//! exactly as the decoder's §5.11.39 pre-loop default produces. The
//! `qm_active` predicate, `q2` derivation, and `dqDenom` lookup are
//! reused from [`crate::cdf`] so the encoder reads the same tables
//! the decoder reads.

use crate::cdf::{
    dequant_denom, get_ac_quant, get_dc_quant, QuantizerParams, IDTX, TX_HEIGHT, TX_SIZES_ALL,
    TX_WIDTH,
};
use crate::qmatrix;

/// Forward quantization — the inverse of [`crate::cdf::dequantize_step1`].
///
/// Given a `coeffs` buffer in row-major order (length `Tx_Width[tx_size]
/// * Tx_Height[tx_size]`, freshly produced by a forward transform), the
/// per-frame [`QuantizerParams`], and the per-block `plane`,
/// `segment_id`, `plane_tx_type`, `seg_qm_level` selectors, return the
/// `Quant[]` array a §5.11.39 `coefficients()` writer would emit.
///
/// The mapping is the §7.12.3 step-1 body solved for `Quant`:
///
/// ```text
///   Quant[i*w+j] = round_half_away(coeffs[i*w+j] * dqDenom / q2)
/// ```
///
/// where `q2`, `dqDenom`, and the `qm_active` predicate are derived
/// the same way the §7.12.3 step-1 dispatcher derives them. Coefficient
/// positions `(i, j)` with `i >= th` or `j >= tw` (the `tw = Min(32,
/// Tx_Width)` / `th = Min(32, Tx_Height)` clamp) are forced to zero —
/// the §5.11.39 walker initialises `Quant[]` to zero before the
/// reverse-scan loop and the spec's step-1 only ever overwrites the
/// active region. Forcing them to zero here keeps the produced
/// `Quant[]` consistent with the decoder's pre-loop default.
///
/// # Bit-exact roundtrip
///
/// For any `Quant[]` whose dequant (via [`crate::cdf::dequantize_step1`])
/// produces in-range values (no `Clip3` clamp triggered), feeding that
/// dequant array as `coeffs` here reproduces the original `Quant[]`
/// bit-exact. This is the encoder's correctness contract: the encoder
/// commits to a `Quant[]` whose dequant matches the coefficients it
/// wants the decoder to reconstruct.
///
/// For arbitrary `coeffs` not aligned to the `q2 / dqDenom` lattice,
/// the roundtrip introduces at most one quantization step of error per
/// coefficient. The §7.12.3 step-1 truncation `(|dq| & 0xFF_FFFF) /
/// dqDenom` rounds toward zero; the forward quantizer here rounds
/// half-away-from-zero, biasing the recovered `Quant[]` upward in
/// magnitude by half a quantizer step relative to the analytic ratio.
/// This bias balances out the decoder's downward (toward-zero) rounding
/// so that `dequant(forward_quantize(c))` is the nearest reachable
/// lattice point to `c`.
///
/// # Arguments
///
/// * `coeffs` — post-forward-transform coefficient buffer
///   (`Tx_Width[tx_size] * Tx_Height[tx_size]` cells, row-major).
/// * `tx_size` — the per-TU `TX_SIZES_ALL` ordinal.
/// * `plane` — `0` (Y), `1` (U), or `2` (V).
/// * `segment_id` — `0..MAX_SEGMENTS`. Drives `get_qindex`.
/// * `plane_tx_type` — the §5.11.40 `PlaneTxType` derivation
///   (`DCT_DCT..H_FLIPADST`). Only the `< IDTX` guard on the QM-active
///   branch consults it.
/// * `seg_qm_level` — `SegQMLevel[plane][segment_id]` from §6.8.13.
///   `15` (or any value `>= 15`) takes the no-QM identity branch.
/// * `quant` — the per-frame [`QuantizerParams`].
///
/// # Panics
///
/// Panics if `tx_size >= TX_SIZES_ALL` or `coeffs.len() != Tx_Width[
/// tx_size] * Tx_Height[tx_size]`.
#[allow(clippy::too_many_arguments)]
pub fn forward_quantize(
    coeffs: &[i64],
    tx_size: usize,
    plane: u8,
    segment_id: u8,
    plane_tx_type: usize,
    seg_qm_level: u8,
    quant: &QuantizerParams,
) -> Vec<i32> {
    assert!(
        tx_size < TX_SIZES_ALL,
        "oxideav-av1 forward_quantize: tx_size {tx_size} out of range (TX_SIZES_ALL = {TX_SIZES_ALL})"
    );
    let w = TX_WIDTH[tx_size];
    let h = TX_HEIGHT[tx_size];
    assert_eq!(
        coeffs.len(),
        w * h,
        "oxideav-av1 forward_quantize: coeffs length {} != Tx_Width[{tx_size}] * Tx_Height[{tx_size}] = {}",
        coeffs.len(),
        w * h
    );

    let tw = core::cmp::min(32, w);
    let th = core::cmp::min(32, h);
    let dq_denom = dequant_denom(tx_size);
    let dc_q_val = get_dc_quant(quant, plane, segment_id) as i64;
    let ac_q_val = get_ac_quant(quant, plane, segment_id) as i64;
    let qm_active = quant.using_qmatrix && plane_tx_type < IDTX && seg_qm_level < 15;

    let mut levels = vec![0i32; w * h];
    for i in 0..th {
        for j in 0..tw {
            let q = if i == 0 && j == 0 { dc_q_val } else { ac_q_val };
            let q2 = if qm_active {
                let qm = qmatrix::qmatrix_value(seg_qm_level, plane, tx_size, i, j) as i64;
                // Round2(q * qm, 5) per §3 p.13: (x + (1 << 4)) >> 5.
                (q * qm + 16) >> 5
            } else {
                q
            };
            // q2 must be positive; the spec's DC/AC lookup tables are
            // strictly positive (`Dc_Qlookup[0][0] = 4`,
            // `Ac_Qlookup[0][0] = 4`) and the QM-active branch's
            // multiplier is in `30..=242`. Guard defensively so a
            // pathological caller-supplied `q2 == 0` does not divide
            // by zero (output a zero level — the §7.12.3 step-3
            // reconstruction would have produced a zero anyway).
            if q2 == 0 {
                levels[i * w + j] = 0;
                continue;
            }
            // Invert dq2 = sign * (|dq| & 0xFF_FFFF) / dq_denom by
            // solving |Quant| = round_half_away(|coeff| * dq_denom /
            // q2). Sign is carried from `coeff`.
            let c = coeffs[i * w + j];
            let sign: i64 = if c < 0 { -1 } else { 1 };
            let abs_c = c.unsigned_abs() as i64;
            // Round-half-away-from-zero: `(2 * |c| * dq_denom + q2)
            // / (2 * q2)` is a single-divide form that matches the
            // analytic round-to-nearest tie-breaking-up rule on the
            // positive ratio. Using the simpler `(|c| * dq_denom +
            // (q2 / 2)) / q2` produces the same value for q2 even
            // (which is the spec's regime — DC/AC lookups are all
            // even integers in row 0) but biases tie-break by `1`
            // when `q2` is odd. The two-step form is robust to both.
            let num = 2 * abs_c * dq_denom + q2;
            let den = 2 * q2;
            let mag = num / den;
            // The spec's truncation `(|dq| & 0xFF_FFFF) / dq_denom`
            // also bounds the recovered magnitude: a `Quant` whose
            // product `Quant * q2` overflows 24 bits would have its
            // dequant wrap. We don't model wrap here — callers feeding
            // a coefficient buffer from a real forward transform stay
            // well under the bound. Detection is the §6.10.34 /
            // §5.11.39 conformance check, not the quantizer's
            // responsibility.
            levels[i * w + j] = (sign * mag) as i32;
        }
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{dequantize_step1, DCT_DCT, TX_4X4, TX_8X8};

    /// At `q_index = 0` the DC + AC quantizers both equal `4` (the
    /// first cell of `Dc_Qlookup[0]` and `Ac_Qlookup[0]`), `dqDenom =
    /// 1` for TX_4X4, no QM. A `Quant` array whose dequant is exactly
    /// `4 * Quant` should round-trip bit-exactly: any `coeff` divisible
    /// by `4` recovers its `coeff / 4` as `Quant`.
    #[test]
    fn forward_quantize_lossless_q_index_zero_tx4x4() {
        let qp = QuantizerParams::neutral(0, 8);
        let coeffs: [i64; 16] = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64];
        let levels = forward_quantize(&coeffs, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        assert_eq!(
            levels,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        // Round-trip through the decoder's §7.12.3 step-1 reproduces
        // the input coefficients exactly.
        let dequant = dequantize_step1(&levels, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        assert_eq!(dequant, coeffs.to_vec());
    }

    /// Negative coefficients round-trip through the sign-preserving
    /// forward quantizer. The decoder's `sign * (|dq| & 0xFF_FFFF) /
    /// dq_denom` rebuilds the same sign on the dequant side.
    #[test]
    fn forward_quantize_preserves_sign_lossless() {
        let qp = QuantizerParams::neutral(0, 8);
        let coeffs: [i64; 16] = [
            -4, 8, -12, 16, -20, 24, -28, 32, -36, 40, -44, 48, -52, 56, -60, 64,
        ];
        let levels = forward_quantize(&coeffs, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        let dequant = dequantize_step1(&levels, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        assert_eq!(dequant, coeffs.to_vec());
    }

    /// At `q_index = 0` `q = 4`. A coefficient not divisible by 4
    /// rounds to the nearest multiple-of-4 lattice point under
    /// `dequant(forward_quantize(_))`. The forward quantizer rounds
    /// the level half-away-from-zero, so `coeff = 6` → `Quant = 2` →
    /// `Dequant = 8` (nearest reachable lattice point above zero,
    /// because the spec's truncation toward zero combines with the
    /// half-away rounding to land on `ceil(coeff / 4) * 4` for
    /// half-integer ties).
    #[test]
    fn forward_quantize_lossy_rounds_half_away_from_zero() {
        let qp = QuantizerParams::neutral(0, 8);
        let coeffs: [i64; 16] = [
            6, // ties-up to 8 (Quant = 2)
            5, // 5/4 = 1.25 -> Quant = 1, Dequant = 4
            7, // 7/4 = 1.75 -> Quant = 2, Dequant = 8
            -6, -5, -7, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0,
        ];
        let levels = forward_quantize(&coeffs, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        assert_eq!(levels[0], 2);
        assert_eq!(levels[1], 1);
        assert_eq!(levels[2], 2);
        assert_eq!(levels[3], -2);
        assert_eq!(levels[4], -1);
        assert_eq!(levels[5], -2);
        // Every recovered coefficient is within `q2 / dq_denom = 4`
        // of the input (the maximum quantization error at `q_index =
        // 0` for TX_4X4).
        let dequant = dequantize_step1(&levels, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        for (k, (orig, recon)) in coeffs.iter().zip(dequant.iter()).enumerate() {
            assert!(
                (orig - recon).abs() <= 4,
                "k={k}: |orig {orig} - recon {recon}| > 4"
            );
        }
    }

    /// At `q_index = 16` the row-0 DC lookup gives `dc_q = 20` and
    /// `ac_q = 23`. TX_4X4 has `dqDenom = 1`. A coefficient that is
    /// an exact multiple of `q` at its position round-trips bit-exactly.
    #[test]
    fn forward_quantize_higher_q_index_aligned_coeffs_roundtrip() {
        let qp = QuantizerParams::neutral(16, 8);
        // Slot 0 uses dc_q = 20; slots 1..16 use ac_q = 23.
        let mut coeffs = [0i64; 16];
        coeffs[0] = 40; // = 2 * 20
        for c in coeffs.iter_mut().skip(1).take(15) {
            *c = 46; // = 2 * 23
        }
        let levels = forward_quantize(&coeffs, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        assert_eq!(levels[0], 2);
        for &lv in &levels[1..] {
            assert_eq!(lv, 2);
        }
        let dequant = dequantize_step1(&levels, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        assert_eq!(dequant, coeffs.to_vec());
    }

    /// `dqDenom == 2` is reached on TX_32X32; the equivalent for the
    /// smaller TX_8X8 stays at `dqDenom == 1`, so the round-trip
    /// property is the same as TX_4X4. Exercise the `tw, th = 8`
    /// active region (still fully in-range, no clamp) for size-shape
    /// coverage. AC lookup at `q_index = 0` gives `ac_q = 4`.
    #[test]
    fn forward_quantize_tx8x8_lossless_q_index_zero() {
        let qp = QuantizerParams::neutral(0, 8);
        let mut coeffs = vec![0i64; 64];
        // DC (slot 0): dc_q = 4 -> Dequant = 4 * Quant. Pick Quant = 5
        // so coeff = 20.
        coeffs[0] = 20;
        // AC slots: ac_q = 4 -> pick a Quant = 3 pattern at strided
        // positions so the resulting Dequant is in-range.
        for (k, slot) in coeffs.iter_mut().enumerate().skip(1).take(63) {
            *slot = ((k as i64) % 7) * 4; // multiples of 4
        }
        let levels = forward_quantize(&coeffs, TX_8X8, 0, 0, DCT_DCT, 15, &qp);
        assert_eq!(levels[0], 5);
        for (k, &lv) in levels.iter().enumerate().skip(1).take(63) {
            let expected = ((k as i64) % 7) as i32;
            assert_eq!(lv, expected, "k={k}");
        }
        let dequant = dequantize_step1(&levels, TX_8X8, 0, 0, DCT_DCT, 15, &qp);
        assert_eq!(dequant, coeffs);
    }

    /// A zero coefficient buffer must produce a zero `Quant[]`
    /// regardless of the quantizer state — the `dq` product is zero,
    /// dequant is zero, and the encoder commits zero levels.
    #[test]
    fn forward_quantize_zero_coeffs_yield_zero_levels() {
        for &qi in &[0u8, 16, 64, 128, 255] {
            let qp = QuantizerParams::neutral(qi, 8);
            let coeffs = vec![0i64; 16];
            let levels = forward_quantize(&coeffs, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
            assert_eq!(levels, vec![0; 16], "q_index = {qi}");
        }
    }

    /// Bit-exact roundtrip drill: for any caller-chosen integer
    /// `Quant[]` (in a safe range), the chain
    /// `dequant -> forward_quantize` recovers the original `Quant[]`.
    /// This is the encoder's correctness contract: the encoder picks
    /// `Quant`, asks the decoder what `Dequant` they imply, and the
    /// `forward_quantize` of those `Dequant` values is the original
    /// `Quant`.
    #[test]
    fn forward_quantize_inverts_dequantize_step1_on_exact_levels() {
        let qp = QuantizerParams::neutral(0, 8);
        // Seven distinct level patterns, well under the 2^16 wrap
        // threshold. Slot 0 is DC; slots 1..16 are AC.
        let patterns: [[i32; 16]; 4] = [
            [10, -3, 7, -2, 0, 5, -1, 4, 0, 0, 8, -6, 3, -9, 2, 11],
            [0; 16],
            [
                100, 50, 25, 10, -100, -50, -25, -10, 1, 2, 3, 4, -1, -2, -3, -4,
            ],
            [
                4096, -4096, 2048, -2048, 1024, -1024, 512, -512, 256, -256, 128, -128, 64, -64,
                32, -32,
            ],
        ];
        for (idx, levels) in patterns.iter().enumerate() {
            let dequant = dequantize_step1(levels, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
            let recovered = forward_quantize(&dequant, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
            assert_eq!(
                recovered,
                levels.to_vec(),
                "pattern {idx} did not round-trip: dequant = {dequant:?}"
            );
        }
    }

    /// QM-active branch — `seg_qm_level < 15`, `using_qmatrix == true`,
    /// `plane_tx_type < IDTX` activates `q2 = Round2(q * qm_value, 5)`.
    /// Drive a TX_4X4 luma block at `q_index = 0` with a non-trivial
    /// `seg_qm_level = 0` and confirm that the `dequantize_step1` →
    /// `forward_quantize` round-trip recovers a caller-chosen `Quant[]`
    /// bit-exactly.
    #[test]
    fn forward_quantize_qm_active_branch_roundtrip() {
        let mut qp = QuantizerParams::neutral(0, 8);
        qp.using_qmatrix = true;
        let levels = [3i32, -1, 2, 0, -2, 1, 4, -3, 0, 0, 1, -1, 2, -2, 3, -3];
        let seg_qm_level = 0;
        let dequant = dequantize_step1(&levels, TX_4X4, 0, 0, DCT_DCT, seg_qm_level, &qp);
        let recovered = forward_quantize(&dequant, TX_4X4, 0, 0, DCT_DCT, seg_qm_level, &qp);
        assert_eq!(recovered, levels.to_vec());
    }

    /// Length mismatch is a caller bug, surfaced as a panic so callers
    /// catch it in development. The §7.12.3 step-1 invariant is that
    /// `Quant` / `Dequant` are dense `Tx_Width * Tx_Height` buffers.
    #[test]
    #[should_panic(expected = "coeffs length")]
    fn forward_quantize_rejects_short_buffer() {
        let qp = QuantizerParams::neutral(0, 8);
        let coeffs = vec![0i64; 15]; // TX_4X4 wants 16
        let _ = forward_quantize(&coeffs, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
    }

    /// Out-of-range `tx_size` is a caller bug, surfaced as a panic.
    #[test]
    #[should_panic(expected = "tx_size")]
    fn forward_quantize_rejects_invalid_tx_size() {
        let qp = QuantizerParams::neutral(0, 8);
        let coeffs = vec![0i64; 16];
        let _ = forward_quantize(&coeffs, TX_SIZES_ALL, 0, 0, DCT_DCT, 15, &qp);
    }
}
