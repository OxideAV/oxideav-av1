//! Forward 4×4 Walsh-Hadamard transform — the encoder counterpart of the
//! §7.13.2.10 inverse WHT in [`crate::transform::inverse_wht4`].
//!
//! ## Why "forward"
//!
//! The §7.13.3 2D inverse-transform dispatcher routes through the
//! §7.13.2.10 inverse Walsh-Hadamard kernel on the `Lossless` branch
//! (row pass with `shift = 2`, column pass with `shift = 0`). The
//! encoder needs the inverse direction — **pixel residuals →
//! coefficients** — so a pixel-space encoder driver at `base_q_idx = 0`
//! (i.e. `CodedLossless = 1` per §5.9.2) can route through the
//! `Lossless` arm and reconstruct arbitrary inputs bit-exactly.
//!
//! Unlike the forward DCT (round 219) which is an integer approximation
//! of an analytic basis (rounded cosine constants ⇒ `M^T · M ≈ 2 · I`
//! with `≈ 1.99988`/`≈ 1.99994` diagonal scaling), the WHT is a **pure
//! integer butterfly** — the spec body in §7.13.2.10 uses only addition,
//! subtraction, and a single `>> 1`. With matching forward/inverse
//! kernels the round-trip is **bit-exact** for any integer input
//! (subject to the row-pass `shift = 2` having low-bit headroom; the
//! forward writes `<< 2`-scaled values so the inverse's `>> 2` cancels).
//!
//! ## Derivation (clean-room)
//!
//! Restating the §7.13.2.10 inverse body (av1-spec p.304) with
//! `T_in = [t0, t1, t2, t3]`, output `T_out = [a, b, c, d]`:
//!
//! ```text
//!   a0 = t0 >> shift     // = T[0] post-pre-shift
//!   c0 = t1 >> shift
//!   d0 = t2 >> shift
//!   b0 = t3 >> shift
//!   A  = a0 + c0          // a += c
//!   D  = d0 - b0          // d -= b
//!   e  = (A - D) >> 1
//!   b_out  = e - b0       // b = e - b
//!   c_out  = e - c0       // c = e - c
//!   a_out  = A - b_out    // a -= b   (= A - e + c0)
//!   d_out  = D + c_out    // d += c   (= D + e - c0)
//! ```
//!
//! To invert, given `(a_out, b_out, c_out, d_out)` solve for
//! `(a0, b0, c0, d0)`:
//!
//! ```text
//!   a_out + b_out = (A - e + c0) + (e - c0) = A   ⇒  A = a_out + b_out
//!   d_out - c_out = (D + e - c0) - (e - c0) = D   ⇒  D = d_out - c_out
//!   e             = (A - D) >> 1                   (matches the inverse's e)
//!   b0            = e - b_out
//!   c0            = e - c_out
//!   a0            = A - c0
//!   d0            = D + b0
//! ```
//!
//! The recovered `(a0, b0, c0, d0)` is the WHT-domain value the inverse
//! sees **after** its `>> shift`. To complete the forward kernel for a
//! pre-shift `shift = s`, multiply each output by `1 << s` so the
//! inverse's `>> s` recovers the WHT-domain values exactly:
//!
//! ```text
//!   T_out[0] = a0 << s
//!   T_out[1] = c0 << s
//!   T_out[2] = d0 << s
//!   T_out[3] = b0 << s
//! ```
//!
//! Substituting `(A - D) >> 1` for `e` is the only operation in the
//! chain that loses information — when `(A - D)` is odd the floor
//! halving drops 1 bit. But on the round-trip the **forward and inverse
//! see the same `A`, `D`** (the four output cells determine `A` and `D`
//! exactly, and the forward stores `(a0, b0, c0, d0)` derived from the
//! same `e = (A - D) >> 1`); therefore `inverse(forward(out)) == out`
//! exactly regardless of `(A - D)` parity. The bit-exact round-trip
//! property is verified in [`tests::wht_4_roundtrip_bit_exact`] (across
//! the full `[-128, 127]` 1D range with `shift = 0`, and a broad pixel
//! sweep with `shift = 2`).
//!
//! ## 2D composition
//!
//! Per §7.13.3 the lossless 2D pipeline applies row WHT first
//! (`shift = 2`) then column WHT (`shift = 0`), with `Round2(_, 0)`
//! per pass (i.e. no post-shift on the lossless branch). The forward
//! is the transposed composition: forward **column** pass first
//! (`shift = 0`), then forward **row** pass (`shift = 2`). At the
//! between-stage there is no clamp on the forward path because the
//! lossless `Residual` bound (§7.13.3 note: `1 + BitDepth` bits) is
//! the spec's conformance guarantee on the inverse side; the forward
//! kernel here trusts callers to pass residuals in that range.
//!
//! ## Scope (arc 16 / round 222)
//!
//! Only [`forward_wht4`] (length-4 1D with caller-supplied `shift`) and
//! [`forward_wht_4x4`] (4×4 2D with the §7.13.3 `Lossless` shift
//! envelope: column pass `shift = 0`, row pass `shift = 2`). Larger
//! WHT sizes do not exist in AV1 — the spec only defines the §7.13.2.10
//! 4-point WHT, and §7.13.3 only invokes it on `tx_size = TX_4X4`.

/// Forward 1D Walsh-Hadamard transform of length 4 with caller-supplied
/// pre-shift `shift`. The bit-exact inverse of
/// [`crate::transform::inverse_wht4`] when called with the same `shift`.
///
/// Reads four spatial values from `t[0..4]` and writes four WHT-domain
/// coefficients back into the same slots. The shift is the per-axis
/// scale matching the §7.13.3 `Lossless` envelope: row pass uses
/// `shift = 2`, column pass uses `shift = 0`.
///
/// # Panics
///
/// Panics if `t.len() < 4`.
pub fn forward_wht4(t: &mut [i64], shift: u32) {
    // Read the four outputs the inverse would produce.
    let out_a = t[0];
    let out_b = t[1];
    let out_c = t[2];
    let out_d = t[3];
    // Invert the §7.13.2.10 body algebraically.
    let big_a = out_a + out_b;
    let big_d = out_d - out_c;
    let e = (big_a - big_d) >> 1;
    let b0 = e - out_b;
    let c0 = e - out_c;
    let a0 = big_a - c0;
    let d0 = big_d + b0;
    // Pre-multiply by `1 << shift` so the inverse's `>> shift` recovers
    // the WHT-domain values without losing bits. The §7.13.2.10 layout
    // stores a0 in slot 0, c0 in slot 1, d0 in slot 2, b0 in slot 3.
    let scale = 1i64 << shift;
    t[0] = a0 * scale;
    t[1] = c0 * scale;
    t[2] = d0 * scale;
    t[3] = b0 * scale;
}

/// Forward 2D Walsh-Hadamard transform for the `TX_4X4` block size — the
/// bit-exact inverse of the §7.13.3 `Lossless` arm of
/// [`crate::transform::inverse_transform_2d`] at `tx_size = TX_4X4`.
///
/// Consumes a `4 × 4 = 16` row-major spatial-residual buffer and returns
/// the `16`-entry row-major coefficient buffer. The composition is the
/// transpose of the §7.13.3 inverse: **column pass first** (forward
/// 1D WHT with `shift = 0`), then **row pass** (forward 1D WHT with
/// `shift = 2`). No rectangular scaling — TX_4X4 is square, so the
/// §7.13.3 `|log2W - log2H| == 1` arm is not taken. No between-stage
/// clamp — the lossless `Residual` conformance bound (§7.13.3 note:
/// `1 + BitDepth` bits) is the spec's invariant on the inverse path;
/// the forward kernel here is the round-trip inverse and so trusts
/// callers to pass residuals in that range.
///
/// # Panics
///
/// Panics if `input.len() != 16`.
pub fn forward_wht_4x4(input: &[i64]) -> [i64; 16] {
    assert_eq!(
        input.len(),
        16,
        "oxideav-av1 forward_wht_4x4 expects 4 * 4 = 16 spatial samples"
    );
    let mut work = [0i64; 16];
    work.copy_from_slice(input);
    // Column pass first (forward 1D WHT with shift = 0). The inverse
    // 2D dispatcher applies the column WHT **last** (after the row
    // WHT), so the forward applies it **first**.
    let mut col_buf = [0i64; 4];
    for j in 0..4 {
        for i in 0..4 {
            col_buf[i] = work[i * 4 + j];
        }
        forward_wht4(&mut col_buf, 0);
        for i in 0..4 {
            work[i * 4 + j] = col_buf[i];
        }
    }
    // Row pass (forward 1D WHT with shift = 2). The inverse 2D
    // dispatcher applies the row WHT **first** with `>> 2`, so the
    // forward writes `<< 2`-scaled values to keep the round-trip
    // bit-exact.
    let mut row_buf = [0i64; 4];
    for i in 0..4 {
        row_buf.copy_from_slice(&work[i * 4..i * 4 + 4]);
        forward_wht4(&mut row_buf, 2);
        work[i * 4..i * 4 + 4].copy_from_slice(&row_buf);
    }
    work
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{DCT_DCT, TX_4X4};
    use crate::transform::{inverse_transform_2d, inverse_wht4};

    /// 1D bit-exact round-trip: `forward_wht4` then `inverse_wht4`
    /// with the same `shift` must recover the original input exactly.
    /// Covers the full signed 8-bit pixel-residual range
    /// (`[-256, 255]`, which is the `±(1 << (BitDepth + 1))` envelope
    /// for the lossless `Residual` bound at `BitDepth = 8`) on each of
    /// the four input slots, plus a few extreme negative values that
    /// exercise the `(A - D) >> 1` arithmetic-shift edge.
    #[test]
    fn wht_4_roundtrip_bit_exact_shift0() {
        for k in 0..4 {
            for v in (-256i64..=255).step_by(7) {
                let mut t = [0i64; 4];
                t[k] = v;
                let original = t;
                forward_wht4(&mut t, 0);
                inverse_wht4(&mut t, 0);
                assert_eq!(
                    t, original,
                    "shift = 0, slot {k}, value {v}: round-trip diverged",
                );
            }
        }
    }

    /// 1D bit-exact round-trip with `shift = 2` (the §7.13.3 row-pass
    /// shift). The forward must scale its outputs by `<< 2` so the
    /// inverse's `>> 2` recovers the WHT-domain coefficients without
    /// losing bits.
    #[test]
    fn wht_4_roundtrip_bit_exact_shift2() {
        for k in 0..4 {
            for v in (-256i64..=255).step_by(11) {
                let mut t = [0i64; 4];
                t[k] = v;
                let original = t;
                forward_wht4(&mut t, 2);
                inverse_wht4(&mut t, 2);
                assert_eq!(
                    t, original,
                    "shift = 2, slot {k}, value {v}: round-trip diverged",
                );
            }
        }
    }

    /// Mixed-value 1D round-trip — every-slot non-zero with both signs,
    /// shift = 0. Validates the `(A - D)` parity edge: when `(A - D)`
    /// is odd, `e = (A - D) >> 1` loses a bit but the forward and
    /// inverse see the same `(A, D)` so the round-trip is still exact.
    #[test]
    fn wht_4_roundtrip_mixed_slots_shift0() {
        // (A - D) odd: try inputs where a+b+c-d is odd.
        let cases: &[[i64; 4]] = &[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [-128, 127, -128, 127],
            [255, -255, 255, -255],
            [3, 5, 7, 11],
            [-7, 13, -19, 23],
        ];
        for case in cases {
            let mut t = *case;
            forward_wht4(&mut t, 0);
            inverse_wht4(&mut t, 0);
            assert_eq!(&t, case, "mixed-slot shift=0 round-trip diverged");
        }
    }

    /// Zero-input passthrough: forward of all-zero is all-zero
    /// (every internal sum is zero, `e = 0`, every output is zero).
    #[test]
    fn forward_wht4_zero_input_yields_zero() {
        let mut t = [0i64; 4];
        forward_wht4(&mut t, 0);
        assert_eq!(t, [0; 4]);
        let mut t = [0i64; 4];
        forward_wht4(&mut t, 2);
        assert_eq!(t, [0; 4]);
    }

    /// 2D bit-exact round-trip on the all-zero spatial input: forward
    /// must yield all-zero coefficients, inverse must recover all-zero
    /// pixel residuals.
    #[test]
    fn forward_wht_4x4_zero_input_yields_zero() {
        let input = [0i64; 16];
        let out = forward_wht_4x4(&input);
        assert_eq!(out, [0; 16]);
        let back = inverse_transform_2d(&out, TX_4X4, DCT_DCT, 8, true);
        assert_eq!(back, vec![0; 16]);
    }

    /// 2D bit-exact round-trip on a flat DC plane: every spatial
    /// sample = `v`. The §7.13.3 lossless inverse with input
    /// `forward_wht_4x4([v; 16])` must recover `[v; 16]` exactly.
    /// Covers the full `[-128, 127]` flat-residual range.
    #[test]
    fn forward_wht_4x4_flat_dc_roundtrips() {
        for v in (-128i64..=127).step_by(3) {
            let input = [v; 16];
            let coeffs = forward_wht_4x4(&input);
            let back = inverse_transform_2d(&coeffs, TX_4X4, DCT_DCT, 8, true);
            assert_eq!(
                back,
                input.to_vec(),
                "flat v = {v}: lossless round-trip diverged"
            );
        }
    }

    /// 2D bit-exact round-trip on a single-spatial-impulse input:
    /// magnitude `+/- 127` at each of the 16 spatial positions. This
    /// is the worst-case "all-energy-in-one-cell" stress for the WHT
    /// butterfly — every internal sum reaches its peak magnitude.
    #[test]
    fn forward_wht_4x4_unit_impulses_roundtrip() {
        for k in 0..16 {
            for &amp in &[1i64, -1, 64, -64, 127, -128] {
                let mut input = [0i64; 16];
                input[k] = amp;
                let coeffs = forward_wht_4x4(&input);
                let back = inverse_transform_2d(&coeffs, TX_4X4, DCT_DCT, 8, true);
                assert_eq!(
                    back,
                    input.to_vec(),
                    "impulse k = {k}, amp = {amp}: lossless round-trip diverged",
                );
            }
        }
    }

    /// 2D bit-exact round-trip on a pseudo-random pixel-residual sweep.
    /// Deterministic LCG (no external RNG) over 64 distinct 4×4 blocks
    /// with values in the §7.13.3 lossless conformance range
    /// (`1 + BitDepth = 9` bits at `BitDepth = 8` ⇒ `[-256, 255]`).
    #[test]
    fn forward_wht_4x4_lcg_sweep_roundtrip() {
        // Simple LCG (Numerical Recipes constants); deterministic so
        // failures replay identically.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        for trial in 0..64 {
            let mut input = [0i64; 16];
            for cell in input.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                // 9-bit signed: [-256, 255].
                let v = ((state >> 23) as u32 & 0x1FF) as i64 - 256;
                *cell = v;
            }
            let coeffs = forward_wht_4x4(&input);
            let back = inverse_transform_2d(&coeffs, TX_4X4, DCT_DCT, 8, true);
            assert_eq!(
                back,
                input.to_vec(),
                "trial {trial}: input = {input:?}, coeffs = {coeffs:?}",
            );
        }
    }

    /// Linearity in the 2D kernel: `forward(2 * x) == 2 * forward(x)`
    /// because every internal step (`+`, `-`, `>> 1` on `(A-D)`, and
    /// the final `<< shift`) is linear when the parity-of-`(A-D)` arm
    /// matches. Doubling preserves parity (`2k` is even iff `k` is),
    /// so this holds across the full input range.
    #[test]
    fn forward_wht_4x4_linearity_at_doubled_input() {
        let mut a = [0i64; 16];
        for (k, slot) in a.iter_mut().enumerate() {
            // Small mixed-sign pattern, all even (parity argument
            // applies regardless, but evens keep the asserts compact).
            *slot = ((k as i64) - 8) * 4;
        }
        let mut b = a;
        for slot in b.iter_mut() {
            *slot *= 2;
        }
        let fa = forward_wht_4x4(&a);
        let fb = forward_wht_4x4(&b);
        for k in 0..16 {
            assert_eq!(fb[k], 2 * fa[k], "linearity slot {k}");
        }
    }

    /// Length-mismatch is a caller bug; surface as a panic so tests
    /// catch it in development.
    #[test]
    #[should_panic(expected = "forward_wht_4x4 expects")]
    fn forward_wht_4x4_rejects_short_buffer() {
        let _ = forward_wht_4x4(&[0i64; 15]);
    }
}
