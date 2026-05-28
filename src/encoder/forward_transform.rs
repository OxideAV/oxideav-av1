//! Forward 4×4 DCT primitive — the encoder counterpart of the §7.13.2.3
//! inverse DCT-4 implemented in [`crate::transform`].
//!
//! ## Why "forward"
//!
//! The §7.13 inverse-transform stack maps **coefficients → pixel
//! residuals**: it takes the per-block §7.12.3 `Dequant[]` array and
//! produces the §7.13.3 `Residual[]` array the §7.12.3 reconstruction
//! step adds to the prediction. The encoder needs the inverse direction
//! — **pixel residuals → coefficients** — so the §5.11.39 coefficient
//! writers (round 212-215) have something to consume.
//!
//! ## Derivation (clean-room)
//!
//! Per the standard DCT-II / DCT-III adjoint relation, the forward DCT
//! is the transpose of the inverse. Concretely, the inverse §7.13.2.3
//! kernel for `n = 2` (size 4) computes an affine map `t' = M · t + e`,
//! where `M` is a fixed 4×4 integer matrix scaled by `1 / 4096` and `e`
//! is the per-stage `Round2(_, 12)` rounding error. The forward kernel
//! evaluates `y = M^T · x / 4096` with a single `Round2(_, 12)` per
//! output coefficient.
//!
//! `M` is obtained by walking [`crate::transform::inverse_dct`] for
//! `n = 2` on each of the four 4-dimensional unit-coefficient inputs
//! `[4096, 0, 0, 0]`, `[0, 4096, 0, 0]`, `[0, 0, 4096, 0]`, `[0, 0, 0,
//! 4096]`. The §7.13.2.3 schedule for `n = 2` reduces to:
//!
//!   1. §7.13.2.2 permutation `brev(2, .) = [0, 2, 1, 3]`.
//!   2. Step 12 i = 0: `B(0, 1, 32, 1, r)` (flip).
//!   3. Step 12 i = 1: `B(2, 3, 48, 0, r)`.
//!   4. Step 17 i = 0: `H(0, 3, 0, r)`.
//!   5. Step 17 i = 1: `H(1, 2, 0, r)`.
//!
//! With `cos128(32) = sin128(32) = 2896`, `cos128(48) = 1567`,
//! `sin128(48) = 3784`, this evaluates `M` (rows = output spatial index,
//! columns = input coefficient index, scaled by `1 / 4096`) as:
//!
//! ```text
//! M = (1 / 4096) * [
//!   [2896,  3784,  2896,  1567],
//!   [2896,  1567, -2896, -3784],
//!   [2896, -1567, -2896,  3784],
//!   [2896, -3784,  2896, -1567],
//! ]
//! ```
//!
//! Therefore `M^T` (forward DCT-II) is:
//!
//! ```text
//! M^T = (1 / 4096) * [
//!   [2896,  2896,  2896,  2896],
//!   [3784,  1567, -1567, -3784],
//!   [2896, -2896, -2896,  2896],
//!   [1567, -3784,  3784, -1567],
//! ]
//! ```
//!
//! ## Scaling
//!
//! `M` is **not** unitary in the strict integer sense, but it is
//! close: the even-index columns (`k = 0, 2`) have squared norm
//! `4 * 2896^2 / 4096^2 ≈ 1.99988`, the odd-index columns (`k = 1,
//! 3`) have squared norm `2 * (3784^2 + 1567^2) / 4096^2 ≈ 1.99994`.
//! Both are close to `2` (the cosine constants `2896 ≈ 4096 / sqrt(2)`,
//! `3784 ≈ 4096 * cos(pi/8)`, `1567 ≈ 4096 * sin(pi/8)` are integer-
//! rounded approximations of the analytic cosines used by the
//! continuous DCT-II, so column norms differ by at most a part in
//! `~2 * 10^4`). `M^T · M` is exactly diagonal (the basis is
//! mutually orthogonal — the off-diagonal entries are exact zeros),
//! with values `≈ 1.99988` (even rows) and `≈ 1.99994` (odd rows)
//! before the per-stage `Round2(_, 12)` rounding.
//!
//! The §7.13.3 dispatcher conceals this asymmetry behind the
//! `Round2(_, rowShift)` / `Round2(_, colShift)` per-axis right-shifts
//! that follow each pass; a real encoder applies a forward equivalent
//! (a quantization step with per-axis quantizers tuned to absorb the
//! basis-vector scale difference) before storing coefficients. The
//! primitive in this module is the bare row/column kernel; the §7.13.3-
//! equivalent forward 2D dispatcher with shift envelope is a
//! subsequent arc.
//!
//! The roundtrip test [`tests::forward_inverse_roundtrip_zeroes_off_diagonal`]
//! verifies the exact diagonal values (`8190` for even columns, `8191`
//! for odd) and exact zeros off-diagonal for unit-coefficient inputs
//! at amplitude `4096`.
//!
//! ## Scope (arc 13 / round 219)
//!
//! Only [`forward_dct_4`] (length-4 1D) and [`forward_dct_4x4`] (4×4
//! 2D, row-then-column composition without the §7.13.3 `Lossless`
//! / rectangular / row-shift / col-shift envelope). Larger sizes
//! (8, 16, 32, 64) and the non-DCT row/column kernels (ADST, FLIPADST,
//! WHT, IDTX, V_*/H_*) are subsequent arcs. The §7.13.3-equivalent 2D
//! dispatcher is a subsequent arc.

use crate::transform::round2;

/// Forward 1D DCT-II for length 4 — the transpose of the §7.13.2.3
/// inverse DCT-4 kernel.
///
/// Reads four spatial values from `t[0..4]` and writes four DCT-II
/// coefficients back into the same slots. Uses one §4.7.2
/// `Round2(_, 12)` per output coefficient. The `r` argument is the
/// row-/column-clamp range argument the parallel inverse kernel
/// accepts; the forward kernel does not require it for correctness
/// because all intermediate sums fit in i64 well within `r` bits for
/// typical 8/10/12-bit pixel residuals, but it is accepted for
/// signature parity with [`crate::transform::inverse_dct`].
///
/// # Panics
///
/// Panics if `t.len() < 4`.
pub fn forward_dct_4(t: &mut [i64], _r: u32) {
    let x0 = t[0];
    let x1 = t[1];
    let x2 = t[2];
    let x3 = t[3];
    // Row 0 of M^T: [2896, 2896, 2896, 2896].
    let y0 = 2896i64 * x0 + 2896 * x1 + 2896 * x2 + 2896 * x3;
    // Row 1 of M^T: [3784, 1567, -1567, -3784].
    let y1 = 3784i64 * x0 + 1567 * x1 - 1567 * x2 - 3784 * x3;
    // Row 2 of M^T: [2896, -2896, -2896, 2896].
    let y2 = 2896i64 * x0 - 2896 * x1 - 2896 * x2 + 2896 * x3;
    // Row 3 of M^T: [1567, -3784, 3784, -1567].
    let y3 = 1567i64 * x0 - 3784 * x1 + 3784 * x2 - 1567 * x3;
    t[0] = round2(y0, 12);
    t[1] = round2(y1, 12);
    t[2] = round2(y2, 12);
    t[3] = round2(y3, 12);
}

/// Forward 2D DCT-II for the `TX_4X4` block size — the transpose of the
/// §7.13.3 `inverse_transform_2d` row-then-column composition restricted
/// to `tx_sz = TX_4X4` and `tx_type = DCT_DCT`.
///
/// Consumes a `4 × 4 = 16` row-major spatial-residual buffer and
/// returns the `16`-entry row-major coefficient buffer. The composition
/// is: row pass applies [`forward_dct_4`] over each row, then column
/// pass applies [`forward_dct_4`] over each column. No rectangular
/// scaling (TX_4X4 is square, so the §7.13.3 `|log2W - log2H| == 1`
/// arm is not taken) and no row-/col-shift (those belong to the §7.13.3
/// dispatcher's `Round2(_, rowShift)` / `Round2(_, colShift)` envelope,
/// which is the inverse direction's pipeline glue, not this primitive).
///
/// # Panics
///
/// Panics if `input.len() != 16`.
pub fn forward_dct_4x4(input: &[i64]) -> [i64; 16] {
    assert_eq!(
        input.len(),
        16,
        "oxideav-av1 forward_dct_4x4 expects 4 * 4 = 16 spatial samples"
    );
    let mut work = [0i64; 16];
    work.copy_from_slice(input);
    // Row pass.
    let mut row_buf = [0i64; 4];
    for i in 0..4 {
        row_buf.copy_from_slice(&work[i * 4..i * 4 + 4]);
        forward_dct_4(&mut row_buf, 16);
        work[i * 4..i * 4 + 4].copy_from_slice(&row_buf);
    }
    // Column pass.
    let mut col_buf = [0i64; 4];
    for j in 0..4 {
        for i in 0..4 {
            col_buf[i] = work[i * 4 + j];
        }
        forward_dct_4(&mut col_buf, 16);
        for i in 0..4 {
            work[i * 4 + j] = col_buf[i];
        }
    }
    work
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{butterfly_b, butterfly_h, inverse_dct_permute};

    /// Re-derive the inverse §7.13.2.3 4-point kernel inline as a
    /// reference for the matrix-transpose check.
    fn inverse_dct_4(t: &mut [i64; 4]) {
        // §7.13.2.2 permutation then §7.13.2.3 step 12 + step 17.
        // r = 16 matches the [`crate::transform`] tests' chosen value.
        let mut tmp = t.to_vec();
        inverse_dct_permute(&mut tmp, 2);
        butterfly_b(&mut tmp, 0, 1, 32, 1, 16);
        butterfly_b(&mut tmp, 2, 3, 48, 0, 16);
        butterfly_h(&mut tmp, 0, 3, 0, 16);
        butterfly_h(&mut tmp, 1, 2, 0, 16);
        for (slot, v) in t.iter_mut().zip(tmp.iter().copied()) {
            *slot = v;
        }
    }

    #[test]
    fn forward_dct_4_zero_input_yields_zero() {
        let mut t = [0i64; 4];
        forward_dct_4(&mut t, 16);
        assert_eq!(t, [0; 4]);
    }

    #[test]
    fn forward_dct_4_dc_only_yields_first_basis() {
        // Forward of a flat DC: [k, k, k, k] -> Round2(4 * 2896 * k, 12)
        // in slot 0, zero in slots 1..4 (orthogonality of the basis).
        // For k = 1024: 4 * 2896 * 1024 = 11862016. Round2(., 12) =
        // (11862016 + 2048) >> 12 = 2896.
        let mut t = [1024i64; 4];
        forward_dct_4(&mut t, 16);
        assert_eq!(t, [2896, 0, 0, 0]);
    }

    #[test]
    fn forward_dct_4_matches_matrix_transpose_on_unit_vectors() {
        // Unit-vector probe through the forward kernel reproduces the
        // four rows of M^T scaled by 1 / 4096; for unit input 4096
        // each row's first nonzero entry should be the matrix value
        // exactly (Round2 of the scaled product).
        for k in 0..4 {
            let mut t = [0i64; 4];
            t[k] = 4096;
            forward_dct_4(&mut t, 16);
            // The response to a unit-vector input at position k reads
            // off the k-th column of M^T (i.e. the k-th row of M):
            //  k = 0: [2896, 3784, 2896, 1567]
            //  k = 1: [2896, 1567, -2896, -3784]
            //  k = 2: [2896, -1567, -2896, 3784]
            //  k = 3: [2896, -3784, 2896, -1567]
            let expected = match k {
                0 => [2896i64, 3784, 2896, 1567],
                1 => [2896, 1567, -2896, -3784],
                2 => [2896, -1567, -2896, 3784],
                3 => [2896, -3784, 2896, -1567],
                _ => unreachable!(),
            };
            assert_eq!(t, expected, "M^T column {k} mismatch");
        }
    }

    #[test]
    fn forward_inverse_roundtrip_scales_dc_by_two() {
        // inverse_dct_4([4096, 0, 0, 0]) = [2896; 4]; forward of that
        // yields the even-column diagonal value 8190 ≈ 2 * 4096 in
        // slot 0, exactly zero elsewhere.
        let mut x = [4096i64, 0, 0, 0];
        inverse_dct_4(&mut x);
        assert_eq!(x, [2896; 4]);
        forward_dct_4(&mut x, 16);
        // 4 * 2896 * 2896 = 33547264; Round2(., 12) = (33547264 +
        // 2048) >> 12 = 33549312 >> 12 = 8190 (NB: 8190, not 8192,
        // because the cosine constant 2896 is itself a rounded
        // approximation of 4096 / sqrt(2)).
        assert_eq!(x, [8190, 0, 0, 0]);
    }

    #[test]
    fn forward_inverse_roundtrip_zeroes_off_diagonal() {
        // For each of the four basis-coefficient unit inputs at
        // amplitude 4096, run inverse then forward and verify the
        // off-diagonal entries are exactly zero (the basis is
        // mutually orthogonal under M^T M). The diagonal entry is
        // approximately 2 * input: column norms of M (scaled by
        // 1 / 4096) are 2 * 2896^2 / 4096^2 ≈ 1.99988 for the even
        // columns (k = 0, 2) and 2 * (3784^2 + 1567^2) / 4096^2 ≈
        // 1.99994 for the odd columns (k = 1, 3) — both very close
        // to 2 (the cosine constants 2896 = round(4096 / sqrt(2)),
        // 3784 ≈ round(4096 * cos(pi/8)), 1567 ≈ round(4096 *
        // sin(pi/8)) are integer-rounded approximations of the
        // analytic cosines). The exact roundtrip is
        // forward(inverse(c)) = (M^T M) c / 4096^2 with per-stage
        // Round2(_, 12), which gives the diagonal values measured
        // here.
        let expected_diag: [i64; 4] = [8190, 8191, 8190, 8191];
        for k in 0..4 {
            let mut c = [0i64; 4];
            c[k] = 4096;
            inverse_dct_4(&mut c);
            forward_dct_4(&mut c, 16);
            for (j, &got) in c.iter().enumerate() {
                if j == k {
                    assert_eq!(
                        got, expected_diag[k],
                        "basis k={k} diag j={j}: got {got}, want {}",
                        expected_diag[k]
                    );
                } else {
                    assert_eq!(got, 0, "basis k={k} off-diag j={j}: got {got}, want 0");
                }
            }
        }
    }

    #[test]
    fn forward_dct_4_linearity_at_aligned_inputs() {
        // Round2 is exact for inputs whose inner product with each
        // row of M^T is a multiple of 4096. The all-equal input
        // satisfies this for row 0 (sum = 4 * input * 2896, divisible
        // by 4096 when 4 * input * 2896 / 4096 is an integer ⇒
        // input * 2896 / 1024 ∈ Z ⇒ input = 1024 * m). For
        // input = 1024: row 0 = 11862016 / 4096 = 2895.5, Round2 =
        // 2896. Scaling input by 2 doubles every row's inner
        // product, so output should double exactly within Round2.
        let mut a = [1024i64; 4];
        let mut b = [2048i64; 4];
        forward_dct_4(&mut a, 16);
        forward_dct_4(&mut b, 16);
        for i in 0..4 {
            assert_eq!(b[i], 2 * a[i], "linearity row {i}");
        }
    }

    #[test]
    fn forward_dct_4x4_zero_input_yields_zero() {
        let input = [0i64; 16];
        let out = forward_dct_4x4(&input);
        assert_eq!(out, [0; 16]);
    }

    #[test]
    fn forward_dct_4x4_dc_only_concentrates_energy() {
        // Flat DC plane: every spatial sample = 1024. After 1D row
        // pass, each row becomes [2896, 0, 0, 0] (per
        // forward_dct_4_dc_only_yields_first_basis). After 1D column
        // pass on column 0 = [2896; 4]: forward_dct_4([2896; 4]) =
        // Round2(4 * 2896 * 2896, 12) = Round2(33547264, 12) = 8190
        // in row 0, zero elsewhere. Columns 1..3 are all-zero so
        // produce all-zero columns.
        let input = [1024i64; 16];
        let out = forward_dct_4x4(&input);
        let mut expected = [0i64; 16];
        expected[0] = 8190;
        assert_eq!(out, expected);
    }

    #[test]
    fn forward_dct_4x4_top_left_impulse_spreads_into_basis() {
        // Spatial impulse at (0, 0) of magnitude 4096; all other
        // positions zero. Row 0 forward: [2896, 3784, 2896, 1567]
        // (the 0-th column of M^T, equivalently the 0-th row of M).
        // Rows 1..3 zero. Column j forward then operates on
        // [first_row_j, 0, 0, 0], for which the k-th output is
        // Round2(M^T[k][0] * first_row_j, 12) and the M^T column 0 =
        // [2896, 3784, 2896, 1567].
        let mut input = [0i64; 16];
        input[0] = 4096;
        let out = forward_dct_4x4(&input);
        // For col 0 (c = 2896): [Round2(2896*2896,12)=2048,
        //   Round2(3784*2896,12)=2675, 2048, Round2(1567*2896,12)=
        //   1108].
        // For col 1 (c = 3784): [2675,
        //   Round2(3784*3784,12)=3496, 2675,
        //   Round2(1567*3784,12)=1448].
        // For col 2 (c = 2896): same as col 0.
        // For col 3 (c = 1567): [1108, 1448, 1108,
        //   Round2(1567*1567,12)=599].
        let expected: [i64; 16] = [
            2048, 2675, 2048, 1108, //
            2675, 3496, 2675, 1448, //
            2048, 2675, 2048, 1108, //
            1108, 1448, 1108, 599, //
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn forward_dct_4x4_basis_concentration_for_each_input_position() {
        // For each (a, b) spatial-position impulse at amplitude 4096,
        // verify the dominant frequency cell matches the expected
        // 1024-times-product-of-M-T-rows pattern. This is a coverage
        // probe across all 16 spatial positions, asserting only that
        // every output cell is bounded — the exact values are
        // covered by the impulse test above.
        for a in 0..4 {
            for b in 0..4 {
                let mut input = [0i64; 16];
                input[a * 4 + b] = 4096;
                let out = forward_dct_4x4(&input);
                // M^T entries are bounded by 3784 in magnitude.
                // After Round2(_, 12) of a product up to 3784 *
                // 3784 = 14318656, the per-cell magnitude is
                // bounded by Round2(3784 * 3784, 12) = 3496. Then
                // the column pass multiplies again by up to 3784
                // and Round2's: bounded by Round2(3784 * 3496, 12)
                // = 3230. So every cell |.| < 3500.
                for (i, v) in out.iter().enumerate() {
                    assert!(v.abs() < 3500, "a={a} b={b} out[{i}]={v} exceeds bound");
                }
            }
        }
    }
}
