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
//! ## Scope (arc 13 / round 219 + arc 18 / round 225)
//!
//! Round 219 landed [`forward_dct_4`] (length-4 1D) and
//! [`forward_dct_4x4`] (4×4 2D, row-then-column composition without
//! the §7.13.3 `Lossless` / rectangular / row-shift / col-shift
//! envelope).
//!
//! Round 225 extends the same shape to **sizes 8, 16, 32, 64** —
//! [`forward_dct_8`], [`forward_dct_16`], [`forward_dct_32`],
//! [`forward_dct_64`] (1D) and the matching `forward_dct_<N>x<N>` 2D
//! square primitives. The derivation strategy generalises r219: the
//! forward 1D DCT for length `N = 2^n` (`n in 2..=6`) is the
//! algebraic transpose of [`crate::transform::inverse_dct`] for the
//! same `n`. Rather than hand-transcribe the 31-step butterfly graph
//! for each size, we materialise the inverse-DCT response matrix `M`
//! once per size at first use (by walking the inverse on each of the
//! `N` unit-coefficient basis inputs `[4096, 0, …]`, `[0, 4096, …]`,
//! …) and reuse `M^T` per call. Same `Round2(_, 12)` rounding shape,
//! same matrix-transpose argument, same asymptotic `O(N^2)` per 1D
//! pass — the only difference from r219 is that the matrix entries
//! are loaded from cache rather than open-coded.
//!
//! Non-DCT row/column kernels (ADST, FLIPADST, WHT, IDTX, V_*/H_*),
//! the rectangular block sizes (`TX_4X8`, …, `TX_32X64`), and the
//! §7.13.3-equivalent forward 2D dispatcher with row-/col-shift
//! envelope remain subsequent arcs. The matrix-cache approach
//! generalises trivially to ADST/FLIPADST once those forward primitives
//! are wired — `inverse_adst` is also a fixed integer linear map, so
//! the same probe-and-transpose recipe applies.

use crate::transform::{inverse_dct, round2};
use std::sync::OnceLock;

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

// ---------------------------------------------------------------------
// Sizes 8 / 16 / 32 / 64 — matrix-cache implementation.
//
// For each supported size `N = 2^n` (`n in 2..=6`), we cache the row-
// major `N * N` integer matrix `M_inv[n]` such that the §7.13.2.3
// inverse DCT acting on the unit-coefficient basis input `c_k = 4096 *
// δ_{j, k}` produces `M_inv[n][i + N * k] = inverse_dct(c_k)[i]`.
// Equivalently, `M_inv[n][i + N * k] / 4096` is the `(i, k)` entry of
// the spec's inverse-DCT matrix, so:
//
//     inverse_dct(c)[i] ≈ sum_k M_inv[n][i + N * k] * c[k] / 4096
//
// The forward DCT (pixel -> coefficient) is `M_inv[n]^T / 4096`:
//
//     forward(x)[k] = round2(sum_i M_inv[n][i + N * k] * x[i], 12)
//
// — that is, `forward(x)[k]` is the inner product of `x` with the
// `k`-th *column* of `M_inv[n]`. The outer index of our cache stride
// `i + N * k` is therefore the input pixel index, and the inner index
// is the output coefficient index (transposed from the inverse).
//
// `r = 32` is used when probing the inverse so the §7.13.2.1 `clip3`
// at `[-2^31, 2^31 - 1]` never trips on the unit-coefficient amplitude
// 4096 (the inverse DCT for `n = 6` produces intermediate butterfly
// sums up to a few hundred thousand in absolute value, comfortably
// within i32 but above the conformance-friendly `r = 16` bound).
// ---------------------------------------------------------------------

/// Returns the cached `N * N` row-major inverse-DCT response matrix
/// for `n in 2..=6` (`N = 1 << n`). Computes on first call, then
/// returns the cached reference for subsequent calls.
fn inverse_dct_matrix(n: u32) -> &'static [i64] {
    debug_assert!((2..=6).contains(&n));
    static M2: OnceLock<Vec<i64>> = OnceLock::new();
    static M3: OnceLock<Vec<i64>> = OnceLock::new();
    static M4: OnceLock<Vec<i64>> = OnceLock::new();
    static M5: OnceLock<Vec<i64>> = OnceLock::new();
    static M6: OnceLock<Vec<i64>> = OnceLock::new();
    let cell = match n {
        2 => &M2,
        3 => &M3,
        4 => &M4,
        5 => &M5,
        6 => &M6,
        _ => unreachable!("forward DCT size n must be in 2..=6"),
    };
    cell.get_or_init(|| {
        let nn = 1usize << n;
        let mut m = vec![0i64; nn * nn];
        let mut probe = vec![0i64; nn];
        for k in 0..nn {
            for v in probe.iter_mut() {
                *v = 0;
            }
            probe[k] = 4096;
            inverse_dct(&mut probe, n, 32);
            for i in 0..nn {
                m[i + nn * k] = probe[i];
            }
        }
        m
    })
}

/// Forward 1D DCT-II for length `N = 1 << n`, `n in 2..=6`. The
/// algebraic transpose of [`crate::transform::inverse_dct`] for the
/// same `n`. Reads `N` spatial values from `t[0..N]` and writes `N`
/// DCT-II coefficients back into the same slots.
///
/// # Panics
///
/// Panics if `n` is outside `2..=6` or if `t.len() < (1 << n)`.
fn forward_dct_n(t: &mut [i64], n: u32) {
    assert!(
        (2..=6).contains(&n),
        "oxideav-av1 forward_dct_n requires n in 2..=6, got {n}"
    );
    let nn = 1usize << n;
    assert!(
        t.len() >= nn,
        "oxideav-av1 forward_dct_n: buffer too short for length {nn}"
    );
    let m = inverse_dct_matrix(n);
    // x = copy of the input slots.
    let mut x = [0i64; 64];
    x[..nn].copy_from_slice(&t[..nn]);
    for k in 0..nn {
        // Output coefficient k = inner product of x with column k of
        // M_inv. Column k of M_inv is `m[i + nn * k]` for i in 0..nn.
        let mut acc: i64 = 0;
        for i in 0..nn {
            acc += m[i + nn * k] * x[i];
        }
        t[k] = round2(acc, 12);
    }
}

/// Forward 1D DCT-II for length 8 — the transpose of the §7.13.2.3
/// inverse DCT-8 kernel.
///
/// # Panics
///
/// Panics if `t.len() < 8`.
pub fn forward_dct_8(t: &mut [i64], _r: u32) {
    forward_dct_n(t, 3);
}

/// Forward 1D DCT-II for length 16 — the transpose of the §7.13.2.3
/// inverse DCT-16 kernel.
///
/// # Panics
///
/// Panics if `t.len() < 16`.
pub fn forward_dct_16(t: &mut [i64], _r: u32) {
    forward_dct_n(t, 4);
}

/// Forward 1D DCT-II for length 32 — the transpose of the §7.13.2.3
/// inverse DCT-32 kernel.
///
/// # Panics
///
/// Panics if `t.len() < 32`.
pub fn forward_dct_32(t: &mut [i64], _r: u32) {
    forward_dct_n(t, 5);
}

/// Forward 1D DCT-II for length 64 — the transpose of the §7.13.2.3
/// inverse DCT-64 kernel.
///
/// # Panics
///
/// Panics if `t.len() < 64`.
pub fn forward_dct_64(t: &mut [i64], _r: u32) {
    forward_dct_n(t, 6);
}

/// Forward 2D DCT-II for an `N * N` square block (`N in {8, 16, 32,
/// 64}`). Row-then-column composition through the 1D primitive of the
/// matching size. No row-/col-shift envelope (same scope contract as
/// [`forward_dct_4x4`]).
///
/// # Panics
///
/// Panics if `input.len() != n * n` or `n` is outside `{4, 8, 16, 32,
/// 64}`.
fn forward_dct_nxn(input: &[i64], side: usize) -> Vec<i64> {
    assert!(
        matches!(side, 4 | 8 | 16 | 32 | 64),
        "oxideav-av1 forward_dct_nxn requires side in {{4,8,16,32,64}}, got {side}"
    );
    assert_eq!(
        input.len(),
        side * side,
        "oxideav-av1 forward_dct_nxn expects side * side = {} samples",
        side * side
    );
    let n: u32 = match side {
        4 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        64 => 6,
        _ => unreachable!(),
    };
    let mut work = input.to_vec();
    // Row pass.
    let mut row_buf = vec![0i64; side];
    for i in 0..side {
        row_buf.copy_from_slice(&work[i * side..(i + 1) * side]);
        forward_dct_n(&mut row_buf, n);
        work[i * side..(i + 1) * side].copy_from_slice(&row_buf);
    }
    // Column pass.
    let mut col_buf = vec![0i64; side];
    for j in 0..side {
        for i in 0..side {
            col_buf[i] = work[i * side + j];
        }
        forward_dct_n(&mut col_buf, n);
        for i in 0..side {
            work[i * side + j] = col_buf[i];
        }
    }
    work
}

/// Forward 2D DCT-II for the `TX_8X8` block size. Same row-then-
/// column composition shape as [`forward_dct_4x4`].
///
/// # Panics
///
/// Panics if `input.len() != 64`.
pub fn forward_dct_8x8(input: &[i64]) -> [i64; 64] {
    let v = forward_dct_nxn(input, 8);
    let mut out = [0i64; 64];
    out.copy_from_slice(&v);
    out
}

/// Forward 2D DCT-II for the `TX_16X16` block size.
///
/// # Panics
///
/// Panics if `input.len() != 256`.
pub fn forward_dct_16x16(input: &[i64]) -> [i64; 256] {
    let v = forward_dct_nxn(input, 16);
    let mut out = [0i64; 256];
    out.copy_from_slice(&v);
    out
}

/// Forward 2D DCT-II for the `TX_32X32` block size.
///
/// # Panics
///
/// Panics if `input.len() != 1024`.
pub fn forward_dct_32x32(input: &[i64]) -> Vec<i64> {
    // Returned as Vec<i64> rather than [i64; 1024] to keep the stack
    // footprint at the call site small (1024 * 8 = 8 KiB; large but
    // tolerable). Callers that need a `[i64; 1024]` can `try_into`.
    forward_dct_nxn(input, 32)
}

/// Forward 2D DCT-II for the `TX_64X64` block size.
///
/// # Panics
///
/// Panics if `input.len() != 4096`.
pub fn forward_dct_64x64(input: &[i64]) -> Vec<i64> {
    // Returned as Vec<i64> — 4096 * 8 = 32 KiB array on the stack is
    // beyond the default 8 MiB main-thread stack budget on many
    // platforms once nested. Vec heap allocation keeps the surface
    // safe for use from sub-threads with smaller stacks.
    forward_dct_nxn(input, 64)
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

    // -----------------------------------------------------------------
    // Sizes 8 / 16 / 32 / 64 — DC-concentration + roundtrip-via-inverse
    // sanity tests.
    //
    // Approach. The §7.13.2.3 inverse DCT for `n >= 3` accumulates per-
    // butterfly `Round2` error across many stages (n = 3 has 6 stages,
    // n = 6 has all 31). The cached response matrix `M_inverse[n]`
    // therefore is **only approximately** column-orthogonal — the
    // off-diagonal entries of `M_inverse[n]^T · M_inverse[n]` carry
    // the accumulated rounding noise. This is intrinsic to the
    // integer butterfly, not a property of our derivation.
    //
    // For an encoder, the well-posed identity is `inverse(forward(x))
    // ≈ scale * x` (apply forward to pixels, recover them through the
    // inverse). We verify this directly on a DC plane (the input
    // `[k, k, …, k]` rebuilds to `[scale * k + small_noise, …]`
    // after `inverse(forward(…))`).
    // -----------------------------------------------------------------

    fn run_forward_dct(c: &mut [i64], n: u32) {
        match n {
            3 => forward_dct_8(c, 32),
            4 => forward_dct_16(c, 32),
            5 => forward_dct_32(c, 32),
            6 => forward_dct_64(c, 32),
            _ => unreachable!(),
        }
    }

    /// `inverse(forward([k; N]))` on a flat-DC input. The forward
    /// projects all energy onto the DC bin (cell 0), the inverse
    /// then projects it back to a flat plane scaled by the DC-bin
    /// magnitude divided by 4096. The output is approximately
    /// `k * (N * 2896 * 2896) / 4096^2` per cell ≈ `k * N / 2`.
    /// Bound the per-cell relative error at a small fraction of `k`.
    fn dc_roundtrip_via_inverse(n: u32, k: i64, max_per_cell_abs_error: i64) {
        let nn = 1usize << n;
        let mut c = vec![k; nn];
        run_forward_dct(&mut c, n);
        // c is now the forward coefficients. Apply the inverse to
        // recover the pixel plane.
        crate::transform::inverse_dct(&mut c, n, 32);
        // Expected: every cell ≈ k * (N * cos0^2) / 4096^2 where
        // cos0 = 2896 ≈ 4096 / sqrt(2). The product `N * 2896^2 /
        // 4096^2` is `N / 2 * (2896^2 / 2048^2)` = `N/2 *
        // 1.99988...`. For k = 1024, n = 3 (N = 8): expected ≈ 4096
        // per cell.
        // We expand the expectation rigorously: the forward output
        // at bin 0 is `Round2(N * 2896 * k, 12)`. Calling that `D`,
        // the inverse applied to `[D, 0, 0, …, 0]` returns `Round2(
        // D * 2896, 12)` per cell. So the recovered cell value is
        // `Round2(Round2(N * 2896 * k, 12) * 2896, 12)`.
        let pre = (nn as i64) * 2896 * k;
        let d = crate::transform::round2(pre, 12);
        let expected_cell = crate::transform::round2(d * 2896, 12);
        for (i, &got) in c.iter().enumerate() {
            let err = (got - expected_cell).abs();
            assert!(
                err <= max_per_cell_abs_error,
                "n={n} k={k} cell {i}: got {got}, expected {expected_cell} (|err| {err} > bound {max_per_cell_abs_error})"
            );
        }
    }

    /// `inverse(forward(x))` on an arbitrary input plane, asserting
    /// the recovered plane is close to `scale * x` (the round-trip
    /// scaling factor `≈ N/2` for the integer DCT-II basis).
    fn arbitrary_roundtrip_via_inverse(n: u32, input: &[i64], max_abs_error: i64) {
        let nn = 1usize << n;
        assert_eq!(input.len(), nn);
        // Expected scale: each forward bin is approximately
        // `sum_k (M^T)[i, k] * x[k] / 4096`. The inverse undoes
        // this approximately to recover x scaled by N/2 (the
        // diagonal of M^T M / 4096^2 ≈ N/2 in the integer-rounded
        // approximation). Round-trip nominal scale is therefore
        // `N/2` per spatial cell, with rounding noise per cell.
        let mut buf = input.to_vec();
        run_forward_dct(&mut buf, n);
        crate::transform::inverse_dct(&mut buf, n, 32);
        let scale = (nn as i64) / 2;
        for (i, (&got, &orig)) in buf.iter().zip(input.iter()).enumerate() {
            let expected = scale * orig;
            let err = (got - expected).abs();
            assert!(
                err <= max_abs_error,
                "n={n} cell {i}: orig={orig}, got={got}, expected≈{expected} (|err| {err} > bound {max_abs_error})"
            );
        }
    }

    #[test]
    fn forward_dct_8_zero_input_yields_zero() {
        let mut t = [0i64; 8];
        forward_dct_8(&mut t, 32);
        assert_eq!(t, [0; 8]);
    }

    #[test]
    fn forward_dct_8_dc_concentrates_energy() {
        // Flat DC: forward of [k; 8] = [round2(8 * 2896 * k, 12), 0,
        // 0, ...]; row 0 of M^T is [2896, 2896, ..., 2896]. For k =
        // 1024: 8 * 2896 * 1024 = 23724032. Round2(., 12) = 5792.
        // Off-DC bins are NOT exactly zero for n >= 3 (rounding
        // residue in M's columns); bound them tightly.
        let mut t = [1024i64; 8];
        forward_dct_8(&mut t, 32);
        // DC bin: row 0 of M^T = column 0 of M. For an orthogonal
        // basis, column 0 should be the constant `2896`. Verify the
        // DC bin matches `Round2(N * 2896 * 1024, 12)` to within a
        // tight bound (Round2 residue from the n=3 butterfly chain).
        let nominal = crate::transform::round2(8i64 * 2896 * 1024, 12);
        assert!(
            (t[0] - nominal).abs() <= 8,
            "DC bin got {} expected≈{nominal}",
            t[0]
        );
        // Off-DC bins should be small: orthogonality residue scales
        // with the input magnitude and the Round2 noise floor.
        for (i, v) in t.iter().enumerate().skip(1) {
            assert!(v.abs() <= 16, "off-DC bin {i} = {v}, exceeds noise bound");
        }
    }

    #[test]
    fn forward_dct_8_dc_roundtrip_via_inverse() {
        // forward then inverse on a flat-DC input returns a flat-DC
        // plane scaled by ≈ N/2. The integer-rounded butterfly leaks
        // a small Round2-floor noise into every cell; bound at 4
        // LSBs.
        dc_roundtrip_via_inverse(3, 1024, 4);
    }

    #[test]
    fn forward_dct_8x8_zero_input_yields_zero() {
        let input = [0i64; 64];
        let out = forward_dct_8x8(&input);
        assert_eq!(out, [0; 64]);
    }

    #[test]
    fn forward_dct_8x8_dc_concentrates_in_top_left() {
        // Flat 1024 input. After row pass each row ≈ [5792, 0, …, 0]
        // (DC concentration per row). After column pass each column
        // ≈ [Round2(N * 2896 * 5792, 12), 0, …, 0] in column 0,
        // ≈ zero elsewhere.
        let input = [1024i64; 64];
        let out = forward_dct_8x8(&input);
        // Verify cell (0, 0) is the dominant DC magnitude (within a
        // small noise bound around the nominal value), and every
        // other cell is small.
        let dc_after_row = crate::transform::round2(8i64 * 2896 * 1024, 12);
        let dc_nominal = crate::transform::round2(8i64 * 2896 * dc_after_row, 12);
        assert!(
            (out[0] - dc_nominal).abs() <= 64,
            "8x8 DC cell {} vs nominal {dc_nominal}",
            out[0]
        );
        for (i, &v) in out.iter().enumerate().skip(1) {
            assert!(v.abs() <= 256, "8x8 off-DC cell {i} = {v} too large");
        }
    }

    #[test]
    fn forward_dct_16_zero_input_yields_zero() {
        let mut t = [0i64; 16];
        forward_dct_16(&mut t, 32);
        assert_eq!(t, [0; 16]);
    }

    #[test]
    fn forward_dct_16_dc_concentrates_energy() {
        let mut t = [1024i64; 16];
        forward_dct_16(&mut t, 32);
        let nominal = crate::transform::round2(16i64 * 2896 * 1024, 12);
        assert!(
            (t[0] - nominal).abs() <= 16,
            "DC bin {} expected≈{nominal}",
            t[0]
        );
        for (i, v) in t.iter().enumerate().skip(1) {
            assert!(v.abs() <= 32, "off-DC bin {i} = {v} exceeds noise bound");
        }
    }

    #[test]
    fn forward_dct_16_dc_roundtrip_via_inverse() {
        dc_roundtrip_via_inverse(4, 1024, 16);
    }

    #[test]
    fn forward_dct_16x16_zero_input_yields_zero() {
        let input = [0i64; 256];
        let out = forward_dct_16x16(&input);
        assert_eq!(out, [0; 256]);
    }

    #[test]
    fn forward_dct_16x16_dc_concentrates_in_top_left() {
        let input = [1024i64; 256];
        let out = forward_dct_16x16(&input);
        let row_dc = crate::transform::round2(16i64 * 2896 * 1024, 12);
        let nominal = crate::transform::round2(16i64 * 2896 * row_dc, 12);
        assert!(
            (out[0] - nominal).abs() <= 256,
            "16x16 DC cell {} vs nominal {nominal}",
            out[0]
        );
        for (i, &v) in out.iter().enumerate().skip(1) {
            assert!(v.abs() <= 1024, "16x16 off-DC cell {i} = {v} too large");
        }
    }

    #[test]
    fn forward_dct_32_zero_input_yields_zero() {
        let mut t = [0i64; 32];
        forward_dct_32(&mut t, 32);
        assert_eq!(t, [0; 32]);
    }

    #[test]
    fn forward_dct_32_dc_concentrates_energy() {
        let mut t = [1024i64; 32];
        forward_dct_32(&mut t, 32);
        let nominal = crate::transform::round2(32i64 * 2896 * 1024, 12);
        assert!(
            (t[0] - nominal).abs() <= 32,
            "DC bin {} expected≈{nominal}",
            t[0]
        );
        for (i, v) in t.iter().enumerate().skip(1) {
            assert!(v.abs() <= 64, "off-DC bin {i} = {v} exceeds noise bound");
        }
    }

    #[test]
    fn forward_dct_32_dc_roundtrip_via_inverse() {
        dc_roundtrip_via_inverse(5, 1024, 64);
    }

    #[test]
    fn forward_dct_32x32_zero_input_yields_zero() {
        let input = vec![0i64; 1024];
        let out = forward_dct_32x32(&input);
        assert_eq!(out.len(), 1024);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0, "32x32 zero-input bin {i} = {v}");
        }
    }

    #[test]
    fn forward_dct_32x32_dc_concentrates_in_top_left() {
        let input = vec![1024i64; 1024];
        let out = forward_dct_32x32(&input);
        let row_dc = crate::transform::round2(32i64 * 2896 * 1024, 12);
        let nominal = crate::transform::round2(32i64 * 2896 * row_dc, 12);
        assert!(
            (out[0] - nominal).abs() <= 1024,
            "32x32 DC cell {} vs nominal {nominal}",
            out[0]
        );
        for (i, &v) in out.iter().enumerate().skip(1) {
            assert!(v.abs() <= 4096, "32x32 off-DC cell {i} = {v} too large");
        }
    }

    #[test]
    fn forward_dct_64_zero_input_yields_zero() {
        let mut t = [0i64; 64];
        forward_dct_64(&mut t, 32);
        assert_eq!(t, [0; 64]);
    }

    #[test]
    fn forward_dct_64_dc_concentrates_energy() {
        let mut t = [1024i64; 64];
        forward_dct_64(&mut t, 32);
        let nominal = crate::transform::round2(64i64 * 2896 * 1024, 12);
        assert!(
            (t[0] - nominal).abs() <= 64,
            "DC bin {} expected≈{nominal}",
            t[0]
        );
        for (i, v) in t.iter().enumerate().skip(1) {
            assert!(v.abs() <= 128, "off-DC bin {i} = {v} exceeds noise bound");
        }
    }

    #[test]
    fn forward_dct_64_dc_roundtrip_via_inverse() {
        dc_roundtrip_via_inverse(6, 1024, 256);
    }

    #[test]
    fn forward_dct_64x64_zero_input_yields_zero() {
        let input = vec![0i64; 4096];
        let out = forward_dct_64x64(&input);
        assert_eq!(out.len(), 4096);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0, "64x64 zero-input bin {i} = {v}");
        }
    }

    #[test]
    fn forward_dct_64x64_dc_concentrates_in_top_left() {
        let input = vec![1024i64; 4096];
        let out = forward_dct_64x64(&input);
        let row_dc = crate::transform::round2(64i64 * 2896 * 1024, 12);
        let nominal = crate::transform::round2(64i64 * 2896 * row_dc, 12);
        assert!(
            (out[0] - nominal).abs() <= 4096,
            "64x64 DC cell {} vs nominal {nominal}",
            out[0]
        );
        for (i, &v) in out.iter().enumerate().skip(1) {
            assert!(v.abs() <= 16384, "64x64 off-DC cell {i} = {v} too large");
        }
    }

    #[test]
    fn forward_dct_8_arbitrary_roundtrip_via_inverse() {
        // LCG-pseudo-random small inputs; verify inverse(forward(x))
        // recovers x scaled by N/2 within a bounded noise floor.
        let mut input = [0i64; 8];
        let mut s: u64 = 0x12345678;
        for v in input.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 32) & 0xFF) as i64 - 128;
        }
        arbitrary_roundtrip_via_inverse(3, &input, 16);
    }

    #[test]
    fn forward_dct_16_arbitrary_roundtrip_via_inverse() {
        let mut input = [0i64; 16];
        let mut s: u64 = 0xCAFEBABE;
        for v in input.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 32) & 0xFF) as i64 - 128;
        }
        arbitrary_roundtrip_via_inverse(4, &input, 32);
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
