//! Forward ADST / FLIPADST primitives — the encoder counterparts of the
//! §7.13.2.6 / §7.13.2.7 / §7.13.2.8 inverse ADST kernels for sizes
//! `4 / 8 / 16` (the only sizes the spec covers — `inverse_adst` is
//! defined for `n in 2..=4`).
//!
//! ## Derivation (clean-room)
//!
//! Identical recipe to [`super::forward_transform`] (r225) for the
//! forward DCT: the §7.13.2 inverse ADST is a fixed integer linear
//! map `inverse_adst(c) ≈ M_inv_adst[n] @ c / 4096` where
//! `M_inv_adst[n]` is the `N * N` integer response matrix the
//! §7.13.2.6/7/8 butterfly schedule materialises (`N = 1 << n`,
//! `n in 2..=4`). The forward ADST is the matrix transpose
//! `M_inv_adst[n]^T @ x / 4096` with one §4.7.2 `Round2(_, 12)` per
//! output coefficient.
//!
//! We build `M_inv_adst[n]` at first use by probing
//! [`crate::transform::inverse_adst`] on each of the `N` unit-coefficient
//! basis inputs `[4096, 0, …]`, `[0, 4096, …]`, … and caching the
//! resulting matrix in a `std::sync::OnceLock`. Per-call cost is then
//! a single `O(N^2)` inner product per output coefficient — same
//! shape as the matrix-cache DCT primitives in [`super::forward_transform`].
//!
//! ## FLIPADST
//!
//! Per §7.13.3, FLIPADST shares the **same butterfly kernel** as
//! ADST — the flip is purely a destination-coordinate transform
//! applied externally (decoder side: §7.12.3 step 3 frame-buffer
//! write with `xx = flipLR ? w - j - 1 : j`, `yy = flipUD ? h - i - 1
//! : i`). Symmetrically, on the encoder side the flip is applied to
//! the residual **before** the forward kernel runs (or to the
//! coefficient output **after**, with the same effect — both
//! describe the matrix product `flip · M^T · flip = M^T` after
//! conjugation by the involutory flip operator).
//!
//! The convention adopted in this module: `forward_flipadst_N(x)`
//! reverses `x` in place, then applies `forward_adst_N`, then
//! reverses the output. The result is the coefficient vector a
//! decoder running `inverse_adst` on this output followed by the
//! §7.12.3 vertical/horizontal flip would reproduce as the original
//! residual. For 2D `forward_flipadst_NxN`, the flip is applied to
//! the residual rows / columns depending on which axis the FLIP
//! direction is on (V_FLIPADST flips rows-of-pixels along the
//! vertical axis ⇒ reverse-each-column-before-row-pass;
//! H_FLIPADST flips along the horizontal axis ⇒
//! reverse-each-row-before-row-pass).
//!
//! For the 2D primitives in this module the convention is
//! **FLIPADST_FLIPADST** on the residual — flip both rows and
//! columns before the forward 2D pass. Callers driving the row-only
//! / column-only FLIP arms (V_FLIPADST, H_FLIPADST,
//! ADST_FLIPADST, FLIPADST_ADST) compose `forward_adst_N` per axis
//! with the appropriate per-axis reverse.
//!
//! ## Scope (arc 19 / round 226)
//!
//! Lands forward ADST + FLIPADST for sizes `4 / 8 / 16` (1D and
//! square 2D). The §7.13.3-equivalent forward 2D dispatcher with
//! row-/col-shift envelope and the per-tx-type kernel-selector remain
//! a subsequent arc (mirrors the same scope split as r219/r225 for
//! the forward DCT primitives).

use crate::transform::{inverse_adst, round2};
use std::sync::OnceLock;

/// Returns the cached `N * N` row-major inverse-ADST response matrix
/// for `n in 2..=4` (`N = 1 << n`). Computes on first call, then
/// returns the cached reference for subsequent calls.
///
/// Layout: `M[i + N * k]` is the spatial output `i` of the inverse
/// ADST applied to the unit-coefficient basis input
/// `c_k = [0, …, 4096 (at index k), …, 0]`. Equivalently
/// `M[i + N * k] / 4096` is the `(i, k)` entry of the spec's
/// inverse-ADST matrix. The forward ADST reads the `k`-th *column*
/// of `M` (`M[i + N * k]` over `i in 0..N`) and inner-products with
/// `x` to produce output coefficient `k`.
fn inverse_adst_matrix(n: u32) -> &'static [i64] {
    debug_assert!((2..=4).contains(&n));
    static M2: OnceLock<Vec<i64>> = OnceLock::new();
    static M3: OnceLock<Vec<i64>> = OnceLock::new();
    static M4: OnceLock<Vec<i64>> = OnceLock::new();
    let cell = match n {
        2 => &M2,
        3 => &M3,
        4 => &M4,
        _ => unreachable!("forward ADST size n must be in 2..=4"),
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
            // r = 32 keeps the §7.13.2.1 H() clamp at `[-2^31, 2^31-1]`
            // well above the worst-case butterfly sum magnitude for the
            // unit-coefficient amplitude 4096 (the n = 4 schedule's
            // step 8 produces intermediate sums of order ~10^5 in
            // absolute value).
            inverse_adst(&mut probe, n, 32);
            for i in 0..nn {
                m[i + nn * k] = probe[i];
            }
        }
        m
    })
}

/// Forward 1D ADST for length `N = 1 << n`, `n in 2..=4`. The
/// algebraic transpose of [`crate::transform::inverse_adst`] for the
/// same `n`. Reads `N` spatial values from `t[0..N]` and writes `N`
/// ADST coefficients back into the same slots.
///
/// # Panics
///
/// Panics if `n` is outside `2..=4` or if `t.len() < (1 << n)`.
fn forward_adst_n(t: &mut [i64], n: u32) {
    assert!(
        (2..=4).contains(&n),
        "oxideav-av1 forward_adst_n requires n in 2..=4, got {n}"
    );
    let nn = 1usize << n;
    assert!(
        t.len() >= nn,
        "oxideav-av1 forward_adst_n: buffer too short for length {nn}"
    );
    let m = inverse_adst_matrix(n);
    let mut x = [0i64; 16];
    x[..nn].copy_from_slice(&t[..nn]);
    for k in 0..nn {
        let mut acc: i64 = 0;
        for i in 0..nn {
            acc += m[i + nn * k] * x[i];
        }
        t[k] = round2(acc, 12);
    }
}

/// Forward 1D ADST for length 4 — the transpose of the §7.13.2.6
/// inverse ADST4 kernel.
///
/// # Panics
///
/// Panics if `t.len() < 4`.
pub fn forward_adst_4(t: &mut [i64], _r: u32) {
    forward_adst_n(t, 2);
}

/// Forward 1D ADST for length 8 — the transpose of the §7.13.2.7
/// inverse ADST8 kernel.
///
/// # Panics
///
/// Panics if `t.len() < 8`.
pub fn forward_adst_8(t: &mut [i64], _r: u32) {
    forward_adst_n(t, 3);
}

/// Forward 1D ADST for length 16 — the transpose of the §7.13.2.8
/// inverse ADST16 kernel.
///
/// # Panics
///
/// Panics if `t.len() < 16`.
pub fn forward_adst_16(t: &mut [i64], _r: u32) {
    forward_adst_n(t, 4);
}

/// Internal helper: forward 2D ADST for `N * N` square block,
/// `N in {4, 8, 16}`. Row-then-column composition; no row-/col-shift
/// envelope (same scope contract as the forward DCT 2D primitives).
fn forward_adst_nxn(input: &[i64], side: usize) -> Vec<i64> {
    assert!(
        matches!(side, 4 | 8 | 16),
        "oxideav-av1 forward_adst_nxn requires side in {{4,8,16}}, got {side}"
    );
    assert_eq!(
        input.len(),
        side * side,
        "oxideav-av1 forward_adst_nxn expects side * side = {} samples",
        side * side
    );
    let n: u32 = match side {
        4 => 2,
        8 => 3,
        16 => 4,
        _ => unreachable!(),
    };
    let mut work = input.to_vec();
    let mut row_buf = vec![0i64; side];
    for i in 0..side {
        row_buf.copy_from_slice(&work[i * side..(i + 1) * side]);
        forward_adst_n(&mut row_buf, n);
        work[i * side..(i + 1) * side].copy_from_slice(&row_buf);
    }
    let mut col_buf = vec![0i64; side];
    for j in 0..side {
        for i in 0..side {
            col_buf[i] = work[i * side + j];
        }
        forward_adst_n(&mut col_buf, n);
        for i in 0..side {
            work[i * side + j] = col_buf[i];
        }
    }
    work
}

/// Forward 2D ADST for the `TX_4X4` block size. Row-then-column
/// composition through the 1D length-4 ADST. No row-/col-shift
/// envelope.
///
/// # Panics
///
/// Panics if `input.len() != 16`.
pub fn forward_adst_4x4(input: &[i64]) -> [i64; 16] {
    let v = forward_adst_nxn(input, 4);
    let mut out = [0i64; 16];
    out.copy_from_slice(&v);
    out
}

/// Forward 2D ADST for the `TX_8X8` block size.
///
/// # Panics
///
/// Panics if `input.len() != 64`.
pub fn forward_adst_8x8(input: &[i64]) -> [i64; 64] {
    let v = forward_adst_nxn(input, 8);
    let mut out = [0i64; 64];
    out.copy_from_slice(&v);
    out
}

/// Forward 2D ADST for the `TX_16X16` block size.
///
/// # Panics
///
/// Panics if `input.len() != 256`.
pub fn forward_adst_16x16(input: &[i64]) -> [i64; 256] {
    let v = forward_adst_nxn(input, 16);
    let mut out = [0i64; 256];
    out.copy_from_slice(&v);
    out
}

// ---------------------------------------------------------------------
// FLIPADST.
//
// FLIPADST shares the inverse-ADST butterfly kernel; per §7.13.3 the
// flip is applied externally during the §7.12.3 frame-buffer write.
// On the encoder side the mirror is: flip the residual before the
// forward ADST kernel runs. We expose `forward_flipadst_*` entry points
// that pre-reverse the input, apply the forward ADST, and post-reverse
// the output — yielding the coefficient vector a decoder running
// `inverse_adst` then the §7.12.3 flip would reproduce as the
// original (un-flipped) residual.
// ---------------------------------------------------------------------

/// Forward 1D FLIPADST for length 4. See module docs for the
/// reverse-then-ADST-then-reverse convention.
///
/// # Panics
///
/// Panics if `t.len() < 4`.
pub fn forward_flipadst_4(t: &mut [i64], r: u32) {
    t[..4].reverse();
    forward_adst_4(t, r);
    t[..4].reverse();
}

/// Forward 1D FLIPADST for length 8.
///
/// # Panics
///
/// Panics if `t.len() < 8`.
pub fn forward_flipadst_8(t: &mut [i64], r: u32) {
    t[..8].reverse();
    forward_adst_8(t, r);
    t[..8].reverse();
}

/// Forward 1D FLIPADST for length 16.
///
/// # Panics
///
/// Panics if `t.len() < 16`.
pub fn forward_flipadst_16(t: &mut [i64], r: u32) {
    t[..16].reverse();
    forward_adst_16(t, r);
    t[..16].reverse();
}

/// Internal helper: forward 2D FLIPADST_FLIPADST for `N * N` square
/// block. Reverses both the row order and column order of the
/// residual before the row-then-column ADST pass, then reverses the
/// output cells accordingly.
fn forward_flipadst_nxn(input: &[i64], side: usize) -> Vec<i64> {
    // Reverse rows (vertical flip) and columns (horizontal flip) of
    // the input plane, then apply forward ADST 2D, then reverse the
    // output rows and columns in the same way (the per-axis reverse
    // commutes with the per-axis ADST per the module-doc derivation).
    let mut flipped = vec![0i64; input.len()];
    for i in 0..side {
        for j in 0..side {
            flipped[i * side + j] = input[(side - 1 - i) * side + (side - 1 - j)];
        }
    }
    let coeffs = forward_adst_nxn(&flipped, side);
    let mut out = vec![0i64; input.len()];
    for i in 0..side {
        for j in 0..side {
            out[i * side + j] = coeffs[(side - 1 - i) * side + (side - 1 - j)];
        }
    }
    out
}

/// Forward 2D FLIPADST_FLIPADST for the `TX_4X4` block size.
///
/// # Panics
///
/// Panics if `input.len() != 16`.
pub fn forward_flipadst_4x4(input: &[i64]) -> [i64; 16] {
    let v = forward_flipadst_nxn(input, 4);
    let mut out = [0i64; 16];
    out.copy_from_slice(&v);
    out
}

/// Forward 2D FLIPADST_FLIPADST for the `TX_8X8` block size.
///
/// # Panics
///
/// Panics if `input.len() != 64`.
pub fn forward_flipadst_8x8(input: &[i64]) -> [i64; 64] {
    let v = forward_flipadst_nxn(input, 8);
    let mut out = [0i64; 64];
    out.copy_from_slice(&v);
    out
}

/// Forward 2D FLIPADST_FLIPADST for the `TX_16X16` block size.
///
/// # Panics
///
/// Panics if `input.len() != 256`.
pub fn forward_flipadst_16x16(input: &[i64]) -> [i64; 256] {
    let v = forward_flipadst_nxn(input, 16);
    let mut out = [0i64; 256];
    out.copy_from_slice(&v);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::inverse_adst;

    // -----------------------------------------------------------------
    // ADST 1D: zero, roundtrip via inverse, basis-vector probes.
    // -----------------------------------------------------------------

    #[test]
    fn forward_adst_4_zero_input_yields_zero() {
        let mut t = [0i64; 4];
        forward_adst_4(&mut t, 32);
        assert_eq!(t, [0; 4]);
    }

    #[test]
    fn forward_adst_8_zero_input_yields_zero() {
        let mut t = [0i64; 8];
        forward_adst_8(&mut t, 32);
        assert_eq!(t, [0; 8]);
    }

    #[test]
    fn forward_adst_16_zero_input_yields_zero() {
        let mut t = [0i64; 16];
        forward_adst_16(&mut t, 32);
        assert_eq!(t, [0; 16]);
    }

    /// `inverse(forward(x))` on a small pixel plane: ADST is a fixed
    /// linear map approximated by integer butterflies that accumulate
    /// `Round2` error across the schedule. The round-trip recovers the
    /// input scaled by the squared L2-norm of the basis (≈ `N/2` per
    /// cell for `n in 2..=4`) up to a bounded per-cell noise floor.
    fn adst_roundtrip_via_inverse(n: u32, input: &[i64], max_abs_error: i64) {
        let nn = 1usize << n;
        assert_eq!(input.len(), nn);
        let mut buf = input.to_vec();
        match n {
            2 => forward_adst_4(&mut buf, 32),
            3 => forward_adst_8(&mut buf, 32),
            4 => forward_adst_16(&mut buf, 32),
            _ => unreachable!(),
        }
        inverse_adst(&mut buf, n, 32);
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
    fn forward_adst_4_roundtrip_via_inverse_dc() {
        adst_roundtrip_via_inverse(2, &[1024i64; 4], 4);
    }

    #[test]
    fn forward_adst_4_roundtrip_via_inverse_arbitrary() {
        // Small LCG-pseudo-random pixel plane.
        let mut input = [0i64; 4];
        let mut s: u64 = 0xDEAD_BEEF_F00D_CAFE;
        for v in input.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 32) & 0xFF) as i64 - 128;
        }
        adst_roundtrip_via_inverse(2, &input, 8);
    }

    #[test]
    fn forward_adst_8_roundtrip_via_inverse_dc() {
        adst_roundtrip_via_inverse(3, &[1024i64; 8], 8);
    }

    #[test]
    fn forward_adst_8_roundtrip_via_inverse_arbitrary() {
        let mut input = [0i64; 8];
        let mut s: u64 = 0x1234_5678_9ABC_DEF0;
        for v in input.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 32) & 0xFF) as i64 - 128;
        }
        adst_roundtrip_via_inverse(3, &input, 16);
    }

    #[test]
    fn forward_adst_16_roundtrip_via_inverse_dc() {
        adst_roundtrip_via_inverse(4, &[1024i64; 16], 16);
    }

    #[test]
    fn forward_adst_16_roundtrip_via_inverse_arbitrary() {
        let mut input = [0i64; 16];
        let mut s: u64 = 0xFEED_FACE_BAAD_F00D;
        for v in input.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 32) & 0xFF) as i64 - 128;
        }
        adst_roundtrip_via_inverse(4, &input, 32);
    }

    #[test]
    fn forward_adst_4_matrix_probe_is_transpose_of_inverse() {
        // Probing the forward ADST with a unit-pixel input `[4096, 0,
        // 0, 0]` produces the coefficient vector whose entries equal
        // the *row 0* of `M_inv_adst[2]` — i.e. the inner product of
        // each column of `M_inv` with the basis vector `e_0`. The
        // canonical anchor: compare the same row directly against an
        // inverse-ADST probe.
        let mut inv_probe = [0i64; 4];
        for (i, slot) in inv_probe.iter_mut().enumerate() {
            *slot = if i == 0 { 4096 } else { 0 };
        }
        // Run inverse_adst on each basis input to materialise the
        // full M_inv matrix locally for comparison.
        let mut mat = [0i64; 16];
        for k in 0..4 {
            let mut probe = [0i64; 4];
            probe[k] = 4096;
            inverse_adst(&mut probe, 2, 32);
            for i in 0..4 {
                mat[i + 4 * k] = probe[i];
            }
        }
        // Forward ADST of `[4096, 0, 0, 0]` should produce
        // `Round2(M_inv[0][k] * 4096, 12) = M_inv[0][k]` for k in 0..4
        // (the multiplication by 4096 cancels the divide).
        let mut t = [0i64; 4];
        t[0] = 4096;
        forward_adst_4(&mut t, 32);
        for k in 0..4 {
            assert_eq!(
                t[k],
                round2(mat[4 * k] * 4096, 12),
                "row 0 col k={k} mismatch"
            );
        }
    }

    // -----------------------------------------------------------------
    // ADST 2D.
    // -----------------------------------------------------------------

    #[test]
    fn forward_adst_4x4_zero_input_yields_zero() {
        let input = [0i64; 16];
        let out = forward_adst_4x4(&input);
        assert_eq!(out, [0; 16]);
    }

    #[test]
    fn forward_adst_8x8_zero_input_yields_zero() {
        let input = [0i64; 64];
        let out = forward_adst_8x8(&input);
        assert_eq!(out, [0; 64]);
    }

    #[test]
    fn forward_adst_16x16_zero_input_yields_zero() {
        let input = [0i64; 256];
        let out = forward_adst_16x16(&input);
        assert_eq!(out, [0; 256]);
    }

    /// 2D roundtrip: forward_adst_NxN then row-then-column inverse
    /// ADST recovers the input scaled by `(N/2)^2` per cell.
    fn adst_2d_roundtrip(n: u32, input: &[i64], max_abs_error: i64) {
        let side = 1usize << n;
        assert_eq!(input.len(), side * side);
        let coeffs: Vec<i64> = match side {
            4 => forward_adst_4x4(input).to_vec(),
            8 => forward_adst_8x8(input).to_vec(),
            16 => forward_adst_16x16(input).to_vec(),
            _ => unreachable!(),
        };
        // Inverse 2D: row pass then column pass through inverse_adst.
        let mut work = coeffs;
        let mut row_buf = vec![0i64; side];
        for i in 0..side {
            row_buf.copy_from_slice(&work[i * side..(i + 1) * side]);
            inverse_adst(&mut row_buf, n, 32);
            work[i * side..(i + 1) * side].copy_from_slice(&row_buf);
        }
        let mut col_buf = vec![0i64; side];
        for j in 0..side {
            for i in 0..side {
                col_buf[i] = work[i * side + j];
            }
            inverse_adst(&mut col_buf, n, 32);
            for i in 0..side {
                work[i * side + j] = col_buf[i];
            }
        }
        let scale = ((side as i64) / 2).pow(2);
        for (i, (&got, &orig)) in work.iter().zip(input.iter()).enumerate() {
            let expected = scale * orig;
            let err = (got - expected).abs();
            assert!(
                err <= max_abs_error,
                "n={n} cell {i}: orig={orig}, got={got}, expected≈{expected} (|err| {err} > bound {max_abs_error})"
            );
        }
    }

    #[test]
    fn forward_adst_4x4_roundtrip_via_inverse_dc() {
        let input = [256i64; 16];
        adst_2d_roundtrip(2, &input, 16);
    }

    #[test]
    fn forward_adst_8x8_roundtrip_via_inverse_dc() {
        let input = [256i64; 64];
        adst_2d_roundtrip(3, &input, 64);
    }

    #[test]
    fn forward_adst_16x16_roundtrip_via_inverse_dc() {
        let input = [256i64; 256];
        adst_2d_roundtrip(4, &input, 256);
    }

    // -----------------------------------------------------------------
    // FLIPADST 1D / 2D: round-trip via the convention that FLIPADST
    // and ADST share the kernel and the flip is the involution
    // around `i -> N - 1 - i`. `forward_flipadst_N(x) ==
    // reverse(forward_adst_N(reverse(x)))`; on flat-DC input this is
    // exactly equal to `forward_adst_N(x)` (flat plane is invariant
    // under reversal). Use a non-symmetric input to exercise the
    // actual reverse path.
    // -----------------------------------------------------------------

    #[test]
    fn forward_flipadst_4_zero_input_yields_zero() {
        let mut t = [0i64; 4];
        forward_flipadst_4(&mut t, 32);
        assert_eq!(t, [0; 4]);
    }

    #[test]
    fn forward_flipadst_4_on_dc_is_reverse_of_adst() {
        // DC input is reversal-invariant, but `forward_flipadst` =
        // reverse → forward_adst → reverse. Pre-reverse of a DC plane
        // is unchanged, post-reverse permutes the output coefficient
        // order ⇒ `forward_flipadst_4(dc) ==
        // reverse(forward_adst_4(dc))`.
        let mut a = [256i64; 4];
        let mut b = [256i64; 4];
        forward_adst_4(&mut a, 32);
        forward_flipadst_4(&mut b, 32);
        a.reverse();
        assert_eq!(a, b);
    }

    #[test]
    fn forward_flipadst_4_on_ramp_is_reverse_of_adst_of_reverse() {
        let input = [10i64, 20, 30, 40];
        let mut flipped_then_adst = input;
        flipped_then_adst.reverse();
        forward_adst_4(&mut flipped_then_adst, 32);
        flipped_then_adst.reverse();
        let mut got = input;
        forward_flipadst_4(&mut got, 32);
        assert_eq!(got, flipped_then_adst);
    }

    #[test]
    fn forward_flipadst_8_on_ramp_is_reverse_of_adst_of_reverse() {
        let input = [10i64, 20, 30, 40, 50, 60, 70, 80];
        let mut flipped_then_adst = input;
        flipped_then_adst.reverse();
        forward_adst_8(&mut flipped_then_adst, 32);
        flipped_then_adst.reverse();
        let mut got = input;
        forward_flipadst_8(&mut got, 32);
        assert_eq!(got, flipped_then_adst);
    }

    #[test]
    fn forward_flipadst_16_on_ramp_is_reverse_of_adst_of_reverse() {
        let mut input = [0i64; 16];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = 10 * (i as i64 + 1);
        }
        let mut flipped_then_adst = input;
        flipped_then_adst.reverse();
        forward_adst_16(&mut flipped_then_adst, 32);
        flipped_then_adst.reverse();
        let mut got = input;
        forward_flipadst_16(&mut got, 32);
        assert_eq!(got, flipped_then_adst);
    }

    #[test]
    fn forward_flipadst_4x4_zero_input_yields_zero() {
        let input = [0i64; 16];
        let out = forward_flipadst_4x4(&input);
        assert_eq!(out, [0; 16]);
    }

    #[test]
    fn forward_flipadst_4x4_on_dc_is_double_reverse_of_adst() {
        // Same shape as the 1D case: DC plane is invariant under
        // pre-reverse-both-axes, so `forward_flipadst_4x4(dc) ==
        // double_reverse(forward_adst_4x4(dc))`. The double reverse
        // mirrors row index `i -> 3 - i` and column index `j -> 3 -
        // j`, which is what `forward_flipadst_nxn` applies to the
        // output cells.
        let input = [256i64; 16];
        let a = forward_adst_4x4(&input);
        let b = forward_flipadst_4x4(&input);
        let mut expected = [0i64; 16];
        for i in 0..4 {
            for j in 0..4 {
                expected[i * 4 + j] = a[(3 - i) * 4 + (3 - j)];
            }
        }
        assert_eq!(b, expected);
    }

    #[test]
    fn forward_flipadst_8x8_zero_input_yields_zero() {
        let input = [0i64; 64];
        let out = forward_flipadst_8x8(&input);
        assert_eq!(out, [0; 64]);
    }

    #[test]
    fn forward_flipadst_16x16_zero_input_yields_zero() {
        let input = [0i64; 256];
        let out = forward_flipadst_16x16(&input);
        assert_eq!(out, [0; 256]);
    }

    /// 2D FLIPADST_FLIPADST is the 2D ADST applied to the
    /// double-flipped (vertical + horizontal) residual, with the
    /// output cells re-flipped accordingly. Verifies the equivalence
    /// directly: `forward_flipadst_NxN(x) ==
    /// double_flip(forward_adst_NxN(double_flip(x)))`.
    #[test]
    fn forward_flipadst_4x4_equals_adst_of_double_flipped() {
        let mut input = [0i64; 16];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = (i as i64) - 8;
        }
        // Build the double-flipped input.
        let mut flipped = [0i64; 16];
        for i in 0..4 {
            for j in 0..4 {
                flipped[i * 4 + j] = input[(3 - i) * 4 + (3 - j)];
            }
        }
        let adst_of_flipped = forward_adst_4x4(&flipped);
        // Double-flip the output back.
        let mut expected = [0i64; 16];
        for i in 0..4 {
            for j in 0..4 {
                expected[i * 4 + j] = adst_of_flipped[(3 - i) * 4 + (3 - j)];
            }
        }
        let got = forward_flipadst_4x4(&input);
        assert_eq!(got, expected);
    }
}
