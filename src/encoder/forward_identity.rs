//! Forward identity (IDTX) primitives — the encoder counterparts of
//! the §7.13.2.11..§7.13.2.14 inverse identity kernels for sizes
//! `4 / 8 / 16 / 32`.
//!
//! ## Derivation (clean-room)
//!
//! Per §7.13.2.11..§7.13.2.14 the inverse identity kernels are pure
//! scalar maps (multiplications by a constant, optionally followed by
//! `Round2(_, 12)`):
//!
//! * `inverse_identity4(t)`  : `t[i] = Round2(t[i] * 5793, 12)`.
//! * `inverse_identity8(t)`  : `t[i] = t[i] * 2`.
//! * `inverse_identity16(t)` : `t[i] = Round2(t[i] * 11586, 12)`.
//! * `inverse_identity32(t)` : `t[i] = t[i] * 4`.
//!
//! These are diagonal `M_inv = diag(c, c, …, c)` linear maps with
//! `c = inverse_identity(unit) = inverse_identity([1])[0]`. The
//! forward kernel is therefore also a diagonal scalar map (the
//! transpose of a diagonal is the diagonal itself). We materialise
//! the per-size `c` via the same matrix-cache probe pattern as the
//! forward DCT / ADST primitives for uniformity, even though the
//! matrix collapses to a single value here.
//!
//! ## Forward identity values
//!
//! By probing `inverse_identity(N, [4096, 0, …])` at amplitude 4096:
//!
//! * `n = 2` (N = 4):  `inverse[0] = Round2(4096 * 5793, 12) =
//!   Round2(23728128, 12) = (23728128 + 2048) >> 12 = 5793`.
//! * `n = 3` (N = 8):  `inverse[0] = 4096 * 2 = 8192`.
//! * `n = 4` (N = 16): `inverse[0] = Round2(4096 * 11586, 12) =
//!   Round2(47456256, 12) = (47456256 + 2048) >> 12 = 11586`.
//! * `n = 5` (N = 32): `inverse[0] = 4096 * 4 = 16384`.
//!
//! `M_inv = diag(c / 4096)`, so the forward kernel is `forward[k] =
//! Round2(c * x[k], 12)` per cell. For the integer-multiply identity
//! arms (n = 3, n = 5) the round-trip `inverse(forward(x))` is
//! `2 * (Round2(8192 * x, 12)) = 2 * (8192 * x + 2048) >> 12 ≈ 4 * x`
//! (n = 3) or `4 * (16384 * x + 2048) >> 12 ≈ 16 * x` (n = 5) — i.e.
//! the input scaled by `(c / 4096)^2`. For the rounded-multiply arms
//! the same identity holds within a `Round2` floor.
//!
//! ## Scope (arc 19 / round 226)
//!
//! Lands forward identity for sizes `4 / 8 / 16 / 32` (1D and square
//! 2D), matching the spec's `inverse_identity` coverage. The square
//! 2D primitive applies the per-axis forward identity on rows then
//! columns — i.e. `forward_identity_NxN(x)[i, j] = forward_identity(
//! forward_identity(x_row_i)_col_j)`. Because the per-axis kernel
//! is a scalar multiply this composes to a single `c^2` scalar
//! multiply per cell.

use crate::transform::{inverse_identity, round2};
use std::sync::OnceLock;

/// Returns the cached forward-identity scalar `c` for `n in 2..=5`
/// (`N = 1 << n`). Computes on first call by probing
/// [`crate::transform::inverse_identity`] at amplitude 4096; the
/// returned `c` is the integer multiplier such that `forward(x)[k] =
/// Round2(c * x[k], 12)`.
fn forward_identity_scalar(n: u32) -> i64 {
    debug_assert!((2..=5).contains(&n));
    static C2: OnceLock<i64> = OnceLock::new();
    static C3: OnceLock<i64> = OnceLock::new();
    static C4: OnceLock<i64> = OnceLock::new();
    static C5: OnceLock<i64> = OnceLock::new();
    let cell = match n {
        2 => &C2,
        3 => &C3,
        4 => &C4,
        5 => &C5,
        _ => unreachable!("forward identity size n must be in 2..=5"),
    };
    *cell.get_or_init(|| {
        let nn = 1usize << n;
        let mut probe = vec![0i64; nn];
        probe[0] = 4096;
        inverse_identity(&mut probe, n);
        probe[0]
    })
}

fn forward_identity_n(t: &mut [i64], n: u32) {
    assert!(
        (2..=5).contains(&n),
        "oxideav-av1 forward_identity_n requires n in 2..=5, got {n}"
    );
    let nn = 1usize << n;
    assert!(
        t.len() >= nn,
        "oxideav-av1 forward_identity_n: buffer too short for length {nn}"
    );
    let c = forward_identity_scalar(n);
    for slot in t.iter_mut().take(nn) {
        *slot = round2(*slot * c, 12);
    }
}

/// Forward 1D identity for length 4 — the transpose (and equal to
/// the inverse, since it's a scalar diagonal map) of §7.13.2.11
/// `inverse_identity4`. Per cell: `out = Round2(5793 * in, 12)`.
///
/// # Panics
///
/// Panics if `t.len() < 4`.
pub fn forward_idtx_4(t: &mut [i64]) {
    forward_identity_n(t, 2);
}

/// Forward 1D identity for length 8 — the transpose of §7.13.2.12
/// `inverse_identity8`. Per cell: `out = Round2(8192 * in, 12) =
/// 2 * in` exactly (the `8192 = 2 * 4096` scalar is an exact power of
/// two).
///
/// # Panics
///
/// Panics if `t.len() < 8`.
pub fn forward_idtx_8(t: &mut [i64]) {
    forward_identity_n(t, 3);
}

/// Forward 1D identity for length 16 — the transpose of §7.13.2.13
/// `inverse_identity16`. Per cell: `out = Round2(11586 * in, 12)`.
///
/// # Panics
///
/// Panics if `t.len() < 16`.
pub fn forward_idtx_16(t: &mut [i64]) {
    forward_identity_n(t, 4);
}

/// Forward 1D identity for length 32 — the transpose of §7.13.2.14
/// `inverse_identity32`. Per cell: `out = Round2(16384 * in, 12) =
/// 4 * in` exactly.
///
/// # Panics
///
/// Panics if `t.len() < 32`.
pub fn forward_idtx_32(t: &mut [i64]) {
    forward_identity_n(t, 5);
}

fn forward_idtx_nxn(input: &[i64], side: usize) -> Vec<i64> {
    assert!(
        matches!(side, 4 | 8 | 16 | 32),
        "oxideav-av1 forward_idtx_nxn requires side in {{4,8,16,32}}, got {side}"
    );
    assert_eq!(
        input.len(),
        side * side,
        "oxideav-av1 forward_idtx_nxn expects side * side = {} samples",
        side * side
    );
    let n: u32 = match side {
        4 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        _ => unreachable!(),
    };
    let mut work = input.to_vec();
    let mut row_buf = vec![0i64; side];
    for i in 0..side {
        row_buf.copy_from_slice(&work[i * side..(i + 1) * side]);
        forward_identity_n(&mut row_buf, n);
        work[i * side..(i + 1) * side].copy_from_slice(&row_buf);
    }
    let mut col_buf = vec![0i64; side];
    for j in 0..side {
        for i in 0..side {
            col_buf[i] = work[i * side + j];
        }
        forward_identity_n(&mut col_buf, n);
        for i in 0..side {
            work[i * side + j] = col_buf[i];
        }
    }
    work
}

/// Forward 2D identity (IDTX) for the `TX_4X4` block size.
///
/// # Panics
///
/// Panics if `input.len() != 16`.
pub fn forward_idtx_4x4(input: &[i64]) -> [i64; 16] {
    let v = forward_idtx_nxn(input, 4);
    let mut out = [0i64; 16];
    out.copy_from_slice(&v);
    out
}

/// Forward 2D identity (IDTX) for the `TX_8X8` block size.
///
/// # Panics
///
/// Panics if `input.len() != 64`.
pub fn forward_idtx_8x8(input: &[i64]) -> [i64; 64] {
    let v = forward_idtx_nxn(input, 8);
    let mut out = [0i64; 64];
    out.copy_from_slice(&v);
    out
}

/// Forward 2D identity (IDTX) for the `TX_16X16` block size.
///
/// # Panics
///
/// Panics if `input.len() != 256`.
pub fn forward_idtx_16x16(input: &[i64]) -> [i64; 256] {
    let v = forward_idtx_nxn(input, 16);
    let mut out = [0i64; 256];
    out.copy_from_slice(&v);
    out
}

/// Forward 2D identity (IDTX) for the `TX_32X32` block size.
/// Returned as `Vec<i64>` to dodge a ~8 KiB stack allocation.
///
/// # Panics
///
/// Panics if `input.len() != 1024`.
pub fn forward_idtx_32x32(input: &[i64]) -> Vec<i64> {
    forward_idtx_nxn(input, 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_idtx_4_zero_input_yields_zero() {
        let mut t = [0i64; 4];
        forward_idtx_4(&mut t);
        assert_eq!(t, [0; 4]);
    }

    #[test]
    fn forward_idtx_8_zero_input_yields_zero() {
        let mut t = [0i64; 8];
        forward_idtx_8(&mut t);
        assert_eq!(t, [0; 8]);
    }

    #[test]
    fn forward_idtx_16_zero_input_yields_zero() {
        let mut t = [0i64; 16];
        forward_idtx_16(&mut t);
        assert_eq!(t, [0; 16]);
    }

    #[test]
    fn forward_idtx_32_zero_input_yields_zero() {
        let mut t = [0i64; 32];
        forward_idtx_32(&mut t);
        assert_eq!(t, [0; 32]);
    }

    #[test]
    fn forward_idtx_4_per_cell_scalar_matches_spec_5793() {
        // forward_idtx_4: per cell out = Round2(5793 * in, 12).
        // For in = 1024: Round2(5793 * 1024, 12) = Round2(5932032, 12) =
        // (5932032 + 2048) >> 12 = 5934080 >> 12 = 1448.
        let mut t = [1024i64; 4];
        forward_idtx_4(&mut t);
        assert_eq!(t, [1448; 4]);
    }

    #[test]
    fn forward_idtx_8_per_cell_doubles_exactly() {
        // n = 3: scalar = 8192. Round2(8192 * in, 12) = 2 * in exactly.
        let mut t = [1000i64; 8];
        forward_idtx_8(&mut t);
        assert_eq!(t, [2000; 8]);
    }

    #[test]
    fn forward_idtx_16_per_cell_scalar_matches_spec_11586() {
        // Round2(11586 * 1024, 12) = (11864064 + 2048) >> 12 =
        // 11866112 >> 12 = 2897.
        let mut t = [1024i64; 16];
        forward_idtx_16(&mut t);
        assert_eq!(t, [2897; 16]);
    }

    #[test]
    fn forward_idtx_32_per_cell_quadruples_exactly() {
        // n = 5: scalar = 16384. Round2(16384 * in, 12) = 4 * in.
        let mut t = [1000i64; 32];
        forward_idtx_32(&mut t);
        assert_eq!(t, [4000; 32]);
    }

    /// `inverse(forward(x))` for the integer-multiply arms (n = 3,
    /// n = 5) is exact: `2 * (2 * x) = 4 * x` for n = 3, `4 * (4 *
    /// x) = 16 * x` for n = 5.
    #[test]
    fn forward_idtx_8_roundtrip_via_inverse_is_exact_4x() {
        let mut buf = [37i64, -22, 18, 0, -100, 64, 7, -1];
        forward_idtx_8(&mut buf);
        crate::transform::inverse_identity(&mut buf, 3);
        let expected: [i64; 8] = [37 * 4, -22 * 4, 18 * 4, 0, -100 * 4, 64 * 4, 7 * 4, -4];
        assert_eq!(buf, expected);
    }

    #[test]
    fn forward_idtx_32_roundtrip_via_inverse_is_exact_16x() {
        let mut buf = [0i64; 32];
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = (i as i64) - 16;
        }
        let orig = buf;
        forward_idtx_32(&mut buf);
        crate::transform::inverse_identity(&mut buf, 5);
        for (i, (&got, &o)) in buf.iter().zip(orig.iter()).enumerate() {
            assert_eq!(got, 16 * o, "n=5 cell {i}: got {got}, expected {}", 16 * o);
        }
    }

    /// `inverse(forward(x))` for the rounded-multiply arms (n = 2,
    /// n = 4) recovers `x` scaled by `(c / 4096)^2` within a small
    /// per-cell Round2 floor.
    #[test]
    fn forward_idtx_4_roundtrip_via_inverse_within_noise() {
        // Forward c = 5793, inverse c = 5793 ⇒ ratio (5793 /
        // 4096)^2 ≈ 2.0. For x = 1024, expected ≈ 2048.
        let orig = [1024i64; 4];
        let mut buf = orig;
        forward_idtx_4(&mut buf);
        crate::transform::inverse_identity(&mut buf, 2);
        for (i, &got) in buf.iter().enumerate() {
            // Tight bound: the per-cell error is at most a few LSBs
            // from two stacked Round2(_, 12) operations.
            assert!(
                (got - 2 * orig[i]).abs() <= 4,
                "n=2 cell {i}: got {got}, expected≈{} (|err| {})",
                2 * orig[i],
                (got - 2 * orig[i]).abs()
            );
        }
    }

    #[test]
    fn forward_idtx_16_roundtrip_via_inverse_within_noise() {
        // c = 11586. (11586 / 4096)^2 ≈ 8.0. For x = 1024,
        // expected ≈ 8192.
        let orig = [1024i64; 16];
        let mut buf = orig;
        forward_idtx_16(&mut buf);
        crate::transform::inverse_identity(&mut buf, 4);
        for (i, &got) in buf.iter().enumerate() {
            assert!(
                (got - 8 * orig[i]).abs() <= 16,
                "n=4 cell {i}: got {got}, expected≈{} (|err| {})",
                8 * orig[i],
                (got - 8 * orig[i]).abs()
            );
        }
    }

    // -----------------------------------------------------------------
    // 2D IDTX.
    // -----------------------------------------------------------------

    #[test]
    fn forward_idtx_4x4_zero_input_yields_zero() {
        let input = [0i64; 16];
        let out = forward_idtx_4x4(&input);
        assert_eq!(out, [0; 16]);
    }

    #[test]
    fn forward_idtx_8x8_zero_input_yields_zero() {
        let input = [0i64; 64];
        let out = forward_idtx_8x8(&input);
        assert_eq!(out, [0; 64]);
    }

    #[test]
    fn forward_idtx_16x16_zero_input_yields_zero() {
        let input = [0i64; 256];
        let out = forward_idtx_16x16(&input);
        assert_eq!(out, [0; 256]);
    }

    #[test]
    fn forward_idtx_32x32_zero_input_yields_zero() {
        let input = vec![0i64; 1024];
        let out = forward_idtx_32x32(&input);
        assert_eq!(out.len(), 1024);
        for &v in out.iter() {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn forward_idtx_8x8_per_cell_quadruples_exactly() {
        // 2D = row then column, each scalar 2; combined scale 4.
        let input = [37i64; 64];
        let out = forward_idtx_8x8(&input);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 148, "cell {i}: got {v}");
        }
    }

    #[test]
    fn forward_idtx_32x32_per_cell_multiplies_by_sixteen_exactly() {
        // Row scale 4, column scale 4, combined 16.
        let input = vec![25i64; 1024];
        let out = forward_idtx_32x32(&input);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 400, "cell {i}: got {v}");
        }
    }
}
