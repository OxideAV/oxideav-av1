//! Spec-correct identity inverse transforms — §7.13.2.11–.14.
//!
//! The legacy [`crate::transform::idtx`] module implements simplified
//! identity kernels (uniform `<<= 1`, with `idtx32` a no-op) that the
//! existing [`crate::transform::inverse_2d`] entry point and its
//! callers in `decode/superblock.rs` are calibrated against. Round 22
//! introduces this **separate** spec-faithful module so the new
//! [`crate::transform::inverse_2d_spec`] entry point can land the
//! §7.13.3 row/col shift accounting (Transform_Row_Shift +
//! `colShift = 4` + rectangular `2896` pre-row scale + spec IDTX
//! magnitudes) without disturbing the PSNR-tuned legacy path.
//!
//! Per the spec each 1-D identity is a per-element multiply with
//! `Round2(.,12)`-rounding for sizes 4/16, an exact `× 2` at size 8
//! and `× 4` at size 32. Size 64 IDTX is **not** instantiated by AV1
//! (the spec's `inverse identity transform process` accepts only
//! `n ∈ {2,3,4,5}`, i.e. lengths 4/8/16/32) and is rejected by
//! [`crate::transform::run_1d_spec`].

use super::cos_pi::round2;

/// 4-point spec IDTX — `T[i] = Round2(T[i] * 5793, 12)`.
///
/// `5793 / 4096 ≈ 1.4143 ≈ √2` — the inverse-identity at length 4
/// carries the same `√2` row-orthonormalisation factor that the spec's
/// rectangular-aspect `2896` (= `√2 * 4096 / 2`) helper uses on the
/// row pre-scale.
pub fn idtx4_spec(x: &mut [i32; 4]) {
    for v in x.iter_mut() {
        *v = round2(*v * 5793, 12);
    }
}

/// 8-point spec IDTX — `T[i] = T[i] * 2`.
///
/// Matches the legacy [`crate::transform::idtx::idtx8`] verbatim.
/// Re-exported here so `inverse_2d_spec` can dispatch through a single
/// table without crossing modules.
pub fn idtx8_spec(x: &mut [i32; 8]) {
    for v in x.iter_mut() {
        *v *= 2;
    }
}

/// 16-point spec IDTX — `T[i] = Round2(T[i] * 11586, 12)`.
///
/// `11586 / 4096 ≈ 2.8286 ≈ 2√2`.
pub fn idtx16_spec(x: &mut [i32; 16]) {
    for v in x.iter_mut() {
        *v = round2(*v * 11586, 12);
    }
}

/// 32-point spec IDTX — `T[i] = T[i] * 4`.
pub fn idtx32_spec(x: &mut [i32; 32]) {
    for v in x.iter_mut() {
        *v *= 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Round2(1 * 5793, 12) = (5793 + 2048) >> 12 = 7841 >> 12 = 1`
    /// — the unit-input scale lands at 1 because the rounding bias
    /// dominates the sub-LSB fraction. Larger inputs scale by
    /// `5793/4096 ≈ √2` per spec §7.13.2.11.
    #[test]
    fn idtx4_spec_unit_and_scale() {
        let mut x = [1, 0, -1, 4096];
        idtx4_spec(&mut x);
        assert_eq!(x[0], 1, "1 -> Round2(5793, 12) = 1");
        assert_eq!(x[1], 0);
        assert_eq!(x[2], -1, "-1 -> Round2(-5793, 12) = -1");
        // 4096 * 5793 = 23,724,128 -> >>12 + round = 5793.
        assert_eq!(x[3], 5793, "4096 -> 5793 (exact spec scale)");
    }

    /// Spec §7.13.2.12 IDTX-8 is `T[i] *= 2` exactly.
    #[test]
    fn idtx8_spec_doubles_each_sample() {
        let mut x = [1, -2, 3, -4, 5, -6, 7, -8];
        idtx8_spec(&mut x);
        assert_eq!(x, [2, -4, 6, -8, 10, -12, 14, -16]);
    }

    /// `Round2(4096 * 11586, 12) = 11586` per spec §7.13.2.13.
    #[test]
    fn idtx16_spec_unit_and_scale() {
        let mut x = [0i32; 16];
        x[0] = 4096;
        x[5] = -4096;
        idtx16_spec(&mut x);
        assert_eq!(x[0], 11586);
        assert_eq!(x[5], -11586);
        for v in x.iter().enumerate().filter(|(i, _)| *i != 0 && *i != 5) {
            assert_eq!(*v.1, 0);
        }
    }

    /// Spec §7.13.2.14 IDTX-32 is `T[i] *= 4` exactly.
    #[test]
    fn idtx32_spec_quadruples_each_sample() {
        let mut x = [0i32; 32];
        for (i, v) in x.iter_mut().enumerate() {
            *v = i as i32 - 16;
        }
        let before = x;
        idtx32_spec(&mut x);
        for (b, a) in before.iter().zip(x.iter()) {
            assert_eq!(*a, *b * 4);
        }
    }
}
