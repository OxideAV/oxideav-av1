//! AV1 transform cosine / sine constants — §7.7.1.
//!
//! `COS_PI[k] = round(cos(k * π / 128) * 2^COS_BITS)` for `k ∈ 0..=64`,
//! with `COS_BITS = 12`. The spec references a handful of landmarks:
//!
//! - `cos_pi[0]  = 4096`
//! - `cos_pi[8]  = 4017`   (`cos(π/16)  * 4096`)
//! - `cos_pi[16] = 3784`   (`cos(π/8)   * 4096`)
//! - `cos_pi[32] = 2896`   (`cos(π/4)   * 4096`)
//! - `cos_pi[48] = 1567`   (`cos(3π/8)  * 4096`)
//! - `cos_pi[56] = 799`    (`cos(7π/16) * 4096`)
//! - `cos_pi[64] = 0`      (`cos(π/2)   * 4096`)
//!
//! `SIN_PI_K_9[k]` (k=1..=4) = `round(sin(k * π / 9) * 2^COS_BITS)`.
//! Used by the 4-point inverse ADST (spec §7.7.2.3).

/// Bit precision of the cosine constants.
pub const COS_BITS: i32 = 12;

/// 65-entry cosine table indexed by the numerator of `k * π / 128`.
///
/// We embed the precomputed integer values rather than computing them
/// at runtime: this keeps the crate free of `std::f64` dependence and
/// matches the spec's reference values exactly.
pub static COS_PI: [i32; 65] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101, 0,
];

/// `sin(k * π / 9) * 2^COS_BITS`, rounded.
pub const SIN_PI_1_9: i32 = 1321;
pub const SIN_PI_2_9: i32 = 2482;
pub const SIN_PI_3_9: i32 = 3344;
pub const SIN_PI_4_9: i32 = 3803;

/// `round((w0 * in0 + w1 * in1) / 2^COS_BITS)` with symmetric
/// rounding. Spec §7.7.1.2.
#[inline]
pub fn half_btf(w0: i32, in0: i32, w1: i32, in1: i32) -> i32 {
    (w0 * in0 + w1 * in1 + (1 << (COS_BITS - 1))) >> COS_BITS
}

/// `round2(x, shift) = (x + (1 << (shift-1))) >> shift`.
#[inline]
pub fn round2(x: i32, shift: u32) -> i32 {
    if shift == 0 {
        x
    } else {
        (x + (1 << (shift - 1))) >> shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cos_pi_landmarks() {
        assert_eq!(COS_PI[0], 4096);
        assert_eq!(COS_PI[8], 4017);
        assert_eq!(COS_PI[16], 3784);
        assert_eq!(COS_PI[32], 2896);
        assert_eq!(COS_PI[48], 1567);
        assert_eq!(COS_PI[56], 799);
        assert_eq!(COS_PI[64], 0);
    }

    #[test]
    fn cos_pi_monotone_descending() {
        let mut prev = i32::MAX;
        for &v in COS_PI.iter() {
            assert!(v <= prev, "non-monotone: {v} > {prev}");
            prev = v;
        }
    }

    #[test]
    fn half_btf_symmetric_rounding() {
        // (0.5 * 4096 + 0.5 * 4096) / 4096 with cos_bits=12 -> 4096.
        let got = half_btf(COS_PI[32], 4096, COS_PI[32], 4096);
        // 2896*4096 + 2896*4096 = 23,724,032 → >>12 + round = 5792.
        assert_eq!(got, 5792);
    }

    #[test]
    fn round2_even() {
        assert_eq!(round2(3, 1), 2);
        assert_eq!(round2(-3, 1), -1); // (-3 + 1) >> 1 = -1 (arith shift)
        assert_eq!(round2(4, 2), 1);
    }
}
