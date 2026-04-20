//! 4-point inverse ADST — §7.7.2.3.

use super::cos_pi::{round2, COS_BITS, SIN_PI_1_9, SIN_PI_2_9, SIN_PI_3_9, SIN_PI_4_9};

/// In-place 4-point inverse ADST. `x` must have exactly 4 entries.
pub fn iadst4(x: &mut [i32; 4]) {
    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let x3 = x[3];

    let mut s0 = SIN_PI_1_9 * x0;
    let mut s1 = SIN_PI_2_9 * x0;
    let s2 = SIN_PI_3_9 * x1;
    let s3 = SIN_PI_4_9 * x2;
    let s4 = SIN_PI_1_9 * x2;
    let s5 = SIN_PI_2_9 * x3;
    let s6 = SIN_PI_4_9 * x3;
    let s7 = x0 - x2 + x3;

    s0 = s0 + s3 + s5;
    s1 = s1 - s4 - s6;
    let s3n = s2;
    let s2n = SIN_PI_3_9 * s7;

    x[0] = round2(s0 + s3n, COS_BITS as u32);
    x[1] = round2(s1 + s3n, COS_BITS as u32);
    x[2] = round2(s2n, COS_BITS as u32);
    x[3] = round2(s0 + s1 - s3n, COS_BITS as u32);
}

/// In-place 4-point inverse flipped ADST. `IADST4` followed by output
/// reversal (spec §7.7.2.2).
pub fn iflipadst4(x: &mut [i32; 4]) {
    iadst4(x);
    x.swap(0, 3);
    x.swap(1, 2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iadst4_linearity_within_tolerance() {
        let a = [100i32, 50, -25, 10];
        let b = [-30i32, 40, 5, -12];
        let mut sum = [0i32; 4];
        for i in 0..4 {
            sum[i] = a[i] + b[i];
        }
        let mut ac = a;
        let mut bc = b;
        iadst4(&mut ac);
        iadst4(&mut bc);
        iadst4(&mut sum);
        for i in 0..4 {
            let want = ac[i] + bc[i];
            let diff = (sum[i] - want).abs();
            assert!(diff <= 1, "linearity at {i}: diff {diff}");
        }
    }

    #[test]
    fn iflipadst4_reverses_iadst4() {
        let mut a = [10i32, 20, -5, 7];
        let mut af = a;
        iadst4(&mut a);
        iflipadst4(&mut af);
        assert_eq!(af[0], a[3]);
        assert_eq!(af[1], a[2]);
        assert_eq!(af[2], a[1]);
        assert_eq!(af[3], a[0]);
    }
}
