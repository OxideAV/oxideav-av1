//! 4-point inverse DCT — §7.7.2.1.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/transform/idct4.go`
//! (MIT, KarpelesLab/goavif).

use super::cos_pi::{half_btf, COS_PI};

/// In-place 4-point inverse DCT. `x` must have exactly 4 entries.
pub fn idct4(x: &mut [i32; 4]) {
    let t0 = half_btf(COS_PI[32], x[0], COS_PI[32], x[2]);
    let t1 = half_btf(COS_PI[32], x[0], -COS_PI[32], x[2]);
    let t2 = half_btf(COS_PI[48], x[1], -COS_PI[16], x[3]);
    let t3 = half_btf(COS_PI[16], x[1], COS_PI[48], x[3]);
    x[0] = t0 + t3;
    x[1] = t1 + t2;
    x[2] = t1 - t2;
    x[3] = t0 - t3;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct4_dc_roundtrip_produces_constant() {
        let mut x = [32768i32, 0, 0, 0];
        idct4(&mut x);
        let first = x[0];
        for (i, v) in x.iter().enumerate().skip(1) {
            assert_eq!(*v, first, "non-constant DC reconstruction at {i}");
        }
        assert_ne!(first, 0);
    }

    #[test]
    fn idct4_linearity_within_tolerance() {
        let a = [100i32, 50, -25, 10];
        let b = [-30i32, 40, 5, -12];
        let mut sum = [0i32; 4];
        for i in 0..4 {
            sum[i] = a[i] + b[i];
        }
        let mut ac = a;
        let mut bc = b;
        idct4(&mut ac);
        idct4(&mut bc);
        idct4(&mut sum);
        for i in 0..4 {
            let want = ac[i] + bc[i];
            let diff = (sum[i] - want).abs();
            assert!(diff <= 1, "linearity at {i}: diff {diff}");
        }
    }
}
